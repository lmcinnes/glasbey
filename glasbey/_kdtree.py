# MIT License
# Leland McInnes, Sergey Alexandrov

"""A minimal 3D k-d tree, replacing scikit-learn's NearestNeighbors for
glasbey's specific usage: build once from a fixed (n, 3) array (a color
grid), then repeatedly query single points for either the single nearest
neighbor or all neighbors within a radius. Every call site in _glasbey.py
and _optimizers.py fits this pattern -- none use k>1 neighbors, and only
one (the theme-palette annealing loop) batches multiple query points,
handled here by simply looping single-point queries.

This is not a general-purpose k-d tree. It exists to drop scikit-learn
(and its scipy dependency -- together the two heaviest packages in
glasbey's dependency tree, larger even than numba/llvmlite) as a runtime
dependency, while keeping the same asymptotic query behavior (O(log n) for
a 1-NN query, O(log n + k) for a radius query returning k results) that
made a tree structure worth using over a brute-force numpy scan in the
first place.

Design notes, from benchmarking against sklearn.neighbors.NearestNeighbors
on a realistic grid (grid_size=64 -> 262,144 points):

  - Leaves bucket LEAF_SIZE points rather than one point per node (the
    standard approach -- scipy.spatial.cKDTree and sklearn.neighbors.KDTree
    do the same, with comparable default leaf sizes). A naive one-point-
    per-node tree measured *slower* than a plain numpy brute-force scan
    for radius queries returning hundreds of points: with ~262k nodes,
    numpy's fixed per-call overhead on tiny 3-element operations dominated.
    Bucketing amortizes that overhead across a vectorized per-leaf
    distance computation instead.
  - Leaf size 256 was swept against 8-2048 and lands at a good balance of
    tree-construction cost, single-neighbor query cost, and multi-hundred-
    result radius query cost, ending up within roughly 1.5x of sklearn's
    compiled implementation end-to-end (build once + a couple hundred
    queries), and comfortably faster than a brute-force numpy scan at
    every leaf size tested down to 8.
"""
import numpy as np

LEAF_SIZE = 256


def _build(points, indices, depth):
    if len(indices) <= LEAF_SIZE:
        return indices
    axis = depth % 3
    vals = points[indices, axis]
    median = np.median(vals)
    left_mask = vals <= median
    # All points share the same value along this axis (or all fall on one
    # side) -- splitting further wouldn't shrink the partition, so stop
    # here rather than recursing forever.
    if left_mask.all() or (~left_mask).all():
        return indices
    left = _build(points, indices[left_mask], depth + 1)
    right = _build(points, indices[~left_mask], depth + 1)
    return (axis, median, left, right)


def _query_1nn(node, points, query, best_idx, best_d2):
    if isinstance(node, np.ndarray):
        leaf_pts = points[node]
        d2 = np.sum((leaf_pts - query) ** 2, axis=1)
        local_best = np.argmin(d2)
        if d2[local_best] < best_d2:
            return node[local_best], d2[local_best]
        return best_idx, best_d2

    axis, median, left, right = node
    q = query[axis]
    near, far = (left, right) if q <= median else (right, left)
    best_idx, best_d2 = _query_1nn(near, points, query, best_idx, best_d2)
    # Only descend into the far side if it could possibly contain a closer
    # point than what we've already found -- the whole point of a tree
    # over a brute-force scan.
    if (q - median) ** 2 < best_d2:
        best_idx, best_d2 = _query_1nn(far, points, query, best_idx, best_d2)
    return best_idx, best_d2


def _query_radius(node, points, query, r2, radius, out):
    if isinstance(node, np.ndarray):
        leaf_pts = points[node]
        d2 = np.sum((leaf_pts - query) ** 2, axis=1)
        out.append(node[d2 <= r2])
        return

    axis, median, left, right = node
    q = query[axis]
    if q - radius <= median:
        _query_radius(left, points, query, r2, radius, out)
    if q + radius > median:
        _query_radius(right, points, query, r2, radius, out)


class KDTree3D:
    """A k-d tree over a fixed set of 3D points, supporting exact
    single-nearest-neighbor and radius queries.

    Parameters
    ----------
    points : array of shape (n, 3)
        The points to index. Copied and cast to float64 internally.
    """

    def __init__(self, points):
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        indices = np.arange(self.points.shape[0])
        self._root = _build(self.points, indices, 0)

    def query_1nn(self, point):
        """Return the index of the point nearest to ``point``."""
        query = np.asarray(point, dtype=np.float64)
        idx, _ = _query_1nn(self._root, self.points, query, -1, np.inf)
        return int(idx)

    def query_1nn_batch(self, query_points):
        """Return an array of nearest-neighbor indices, one per row of
        ``query_points``."""
        query_points = np.asarray(query_points, dtype=np.float64)
        return np.array([self.query_1nn(q) for q in query_points], dtype=np.int64)

    def query_radius(self, point, radius):
        """Return the indices of all points within ``radius`` of
        ``point`` (unsorted)."""
        query = np.asarray(point, dtype=np.float64)
        out = []
        _query_radius(self._root, self.points, query, radius * radius, radius, out)
        if not out:
            return np.array([], dtype=np.int64)
        return np.concatenate(out)


def min_pairwise_distances(a, b):
    """For each row of ``a``, the Euclidean distance to its closest row in
    ``b``. Replaces sklearn.metrics.pairwise_distances(a, b).min(axis=1),
    the only pairwise_distances usage in this package -- always called on
    small arrays (a handful to a few hundred rows), so a full broadcasted
    distance matrix is cheap and the simplest correct approach.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a[:, None, :] - b[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    return np.sqrt(d2.min(axis=1))
