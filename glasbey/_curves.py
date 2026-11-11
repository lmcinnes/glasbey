"""Curve primitives in CAM02-UCS shared by continuous colormaps and theme arcs.

Conventions: colours are CAM02-UCS (J', a', b') unless a name says otherwise;
user bounds (lightness, chroma, hue) are CIECAM02 J, C and h.
"""

from dataclasses import dataclass

import numpy as np
from colorspacious import cspace_convert, CIECAM02Space
from scipy.integrate import trapezoid
from scipy.interpolate import BSpline, PchipInterpolator
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

from ._grids import rgb_grid, jch_grid


_FEASIBILITY_TOL = 1e-7


def _safe_convert(colors, start, end):
    # colorspacious can fail a whole batch because of a single undefined color.
    colors = np.asarray(colors, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        try:
            return cspace_convert(colors, start, end)
        except (ValueError, FloatingPointError, IndexError):
            if len(colors) <= 1:
                return np.full_like(colors, np.nan)
            half = len(colors) // 2
            return np.vstack([
                _safe_convert(colors[:half], start, end),
                _safe_convert(colors[half:], start, end),
            ])


def _cam02ucs_to_jch(colors):
    # Closed-form inverse of colorspacious's CAM02-UCS; its XYZ round trip fails off-gamut.
    colors = np.asarray(colors, dtype=float)
    lightness, a, b = colors[..., 0], colors[..., 1], colors[..., 2]
    colorfulness = np.expm1(0.0228 * np.hypot(a, b)) / 0.0228
    return np.stack([
        lightness / (1.7 - 0.007 * lightness),
        colorfulness / CIECAM02Space.sRGB.F_L**0.25,
        np.degrees(np.arctan2(b, a)) % 360,
    ], axis=-1)


def _color_spaces(colors):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return _safe_convert(colors, "CAM02-UCS", "sRGB1"), _cam02ucs_to_jch(colors)


def _fast_color_spaces(colors):
    from ._fastcolor import cam02ucs_to_srgb1

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return cam02ucs_to_srgb1(colors), _cam02ucs_to_jch(colors)


@dataclass
class _Domain:
    lightness_bounds: object = None
    chroma_bounds: object = None
    hue_bounds: tuple = (0.0, 360.0)
    rgb_bounds: tuple = ((0.0, 1.0),) * 3
    # Optional extra margins, a callable mapping (colors, jch) to an (m, k) array.
    extra_margins: object = None
    # Use the closed-form NumPy conversion instead of colorspacious.
    fast: bool = False

    def margins(self, colors):
        colors = np.atleast_2d(colors)
        rgb, jch = (_fast_color_spaces if self.fast else _color_spaces)(colors)
        bounds = np.asarray(self.rgb_bounds)
        margins = [rgb - bounds[:, 0], bounds[:, 1] - rgb]
        for channel, limits in enumerate((self.lightness_bounds, self.chroma_bounds)):
            if limits is not None:
                margins.extend([
                    (jch[:, channel:channel + 1] - limits[0]) / 100,
                    (limits[1] - jch[:, channel:channel + 1]) / 100,
                ])
        if self.hue_bounds != (0.0, 360.0):
            width = (self.hue_bounds[1] - self.hue_bounds[0]) % 360
            middle = self.hue_bounds[0] + width / 2
            angular = np.cos(np.deg2rad(jch[:, 2] - middle)) - np.cos(
                np.deg2rad(width / 2)
            )
            angular[jch[:, 1] < 1e-7] = 1
            margins.append(angular[:, None])
        if self.extra_margins is not None:
            margins.append(np.atleast_2d(self.extra_margins(colors, jch)).reshape(len(colors), -1))
        result = np.column_stack(margins)
        result[~np.isfinite(colors).all(axis=1)] = -1e6
        return np.nan_to_num(result, nan=-1e6, posinf=-1e6, neginf=-1e6)

    def contains(self, colors):
        return np.all(self.margins(colors) >= -_FEASIBILITY_TOL, axis=1)


class _Arc:
    """A spline parameterized by its numerical perceptual arc length."""

    def __init__(self, spline, anchor_parameters):
        self.spline = spline
        self.anchor_parameters = np.asarray(anchor_parameters)
        self.parameters = np.unique(np.concatenate([
            np.linspace(0, 1, 2049), spline.t, self.anchor_parameters,
        ]))
        left = self.parameters[:-1]
        right = self.parameters[1:]
        nodes, weights = np.polynomial.legendre.leggauss(8)
        locations = (left[:, None] + right[:, None]) / 2 + (
            right - left
        )[:, None] * nodes / 2
        speed = np.linalg.norm(spline.derivative()(locations), axis=-1)
        lengths = (speed @ weights) * (right - left) / 2
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        self.length = float(cumulative[-1])
        if not np.isfinite(self.length) or self.length <= 1e-10:
            raise ValueError("A colormap arm must have nonzero perceptual length")
        self.fractions = cumulative / self.length
        if np.any(np.diff(self.fractions) <= 0):
            raise ValueError("A colormap arm cannot contain a constant interval")
        self.anchor_positions = np.interp(
            self.anchor_parameters, self.parameters, self.fractions
        )
        self._inverse = PchipInterpolator(self.fractions, self.parameters)

    def __call__(self, positions):
        return self.spline(self._inverse(positions))


def _chroma_reward(colors, fractions, chroma_scale):
    chroma = _cam02ucs_to_jch(colors)[:, 1]
    if not np.isfinite(chroma).all():
        raise ValueError("Chroma is undefined for a candidate color")
    weight = 16 * fractions**2 * (1 - fractions)**2
    return float(trapezoid(weight * chroma, fractions) / (
        max(trapezoid(weight, fractions), 1e-12) * chroma_scale
    ))


class _Grid:
    def __init__(self, domain, grid_size, grid_space, seeds):
        if grid_space == "RGB":
            colors = rgb_grid(
                grid_size, *domain.rgb_bounds, output_colorspace="CAM02-UCS"
            )
        else:
            pilot = rgb_grid(16, *domain.rgb_bounds, output_colorspace="JCh")
            seed_jch = cspace_convert(seeds, "CAM02-UCS", "JCh")
            pilot = np.vstack([pilot, seed_jch])
            lightness = domain.lightness_bounds or (
                max(1e-6, float(pilot[:, 0].min())), float(pilot[:, 0].max())
            )
            chroma = domain.chroma_bounds or (0, float(pilot[:, 1].max()) * 1.05)
            with np.errstate(invalid="ignore", divide="ignore"):
                colors = jch_grid(
                    grid_size, lightness, chroma, domain.hue_bounds,
                    output_colorspace="CAM02-UCS",
                )
        cam = colors[domain.contains(colors)]
        self._set(np.unique(np.vstack([cam, seeds]), axis=0), domain)

    def _set(self, colors, domain, chroma=None):
        self.colors = colors
        self.chroma = (
            cspace_convert(self.colors, "CAM02-UCS", "JCh")[:, 1] if chroma is None else chroma
        )
        self.chroma_scale = max(1.0, float(self.chroma.max())) if len(self.chroma) else 1.0
        self.domain = domain

    @classmethod
    def from_colors(cls, domain, colors, filter=True):
        """Build a grid from prebuilt CAM02-UCS colours, optionally filtered by the domain."""
        grid = cls.__new__(cls)
        colors = np.asarray(colors, dtype=float)
        if filter and len(colors):
            colors = colors[domain.contains(colors)]
        grid._set(np.unique(colors, axis=0) if len(colors) else colors.reshape(0, 3), domain,
                  _cam02ucs_to_jch(colors)[:, 1] if len(colors) else np.zeros(0))
        return grid

    def restricted(self, domain, extra_colors=None):
        """A view of this grid's colours inside a (sub)domain; no grid is rebuilt."""
        keep = domain.contains(self.colors) if len(self.colors) else np.zeros(0, bool)
        colors = self.colors[keep]
        chroma = self.chroma[keep]
        if extra_colors is not None and len(extra_colors):
            extra = np.atleast_2d(np.asarray(extra_colors, dtype=float))
            colors = np.vstack([colors, extra])
            chroma = np.concatenate([chroma, _cam02ucs_to_jch(extra)[:, 1]])
        grid = type(self).__new__(type(self))
        grid._set(colors, domain, chroma)
        return grid

    def connect(self, start, end):
        fraction = np.linspace(0, 1, 65)[:, None]
        if self.domain.contains(start + fraction * (end - start)).all():
            return np.vstack([start, end])
        direction = 1 if end[0] >= start[0] else -1
        eligible = self.colors[
            (self.colors[:, 0] >= min(start[0], end[0]) - 1e-8)
            & (self.colors[:, 0] <= max(start[0], end[0]) + 1e-8)
        ]
        for limit in (4096, 16384):
            subset = eligible[np.linspace(
                0, max(len(eligible) - 1, 0), min(limit, len(eligible)), dtype=int
            )]
            points = np.vstack([start, end, subset])
            k = min(24, len(points))
            # Layered JCh grids need lightness-compressed neighbours to link layers.
            neighbors = np.hstack([
                NearestNeighbors(n_neighbors=k).fit(points * scale).kneighbors(
                    points * scale, return_distance=False
                )
                for scale in (np.ones(3), np.array([0.2, 1.0, 1.0]))
            ])
            sources = np.repeat(np.arange(len(points)), neighbors.shape[1])
            targets = neighbors.ravel()
            sources, targets = np.unique(np.column_stack([sources, targets]), axis=0).T
            weights = np.maximum(np.linalg.norm(points[targets] - points[sources], axis=1), 1e-10)
            allowed = (sources != targets) & (
                direction * (points[targets, 0] - points[sources, 0]) >= -1e-9
            )
            sources, targets, weights = sources[allowed], targets[allowed], weights[allowed]
            coarse = np.linspace(0, 1, 9)[None, :, None]
            segments = points[sources, None] + coarse * (
                points[targets] - points[sources]
            )[:, None]
            valid = self.domain.contains(segments.reshape(-1, 3)).reshape(len(sources), -1).all(axis=1)
            graph = csr_matrix(
                (weights[valid], (sources[valid], targets[valid])),
                shape=(len(points), len(points)),
            )
            for attempt in range(32):
                lengths, predecessors = dijkstra(
                    graph, indices=0, return_predecessors=True
                )
                if not np.isfinite(lengths[1]):
                    break
                route = [1]
                while route[-1] != 0:
                    route.append(int(predecessors[route[-1]]))
                route.reverse()
                invalid = []
                for source, target in zip(route[:-1], route[1:]):
                    segment = points[source] + fraction * (points[target] - points[source])
                    if not self.domain.contains(segment).all():
                        invalid.append((source, target))
                if not invalid:
                    return points[route]
                for source, target in invalid:
                    graph[source, target] = 0
                graph.eliminate_zeros()
        raise ValueError(
            "No feasible monotone route found between seeds; try a finer grid "
            "or different seeds/bounds. This is a search failure, not proof of infeasibility."
        )

    def waypoints(self, anchors):
        pieces = [anchors[:1]]
        indices = [0]
        for start, end in zip(anchors[:-1], anchors[1:]):
            piece = self.connect(start, end)[1:]
            pieces.append(piece)
            indices.append(indices[-1] + len(piece))
        points = np.vstack(pieces)
        lengths = np.concatenate([[0], np.cumsum(np.linalg.norm(
            np.diff(points, axis=0), axis=1
        ))])
        parameters = lengths / lengths[-1]
        return points, parameters, parameters[indices]


def _validate_curve(spline, domain, anchor_parameters, samples=129):
    parameters = np.unique(np.concatenate([
        np.linspace(0, 1, samples), spline.t, anchor_parameters,
    ]))
    for depth in range(10):
        colors = spline(parameters)
        margins = domain.margins(colors).min(axis=1)
        if margins.min() < -_FEASIBILITY_TOL:
            return False, parameters[margins < -_FEASIBILITY_TOL], depth
        midpoints = (parameters[:-1] + parameters[1:]) / 2
        middle_colors = spline(midpoints)
        middle_margins = domain.margins(middle_colors).min(axis=1)
        if middle_margins.min() < -_FEASIBILITY_TOL:
            return False, midpoints[middle_margins < -_FEASIBILITY_TOL], depth
        error = np.linalg.norm(middle_colors - (colors[:-1] + colors[1:]) / 2, axis=1)
        refine = (error > 1e-3) | (
            (np.minimum(margins[:-1], margins[1:]) < 1e-4) & (error > 1e-6)
        )
        if not refine.any():
            return True, np.array([]), depth
        parameters = np.unique(np.concatenate([parameters, midpoints[refine]]))
    return False, midpoints[refine], depth


# Solver profiles. "default" is what create_colormap has always used.
_PROFILES = {
    "default": dict(mesh_points=None, refinements=4, validation_samples=129, quadrature=8),
    # Batched finite differences with explicit Jacobians; extras must accept a leading batch axis.
    "fast": dict(mesh_points=65, refinements=3, validation_samples=65, quadrature=4),
}


class _SplineProblem:
    """A cubic B-spline fit in CAM02-UCS with domain constraints.

    Anchors with weight ``None`` are pinned exactly through a null-space
    parameterisation of the coefficients; anchors with a finite weight add a
    quadratic penalty instead. Any number of pinned anchors (including a single
    interior one, or none) is supported.

    ``extra_objective(problem, variables, geometry)`` and
    ``extra_constraints(problem, variables, geometry)`` may be passed to
    :meth:`solve` to add terms without forking the solver; ``geometry`` is the
    tuple returned by :meth:`geometry`.
    """

    def __init__(self, anchors, points, parameters, anchor_parameters, count, domain,
                 *, anchor_weights=None, direction=None, scale=None,
                 monotone_lightness=True, profile="default", chroma_profile="mid"):
        anchors = np.atleast_2d(np.asarray(anchors, dtype=float))
        anchor_parameters = np.asarray(anchor_parameters, dtype=float)
        if profile not in _PROFILES:
            raise ValueError(f"Unknown solver profile {profile!r}")
        self.profile = profile
        self._settings = _PROFILES[profile]
        # "mid" rewards chroma around the arc's middle; "uniform" along its whole length;
        # a callable maps J' to a [0, 1] lightness profile (fast profile only).
        self.chroma_profile = chroma_profile
        if len(anchor_parameters) >= 2 and anchor_parameters[0] == 0 and anchor_parameters[-1] == 1:
            interior = list(anchor_parameters[1:-1])
        else:
            interior = [value for value in anchor_parameters if 0 < value < 1]
        while len(interior) < count - 4:
            edges = np.array([0] + sorted(interior) + [1])
            widest = int(np.argmax(np.diff(edges)))
            interior.append((edges[widest] + edges[widest + 1]) / 2)
        self.knots = np.array([0.0] * 4 + sorted(interior) + [1.0] * 4)
        self.basis = BSpline(self.knots, np.eye(count), 3)
        self.anchor_parameters = anchor_parameters
        self.anchors = anchors
        self.domain = domain
        self.monotone_lightness = monotone_lightness
        if direction is None:
            direction = 1 if anchors[-1, 0] >= anchors[0, 0] else -1
        self.direction = direction
        if anchor_weights is None:
            anchor_weights = [None] * len(anchors)
        pinned = np.array([weight is None for weight in anchor_weights], dtype=bool)
        self.soft_weights = np.array(
            [0.0 if weight is None else float(weight) for weight in anchor_weights]
        )[~pinned]
        self.soft_anchors = anchors[~pinned]
        self.soft_matrix = (
            self.basis(anchor_parameters[~pinned]) if (~pinned).any() else np.zeros((0, count))
        )
        if pinned.all():
            anchor_matrix = self.basis(anchor_parameters)
            self.fixed = np.linalg.lstsq(anchor_matrix, anchors, rcond=None)[0]
            self.free = null_space(anchor_matrix)
        elif pinned.any():
            anchor_matrix = self.basis(anchor_parameters[pinned])
            self.fixed = np.linalg.lstsq(anchor_matrix, anchors[pinned], rcond=None)[0]
            self.free = null_space(anchor_matrix)
        else:
            self.fixed = np.zeros((count, 3))
            self.free = np.eye(count)
        if scale is None:
            scale = max(1.0, float(np.linalg.norm(np.diff(anchors, axis=0), axis=1).sum()))
        self.scale = scale
        fitting_parameters = np.unique(np.concatenate([parameters, np.linspace(0, 1, 129)]))
        target = np.column_stack([
            np.interp(fitting_parameters, parameters, points[:, channel])
            for channel in range(3)
        ])
        self.initial = self.fit_variables(fitting_parameters, target)
        mesh_points = self._settings["mesh_points"] or max(129, 16 * count)
        self.mesh = np.unique(np.concatenate([
            np.linspace(0, 1, mesh_points), self.knots, anchor_parameters,
        ]))
        self._update_basis()

    def fit_variables(self, fitting_parameters, target):
        """Least-squares variables whose spline approximates ``target`` at the parameters."""
        fitting_basis = self.basis(fitting_parameters)
        return np.linalg.lstsq(
            fitting_basis @ self.free,
            (target - fitting_basis @ self.fixed) / self.scale,
            rcond=None,
        )[0].ravel()

    def _update_basis(self):
        self.values = self.basis(self.mesh)
        # Interior safety margin stops the solver hugging bounds between mesh points.
        to_anchor = np.min(np.abs(self.mesh[:, None] - self.anchor_parameters[None]), axis=1)
        self.safety = 1e-4 * np.clip(to_anchor / 0.02, 0, 1)[:, None]
        self.first = self.basis.derivative()(self.mesh)
        self.second = self.basis.derivative(2)(self.mesh)
        nodes, weights = np.polynomial.legendre.leggauss(self._settings["quadrature"])
        half_width = np.diff(self.mesh) / 2
        locations = (self.mesh[:-1] + half_width)[:, None] + half_width[:, None] * nodes
        self.quadrature_weights = half_width[:, None] * weights
        self.quadrature_first = self.basis.derivative()(locations)
        self.quadrature_second = self.basis.derivative(2)(locations)
        self._q1 = np.ascontiguousarray(self.quadrature_first.reshape(-1, self.quadrature_first.shape[-1]))
        self._q2 = np.ascontiguousarray(self.quadrature_second.reshape(-1, self.quadrature_second.shape[-1]))

    def coefficients(self, variables):
        return self.fixed + self.scale * self.free @ variables.reshape(-1, 3)

    def spline(self, variables):
        return BSpline(self.knots, self.coefficients(variables), 3)

    def geometry(self, variables):
        coefficients = self.coefficients(variables)
        colors = self.values @ coefficients
        velocity = self.quadrature_first @ coefficients
        acceleration = self.quadrature_second @ coefficients
        speed = np.maximum(np.linalg.norm(velocity, axis=-1), 1e-9)
        cumulative = np.concatenate([[0], np.cumsum(
            np.sum(speed * self.quadrature_weights, axis=1)
        )])
        length = max(float(cumulative[-1]), 1e-9)
        bending = np.sum(
            np.sum(np.cross(velocity, acceleration)**2, axis=-1) / speed**5
            * self.quadrature_weights
        )
        return colors, length, cumulative / length, bending

    def soft_penalty(self, variables):
        if not len(self.soft_weights):
            return 0.0
        residual = self.soft_matrix @ self.coefficients(variables) - self.soft_anchors
        return float(np.sum(self.soft_weights * np.sum(residual**2, axis=1)) / self.scale**2)

    # ------------------------------------------------------------ batched (fast profile)

    def coefficients_batch(self, variables):
        """Coefficients for a batch of variable vectors, shape (B, count, 3)."""
        variables = np.asarray(variables).reshape(len(variables), -1, 3)
        return self.fixed + self.scale * np.matmul(self.free, variables)

    def geometry_batch(self, variables):
        coefficients = self.coefficients_batch(variables)
        batch = len(coefficients)
        colors = np.matmul(self.values, coefficients)
        intervals, nodes, count = self.quadrature_first.shape
        velocity = np.matmul(self._q1, coefficients).reshape(batch, intervals, nodes, 3)
        acceleration = np.matmul(self._q2, coefficients).reshape(batch, intervals, nodes, 3)
        speed = np.maximum(np.sqrt(np.sum(velocity**2, axis=-1)), 1e-9)
        cumulative = np.concatenate([
            np.zeros((batch, 1)), np.cumsum(np.sum(speed * self.quadrature_weights, axis=2), axis=1),
        ], axis=1)
        length = np.maximum(cumulative[:, -1], 1e-9)
        v, a = velocity, acceleration
        cross = np.stack([
            v[..., 1] * a[..., 2] - v[..., 2] * a[..., 1],
            v[..., 2] * a[..., 0] - v[..., 0] * a[..., 2],
            v[..., 0] * a[..., 1] - v[..., 1] * a[..., 0],
        ], axis=-1)
        bending = np.sum(np.sum(cross**2, axis=-1) / speed**5 * self.quadrature_weights, axis=(1, 2))
        return colors, length, cumulative / length[:, None], bending, coefficients

    def _solve_batched(self, initial, max_iter, smoothness, chroma_weight, chroma_scale,
                       reference_length, arc_length_factor, length_weight, extra_objective,
                       extra_constraints):
        epsilon = np.sqrt(np.finfo(float).eps)

        def objective_batch(batch):
            colors, length, fractions, bending, coefficients = self.geometry_batch(batch)
            geometry = (colors, length, fractions, bending)
            reference = reference_length or self.scale
            value = smoothness * reference * bending / (4 * np.pi**2)
            if length_weight:
                value = value + length_weight * (
                    length / reference if reference_length is None else -length / reference
                )
            with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                if chroma_weight:
                    chroma = _cam02ucs_to_jch(colors)[..., 1]
                    if callable(self.chroma_profile):
                        # Signed weight from a lightness profile in [0, 1]: chroma is
                        # rewarded where the profile is high and penalised where it is low.
                        weight = 2 * self.chroma_profile(colors[..., 0]) - 1
                        reward = trapezoid(weight * chroma, fractions, axis=-1) / chroma_scale
                    else:
                        if self.chroma_profile == "uniform":
                            weight = np.ones_like(fractions)
                        else:
                            weight = 16 * fractions**2 * (1 - fractions)**2
                        reward = trapezoid(weight * chroma, fractions, axis=-1) / (
                            np.maximum(trapezoid(weight, fractions, axis=-1), 1e-12) * chroma_scale
                        )
                    value = value - chroma_weight * reward
                sampled = colors[:, ::8]
                separation = np.sqrt(np.sum((sampled[:, :, None] - sampled[:, None, :])**2, axis=-1))
                along = np.abs(fractions[:, ::8, None] - fractions[:, None, ::8]) * length[:, None, None]
                mask = along > length[:, None, None] * 0.2
                ratio = np.where(mask, separation / np.where(mask, along, 1.0), 1.0)
                penalty = np.where(mask, np.square(np.maximum(0, 0.2 - ratio)), 0.0)
                counts = mask.sum(axis=(1, 2))
                value = value + np.where(counts > 0, penalty.sum(axis=(1, 2)) / np.maximum(counts, 1), 0.0)
                if len(self.soft_weights):
                    residual = np.matmul(self.soft_matrix, coefficients) - self.soft_anchors
                    value = value + np.sum(self.soft_weights * np.sum(residual**2, axis=-1), axis=-1) / self.scale**2
                if extra_objective is not None:
                    value = value + extra_objective(self, batch, geometry)
            return np.where(np.isfinite(value), value, 1e12)

        def constraints_batch(batch, safety=True):
            colors, length, fractions, bending, coefficients = self.geometry_batch(batch)
            geometry = (colors, length, fractions, bending)
            count = len(batch)
            margins = self.domain.margins(colors.reshape(-1, 3)).reshape(count, colors.shape[1], -1)
            if safety:
                margins = margins - self.safety[None]
            result = [margins.reshape(count, -1)]
            if self.monotone_lightness:
                result.append(self.direction * np.diff(coefficients[:, :, 0], axis=1) / 100)
            if reference_length is not None:
                result.append((arc_length_factor - length / reference_length)[:, None])
            if extra_constraints is not None:
                result.append(np.asarray(extra_constraints(self, batch, geometry)).reshape(count, -1))
            return np.concatenate(result, axis=1)

        # SLSQP, its callback and the finite differences revisit the same points.
        objective_memo, constraint_memo = {}, {}

        def remember(memo, key, value):
            if len(memo) > 64:
                memo.clear()
            memo[key] = value
            return value

        def raw_constraints(stack):
            return constraints_batch(stack, safety=False)

        def perturbed(variables):
            steps = epsilon * np.maximum(1.0, np.abs(variables))
            return variables[None] + np.diag(steps), steps

        def objective(variables):
            key = (len(self.mesh), variables.tobytes())
            if key not in objective_memo:
                remember(objective_memo, key, float(objective_batch(variables[None])[0]))
            return objective_memo[key]

        def gradient(variables):
            stack, steps = perturbed(variables)
            values = objective_batch(np.vstack([variables[None], stack]))
            remember(objective_memo, (len(self.mesh), variables.tobytes()), float(values[0]))
            return (values[1:] - values[0]) / steps

        def safety_offsets():
            # Safety margins only apply to the domain-margin block of the constraint vector.
            margins = self.domain.margins(self.values[:1] @ self.fixed).shape[1]
            offsets = np.zeros(constraint_size())
            offsets[: self.values.shape[0] * margins] = np.repeat(self.safety[:, 0], margins)
            return offsets

        size_cache = {}

        def constraint_size():
            return size_cache["size"]

        def raw(variables):
            key = (len(self.mesh), variables.tobytes())
            if key not in constraint_memo:
                values = raw_constraints(variables[None])[0]
                size_cache["size"] = len(values)
                remember(constraint_memo, key, values)
            return constraint_memo[key]

        def constraints(variables, safety=True):
            values = raw(variables)
            if not safety:
                return values
            mesh_size = len(self.mesh)
            if size_cache.get("mesh") != mesh_size:
                size_cache["mesh"] = mesh_size
                size_cache["offsets"] = safety_offsets()
            return values - size_cache["offsets"]

        def jacobian(variables):
            stack, steps = perturbed(variables)
            values = raw_constraints(np.vstack([variables[None], stack]))
            size_cache["size"] = values.shape[1]
            remember(constraint_memo, (len(self.mesh), variables.tobytes()), values[0])
            return ((values[1:] - values[0]) / steps[:, None]).T

        return self._run_slsqp(initial, max_iter, objective, constraints, reference_length,
                               arc_length_factor, gradient, jacobian)

    def _run_slsqp(self, initial, max_iter, objective, constraints, reference_length,
                   arc_length_factor, gradient=None, jacobian=None):
        candidates = []

        def retain(variables):
            if constraints(variables, False).min() >= -_FEASIBILITY_TOL:
                candidates.append((objective(variables), variables.copy()))
                candidates.sort(key=lambda item: item[0])
                del candidates[8:]

        samples = self._settings["validation_samples"]
        current = initial
        retain(current)
        remaining = max_iter
        result = None
        for refinement in range(self._settings["refinements"]):
            constraint = {"type": "ineq", "fun": constraints}
            if jacobian is not None:
                constraint["jac"] = jacobian
            result = minimize(
                objective, current, method="SLSQP", jac=gradient,
                constraints=constraint, callback=retain,
                options={"maxiter": remaining, "ftol": 1e-9},
            )
            current = result.x
            remaining -= max(1, result.nit)
            valid, extra, depth = _validate_curve(
                self.spline(current), self.domain, self.anchor_parameters, samples
            )
            if valid and constraints(current, False).min() >= -_FEASIBILITY_TOL:
                arc = _Arc(self.spline(current), self.anchor_parameters)
                if reference_length is None or arc.length <= (
                    arc_length_factor * reference_length + 1e-5
                ):
                    return current, arc, result, depth
            if remaining <= 0:
                break
            interval = np.clip(np.searchsorted(self.mesh, extra), 1, len(self.mesh) - 1)
            neighbours = np.unique(np.concatenate([interval - 1, interval, np.minimum(interval + 1, len(self.mesh) - 1)]))
            neighbours = neighbours[neighbours > 0]
            self.mesh = np.unique(np.concatenate([
                self.mesh, extra, (self.mesh[neighbours - 1] + self.mesh[neighbours]) / 2,
            ]))
            self._update_basis()
        for score, candidate in sorted(candidates, key=lambda item: item[0]):
            valid, extra, depth = _validate_curve(
                self.spline(candidate), self.domain, self.anchor_parameters, samples
            )
            if valid and constraints(candidate, False).min() >= -_FEASIBILITY_TOL:
                arc = _Arc(self.spline(candidate), self.anchor_parameters)
                if reference_length is None or arc.length <= arc_length_factor * reference_length + 1e-5:
                    if result is not None:
                        result.success = False
                        result.message = "Retained a feasible iterate; " + str(result.message)
                    return candidate, arc, result, depth
        return None

    def solve(self, initial, max_iter, smoothness=1.0, chroma_weight=0.0,
              chroma_scale=1.0, reference_length=None, arc_length_factor=1.5,
              *, length_weight=1.0, extra_objective=None, extra_constraints=None):
        if self.profile == "fast":
            return self._solve_batched(
                initial, max_iter, smoothness, chroma_weight, chroma_scale, reference_length,
                arc_length_factor, length_weight, extra_objective, extra_constraints,
            )

        def objective(variables):
            colors, length, fractions, bending = geometry = self.geometry(variables)
            reference = reference_length or self.scale
            value = smoothness * reference * bending / (4 * np.pi**2)
            if length_weight == 1.0:
                value += length / reference if reference_length is None else -length / reference
            elif length_weight:
                value += length_weight * (
                    length / reference if reference_length is None else -length / reference
                )
            if chroma_weight:
                with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                    try:
                        reward = _chroma_reward(colors, fractions, chroma_scale)
                    except (ValueError, FloatingPointError, IndexError):
                        return 1e12
                value -= chroma_weight * reward
            sampled = colors[::8]
            separation = np.linalg.norm(sampled[:, None] - sampled[None, :], axis=-1)
            along = np.abs(fractions[::8, None] - fractions[None, ::8]) * length
            nonlocal_pairs = along > length * 0.2
            if nonlocal_pairs.any():
                value += np.square(np.maximum(
                    0, 0.2 - separation[nonlocal_pairs] / along[nonlocal_pairs]
                )).mean()
            if len(self.soft_weights):
                value += self.soft_penalty(variables)
            if extra_objective is not None:
                with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                    value += extra_objective(self, variables, geometry)
            return float(value) if np.isfinite(value) else 1e12

        def constraints(variables, safety=True):
            coefficients = self.coefficients(variables)
            margins = self.domain.margins(self.values @ coefficients)
            result = [(margins - self.safety if safety else margins).ravel()]
            if self.monotone_lightness:
                result.append(self.direction * np.diff(coefficients[:, 0]) / 100)
            geometry = None
            if reference_length is not None:
                geometry = self.geometry(variables)
                result.append(np.array([
                    arc_length_factor - geometry[1] / reference_length
                ]))
            if extra_constraints is not None:
                if geometry is None:
                    geometry = self.geometry(variables)
                result.append(np.ravel(extra_constraints(self, variables, geometry)))
            return np.concatenate(result)

        candidates = []

        def retain(variables):
            if constraints(variables, False).min() >= -_FEASIBILITY_TOL:
                candidates.append((objective(variables), variables.copy()))
                candidates.sort(key=lambda item: item[0])
                del candidates[8:]

        samples = self._settings["validation_samples"]
        current = initial
        retain(current)
        remaining = max_iter
        for refinement in range(self._settings["refinements"]):
            result = minimize(
                objective, current, method="SLSQP",
                constraints={"type": "ineq", "fun": constraints},
                callback=retain,
                options={"maxiter": remaining, "ftol": 1e-9},
            )
            current = result.x
            remaining -= max(1, result.nit)
            valid, extra, depth = _validate_curve(
                self.spline(current), self.domain, self.anchor_parameters, samples
            )
            if valid and constraints(current, False).min() >= -_FEASIBILITY_TOL:
                arc = _Arc(self.spline(current), self.anchor_parameters)
                if reference_length is None or arc.length <= (
                    arc_length_factor * reference_length + 1e-5
                ):
                    return current, arc, result, depth
            if remaining <= 0:
                break
            # Refine only around detected violations to keep the constraint count bounded.
            interval = np.clip(np.searchsorted(self.mesh, extra), 1, len(self.mesh) - 1)
            neighbours = np.unique(np.concatenate([interval - 1, interval, np.minimum(interval + 1, len(self.mesh) - 1)]))
            neighbours = neighbours[neighbours > 0]
            self.mesh = np.unique(np.concatenate([
                self.mesh, extra, (self.mesh[neighbours - 1] + self.mesh[neighbours]) / 2,
            ]))
            self._update_basis()
        for score, candidate in sorted(candidates, key=lambda item: item[0]):
            valid, extra, depth = _validate_curve(
                self.spline(candidate), self.domain, self.anchor_parameters, samples
            )
            if valid and constraints(candidate, False).min() >= -_FEASIBILITY_TOL:
                arc = _Arc(self.spline(candidate), self.anchor_parameters)
                if reference_length is None or arc.length <= arc_length_factor * reference_length + 1e-5:
                    result.success = False
                    result.message = "Retained a feasible iterate; " + str(result.message)
                    return candidate, arc, result, depth
        return None
