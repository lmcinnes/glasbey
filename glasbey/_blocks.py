"""Block palettes built from constrained theme arcs.

Pipeline: plan -> select bases -> allocate hue sweep -> refine -> fit -> polish.
Stages before fitting work on cheap *footprints*: small point sets predicting
where a block will lie in CAM02-UCS.

Naming: ``jp`` is CAM02-UCS lightness J'; ``lightness`` / ``J`` is CIECAM02 J.
"""

from dataclasses import dataclass, field
import time

import numpy as np
from colorspacious import cspace_convert
from scipy.spatial import cKDTree

from ._curves import _Domain, _Grid, _cam02ucs_to_jch
from ._fastcolor import cam02ucs_to_srgb1
from ._grids import rgb_grid, jch_grid
from ._theme import (
    DEFAULT_STEP, GREY_CHROMA, LIGHTNESS_DISCOUNT, MIN_STEP, _sample_window, chroma_to_colorfulness,
    discounted_distance, fit_theme_arc, gamut_table, hue_offset,
)
from ._converters import get_rgb_palette

__all__ = [
    "BlockSpec", "Footprint", "discounted_distance", "plan_blocks", "select_bases",
    "build_block_palette",
]

# Tunable defaults (prototype values; see the implementation plan).
LONG_BLOCK_MIN = 6
MID_BAND = (45.0, 65.0)
MID_PENALTY = 0.5
SEP_MIN = 6.0
CROWDING_WIDEN = 10.0
HELIX_POINTS = 9
HELIX_HUE_STEP = 2.0
HELIX_COLORFULNESS = 0.85
LONG_BASE_SCALE = 0.95
SHORT_POINTS = 5
SHORT_LIGHTNESS_SPREAD = 0.45
SHORT_TAPER = 0.3
SHORT_CANDIDATE_LIGHTNESS = (30.0, 80.0)
SHORT_CANDIDATE_CHROMA = (25.0, 70.0)
SHORT_CANDIDATE_GRID = 24
MAX_SWEEP = 150.0
LONG_ENDPOINT_WEIGHT = 2.0


def jp_from_j(lightness):
    """CIECAM02 J to CAM02-UCS J'."""
    lightness = np.asarray(lightness, dtype=float)
    return 1.7 * lightness / (1 + 0.007 * lightness)


def j_from_jp(jp):
    jp = np.asarray(jp, dtype=float)
    return jp / (1.7 - 0.007 * jp)


@dataclass
class BlockSpec:
    """Planning information for one block."""

    index: int
    size: int
    kind: str  # "long", "short" or "point"
    length: float  # target CAM02-UCS length (n - 1) * step
    step: float
    demand: float = 0.0  # wanted hue sweep in degrees
    lightness_span: float = 0.0  # reachable J' extent
    non_lightness: float = 0.0  # wanted a'b' length L_ab


@dataclass
class Footprint:
    """Predicted CAM02-UCS points of a block plus the parameters that made them."""

    points: np.ndarray
    base: np.ndarray
    hue: float
    sweep: float
    twist: int
    kind: str
    lightness: float = 0.0  # CIECAM02 J of the footprint centre


@dataclass
class _Context:
    """Shared, per-call state: bounds, grid, gamut table and CVD settings."""

    domain: _Domain
    gamut: object
    jp_bounds: tuple
    lightness_bounds: tuple
    chroma_bounds: tuple
    hue_bounds: tuple
    grid: object = None
    colorblind_safe: bool = False
    cvd_space: dict = None
    hue_sweep: object = "auto"
    diagnostics: dict = field(default_factory=dict)
    grid_space: str = "RGB"

    def cvd(self, points):
        points = np.asarray(points, dtype=float)
        shape = points.shape
        rgb = np.clip(np.nan_to_num(cam02ucs_to_srgb1(points.reshape(-1, 3))), 0, 1)
        simulated = cspace_convert(rgb, self.cvd_space, "CAM02-UCS")
        return simulated.reshape(shape)


def make_context(lightness_bounds=(10, 90), chroma_bounds=(0, 80), hue_bounds=(0, 360),
                 rgb_bounds=((0.0, 1.0),) * 3, *, grid_size=32, grid_space="RGB",
                 colorblind_safe=False, cvd_type="deuteranomaly", cvd_severity=50.0,
                 hue_sweep="auto"):
    lightness_bounds = tuple(float(v) for v in lightness_bounds)
    chroma_bounds = tuple(float(v) for v in chroma_bounds)
    hue_bounds = tuple(float(v) for v in hue_bounds)
    rgb_bounds = tuple(tuple(float(v) for v in pair) for pair in rgb_bounds)
    if hue_bounds == (0.0, 360.0) or (hue_bounds[1] - hue_bounds[0]) % 360 == 0:
        hue_bounds = (0.0, 360.0)
    domain = _Domain(lightness_bounds, chroma_bounds, hue_bounds, rgb_bounds, fast=True)
    grid = _Grid(domain, grid_size, grid_space, np.zeros((0, 3)))
    if len(grid.colors) < 8:
        raise ValueError("The colour bounds leave too few representable colours")
    context = _Context(
        domain, gamut_table(lightness_bounds, chroma_bounds, rgb_bounds),
        tuple(float(v) for v in jp_from_j(lightness_bounds)), lightness_bounds, chroma_bounds,
        hue_bounds, grid, colorblind_safe,
        {"name": "sRGB1+CVD", "cvd_type": cvd_type, "severity": cvd_severity},
        hue_sweep, grid_space=grid_space,
    )
    return context


# ---------------------------------------------------------------- planning


def _mean_colorfulness(context, jp_low, jp_high):
    jp = np.linspace(jp_low, jp_high, 9)
    hues = np.arange(0, 360, 10.0)
    if context.hue_bounds != (0.0, 360.0):
        width = (context.hue_bounds[1] - context.hue_bounds[0]) % 360
        hues = (context.hue_bounds[0] + np.linspace(0, width, 12)) % 360
    values = HELIX_COLORFULNESS * context.gamut.max_colorfulness(jp[:, None], hues[None, :])
    if context.chroma_bounds is not None:
        values = np.minimum(values, chroma_to_colorfulness(context.chroma_bounds[1]))
    return max(float(values.mean()), 5.0)


def non_lightness_length(size, length, lightness_span):
    """Wanted a'b' length L_ab = max(rho(n) L, sqrt(L^2 - L_J^2))."""
    if size <= 1:
        return 0.0
    rho = min(0.6, 0.15 * np.sqrt(size - 1))
    return float(max(rho * length, np.sqrt(max(0.0, length**2 - lightness_span**2))))


def block_demand(size, length, lightness_span, mean_colorfulness, cap=None):
    """Wanted hue sweep in degrees for a block (Phase 4 demand)."""
    if size <= 1:
        return 0.0
    non_lightness = non_lightness_length(size, length, lightness_span)
    sweep = float(np.degrees(non_lightness / max(mean_colorfulness, 5.0)))
    sweep = min(sweep, MAX_SWEEP)
    if cap is not None:
        sweep = min(sweep, float(cap))
    return sweep


def usable_share(chromatic_blocks, lightness_bounds):
    width = lightness_bounds[1] - lightness_bounds[0]
    return float(
        np.clip(0.8 - 0.05 * (chromatic_blocks - 3), 0.4, 0.8) * np.clip(width / 80, 0.6, 1.0)
    )


def plan_blocks(block_sizes, context, step=DEFAULT_STEP):
    jp_low, jp_high = context.jp_bounds
    span = max(jp_high - jp_low - 4.0, 1.0)
    colorfulness = _mean_colorfulness(context, jp_low + 2, jp_high - 2)
    cap = None
    if context.hue_sweep not in (None, "auto"):
        cap = float(context.hue_sweep)
    specs = []
    for index, size in enumerate(block_sizes):
        size = int(size)
        kind = "point" if size == 1 else ("long" if size >= LONG_BLOCK_MIN else "short")
        length = (size - 1) * step
        demand = 0.0 if cap == 0 else block_demand(size, length, span, colorfulness, cap)
        specs.append(BlockSpec(index, size, kind, length, step, demand, span,
                               non_lightness_length(size, length, span)))
    return specs


def reserved_sweeps(specs, context):
    demands = np.array([spec.demand for spec in specs], dtype=float)
    chromatic = int(np.sum(demands > 0))
    total = demands.sum()
    if total <= 0:
        return np.zeros(len(specs))
    share = usable_share(chromatic, context.lightness_bounds)
    return np.minimum(demands, share * 360 * demands / total)


# ---------------------------------------------------------------- footprints


def _polar_to_cam(jp, colorfulness, hue):
    radians = np.deg2rad(hue)
    return np.stack([jp, colorfulness * np.cos(radians), colorfulness * np.sin(radians)], axis=-1)


def _hue_allowed(hue, context):
    if context.hue_bounds == (0.0, 360.0):
        return np.ones(np.shape(hue), dtype=bool)
    low, high = context.hue_bounds
    hue = np.asarray(hue) % 360
    if low <= high:
        return (hue >= low) & (hue <= high)
    return (hue >= low) | (hue <= high)


def helix_footprints(spec, context, sweep, twist, hues=None):
    """Long-block helix footprints for candidate mid-lightness hues; shape (k, 9, 3)."""
    jp_low, jp_high = context.jp_bounds
    jp_mid = (jp_low + jp_high) / 2
    half = max(0.0, min(spec.length / 2, (jp_high - jp_low) / 2 - 2))
    if hues is None:
        hues = np.arange(0, 360, HELIX_HUE_STEP)
    hues = np.asarray(hues, dtype=float)
    t = np.linspace(-1, 1, HELIX_POINTS)
    jp = jp_mid + t * half
    hue = hues[:, None] + twist * t[None, :] * sweep / 2
    colorfulness = HELIX_COLORFULNESS * context.gamut.max_colorfulness(
        np.broadcast_to(jp, hue.shape), hue % 360
    )
    if context.chroma_bounds is not None:
        colorfulness = np.minimum(colorfulness, chroma_to_colorfulness(context.chroma_bounds[1]))
    points = _polar_to_cam(np.broadcast_to(jp, hue.shape), colorfulness, hue)
    # The base (helix midpoint, scaled toward neutral) must clear the chroma lower bound.
    minimum = chroma_to_colorfulness(context.chroma_bounds[0] + 2.0) / LONG_BASE_SCALE \
        if context.chroma_bounds is not None else 1.0
    valid = _hue_allowed(hue % 360, context).all(axis=1) & (
        colorfulness[:, HELIX_POINTS // 2] > max(1.0, float(minimum)))
    return points, hues, valid


def short_candidates(context):
    """Candidate (J', a', b') centres for short blocks."""
    cache = context.diagnostics.setdefault("_short_candidates", {})
    if "colors" in cache:
        return cache["colors"], cache["jch"]
    if context.grid_space == "JCh":
        cam = jch_grid(SHORT_CANDIDATE_GRID, context.lightness_bounds, context.chroma_bounds,
                       context.hue_bounds, output_colorspace="CAM02-UCS").astype(float)
        cam = cam[context.domain.contains(cam)]
        jch = _cam02ucs_to_jch(cam)
    else:
        jch = rgb_grid(SHORT_CANDIDATE_GRID, *context.domain.rgb_bounds,
                       output_colorspace="JCh").astype(float)
        cam = cspace_convert(jch, "JCh", "CAM02-UCS")

    def mask(lightness, chroma):
        keep = (jch[:, 0] >= lightness[0]) & (jch[:, 0] <= lightness[1])
        keep &= (jch[:, 1] >= chroma[0]) & (jch[:, 1] <= chroma[1])
        return keep & _hue_allowed(jch[:, 2], context)

    lightness = (max(SHORT_CANDIDATE_LIGHTNESS[0], context.lightness_bounds[0]),
                 min(SHORT_CANDIDATE_LIGHTNESS[1], context.lightness_bounds[1]))
    chroma = (max(SHORT_CANDIDATE_CHROMA[0], context.chroma_bounds[0]),
              min(SHORT_CANDIDATE_CHROMA[1], context.chroma_bounds[1]))
    keep = mask(lightness, chroma)
    if keep.sum() < 16:
        keep = mask(context.lightness_bounds, context.chroma_bounds)
    if keep.sum() < 1:
        keep = mask(context.lightness_bounds, (0, np.inf))
    cache["colors"], cache["jch"] = cam[keep], jch[keep]
    return cam[keep], jch[keep]


def short_footprints(spec, centres, centre_jch, context, sweep, twist):
    """Short-block footprints around candidate centres; shape (k, 5, 3)."""
    if spec.kind == "point":
        return centres[:, None, :]
    t = np.linspace(-1, 1, SHORT_POINTS)
    jp = centres[:, :1] + t[None, :] * SHORT_LIGHTNESS_SPREAD * spec.length
    jp = np.clip(jp, *context.jp_bounds)
    colorfulness = np.hypot(centres[:, 1], centres[:, 2])[:, None] * (1 - SHORT_TAPER * np.abs(t))[None]
    hue = centre_jch[:, 2:3] + twist * t[None, :] * sweep / 2
    return _polar_to_cam(jp, colorfulness, hue)


def mid_band(context, widen=0.0):
    low, high = context.lightness_bounds
    if high - low < 2 * (MID_BAND[1] - MID_BAND[0]):
        width = high - low
        band = (low + width / 4, high - width / 4)
    else:
        band = MID_BAND
    band = (max(low, band[0] - widen), min(high, band[1] + widen))
    if band[0] > band[1]:
        band = (low, high)
    return band


# ---------------------------------------------------------------- selection


class _Placed:
    """Points already placed, with optional CVD-simulated copies."""

    def __init__(self, context, fixed=None):
        self.context = context
        white_black = cspace_convert([[1, 1, 1], [0, 0, 0]], "sRGB1", "CAM02-UCS")
        self.extremes = white_black
        self.points = np.zeros((0, 3))
        self.owner = np.zeros(0, dtype=int)
        self.cvd_points = np.zeros((0, 3))
        self.cvd_extremes = white_black.copy()
        if fixed is not None and len(fixed):
            self.add(np.asarray(fixed, dtype=float), -1)

    def add(self, points, owner):
        points = np.atleast_2d(points)
        self.points = np.vstack([self.points, points])
        self.owner = np.concatenate([self.owner, np.full(len(points), owner)])
        if self.context.colorblind_safe:
            self.cvd_points = np.vstack([self.cvd_points, self.context.cvd(points)])
        self._trees = None

    def remove(self, owner):
        self._minima = None
        keep = self.owner != owner
        self.points = self.points[keep]
        if self.context.colorblind_safe:
            self.cvd_points = self.cvd_points[keep]
        self.owner = self.owner[keep]
        self._trees = None

    _trees = None
    _scale = np.array([LIGHTNESS_DISCOUNT, 1.0, 1.0])

    def _tree(self, which):
        if self._trees is None:
            self._trees = {}
        if which not in self._trees:
            points = self.points if which == "normal" else self.cvd_points
            self._trees[which] = cKDTree(points * self._scale) if len(points) else None
        return self._trees[which]

    def score(self, footprints, cvd_footprints=None, extremes=None, key=None):
        """Minimum separation of each candidate footprint (k, m, 3) from everything placed.

        ``extremes`` optionally gives precomputed distances to black and white
        (normal, CVD). With a ``key`` the per-point minima are kept and updated
        incrementally as points are added, as in Glasbey's algorithm.
        """
        extremes = extremes or (None, None)
        result = self._running(footprints, "normal", key, extremes[0], self.extremes)
        if self.context.colorblind_safe:
            if cvd_footprints is None:
                cvd_footprints = self.context.cvd(footprints)
            result = np.minimum(result, self._running(
                cvd_footprints, "cvd", key, extremes[1], self.cvd_extremes))
        return result

    def _running(self, footprints, which, key, to_extremes, extremes):
        points = self.points if which == "normal" else self.cvd_points
        if key is None:
            return self._score(footprints, self._tree(which), extremes, to_extremes)
        if self._minima is None:
            self._minima = {}
        state = self._minima.get((key, which))
        if state is None:
            if to_extremes is None:
                to_extremes = self.extreme_distances(footprints, extremes)
            minima = np.broadcast_to(to_extremes[:, None], footprints.shape[:-1]).copy()
            seen = 0
        else:
            seen, minima = state
        if seen < len(points):
            new = points[seen:]
            if len(new) > 32:
                tree = cKDTree(new * self._scale)
                distances = tree.query((footprints * self._scale).reshape(-1, 3))[0]
                distances = distances.reshape(footprints.shape[:-1])
            else:
                distances = discounted_distance(footprints, new).min(axis=-1)
            minima = np.minimum(minima, distances)
        self._minima[(key, which)] = (len(points), minima)
        return minima.min(axis=-1)

    _minima = None

    @staticmethod
    def extreme_distances(footprints, extremes):
        return np.linalg.norm(footprints[..., None, :] - extremes, axis=-1).min(axis=(-1, -2))

    def _score(self, footprints, tree, extremes, to_extremes=None):
        if to_extremes is None:
            to_extremes = self.extreme_distances(footprints, extremes)
        if tree is None:
            return to_extremes
        flat = (footprints * self._scale).reshape(-1, 3)
        distances, _ = tree.query(flat)
        to_points = distances.reshape(footprints.shape[:-1]).min(axis=-1)
        return np.minimum(to_points, to_extremes)


_EXTREMES = None


def _extremes():
    global _EXTREMES
    if _EXTREMES is None:
        _EXTREMES = cspace_convert([[1, 1, 1], [0, 0, 0]], "sRGB1", "CAM02-UCS")
    return _EXTREMES


def _cached_footprints(context, key, build):
    """Candidate footprints with their CVD simulation and distances to black and white."""
    cache = context.diagnostics.setdefault("_footprint_cache", {})
    if key not in cache:
        built = build()
        footprints = built[0] if isinstance(built, tuple) else built
        simulated = context.cvd(footprints) if context.colorblind_safe else None
        extremes = (
            _Placed.extreme_distances(footprints, _extremes()),
            _Placed.extreme_distances(simulated, _extremes()) if simulated is not None else None,
        )
        cache[key] = (built, simulated, extremes)
    return cache[key]


def _place_long(spec, context, placed, sweep, twist):
    (points, hues, valid), simulated, extremes = _cached_footprints(
        context, ("long", spec.length, round(float(sweep), 6), twist),
        lambda: helix_footprints(spec, context, sweep, twist),
    )
    key = ("long", spec.length, round(float(sweep), 6), twist)
    scores = np.where(valid, placed.score(points, simulated, extremes, key), -np.inf)
    best = int(np.argmax(scores))
    footprint = points[best]
    centre = footprint[HELIX_POINTS // 2]
    base = np.array([centre[0], centre[1] * LONG_BASE_SCALE, centre[2] * LONG_BASE_SCALE])
    order = np.argsort(-scores)
    return Footprint(footprint, base, float(hues[best]), float(sweep), twist, spec.kind,
                     float(j_from_jp(centre[0]))), float(scores[best]), {
        "alternatives": [float(hues[i]) for i in order[:10] if np.isfinite(scores[i])],
    }


def _place_short(spec, context, placed, sweep, twist, exclude=()):
    centres, centre_jch = short_candidates(context)
    footprints, simulated, extremes = _cached_footprints(
        context, (spec.kind, spec.length, round(float(sweep), 6), twist),
        lambda: short_footprints(spec, centres, centre_jch, context, sweep, twist),
    )
    raw = placed.score(footprints, simulated, extremes,
                       (spec.kind, spec.length, round(float(sweep), 6), twist))
    crowded = False
    widen = 0.0
    while True:
        band = mid_band(context, widen)
        lightness = centre_jch[:, 0]
        penalty = MID_PENALTY * np.maximum(0, np.maximum(band[0] - lightness, lightness - band[1]))
        scores = raw - penalty
        if exclude:
            scores[list(exclude)] = -np.inf
        best = int(np.argmax(scores))
        if raw[best] >= SEP_MIN or band == context.lightness_bounds or widen > 100:
            break
        crowded = True
        widen += CROWDING_WIDEN
    order = np.argsort(-scores)
    # Refinement may only move to alternatives inside the band (never optimise the penalty away).
    in_band = (lightness >= band[0]) & (lightness <= band[1])
    order = order[in_band[order] & np.isfinite(scores[order])]
    return Footprint(footprints[best], centres[best].copy(), float(centre_jch[best, 2]),
                     float(sweep), twist, spec.kind, float(centre_jch[best, 0])), float(raw[best]), {
        "crowded": crowded, "band": band, "candidate": best,
        "alternatives": [int(i) for i in order[:10]],
    }


def select_bases(specs, context, sweeps, twist, fixed_colors=None):
    """Greedy max-min placement of block footprints, largest blocks first.

    Returns a list of :class:`Footprint` (in ``specs`` order) and per-block info.
    """
    placed = _Placed(context, fixed_colors)
    order = sorted(range(len(specs)), key=lambda i: -specs[i].size)
    footprints = [None] * len(specs)
    info = [None] * len(specs)
    for i in order:
        spec = specs[i]
        if spec.kind == "long":
            footprint, score, details = _place_long(spec, context, placed, sweeps[i], twist)
        else:
            footprint, score, details = _place_short(spec, context, placed, sweeps[i], twist)
        details["score"] = score
        footprints[i], info[i] = footprint, details
        placed.add(footprint.points, i)
    return footprints, info


def footprint_separation(footprints, context=None, w_j=LIGHTNESS_DISCOUNT):
    """Minimum lightness-discounted distance between any two blocks' footprints."""
    best = np.inf
    for i in range(len(footprints)):
        for j in range(i + 1, len(footprints)):
            distance = discounted_distance(footprints[i].points, footprints[j].points, w_j).min()
            best = min(best, float(distance))
    return best


# ---------------------------------------------------------------- allocation

REFINE_ROUNDS = 3
REFINE_PAIRS = 3
SHRINK_FACTOR = 0.8
SHRINK_ROUNDS = 5
YELLOW_HUE = 100.0


def rebuild_footprint(spec, footprint, context, sweep, twist=None, position=None):
    """Recompute a footprint at the same (or a new) position with a new sweep."""
    twist = footprint.twist if twist is None else twist
    if position is None:
        position = footprint.hue if spec.kind == "long" else footprint.base
    key = (spec.kind, spec.length, round(float(sweep), 9), twist,
           tuple(np.round(np.atleast_1d(position), 9)))
    cache = context.diagnostics.setdefault("_rebuild_cache", {})
    if key not in cache:
        cache[key] = _rebuild_footprint(spec, footprint, context, sweep, twist, position)
    return cache[key]


def _rebuild_footprint(spec, footprint, context, sweep, twist, position):
    if spec.kind == "long":
        hue = footprint.hue if position is None else float(position)
        points = helix_footprints(spec, context, sweep, twist, hues=[hue])[0][0]
        centre = points[HELIX_POINTS // 2]
        base = np.array([centre[0], centre[1] * LONG_BASE_SCALE, centre[2] * LONG_BASE_SCALE])
        return Footprint(points, base, hue, float(sweep), twist, spec.kind, float(j_from_jp(centre[0])))
    centre = footprint.base if position is None else np.asarray(position, dtype=float)
    centre_jch = _cam02ucs_to_jch(centre)[None]
    points = short_footprints(spec, centre[None], centre_jch, context, sweep, twist)[0]
    return Footprint(points, centre.copy(), float(centre_jch[0, 2]), float(sweep), twist, spec.kind,
                     float(centre_jch[0, 0]))


def block_demand_at(spec, footprint, context):
    """Demand converted to degrees with the block's own footprint colourfulness."""
    if spec.demand <= 0:
        return 0.0
    colorfulness = float(np.hypot(footprint.points[:, 1], footprint.points[:, 2]).mean())
    cap = None if context.hue_sweep in (None, "auto") else float(context.hue_sweep)
    return block_demand(spec.size, spec.length, spec.lightness_span, colorfulness, cap)


def _chromatic(spec, footprint):
    return spec.demand > 0 and _cam02ucs_to_jch(footprint.base)[1] >= GREY_CHROMA


def _lightness_extent(footprint):
    return float(footprint.points[:, 0].min()), float(footprint.points[:, 0].max())


def allocate_sweeps(specs, footprints, context, reserved, twist):
    """Hue sweeps from realised hue gaps between lightness-overlapping neighbours."""
    count = len(specs)
    demand = np.array([block_demand_at(spec, footprint, context)
                       for spec, footprint in zip(specs, footprints)], dtype=float)
    sweeps = np.minimum(np.asarray(reserved, dtype=float), demand)
    chromatic = [i for i in range(count) if _chromatic(specs[i], footprints[i])]
    share = usable_share(len(chromatic), context.lightness_bounds)
    hues = np.array([footprint.hue for footprint in footprints])
    extents = [_lightness_extent(footprint) for footprint in footprints]
    chromatic_array = np.array(chromatic, dtype=int)
    low = np.array([extent[0] for extent in extents])
    high = np.array([extent[1] for extent in extents])
    for i in chromatic:
        others = chromatic_array[chromatic_array != i]
        overlap = np.minimum(high[i], high[others]) > np.maximum(low[i], low[others])
        neighbours = others[overlap]
        if not len(neighbours):
            sweeps[i] = demand[i]
            continue
        offsets = hue_offset(hues[neighbours], hues[i])
        halves = []
        for direction in (twist, -twist):  # the light end's gap, then the dark end's gap
            gaps = (direction * offsets) % 360
            nearest = int(np.argmin(gaps))
            gap, j = gaps[nearest], neighbours[nearest]
            fraction = demand[i] / max(demand[i] + demand[j], 1e-9)
            halves.append(share * gap * fraction)
        allowance = 2 * min(halves)
        sweeps[i] = min(demand[i], max(sweeps[i], allowance))
    for i in range(count):
        if i not in chromatic:
            sweeps[i] = 0.0
    footprints = [rebuild_footprint(spec, footprint, context, sweep, twist)
                  for spec, footprint, sweep in zip(specs, footprints, sweeps)]
    for round_ in range(SHRINK_ROUNDS):
        close = _close_pairs(footprints, SEP_MIN)
        if not close:
            break
        shrink = {index for pair in close for index in pair if sweeps[index] > 0}
        if not shrink:
            break
        for index in shrink:
            sweeps[index] *= SHRINK_FACTOR
            footprints[index] = rebuild_footprint(specs[index], footprints[index], context,
                                                  sweeps[index], twist)
    return footprints, sweeps


def _pair_minima(footprints, w_j=LIGHTNESS_DISCOUNT):
    """(K, K) matrix of minimum discounted distances between footprints (inf on the diagonal)."""
    points = np.vstack([footprint.points for footprint in footprints])
    starts = np.cumsum([0] + [len(footprint.points) for footprint in footprints[:-1]])
    distances = discounted_distance(points, points, w_j)
    result = np.minimum.reduceat(np.minimum.reduceat(distances, starts, axis=0), starts, axis=1)
    np.fill_diagonal(result, np.inf)
    return result


def _close_pairs(footprints, threshold):
    minima = _pair_minima(footprints)
    return [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(minima < threshold, 1)))]


def separation_objective(specs, footprints, context, fixed_colors=None):
    """(minimum discounted footprint separation, size-weighted satisfied demand)."""
    def minimum(points_list, fixed, extremes):
        points = np.vstack(points_list)
        owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(points_list)])
        distances = discounted_distance(points, points)
        distances[owner[:, None] == owner[None, :]] = np.inf
        best = distances.min() if len(points_list) > 1 else np.inf
        if fixed is not None and len(fixed):
            best = min(best, discounted_distance(points, fixed).min())
        return min(best, np.linalg.norm(points[:, None] - extremes, axis=-1).min())

    extremes = cspace_convert([[1, 1, 1], [0, 0, 0]], "sRGB1", "CAM02-UCS")
    points = [footprint.points for footprint in footprints]
    separation = minimum(points, fixed_colors, extremes)
    if context.colorblind_safe:
        fixed = context.cvd(fixed_colors) if fixed_colors is not None and len(fixed_colors) else None
        separation = min(separation, minimum([context.cvd(p) for p in points], fixed, extremes))
    satisfied = sum(
        spec.size * (1.0 if spec.demand <= 0 else min(1.0, footprint.sweep / spec.demand))
        for spec, footprint in zip(specs, footprints)
    )
    return float(separation), satisfied


def _better(candidate, incumbent, tolerance=1e-6):
    if candidate[0] > incumbent[0] + tolerance:
        return True
    return abs(candidate[0] - incumbent[0]) <= tolerance and candidate[1] > incumbent[1] + tolerance


def choose_twist(specs, footprints):
    """Global twist sending the most sweep-weighted light ends toward yellow."""
    total = 0.0
    for spec, footprint in zip(specs, footprints):
        if spec.demand > 0:
            total += footprint.sweep * np.sin(np.deg2rad(hue_offset(YELLOW_HUE, footprint.hue)))
    return -1 if total < -1e-9 else 1


def refine(specs, footprints, info, context, reserved, twist, fixed_colors=None,
           rounds=REFINE_ROUNDS):
    """Local moves on footprints: swaps between same-class blocks and moves to alternatives."""
    footprints, sweeps = allocate_sweeps(specs, footprints, context, reserved, twist)
    best = separation_objective(specs, footprints, context, fixed_colors)
    positions = [footprint.hue if spec.kind == "long" else footprint.base.copy()
                 for spec, footprint in zip(specs, footprints)]
    centres, _ = short_candidates(context)
    accepted = 0

    def evaluate(new_positions):
        moved = [rebuild_footprint(spec, footprint, context, reserved[i], twist, new_positions[i])
                 for i, (spec, footprint) in enumerate(zip(specs, footprints))]
        allocated, new_sweeps = allocate_sweeps(specs, moved, context, reserved, twist)
        return separation_objective(specs, allocated, context, fixed_colors), allocated, new_sweeps

    for round_ in range(rounds):
        improved = False
        moves = []
        # Only moves touching blocks in the closest pairs can raise the minimum separation.
        minima = _pair_minima(footprints) if len(footprints) > 1 else np.zeros((1, 1))
        order = np.argsort(minima, axis=None)[: 2 * REFINE_PAIRS]
        involved = {int(k) for flat in order for k in np.unravel_index(flat, minima.shape)}
        for i in range(len(specs)):
            for j in range(i + 1, len(specs)):
                if (i in involved or j in involved) and specs[i].kind == specs[j].kind \
                        and specs[i].size != specs[j].size:
                    moves.append(("swap", i, j))
        for i, details in enumerate(info):
            if i not in involved:
                continue
            for alternative in details.get("alternatives", [])[1:]:
                moves.append(("move", i, alternative))
        for move in moves:
            candidate = list(positions)
            if move[0] == "swap":
                _, i, j = move
                candidate[i], candidate[j] = positions[j], positions[i]
            else:
                _, i, alternative = move
                candidate[i] = (float(alternative) if specs[i].kind == "long"
                                else centres[int(alternative)].copy())
            score, allocated, new_sweeps = evaluate(candidate)
            if _better(score, best):
                best, footprints, sweeps, positions = score, allocated, new_sweeps, candidate
                improved = True
                accepted += 1
        if not improved:
            break
    return footprints, sweeps, {"objective": best, "accepted_moves": accepted}


# ---------------------------------------------------------------- fitting


def fit_blocks(specs, footprints, context, fixed_colors=None, max_iter=None):
    """Fit each block's theme arc, largest first, avoiding colours already fitted."""
    order = sorted(range(len(specs)), key=lambda i: -specs[i].size)
    themes = [None] * len(specs)
    others = [np.asarray(fixed_colors, dtype=float)] if fixed_colors is not None and len(fixed_colors) else []
    for i in order:
        spec, footprint = specs[i], footprints[i]
        sweep = footprint.sweep if _cam02ucs_to_jch(footprint.base)[1] >= GREY_CHROMA else 0.0
        kwargs = {} if max_iter is None else {"max_iter": max_iter}
        themes[i] = fit_theme_arc(
            footprint.base, spec.size, step=spec.step, domain=context.domain, gamut=context.gamut,
            hue_allowance=(sweep / 2, sweep / 2), twist=footprint.twist,
            other_colors=np.vstack(others) if others else None, grid=context.grid,
            endpoint_targets=(footprint.points[0], footprint.points[-1]),
            soft_endpoints=LONG_ENDPOINT_WEIGHT if spec.kind == "long" else True, **kwargs,
        )
        themes[i].diagnostics["allowance"] = (sweep / 2, sweep / 2)
        themes[i].diagnostics["twist"] = footprint.twist
        others.append(themes[i].colors)
    return themes


def block_separation(blocks, w_j=LIGHTNESS_DISCOUNT):
    """Minimum lightness-discounted distance between colours of different blocks."""
    best = np.inf
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            best = min(best, float(discounted_distance(blocks[i], blocks[j], w_j).min()))
    return best


# ---------------------------------------------------------------- polish

POLISH_SLIDES = 9
POLISH_SHRINKS = (1.0, 0.95, 0.9, 0.85)
POLISH_MAX_REFITS = 4
ROUNDING_NUDGE = 0.005


@dataclass
class _BlockState:
    theme: object
    length: float
    offset: float = 0.0

    def positions(self, n=None):
        n = self.theme.size if n is None else n
        if self.theme.arc is None or n == 1:
            return None
        length = min(self.length, 0.98 * self.theme.arc.length)
        return _sample_window(self.theme.arc, n, length, self.theme.base_position, self.offset)[0]

    def colors(self):
        positions = self.positions()
        if positions is None:
            return self.theme.colors
        return self.theme.arc(positions)


def _minimum_between(blocks, fixed=None):
    """Minimum discounted distance between colours of different blocks (and fixed colours)."""
    points = np.vstack(blocks)
    starts = np.cumsum([0] + [len(block) for block in blocks[:-1]])
    best = np.inf
    if len(blocks) > 1:
        distances = discounted_distance(points, points)
        minima = np.minimum.reduceat(np.minimum.reduceat(distances, starts, axis=0), starts, axis=1)
        np.fill_diagonal(minima, np.inf)
        best = float(minima.min())
    if fixed is not None and len(fixed):
        best = min(best, float(discounted_distance(points, fixed).min()))
    return best


def palette_separation(blocks, context, fixed_colors=None):
    """Global minimum lightness-discounted between-block distance (min with CVD if enabled)."""
    result = _minimum_between(blocks, fixed_colors)
    if context.colorblind_safe:
        simulated = [context.cvd(block) for block in blocks]
        fixed = context.cvd(fixed_colors) if fixed_colors is not None and len(fixed_colors) else None
        result = min(result, _minimum_between(simulated, fixed))
    return result


def _closest_pair(blocks, fixed_colors=None):
    best, pair = np.inf, None
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            distance = float(discounted_distance(blocks[i], blocks[j]).min())
            if distance < best:
                best, pair = distance, (i, j)
        if fixed_colors is not None and len(fixed_colors):
            distance = float(discounted_distance(blocks[i], fixed_colors).min())
            if distance < best:
                best, pair = distance, (i, None)
    return pair


def polish_blocks(specs, themes, context, n_iter=10, search_radius=5.0, fixed_colors=None):
    """Block-level moves that raise the global minimum without breaking block order.

    Each round takes the closest block pair and tries, for each block of the
    pair, sliding its window along its arc by up to ``search_radius``,
    shrinking its step by up to 15% and (short blocks only) refitting from a
    base nudged by up to ``search_radius``. The single best improving move is
    accepted.
    """
    states = [_BlockState(theme, theme.step * (theme.size - 1),
                          theme.diagnostics.get("window_offset", 0.0)) for theme in themes]
    blocks = [state.colors() for state in states]
    current = palette_separation(blocks, context, fixed_colors)
    accepted, refits = 0, 0
    for iteration in range(int(n_iter)):
        pair = _closest_pair(blocks, fixed_colors)
        if pair is None:
            break
        best_score, best_move = current, None
        for index in (i for i in pair if i is not None):
            state = states[index]
            if state.theme.arc is None or state.theme.size < 2:
                continue
            base_length = state.theme.step * (state.theme.size - 1)
            for scale in POLISH_SHRINKS:
                if scale < 1.0 and state.theme.step * scale < MIN_STEP:
                    continue  # never shrink an already short step further
                for slide in np.linspace(-search_radius, search_radius, POLISH_SLIDES):
                    trial = _BlockState(state.theme, base_length * scale, state.offset + slide)
                    if scale == state.length / max(base_length, 1e-9) and slide == 0:
                        continue
                    candidate = list(blocks)
                    candidate[index] = trial.colors()
                    score = palette_separation(candidate, context, fixed_colors)
                    if score > best_score + 1e-6:
                        best_score, best_move = score, (index, trial)
        if best_move is None and refits < POLISH_MAX_REFITS:
            for index in (i for i in pair if i is not None):
                if specs[index].kind == "long" or states[index].theme.arc is None:
                    continue
                theme = states[index].theme
                others = [blocks[k] for k in range(len(blocks)) if k != index]
                if fixed_colors is not None and len(fixed_colors):
                    others.append(fixed_colors)
                others = np.vstack(others)
                nearest = others[np.argmin(discounted_distance(blocks[index], others).min(axis=0))]
                away = theme.base - nearest
                away[0] = 0.0
                norm = np.linalg.norm(away)
                if norm < 1e-9:
                    continue
                nudged = theme.base + away / norm * search_radius
                refits += 1
                refit = fit_theme_arc(
                    nudged, theme.size, step=specs[index].step, domain=context.domain,
                    gamut=context.gamut, grid=context.grid, warn=False,
                    hue_allowance=(theme.diagnostics.get("allowance", (0.0, 0.0))),
                    twist=theme.diagnostics.get("twist", 1), other_colors=others,
                )
                if refit.diagnostics["fallback"]:
                    continue
                trial = _BlockState(refit, refit.step * (refit.size - 1),
                                    refit.diagnostics.get("window_offset", 0.0))
                candidate = list(blocks)
                candidate[index] = trial.colors()
                score = palette_separation(candidate, context, fixed_colors)
                if score > best_score + 1e-6:
                    best_score, best_move = score, (index, trial)
        if best_move is None:
            break
        index, trial = best_move
        states[index] = trial
        blocks[index] = trial.colors()
        current = best_score
        accepted += 1
    return states, {"separation": current, "accepted_moves": accepted, "refits": refits}


def _hex_lightness(colors):
    from matplotlib.colors import to_rgb

    rgb = np.array([to_rgb(color) for color in colors])
    return cspace_convert(rgb, "sRGB1", "CAM02-UCS")[:, 0]


def blocks_to_output(states, as_hex=True):
    """Convert block states to RGB output, repairing collisions caused by hex rounding."""
    output = []
    for state in states:
        colors = state.colors()
        if not as_hex:
            output.extend(get_rgb_palette(colors, as_hex=False))
            continue
        block = list(get_rgb_palette(colors, as_hex=True))
        positions = state.positions()
        if positions is not None and len(block) > 1:
            for attempt in range(8):
                lightness = _hex_lightness(block)
                bad = [k for k in range(1, len(block))
                       if block[k] == block[k - 1] or lightness[k] <= lightness[k - 1]]
                if not bad:
                    break
                for k in bad:
                    # Push the pair apart along the arc by +-0.5%.
                    positions[k] = min(1.0, positions[k] + ROUNDING_NUDGE)
                    positions[k - 1] = max(0.0, positions[k - 1] - ROUNDING_NUDGE)
                block = list(get_rgb_palette(state.theme.arc(positions), as_hex=True))
        output.extend(block)
    return output


def build_block_palette(block_sizes, *, context=None, step=DEFAULT_STEP, twist=1,
                        fixed_colors=None, refine_rounds=None, optimize=False, n_iter=10,
                        search_radius=5.0, **context_options):
    """Run the block-palette pipeline; return (CAM02-UCS colours, diagnostics).

    Stages: plan -> select bases -> allocate hue sweep -> refine -> fit -> polish.
    ``diagnostics["states"]`` holds per-block sampling states for output.
    """
    timings = {}
    start = time.perf_counter()
    if context is None:
        context = make_context(**context_options)
    if fixed_colors is not None:
        fixed_colors = np.asarray(fixed_colors, dtype=float).reshape(-1, 3)
    timings["setup"] = time.perf_counter() - start
    start = time.perf_counter()
    specs = plan_blocks(block_sizes, context, step)
    reserved = reserved_sweeps(specs, context)
    timings["plan"] = time.perf_counter() - start
    start = time.perf_counter()
    sign = 1 if twist == "auto" else int(twist)
    footprints, info = select_bases(specs, context, reserved, sign, fixed_colors)
    if twist == "auto":
        chosen = choose_twist(specs, footprints)
        if chosen != sign:
            sign = chosen
            footprints, info = select_bases(specs, context, reserved, sign, fixed_colors)
    timings["select"] = time.perf_counter() - start
    start = time.perf_counter()
    rounds = REFINE_ROUNDS if refine_rounds is None else refine_rounds
    if rounds > 0 and len(specs) > 1:
        footprints, sweeps, refinement = refine(specs, footprints, info, context, reserved, sign,
                                                fixed_colors, rounds)
    else:
        footprints, sweeps = allocate_sweeps(specs, footprints, context, reserved, sign)
        refinement = {"objective": separation_objective(specs, footprints, context, fixed_colors),
                      "accepted_moves": 0}
    timings["allocate_refine"] = time.perf_counter() - start
    start = time.perf_counter()
    themes = fit_blocks(specs, footprints, context, fixed_colors)
    timings["fit"] = time.perf_counter() - start
    start = time.perf_counter()
    if optimize and len(specs) + (fixed_colors is not None and len(fixed_colors) > 0) > 1:
        states, polish = polish_blocks(specs, themes, context, n_iter, search_radius, fixed_colors)
    else:
        states = [_BlockState(theme, theme.step * (theme.size - 1),
                              theme.diagnostics.get("window_offset", 0.0)) for theme in themes]
        polish = {"accepted_moves": 0, "refits": 0}
    timings["polish"] = time.perf_counter() - start
    blocks = [state.colors() for state in states]
    diagnostics = {
        "specs": specs, "footprints": footprints, "selection": info, "themes": themes,
        "states": states, "reserved": [float(value) for value in reserved],
        "sweeps": [float(value) for value in sweeps], "twist": sign,
        "refinement": refinement, "polish": polish, "timings": timings,
        "separation": block_separation(blocks) if len(blocks) > 1 else np.inf,
        "palette_separation": palette_separation(blocks, context, fixed_colors),
        "footprint_separation": footprint_separation(footprints) if len(blocks) > 1 else np.inf,
        "crowding": any(item.get("crowded", False) for item in info),
        "fallbacks": int(sum(theme.diagnostics.get("fallback", False) for theme in themes)),
    }
    return np.vstack(blocks), diagnostics
