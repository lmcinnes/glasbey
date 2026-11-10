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
from ._grids import rgb_grid
from ._theme import (
    DEFAULT_STEP, GREY_CHROMA, LIGHTNESS_DISCOUNT, chroma_to_colorfulness, discounted_distance,
    fit_theme_arc, gamut_table, hue_offset,
)

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
        hue_sweep,
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


def block_demand(size, length, lightness_span, mean_colorfulness, cap=None):
    """Wanted hue sweep in degrees for a block (Phase 4 demand)."""
    if size <= 1:
        return 0.0
    rho = min(0.6, 0.15 * np.sqrt(size - 1))
    non_lightness = max(rho * length, np.sqrt(max(0.0, length**2 - lightness_span**2)))
    sweep = float(np.degrees(non_lightness / mean_colorfulness))
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
        specs.append(BlockSpec(index, size, kind, length, step, demand, span))
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
    valid = _hue_allowed(hue % 360, context).all(axis=1) & (colorfulness[:, HELIX_POINTS // 2] > 1)
    return points, hues, valid


def short_candidates(context):
    """Candidate (J', a', b') centres for short blocks."""
    cache = context.diagnostics.setdefault("_short_candidates", {})
    if "colors" in cache:
        return cache["colors"], cache["jch"]
    jch = rgb_grid(SHORT_CANDIDATE_GRID, *context.domain.rgb_bounds, output_colorspace="JCh").astype(float)
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

    def score(self, footprints, cvd_footprints=None):
        """Minimum separation of each candidate footprint (k, m, 3) from everything placed."""
        result = self._score(footprints, self._tree("normal"), self.extremes)
        if self.context.colorblind_safe:
            if cvd_footprints is None:
                cvd_footprints = self.context.cvd(footprints)
            result = np.minimum(
                result, self._score(cvd_footprints, self._tree("cvd"), self.cvd_extremes)
            )
        return result

    def _score(self, footprints, tree, extremes):
        to_extremes = np.linalg.norm(footprints[..., None, :] - extremes, axis=-1).min(axis=(-1, -2))
        if tree is None:
            return to_extremes
        flat = (footprints * self._scale).reshape(-1, 3)
        distances, _ = tree.query(flat)
        to_points = distances.reshape(footprints.shape[:-1]).min(axis=-1)
        return np.minimum(to_points, to_extremes)


def _cached_footprints(context, key, build):
    cache = context.diagnostics.setdefault("_footprint_cache", {})
    if key not in cache:
        footprints = build()
        simulated = context.cvd(footprints) if context.colorblind_safe else None
        cache[key] = (footprints, simulated)
    return cache[key]


def _place_long(spec, context, placed, sweep, twist):
    points, hues, valid = helix_footprints(spec, context, sweep, twist)
    simulated = _cached_footprints(
        context, ("long", spec.length, round(float(sweep), 6), twist), lambda: points
    )[1]
    scores = np.where(valid, placed.score(points, simulated), -np.inf)
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
    footprints, simulated = _cached_footprints(
        context, (spec.kind, spec.length, round(float(sweep), 6), twist),
        lambda: short_footprints(spec, centres, centre_jch, context, sweep, twist),
    )
    raw = placed.score(footprints, simulated)
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
            other_colors=np.vstack(others) if others else None, grid=context.grid, **kwargs,
        )
        others.append(themes[i].colors)
    return themes


def block_separation(blocks, w_j=LIGHTNESS_DISCOUNT):
    """Minimum lightness-discounted distance between colours of different blocks."""
    best = np.inf
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            best = min(best, float(discounted_distance(blocks[i], blocks[j], w_j).min()))
    return best


def build_block_palette(block_sizes, *, context=None, step=DEFAULT_STEP, twist=1,
                        fixed_colors=None, **context_options):
    """Run the block-palette pipeline; return (CAM02-UCS colours, diagnostics)."""
    timings = {}
    start = time.perf_counter()
    if context is None:
        context = make_context(**context_options)
    timings["setup"] = time.perf_counter() - start
    start = time.perf_counter()
    specs = plan_blocks(block_sizes, context, step)
    sweeps = reserved_sweeps(specs, context)
    footprints, info = select_bases(specs, context, sweeps, twist, fixed_colors)
    timings["select"] = time.perf_counter() - start
    start = time.perf_counter()
    themes = fit_blocks(specs, footprints, context, fixed_colors)
    timings["fit"] = time.perf_counter() - start
    blocks = [theme.colors for theme in themes]
    diagnostics = {
        "specs": specs, "footprints": footprints, "selection": info, "themes": themes,
        "sweeps": [footprint.sweep for footprint in footprints], "twist": twist,
        "timings": timings,
        "separation": block_separation(blocks) if len(blocks) > 1 else np.inf,
        "footprint_separation": footprint_separation(footprints) if len(blocks) > 1 else np.inf,
        "crowding": any(item.get("crowded", False) for item in info),
    }
    return np.vstack(blocks), diagnostics
