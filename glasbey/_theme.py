"""Theme arcs: one block of a block palette as a constrained, arc-length-sampled curve.

A theme arc is a smooth curve in CAM02-UCS through a pinned base colour with
free endpoints, a target perceptual length, a hue window and a chroma floor.
Colours are sampled evenly by arc length.

Naming: ``jp`` is CAM02-UCS lightness J', ``lightness`` / ``J`` is CIECAM02 J.
Hues are CIECAM02 h in degrees; offsets are unwrapped relative to the base hue.
"""

from dataclasses import dataclass, field
from functools import lru_cache
import warnings

import numpy as np
from colorspacious import CIECAM02Space
from scipy.integrate import trapezoid
from scipy.spatial.distance import cdist

from ._curves import _Domain, _Grid, _SplineProblem, _cam02ucs_to_jch
from ._grids import rgb_grid

# Tunable defaults (prototype values; see the implementation plan).
DEFAULT_STEP = 10.0  # fixed-step fallback; block palettes use block_step() by default
# Block length grows sublinearly with size: L(n) = LENGTH_MAX (1 - exp(-(n - 1) STEP_INITIAL / LENGTH_MAX)),
# so pairs get a large step (~32) and long blocks ~10, much as the 0.4.0 blocks did.
STEP_INITIAL = 32.0
LENGTH_MAX = 105.0
MIN_STEP = 4.0
BASE_MARGIN = 0.01
INSET_FACTORS = (1.0, 0.97, 0.93, 0.88)
# Tried when a chroma lower bound stops the inset toward neutral.
OUTSET_FACTORS = (1.04, 1.08, 1.12)
# CIECAM02 chroma above the lower bound a below-bound base is projected to.
CHROMA_PROJECT_MARGINS = (1.0, 2.5, 5.0)
GREY_CHROMA = 12.0
HUE_WINDOW_PAD = 8.0
ONE_SIDED_ROOM = 3.0
CONTROL_POINTS = 6
MAX_ITER = 30
LENGTH_RETRIES = (1.0, 0.88, 0.75, 0.6)
REACH_STRETCH = 1.15
SIDE_SLACK = 0.85
SOFT_ENDPOINT_WEIGHT = 2.0
MIN_TURN_FRACTION = 0.3
CHORD_ITERATIONS = 4
ENDPOINT_HUE_WEIGHT = 0.4
HUE_PROFILE_WEIGHT = 1.0
HUE_PROFILE_SCALE = 20.0
CHROMA_REWARD = 0.5
# Lightness profile for chroma: full chroma up to J' = LIGHT_CHROMA_PIVOT, easing down
# by LIGHT_CHROMA_DROP (a fraction) by J' = LIGHT_CHROMA_TOP, so light ends are paler
# tints. Used by footprints, the arc's chroma reward and the chroma floor.
LIGHT_CHROMA_DROP = 0.7
LIGHT_CHROMA_PIVOT = 70.0
LIGHT_CHROMA_TOP = 97.0
CHROMA_FLOOR = 0.7
MIN_LIGHTNESS_SLOPE = 0.15
MAX_LENGTH_FACTOR = 1.5
LIGHTNESS_DISCOUNT = 0.35
CHROMATIC_SPREAD_CHROMA = 15.0

_FL4 = CIECAM02Space.sRGB.F_L**0.25


def lightness_chroma_profile(jp):
    """Relative chroma wanted at CAM02-UCS lightness ``jp`` (1 = full, lower at the light end)."""
    x = np.clip((np.asarray(jp, dtype=float) - LIGHT_CHROMA_PIVOT)
                / (LIGHT_CHROMA_TOP - LIGHT_CHROMA_PIVOT), 0, 1)
    return 1 - LIGHT_CHROMA_DROP * x * x * (3 - 2 * x)


def block_length(n, step=None):
    """Target CAM02-UCS length of a block of ``n`` colours.

    With ``step`` given the length is ``(n - 1) * step``; otherwise it follows
    the size-dependent schedule (large steps for small blocks).
    """
    if n <= 1:
        return 0.0
    if step is not None:
        return (n - 1) * float(step)
    return float(LENGTH_MAX * -np.expm1(-(n - 1) * STEP_INITIAL / LENGTH_MAX))


def block_step(n, step=None):
    """Target distance between neighbouring colours in a block of ``n`` colours."""
    if step is not None:
        return float(step)
    return block_length(n) / (n - 1) if n > 1 else 0.0


def discounted_distance(a, b, w_j=LIGHTNESS_DISCOUNT):
    """Pairwise lightness-discounted CAM02-UCS distance ||(w_J dJ', da', db')||.

    ``a`` has shape (..., 3) and ``b`` shape (p, 3); the result has shape (..., p).
    """
    a = np.asarray(a, dtype=float)
    b = np.atleast_2d(np.asarray(b, dtype=float))
    weights = np.array([w_j, 1.0, 1.0])
    flat = (a * weights).reshape(-1, 3)
    return cdist(flat, b * weights).reshape(a.shape[:-1] + (len(b),))


def hue_offset(hue, base_hue):
    """Signed hue difference in degrees, wrapped to [-180, 180)."""
    return (np.asarray(hue) - base_hue + 180.0) % 360.0 - 180.0


def colorfulness_to_chroma(mp):
    """CAM02-UCS colourfulness M' to CIECAM02 chroma C."""
    return np.expm1(0.0228 * np.asarray(mp)) / 0.0228 / _FL4


def chroma_to_colorfulness(chroma):
    return np.log1p(0.0228 * np.asarray(chroma) * _FL4) / 0.0228


class _GamutTable:
    """Maximum CAM02-UCS colourfulness M' per (J' bin, hue bin) inside the bounds."""

    JP_BIN = 4.0
    HUE_BIN = 6.0

    def __init__(self, lightness_bounds=None, chroma_bounds=None, rgb_bounds=None):
        rgb_bounds = rgb_bounds or ((0.0, 1.0),) * 3
        cam = rgb_grid(40, *rgb_bounds, output_colorspace="CAM02-UCS").astype(float)
        jch = _cam02ucs_to_jch(cam)
        keep = np.ones(len(cam), dtype=bool)
        if lightness_bounds is not None:
            keep &= (jch[:, 0] >= lightness_bounds[0]) & (jch[:, 0] <= lightness_bounds[1])
        if chroma_bounds is not None:
            keep &= (jch[:, 1] >= chroma_bounds[0]) & (jch[:, 1] <= chroma_bounds[1])
        cam, jch = cam[keep], jch[keep]
        self.jp_edges = np.arange(0.0, 100.0 + self.JP_BIN, self.JP_BIN)
        self.hue_count = int(round(360 / self.HUE_BIN))
        table = np.zeros((len(self.jp_edges) - 1, self.hue_count))
        rows = np.clip((cam[:, 0] // self.JP_BIN).astype(int), 0, len(table) - 1)
        columns = (jch[:, 2] // self.HUE_BIN).astype(int) % self.hue_count
        np.maximum.at(table, (rows, columns), np.hypot(cam[:, 1], cam[:, 2]))
        self.table = table
        self.jp_range = (float(cam[:, 0].min()), float(cam[:, 0].max())) if len(cam) else (0.0, 0.0)

    def max_colorfulness(self, jp, hue):
        """Bilinear-interpolated M'_max at CAM02-UCS lightness ``jp`` and hue (degrees)."""
        jp = np.asarray(jp, dtype=float)
        hue = np.asarray(hue, dtype=float)
        row = np.clip(jp / self.JP_BIN - 0.5, 0, self.table.shape[0] - 1)
        column = (hue / self.HUE_BIN - 0.5) % self.hue_count
        r0 = np.floor(row).astype(int)
        r1 = np.minimum(r0 + 1, self.table.shape[0] - 1)
        c0 = np.floor(column).astype(int) % self.hue_count
        c1 = (c0 + 1) % self.hue_count
        fr = row - np.floor(row)
        fc = column - np.floor(column)
        top = self.table[r0, c0] * (1 - fc) + self.table[r0, c1] * fc
        bottom = self.table[r1, c0] * (1 - fc) + self.table[r1, c1] * fc
        return top * (1 - fr) + bottom * fr

    def max_chroma(self, jp, hue):
        return colorfulness_to_chroma(self.max_colorfulness(jp, hue))


@lru_cache(maxsize=16)
def gamut_table(lightness_bounds=None, chroma_bounds=None, rgb_bounds=None):
    """Cached :class:`_GamutTable` per bounds tuple."""
    return _GamutTable(lightness_bounds, chroma_bounds, rgb_bounds)


class _Polyline:
    """Arc-length parameterised polyline with the :class:`_Arc` call interface."""

    def __init__(self, points, anchor_index):
        self.points = np.asarray(points, dtype=float)
        lengths = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))])
        self.length = float(lengths[-1])
        self.fractions = lengths / max(self.length, 1e-12)
        self.anchor_positions = np.array([self.fractions[anchor_index]])

    def __call__(self, positions):
        positions = np.asarray(positions, dtype=float)
        return np.column_stack([
            np.interp(positions, self.fractions, self.points[:, channel]) for channel in range(3)
        ])


@dataclass
class ThemeArc:
    """A fitted theme arc for one block."""

    arc: object
    base: np.ndarray
    base_position: float
    colors: np.ndarray
    step: float
    window: tuple
    domain: object = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def size(self):
        return len(self.colors)


def _hue_margin(hue, chroma, bounds):
    width = (bounds[1] - bounds[0]) % 360
    if width == 0:
        width = 360.0
    middle = bounds[0] + width / 2
    angular = np.cos(np.deg2rad(hue - middle)) - np.cos(np.deg2rad(width / 2))
    angular[chroma < 1e-7] = 1
    return angular


def block_domain(base, domain, gamut, hue_allowance=(0.0, 0.0), chroma_floor=None):
    """The global domain intersected with a block's hue window and chroma floor."""
    chroma_floor = CHROMA_FLOOR if chroma_floor is None else chroma_floor
    base_jch = _cam02ucs_to_jch(base)
    # Above a positive chroma lower bound the arc cannot pass through neutral, so
    # the base hue is meaningful (and a free hue would swing to the opposite side).
    chromatic = base_jch[1] >= min(GREY_CHROMA, _chroma_lower(domain) or np.inf)
    hue_bounds = domain.hue_bounds
    extra = []
    if domain.extra_margins is not None:
        extra.append(domain.extra_margins)
    window = None
    if chromatic:
        minus = hue_allowance[0] + HUE_WINDOW_PAD
        plus = hue_allowance[1] + HUE_WINDOW_PAD
        if minus + plus < 340:
            window = ((base_jch[2] - minus) % 360, (base_jch[2] + plus) % 360)
            if tuple(hue_bounds) != (0.0, 360.0):
                global_hue = tuple(hue_bounds)
                extra.append(lambda colors, jch: _hue_margin(jch[:, 2], jch[:, 1], global_hue)[:, None])
            hue_bounds = window
        floor_chroma = chroma_floor * base_jch[1]

        def floor_margin(colors, jch):
            limit = np.minimum(floor_chroma, chroma_floor * gamut.max_chroma(colors[:, 0], jch[:, 2]))
            limit = limit * lightness_chroma_profile(colors[:, 0])
            return ((jch[:, 1] - limit) / 100)[:, None]

        if chroma_floor > 0:
            extra.append(floor_margin)

    def combined(colors, jch):
        return np.hstack([function(colors, jch) for function in extra])

    result = _Domain(
        domain.lightness_bounds, domain.chroma_bounds, tuple(float(v) for v in hue_bounds),
        domain.rgb_bounds, combined if extra else None, fast=True,
    )
    result.hue_window = window
    result.chromatic = bool(chromatic)
    return result


def _chroma_lower(domain):
    if domain.chroma_bounds is None or domain.chroma_bounds[0] <= 0:
        return 0.0
    return float(domain.chroma_bounds[0])


def _with_chroma(base, chroma):
    """``base`` moved radially in a'b' to CIECAM02 chroma ``chroma`` (same J', hue)."""
    ab = np.asarray(base[1:], dtype=float)
    norm = float(np.hypot(*ab))
    unit = ab / norm if norm > 1e-9 else np.array([1.0, 0.0])
    low, high = 0.0, 100.0
    for _ in range(40):
        middle = (low + high) / 2
        if _cam02ucs_to_jch(np.array([base[0], *(middle * unit)]))[1] < chroma:
            low = middle
        else:
            high = middle
    return np.array([base[0], *(high * unit)])


def _inset_base(base, domain, gamut, hue_allowance, chroma_floor):
    # Margin to the global bounds only: the hue window and chroma floor are
    # centred on the base by construction.
    plain = _Domain(domain.lightness_bounds, domain.chroma_bounds, domain.hue_bounds,
                    domain.rgb_bounds, fast=True)
    for factor in INSET_FACTORS:
        candidate = np.array([base[0], base[1] * factor, base[2] * factor])
        local = block_domain(candidate, domain, gamut, hue_allowance, chroma_floor)
        if (plain.margins(candidate).min() >= BASE_MARGIN and local.contains(candidate)[0]) or (
            factor == INSET_FACTORS[-1] and local.contains(candidate)[0]
        ):
            yield factor, candidate, local
    lower = _chroma_lower(domain)
    if lower and _cam02ucs_to_jch(base)[1] < lower + CHROMA_PROJECT_MARGINS[-1]:
        # A base below (or on) the chroma lower bound, e.g. a grey: push it radially
        # out past the bound, keeping its lightness and hue.
        for margin in CHROMA_PROJECT_MARGINS:
            candidate = _with_chroma(base, lower + margin)
            local = block_domain(candidate, domain, gamut, hue_allowance, chroma_floor)
            if plain.margins(candidate).min() >= BASE_MARGIN and local.contains(candidate)[0]:
                yield 1.0, candidate, local
    if lower:
        # A base on the chroma lower bound cannot move toward neutral; move outward.
        for factor in OUTSET_FACTORS:
            candidate = np.array([base[0], base[1] * factor, base[2] * factor])
            local = block_domain(candidate, domain, gamut, hue_allowance, chroma_floor)
            if local.contains(candidate)[0]:
                yield factor, candidate, local
    # Last resort: clamp into the lightness bounds and pull toward neutral.
    jch = _cam02ucs_to_jch(base)
    if domain.lightness_bounds is not None:
        low, high = domain.lightness_bounds
        if not low <= jch[0] <= high:
            target = np.clip(jch[0], low + 1, high - 1)
            jp = 1.7 * target / (1 + 0.007 * target)
            base = np.array([jp, base[1], base[2]])
    for factor in (0.75, 0.5, 0.25):
        candidate = np.array([base[0], base[1] * factor, base[2] * factor])
        local = block_domain(candidate, domain, gamut, hue_allowance, chroma_floor)
        if local.contains(candidate)[0]:
            yield factor, candidate, local
            return


def _side_targets(total, reach_dark, reach_light):
    dark = light = total / 2
    if reach_dark < dark:
        light = min(reach_light, total - reach_dark)
        dark = reach_dark
    elif reach_light < light:
        dark = min(reach_dark, total - reach_light)
        light = reach_light
    if reach_dark < ONE_SIDED_ROOM:
        dark, light = 0.0, min(reach_light, total)
    if reach_light < ONE_SIDED_ROOM:
        light, dark = 0.0, min(reach_dark, total)
    return dark, light


def _target_offsets(hue_allowance, twist):
    # Light end rotates by ``twist``; dark end rotates the other way.
    if twist >= 0:
        return -hue_allowance[0], hue_allowance[1]
    return hue_allowance[1], -hue_allowance[0]


def _pick_endpoint(base, candidates, side_length, target_offset, base_hue, chromatic,
                   target_point=None):
    distances = np.linalg.norm(candidates - base, axis=1)
    if target_point is not None:
        # Follow the predicted direction (e.g. a footprint end) at the side's length.
        direction = np.asarray(target_point, dtype=float) - base
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            goal = base + direction * side_length / norm
            return candidates[int(np.argmin(np.linalg.norm(candidates - goal, axis=1)))]
    score = np.abs(distances - side_length)
    if chromatic:
        offsets = hue_offset(_cam02ucs_to_jch(candidates)[:, 2], base_hue)
        score = score + ENDPOINT_HUE_WEIGHT * np.abs(offsets - target_offset)
    return candidates[int(np.argmin(score))]


def _hue_profile(base_jch, dark_jp, light_jp, offsets, chromatic):
    """Return a callable giving the target hue offset at CAM02-UCS lightness."""
    base_jp = None

    def target(jp):
        dark_rate = offsets[0] / min(dark_jp - base_jp, -1e-6) if dark_jp is not None else 0.0
        light_rate = offsets[1] / max(light_jp - base_jp, 1e-6) if light_jp is not None else 0.0
        delta = jp - base_jp
        result = np.where(delta < 0, dark_rate * delta, light_rate * delta)
        return np.clip(result, min(offsets[0], 0, offsets[1]), max(offsets[0], 0, offsets[1]))

    def bind(jp_base):
        nonlocal base_jp
        base_jp = jp_base
        return target

    return bind


def _make_terms(base_jch, profile, chromatic, use_profile, length_target, slope, twist=0,
                side_goals=(0.0, 0.0), min_turn=0.0):
    """Extra objective / constraint terms for the batched ("fast") spline solver.

    Both callables receive a batch of variable vectors (B, k) and the batched
    geometry ``(colors (B, M, 3), length (B,), fractions (B, M), bending (B,))``.
    """
    base_chroma = max(base_jch[1], 1e-6)
    scale = max(length_target, 1.0)

    def objective(problem, variables, geometry):
        colors, length, fractions, bending = geometry
        if not (chromatic and use_profile):
            return 0.0
        jch = _cam02ucs_to_jch(colors)
        weight = np.clip(jch[..., 1] / base_chroma, 0, 1)
        error = (hue_offset(jch[..., 2], base_jch[2]) - profile(colors[..., 0])) / HUE_PROFILE_SCALE
        return HUE_PROFILE_WEIGHT * trapezoid(weight * error**2, fractions, axis=-1)

    def constraints(problem, variables, geometry):
        colors, length, fractions, bending = geometry
        coefficients = problem.coefficients_batch(variables)
        velocity = np.matmul(problem.first, coefficients)
        speed = np.sqrt(np.sum(velocity**2, axis=-1))
        parts = [
            (length / length_target - 1)[:, None],
            (velocity[..., 0] - slope * speed) / scale,
        ]
        # Each side of the base must carry its share, so the window stays centred.
        position = fractions[:, np.searchsorted(problem.mesh, problem.anchor_parameters[0])]
        for goal, side in zip(side_goals, (position, 1 - position)):
            if goal > 0:
                parts.append((side * length / goal - 1)[:, None])
        if twist:
            # Hue turns only in the twist direction, M'^2 dh/dt = a' b'_t - b' a'_t, at no
            # less than min_turn radians per unit parameter (scaled by M'^2 so it fades at grey).
            turning = colors[..., 1] * velocity[..., 2] - colors[..., 2] * velocity[..., 1]
            squared = colors[..., 1]**2 + colors[..., 2]**2
            parts.append((twist * turning - min_turn * squared) / scale**2)
        return np.concatenate(parts, axis=1)

    return objective, constraints


_SEGMENT = np.linspace(0, 1, 33)[:, None]


def _route(grid, start, end):
    """Feasible monotone polyline from ``start`` to ``end``.

    Tries the straight segment, then a single bend pulled toward the neutral
    axis (cheap and usually enough on gamut surfaces), then the grid search.
    """
    if grid.domain.contains(start + _SEGMENT * (end - start)).all():
        return np.vstack([start, end])
    middle = (start + end) / 2
    for pull in (0.85, 0.7, 0.5):
        bend = np.array([middle[0], middle[1] * pull, middle[2] * pull])
        if grid.domain.contains(np.vstack([
            start + _SEGMENT * (bend - start), bend + _SEGMENT * (end - bend),
        ])).all():
            return np.vstack([start, bend, end])
    try:
        return grid.connect(start, end)
    except ValueError:
        return None


def _sample_window(arc, n, length, base_position, offset=0.0):
    """Fractions of ``n`` samples covering ``length`` of the arc with equal chords.

    Samples start evenly spaced by arc length, then are respaced so the straight
    distances between neighbours are equal (they differ on tightly curved arcs).
    """
    width = min(1.0, length / arc.length)
    start = np.clip(base_position - width / 2 + offset / arc.length, 0.0, 1.0 - width)
    positions = np.linspace(start, start + width, n)
    if n > 2:
        for _ in range(CHORD_ITERATIONS):
            chords = np.linalg.norm(np.diff(arc(positions), axis=0), axis=1)
            if chords.min() <= 0 or chords.std() < 1e-4 * chords.mean():
                break
            cumulative = np.concatenate([[0.0], np.cumsum(chords)])
            targets = np.linspace(0, cumulative[-1], n)
            positions = np.interp(targets, cumulative, positions)
    return positions, (float(start), float(start + width))


def _best_offset(arc, n, length, base_position, other_colors):
    width = min(1.0, length / arc.length)
    if other_colors is None or len(other_colors) == 0 or width >= 0.999:
        return 0.0
    slack = min(0.25 * length, (1 - width) * arc.length)
    best, best_score = 0.0, -np.inf
    for offset in np.linspace(-slack, slack, 9):
        positions, _ = _sample_window(arc, n, length, base_position, offset)
        score = discounted_distance(arc(positions), other_colors).min()
        # Prefer the centred window unless moving clearly helps.
        score -= 1e-3 * abs(offset)
        if score > best_score + 1e-9:
            best, best_score = offset, score
    return best


def _diagnostics(colors, base_jch, target_length, arc_length, inset, fallback, step):
    jch = _cam02ucs_to_jch(colors)
    steps = np.linalg.norm(np.diff(colors, axis=0), axis=1)
    chromatic = jch[:, 1] > CHROMATIC_SPREAD_CHROMA
    if chromatic.sum() >= 2:
        offsets = hue_offset(jch[chromatic, 2], base_jch[2])
        spread = float(offsets.max() - offsets.min())
    else:
        spread = 0.0
    return {
        "step": float(steps.mean()) if len(steps) else 0.0,
        "target_step": float(step),
        "step_cv": float(steps.std() / steps.mean()) if len(steps) and steps.mean() > 0 else 0.0,
        "monotone_lightness": bool(np.all(np.diff(colors[:, 0]) > 0)) if len(colors) > 1 else True,
        "hue_spread": spread,
        "min_chroma": float(jch[:, 1].min()),
        "inset": float(inset),
        "target_length": float(target_length),
        "effective_length": float(steps.sum()),
        "arc_length": float(arc_length),
        "fallback": bool(fallback),
    }


def fit_theme_arc(base, n, *, step=None, domain=None, gamut=None, hue_allowance=(0.0, 0.0),
                  twist=1, chroma_floor=None, other_colors=None, grid=None, max_iter=MAX_ITER,
                  warn=True, endpoint_targets=None, soft_endpoints=False):
    """Fit one block as a smooth arc through ``base`` and sample ``n`` colours.

    Parameters
    ----------
    base : array of shape (3,)
        Base colour in CAM02-UCS. It may be inset toward neutral to make a
        smooth curve through it feasible.
    n : int
        Number of colours to sample.
    step : float or None
        Target CAM02-UCS distance between adjacent colours; ``None`` uses the
        size-dependent :func:`block_step` schedule.
    domain : _Domain
        Global colour domain (CIECAM02 bounds plus sRGB bounds).
    gamut : _GamutTable
        Gamut table for the domain's bounds; used by the chroma floor.
    hue_allowance : (float, float)
        Degrees the block may rotate below and above the base hue.
    twist : {1, -1}
        Direction the light end rotates in hue.
    chroma_floor : float
        Fraction of ``min(C_base, C_max(J, h))`` chroma that colours must keep.
    other_colors : array, optional
        Already-placed colours; the sample window is slid to avoid them.
    grid : _Grid, optional
        A prebuilt grid over ``domain`` used for routing and endpoint search.
    endpoint_targets : (array, array), optional
        Predicted dark and light ends (e.g. from a block footprint); provisional
        endpoints follow their directions from the base.
    soft_endpoints : bool or float
        Also hold the arc's ends near those provisional endpoints with soft
        anchors (``True`` uses ``SOFT_ENDPOINT_WEIGHT``; a float is the weight),
        so the block stays close to its predicted footprint.
    """
    base = np.asarray(base, dtype=float).reshape(3)
    domain = domain or _Domain(fast=True)
    if gamut is None:
        gamut = gamut_table(
            None if domain.lightness_bounds is None else tuple(domain.lightness_bounds),
            None if domain.chroma_bounds is None else tuple(domain.chroma_bounds),
            tuple(map(tuple, domain.rgb_bounds)),
        )
    step = block_step(n, step)
    target_length = (n - 1) * step
    base_jch = _cam02ucs_to_jch(base)
    if n == 1:
        colors = base[None].copy()
        return ThemeArc(None, base, 0.5, colors, 0.0, (0.5, 0.5), domain,
                        _diagnostics(colors, base_jch, 0.0, 0.0, 1.0, False, step))
    if grid is None:
        grid = _Grid(domain, 32, "RGB", base[None])

    fallback_route = None
    for inset, inset_base, local in _inset_base(base, domain, gamut, hue_allowance, chroma_floor):
        local_jch = _cam02ucs_to_jch(inset_base)
        chromatic = local.chromatic
        view = grid.restricted(local)
        if len(view.colors) < 4:
            continue
        candidates = view.colors
        dark_side = candidates[candidates[:, 0] < inset_base[0] - 1.0]
        light_side = candidates[candidates[:, 0] > inset_base[0] + 1.0]
        reach_dark = float(np.linalg.norm(dark_side - inset_base, axis=1).max()) if len(dark_side) else 0.0
        reach_light = float(np.linalg.norm(light_side - inset_base, axis=1).max()) if len(light_side) else 0.0
        dark_length, light_length = _side_targets(target_length, 0.9 * reach_dark, 0.9 * reach_light)
        offsets = _target_offsets(hue_allowance, twist)
        routes = []
        targets = endpoint_targets if endpoint_targets is not None else (None, None)
        for side_length, side, offset, target in ((dark_length, dark_side, offsets[0], targets[0]),
                                                  (light_length, light_side, offsets[1], targets[1])):
            if side_length <= 0 or not len(side):
                routes.append(None)
                continue
            endpoint = _pick_endpoint(inset_base, side, side_length, offset, local_jch[2], chromatic,
                                      target)
            routes.append(_route(view, inset_base, endpoint))
        if routes[0] is None and routes[1] is None:
            continue
        dark_route = routes[0][::-1] if routes[0] is not None else inset_base[None]
        light_route = routes[1][1:] if routes[1] is not None else np.zeros((0, 3))
        points = np.vstack([dark_route, light_route])
        anchor_index = len(dark_route) - 1
        fallback_route = (points, anchor_index, inset, inset_base)
        chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
        if chord[-1] <= 1e-6:
            continue
        parameters = chord / chord[-1]
        side_reach = [
            min(dark_length, np.linalg.norm(points[0] - inset_base)),
            min(light_length, np.linalg.norm(points[-1] - inset_base)),
        ]
        # A smooth arc may bow beyond the straight-line reach; retries step down.
        reachable = min(target_length, REACH_STRETCH * max(chord[-1], sum(side_reach)))
        profile = _hue_profile(local_jch, points[0, 0] if anchor_index > 0 else None,
                               points[-1, 0] if anchor_index < len(points) - 1 else None,
                               offsets, chromatic)(inset_base[0])
        use_profile = True
        turn = int(np.sign(twist)) if chromatic and max(hue_allowance) > 0 else 0
        for factor in LENGTH_RETRIES:
            length_goal = max(reachable * factor, 1e-3)
            anchors, anchor_parameters, weights = [inset_base], [parameters[anchor_index]], [None]
            endpoint_weight = (SOFT_ENDPOINT_WEIGHT if soft_endpoints is True else float(soft_endpoints or 0))
            if endpoint_targets is not None and endpoint_weight > 0:
                # Keep the arc near the footprint the block was selected for.
                if anchor_index > 0:
                    anchors.append(points[0]); anchor_parameters.append(0.0)
                    weights.append(endpoint_weight)
                if anchor_index < len(points) - 1:
                    anchors.append(points[-1]); anchor_parameters.append(1.0)
                    weights.append(endpoint_weight)
            problem = _SplineProblem(
                np.vstack(anchors), points, parameters, np.array(anchor_parameters),
                CONTROL_POINTS, local, anchor_weights=weights, direction=1,
                scale=max(length_goal, 1.0), profile="fast", chroma_profile=lightness_chroma_profile,
            )
            share = SIDE_SLACK * length_goal / max(sum(side_reach), 1e-9)
            objective, constraints = _make_terms(
                local_jch, profile, chromatic, use_profile, length_goal, MIN_LIGHTNESS_SLOPE, turn,
                (share * side_reach[0] if anchor_index > 0 else 0.0,
                 share * side_reach[1] if anchor_index < len(points) - 1 else 0.0),
                # One-sided (warm) allowances keep the dark half at the base hue.
                MIN_TURN_FRACTION * np.deg2rad(sum(hue_allowance))
                if turn and min(hue_allowance) > 0 else 0.0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = problem.solve(
                    problem.initial, max_iter, smoothness=1.0, chroma_weight=CHROMA_REWARD,
                    chroma_scale=max(view.chroma_scale, 1.0), reference_length=length_goal,
                    arc_length_factor=MAX_LENGTH_FACTOR, length_weight=0.0,
                    extra_objective=objective, extra_constraints=constraints,
                )
            if fitted is None:
                continue
            arc = fitted[1]
            result = _finish(arc, n, step, target_length, inset_base, local_jch, inset, False,
                             local, other_colors, warn)
            if result is not None:
                result.diagnostics["length_factor"] = factor
                result.diagnostics["converged"] = bool(fitted[2].success)
                return result

    if fallback_route is None:
        # No feasible region around the base: use in-domain grid colours near it,
        # spread in lightness, so the block is still valid and ordered.
        colors = _nearest_ramp(base, n, step, grid)
        return ThemeArc(None, base, 0.5, colors, 0.0, (0.0, 1.0), domain,
                        _diagnostics(colors, base_jch, target_length, 0.0, 1.0, True, step))
    points, anchor_index, inset, inset_base = fallback_route
    arc = _Polyline(points, anchor_index)
    result = _finish(arc, n, step, target_length, inset_base, _cam02ucs_to_jch(inset_base), inset,
                     True, domain, other_colors, warn, require_monotone=False)
    return result


def _nearest_ramp(base, n, step, grid):
    """n grid colours near ``base`` with distinct lightness, ordered dark to light."""
    colors = grid.colors
    gap = max(2.5, min(step, 6.0))
    order = np.argsort(discounted_distance(colors, base[None])[:, 0])
    chosen = []
    for index in order:
        if all(abs(colors[index, 0] - colors[other, 0]) >= gap for other in chosen):
            chosen.append(index)
            if len(chosen) == n:
                break
    if len(chosen) < n:  # very narrow domain: relax the lightness gap
        chosen = list(order[np.linspace(0, len(order) - 1, n).astype(int)])
    result = colors[chosen]
    return result[np.argsort(result[:, 0], kind="stable")]


def _finish(arc, n, step, target_length, base, base_jch, inset, fallback, domain, other_colors,
            warn, require_monotone=True):
    length = min(target_length, 0.98 * arc.length)
    base_position = float(arc.anchor_positions[0])
    offset = _best_offset(arc, n, length, base_position, other_colors)
    positions, window = _sample_window(arc, n, length, base_position, offset)
    colors = arc(positions)
    if require_monotone and not np.all(np.diff(colors[:, 0]) > 0):
        return None
    effective_step = length / (n - 1)
    if warn and effective_step < MIN_STEP:
        warnings.warn(
            f"A block of {n} colours could only reach a step of {effective_step:.1f} "
            f"(target {step:.1f}); consider fewer colours per block or wider bounds.",
            RuntimeWarning, stacklevel=3,
        )
    diagnostics = _diagnostics(colors, base_jch, target_length, arc.length, inset, fallback, step)
    diagnostics["window_offset"] = float(offset)
    return ThemeArc(arc, base, base_position, colors, effective_step, window, domain, diagnostics)


def sample_theme_arc(theme, n=None, *, length=None, offset=0.0):
    """Resample a fitted theme arc, optionally with a different length or window offset.

    ``offset`` slides the window along the arc in CAM02-UCS distance units.
    Returns an (n, 3) array of CAM02-UCS colours.
    """
    n = theme.size if n is None else n
    if theme.arc is None or n == 1:
        return theme.colors[:n].copy()
    if length is None:
        length = theme.step * (n - 1)
    length = min(length, 0.98 * theme.arc.length)
    positions, _ = _sample_window(theme.arc, n, length, theme.base_position, offset)
    return theme.arc(positions)
