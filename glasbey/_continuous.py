"""Constrained, seeded curves in CAM02-UCS."""

from dataclasses import dataclass
import warnings

import numpy as np
from colorspacious import cspace_convert
from matplotlib.colors import ListedColormap, to_rgb
from scipy.optimize import minimize

from ._converters import palette_to_sRGB1, get_rgb_palette
from ._curves import (  # noqa: F401  (re-exported for backwards compatibility)
    _FEASIBILITY_TOL,
    _Arc,
    _Domain,
    _Grid,
    _SplineProblem,
    _cam02ucs_to_jch,
    _chroma_reward,
    _color_spaces,
    _safe_convert,
    _validate_curve,
)


@dataclass
class _ArmFit:
    problem: _SplineProblem
    variables: np.ndarray
    arc: _Arc
    reference_length: float
    result: object
    validation_depth: int
    used_reference: bool


def _arm_quality(problem, variables, reference_length, smoothness, chroma_weight, chroma_scale):
    colors, length, fractions, bending = problem.geometry(variables)
    value = smoothness * reference_length * bending / (4 * np.pi**2) - length / reference_length
    if chroma_weight:
        value -= chroma_weight * _chroma_reward(colors, fractions, chroma_scale)
    sampled = colors[::8]
    separation = np.linalg.norm(sampled[:, None] - sampled[None, :], axis=-1)
    along = np.abs(fractions[::8, None] - fractions[None, ::8]) * length
    nonlocal_pairs = along > length * 0.2
    if nonlocal_pairs.any():
        value += np.square(np.maximum(
            0, 0.2 - separation[nonlocal_pairs] / along[nonlocal_pairs]
        )).mean()
    return float(value)


def _fit_arm(anchors, grid, max_control_points, arc_length_factor,
             smoothness, chroma_weight, max_iter):
    points, parameters, anchor_parameters = grid.waypoints(anchors)
    minimum = len(anchors) + 2
    budget = max_control_points or max(12, 2 * len(anchors) + 4)
    if budget < minimum:
        raise ValueError(f"max_control_points must be at least {minimum} for this arm")
    reference = None
    for count in sorted({min(budget, minimum + 2), budget}):
        problem = _SplineProblem(
            anchors, points, parameters, anchor_parameters, count, grid.domain
        )
        reference = problem.solve(problem.initial, max_iter)
        if reference is not None:
            break
    if reference is None:
        raise ValueError(
            "No feasible monotone spline found; increase max_control_points, grid_size "
            "or max_iter, or reconsider the seeds/bounds. Constraints were not relaxed."
        )
    variables, reference_arc, result, depth = reference
    best = reference
    reference_length = reference_arc.length
    starts = [variables]
    if chroma_weight > 0 and arc_length_factor > 1:
        midpoint = reference_arc(0.5)
        nearby = np.abs(grid.colors[:, 0] - midpoint[0]) <= max(
            2, abs(anchors[-1, 0] - anchors[0, 0]) / 4
        )
        candidates = np.flatnonzero(nearby)
        ranked = candidates[np.argsort(-grid.chroma[candidates], kind="stable")]
        chosen = []
        for candidate in ranked:
            guide = grid.colors[candidate]
            if any(np.linalg.norm(guide[1:] - prior[1:]) < 15 for prior in chosen):
                continue
            chosen.append(guide)
            target = problem.values @ problem.coefficients(variables)
            bow = 16 * problem.mesh**2 * (1 - problem.mesh)**2
            target[:, 1:] += bow[:, None] * (guide[1:] - midpoint[1:]) * 0.5
            alternative = np.linalg.lstsq(
                problem.values @ problem.free,
                (target - problem.values @ problem.fixed) / problem.scale,
                rcond=None,
            )[0].ravel()
            starts.append(alternative)
            if len(chosen) == 2:
                break

    def quality(fit):
        return _arm_quality(
            problem, fit[0], reference_length, smoothness, chroma_weight, grid.chroma_scale
        )

    best_quality = quality(best)
    for initial in starts:
        fitted = problem.solve(
            initial, max_iter, smoothness, chroma_weight,
            grid.chroma_scale, reference_length, arc_length_factor,
        )
        if fitted is not None and quality(fitted) < best_quality:
            best, best_quality = fitted, quality(fitted)
    variables, arc, result, depth = best
    return _ArmFit(problem, variables, arc, reference_length, result, depth, best is reference)


def _arm_diagnostics(fit):
    arc = fit.arc
    samples = arc(np.linspace(0, 1, 257))
    distances = np.linalg.norm(np.diff(samples, axis=0), axis=1)
    velocity = arc.spline.derivative()(fit.problem.mesh)
    acceleration = arc.spline.derivative(2)(fit.problem.mesh)
    curvature = np.linalg.norm(np.cross(velocity, acceleration), axis=1) / np.maximum(
        np.linalg.norm(velocity, axis=1), 1e-9
    )**3
    diagnostics = {
        "converged": bool(fit.result.success),
        "message": str(fit.result.message),
        "control_points": len(arc.spline.c),
        "arc_length": arc.length,
        "reference_length": fit.reference_length,
        "length_ratio": arc.length / fit.reference_length,
        "seed_error": float(np.linalg.norm(
            arc(arc.anchor_positions) - fit.problem.anchors, axis=1
        ).max()),
        "step_distance_cv": float(distances.std() / distances.mean()),
        "max_relative_step_deviation": float(np.max(np.abs(distances / distances.mean() - 1))),
        "max_curvature": float(curvature.max()),
        "max_constraint_violation": float(max(0, -fit.problem.domain.margins(samples).min())),
        "weighted_chroma": _chroma_reward(samples, np.linspace(0, 1, 257), 1),
        "midpoint_chroma": float(cspace_convert(arc(0.5), "CAM02-UCS", "JCh")[1]),
        "validation_depth": fit.validation_depth,
        "used_reference": fit.used_reference,
    }
    if not fit.result.success:
        warnings.warn(
            "Returning a validated feasible colormap before optimizer convergence: "
            + str(fit.result.message), RuntimeWarning, stacklevel=3,
        )
    return diagnostics


def _balance_arms(fits, grid, arc_length_factor, smoothness, chroma_weight, max_iter):
    split = fits[0].variables.size
    initial = np.concatenate([fit.variables for fit in fits])

    def unpack(variables):
        return variables[:split], variables[split:]

    def objective(variables):
        parts = unpack(variables)
        try:
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                score = np.mean([
                    _arm_quality(fit.problem, part, fit.reference_length, smoothness,
                                 chroma_weight, grid.chroma_scale)
                    for fit, part in zip(fits, parts)
                ])
                lengths = [fit.problem.geometry(part)[1] for fit, part in zip(fits, parts)]
                tangents = [fit.problem.spline(part).derivative()(0) for fit, part in zip(fits, parts)]
                alignment = np.dot(tangents[0], tangents[1]) / max(
                    np.linalg.norm(tangents[0]) * np.linalg.norm(tangents[1]), 1e-12
                )
                score += 0.1 * ((lengths[0] - lengths[1]) / sum(lengths))**2
                score += 0.1 * (1 + np.clip(alignment, -1, 1))
            return float(score) if np.isfinite(score) else 1e12
        except (ValueError, FloatingPointError, IndexError):
            return 1e12

    def constraints(variables):
        margins = []
        for fit, part in zip(fits, unpack(variables)):
            problem = fit.problem
            coefficients = problem.coefficients(part)
            margins.extend([
                (grid.domain.margins(problem.values @ coefficients) - problem.safety).ravel(),
                problem.direction * np.diff(coefficients[:, 0]) / 100,
                [arc_length_factor - problem.geometry(part)[1] / fit.reference_length],
            ])
        return np.concatenate(margins)

    result = minimize(
        objective, initial, method="SLSQP",
        constraints={"type": "ineq", "fun": constraints},
        options={"maxiter": max_iter, "ftol": 1e-8},
    )
    if constraints(result.x).min() < -_FEASIBILITY_TOL or objective(result.x) >= objective(initial):
        return fits, False
    refined = []
    for fit, part in zip(fits, unpack(result.x)):
        spline = fit.problem.spline(part)
        valid, extra, depth = _validate_curve(spline, grid.domain, fit.problem.anchor_parameters)
        if not valid:
            return fits, False
        arc = _Arc(spline, fit.problem.anchor_parameters)
        if arc.length > arc_length_factor * fit.reference_length + 1e-5:
            return fits, False
        refined.append(_ArmFit(fit.problem, part, arc, fit.reference_length, result, depth, False))
    return refined, True


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _bounds(value, name, minimum, maximum=None, allow_none=False, wrap=False):
    if value is None and allow_none:
        return None
    try:
        values = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must contain two finite bounds") from None
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain two finite bounds")
    if (values < minimum).any() or (maximum is not None and (values > maximum).any()):
        raise ValueError(f"{name} is outside its valid range")
    if not wrap and values[0] > values[1]:
        raise ValueError(f"{name} must be in increasing order")
    return tuple(values)


def _parse_seeds(seed_colors):
    try:
        if isinstance(seed_colors, str):
            colors = palette_to_sRGB1(seed_colors)
        else:
            items = list(seed_colors)
            for color in items:
                if not isinstance(color, str):
                    channels = np.asarray(color, dtype=float)
                    if channels.shape not in ((3,), (4,)) or not np.isfinite(channels).all():
                        raise ValueError("Each seed must be a color string or a finite RGB/RGBA tuple")
                    if np.any(channels < 0) or np.any(channels > 255):
                        raise ValueError("Seed channels must be in [0, 1] or [0, 255]")
            colors = palette_to_sRGB1(items)
            if len(colors) != len(items):
                raise ValueError("Every supplied seed must be a valid color")
    except (TypeError, IndexError, ValueError) as error:
        raise ValueError(f"Invalid seed colors: {error}") from error
    if colors.ndim != 2 or colors.shape[1] != 3 or not np.isfinite(colors).all():
        raise ValueError("seed_colors must contain finite RGB colors")
    return np.asarray(colors, dtype=float)


def _ordered(indices, seeds, increasing=True):
    ordered = sorted(indices, key=lambda index: (
        seeds[index, 0] if increasing else -seeds[index, 0], index
    ))
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and abs(seeds[ordered[end], 0] - seeds[ordered[start], 0]) < 1e-10:
            end += 1
        if end - start > 1 and start:
            remaining = ordered[start:end]
            previous = ordered[start - 1]
            for position in range(start, end):
                chosen = min(remaining, key=lambda index: (
                    np.linalg.norm(seeds[index] - seeds[previous]), index
                ))
                ordered[position] = chosen
                remaining.remove(chosen)
                previous = chosen
        start = end
    return ordered


def _partition_arms(seeds, center, light_center):
    remaining = [index for index in range(len(seeds)) if index != center]
    chromatic = seeds[remaining, 1:]
    distances = np.linalg.norm(chromatic[:, None] - chromatic[None, :], axis=-1)
    np.fill_diagonal(distances, -np.inf)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    groups = [[remaining[first]], [remaining[second]]]

    def ordered(group):
        return [center] + _ordered(group, seeds, increasing=not light_center)

    def cost(group):
        points = seeds[ordered(group)]
        steps = np.diff(points, axis=0)
        lengths = np.linalg.norm(steps, axis=1)
        unit = steps / np.maximum(lengths[:, None], 1e-9)
        return float(lengths.sum() + 5 * np.square(np.diff(unit, axis=0)).sum())

    for index in remaining:
        if index in groups[0] or index in groups[1]:
            continue
        arm = min(range(2), key=lambda side: cost(groups[side] + [index]) - cost(groups[side]))
        groups[arm].append(index)
    for iteration in range(min(16, len(seeds) * 2)):
        best = groups
        best_cost = cost(groups[0]) + cost(groups[1])
        for side in range(2):
            if len(groups[side]) > 1:
                for index in groups[side]:
                    candidate = [group.copy() for group in groups]
                    candidate[side].remove(index)
                    candidate[1 - side].append(index)
                    candidate_cost = cost(candidate[0]) + cost(candidate[1])
                    if candidate_cost < best_cost - 1e-9:
                        best, best_cost = candidate, candidate_cost
        for left in groups[0]:
            for right in groups[1]:
                candidate = [
                    [index for index in groups[0] if index != left] + [right],
                    [index for index in groups[1] if index != right] + [left],
                ]
                candidate_cost = cost(candidate[0]) + cost(candidate[1])
                if candidate_cost < best_cost - 1e-9:
                    best, best_cost = candidate, candidate_cost
        if best is groups:
            break
        groups = best
    return sorted([ordered(group) for group in groups], key=lambda group: group[-1])


class ContinuousColormap:
    """A fitted CAM02-UCS curve with perceptual arc-length coordinates.

    Instances are constructed by :func:`create_colormap`. Calling an instance
    with coordinates in [0, 1] returns float sRGB colors. ``seed_positions`` is
    in the original input order, including duplicate seeds. ``seed_order`` and
    ``arm_assignments`` identify original input indices. ``diagnostics`` reports
    per-arm fitting and sampling quality. The diverging center is exactly 0.5.

    Feasibility is checked numerically during fitting and on every evaluation;
    the representation is not an analytical certificate of gamut containment.
    """

    def __init__(self, arcs, domain, seed_positions, seed_order, assignments, diagnostics):
        self._arcs = tuple(arcs)
        self._domain = domain
        self.seed_positions = np.asarray(seed_positions)
        self.seed_order = tuple(seed_order)
        self.arm_assignments = tuple(tuple(arm) for arm in assignments)
        self.mode = "sequential" if len(arcs) == 1 else "diverging"
        self.diagnostics = diagnostics

    def __call__(self, positions):
        positions = np.asarray(positions, dtype=float)
        if not np.isfinite(positions).all() or np.any((positions < 0) | (positions > 1)):
            raise ValueError("Colormap positions must be finite and in [0, 1]")
        flat = positions.ravel()
        if self.mode == "sequential":
            colors = self._arcs[0](flat)
        else:
            colors = np.empty((len(flat), 3))
            left = flat <= 0.5
            colors[left] = self._arcs[0](1 - 2 * flat[left])
            colors[~left] = self._arcs[1](2 * flat[~left] - 1)
        if not self._domain.contains(colors).all():
            raise ValueError("Evaluation exposed a color outside the fitted constraints")
        rgb = cspace_convert(colors, "CAM02-UCS", "sRGB1")
        return np.clip(rgb, 0, 1).reshape(positions.shape + (3,))

    def sample(self, n_colors=256, *, as_hex=False, include_seeds=False):
        """Return (positions, colors), optionally inserting all exact seed positions.

        Inserting seeds may increase the count and makes positions nonuniform.
        Hex output is quantized and is rejected if rounding violates bounds.
        """
        n_colors = _positive_integer(n_colors, "n_colors", 2)
        positions = np.linspace(0, 1, n_colors)
        if include_seeds:
            positions = np.unique(np.concatenate([positions, self.seed_positions]))
        colors = self(positions)
        if as_hex:
            colors = get_rgb_palette(cspace_convert(colors, "sRGB1", "CAM02-UCS"))
            rounded = cspace_convert([to_rgb(color) for color in colors], "sRGB1", "CAM02-UCS")
            if not self._domain.contains(rounded).all():
                raise ValueError("Hex rounding violates color bounds; use as_hex=False")
        return positions, colors

    def to_matplotlib(self, name="glasbey", n_colors=256):
        """Return a finite ListedColormap sampled without additional RGB interpolation."""
        return ListedColormap(self.sample(n_colors)[1], name=name)


def create_colormap(
    seed_colors,
    *,
    mode="sequential",
    direction="increasing",
    center_lightness="light",
    center_color=None,
    grid_size=64,
    grid_space="RGB",
    lightness_bounds=None,
    chroma_bounds=None,
    hue_bounds=(0, 360),
    red_bounds=(0, 1),
    green_bounds=(0, 1),
    blue_bounds=(0, 1),
    max_control_points=None,
    arc_length_factor=1.5,
    smoothness=1.0,
    chroma_weight=1.0,
    max_iter=100,
):
    """Fit a perceptually sampled continuous colormap through unordered colors.

    Parameters
    ----------
    seed_colors : sequence of colors or matplotlib palette name
        Fixed anchors. Color parsing follows ``extend_palette``. Sequential maps
        need two distinct colors; diverging maps need three. Duplicates are kept
        in ``seed_positions`` but fitted only once.
    mode : {"sequential", "diverging"}
        Sequential maps have monotone lightness. Diverging maps have two
        independently uniform arms meeting at a supplied seed at position 0.5.
    direction : {"increasing", "decreasing"}
        Lightness direction for sequential maps.
    center_lightness : {"light", "dark"}
        Whether a diverging center is a lightness maximum or minimum.
    center_color : color, optional
        A supplied seed to use as center; it must have the requested extreme
        lightness. Otherwise choose an extreme seed, breaking ties by chroma.
    grid_size : int or triple of int
        Grid resolution used to guide routing and chromatic exploration.
    grid_space : {"RGB", "JCh"}
        Grid sampling space, independent of which bounds are enforced.
    lightness_bounds, chroma_bounds : pair of float, optional
        JCh limits. Unlike categorical palettes, these are unrestricted by default.
    hue_bounds : pair of float
        JCh hue limits in degrees, allowing wrapped intervals such as (350, 20).
    red_bounds, green_bounds, blue_bounds : pair of float
        sRGB channel limits, always enforced regardless of grid space.
    max_control_points : int, optional
        Maximum cubic spline coefficients per arm. A bounded automatic budget
        is selected when omitted. Too-small budgets raise ValueError.
    arc_length_factor : float >= 1
        Maximum length relative to an initial feasible reference arm.
    smoothness : float > 0
        Geometric bending penalty, normalized so a full circular turn has unit
        bending energy when its length equals the reference length.
    chroma_weight : float >= 0
        Reward for JCh chroma around each arm's perceptual midpoint, at 0.5 for
        sequential maps and 0.25/0.75 for diverging maps. Zero disables it.
    max_iter : int
        SLSQP iteration budget per candidate fit, including constraint refinement.

    Returns
    -------
    colormap : ContinuousColormap
        Callable curve with sampling, Matplotlib export, seed positions and
        diagnostics. No global optimum or analytic gamut certificate is claimed.
        No seed or explicit bound is moved to make an infeasible fit succeed.
    """
    if mode not in ("sequential", "diverging"):
        raise ValueError("mode must be 'sequential' or 'diverging'")
    if direction not in ("increasing", "decreasing"):
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    if center_lightness not in ("light", "dark"):
        raise ValueError("center_lightness must be 'light' or 'dark'")
    if mode == "sequential" and center_color is not None:
        raise ValueError("center_color applies only to diverging maps")
    if grid_space not in ("RGB", "JCh"):
        raise ValueError("grid_space must be 'RGB' or 'JCh'")
    if isinstance(grid_size, (int, np.integer)):
        grid_size = _positive_integer(grid_size, "grid_size", 2)
    else:
        try:
            grid_size = tuple(_positive_integer(size, "grid_size", 2) for size in grid_size)
        except TypeError:
            raise ValueError("grid_size must be an integer or triple of integers") from None
        if len(grid_size) != 3:
            raise ValueError("grid_size must be an integer or triple of integers")
    max_iter = _positive_integer(max_iter, "max_iter")
    if max_control_points is not None:
        max_control_points = _positive_integer(max_control_points, "max_control_points", 4)
    for name, value, lower in (
        ("arc_length_factor", arc_length_factor, 1),
        ("smoothness", smoothness, 0), ("chroma_weight", chroma_weight, 0),
    ):
        if not np.isscalar(value) or not np.isfinite(value) or value < lower:
            raise ValueError(f"{name} must be finite and >= {lower}")
    if smoothness == 0:
        raise ValueError("smoothness must be positive")
    domain = _Domain(
        _bounds(lightness_bounds, "lightness_bounds", 0, 100, allow_none=True),
        _bounds(chroma_bounds, "chroma_bounds", 0, allow_none=True),
        _bounds(hue_bounds, "hue_bounds", 0, 360, wrap=True),
        tuple(_bounds(value, name, 0, 1) for value, name in (
            (red_bounds, "red_bounds"), (green_bounds, "green_bounds"), (blue_bounds, "blue_bounds")
        )),
    )
    parsed = _parse_seeds(seed_colors)
    unique, original_indices, inverse = np.unique(
        parsed, axis=0, return_index=True, return_inverse=True
    )
    if len(unique) < (2 if mode == "sequential" else 3):
        raise ValueError(f"Not enough distinct seed colors for a {mode} colormap")
    seeds = cspace_convert(unique, "sRGB1", "CAM02-UCS")
    if not domain.contains(seeds).all():
        raise ValueError("All seed colors must satisfy the supplied color bounds")
    if mode == "sequential":
        assignments = [_ordered(range(len(seeds)), seeds, direction == "increasing")]
    else:
        light = center_lightness == "light"
        extreme = seeds[:, 0].max() if light else seeds[:, 0].min()
        candidates = np.flatnonzero(np.abs(seeds[:, 0] - extreme) < 1e-8)
        chroma = cspace_convert(seeds, "CAM02-UCS", "JCh")[:, 1]
        center = min(candidates, key=lambda index: (chroma[index], index))
        if center_color is not None:
            center_rgb = _parse_seeds([center_color])[0]
            matches = np.flatnonzero(np.max(np.abs(unique - center_rgb), axis=1) < 1e-8)
            if not len(matches) or matches[0] not in candidates:
                raise ValueError("center_color must be a supplied seed with extreme lightness")
            center = int(matches[0])
        assignments = _partition_arms(seeds, center, light)
    grid = _Grid(domain, grid_size, grid_space, seeds)
    fits = [
        _fit_arm(seeds[indices], grid, max_control_points, arc_length_factor,
                 smoothness, chroma_weight, max_iter)
        for indices in assignments
    ]
    balanced = False
    if mode == "diverging":
        fits, balanced = _balance_arms(
            fits, grid, arc_length_factor, smoothness, chroma_weight, max_iter
        )
    arcs = [fit.arc for fit in fits]
    reports = [_arm_diagnostics(fit) for fit in fits]
    positions = np.empty(len(seeds))
    for side, (indices, arc) in enumerate(zip(assignments, arcs)):
        if mode == "sequential":
            positions[indices] = arc.anchor_positions
        elif side == 0:
            positions[indices] = (1 - arc.anchor_positions) / 2
        else:
            positions[indices] = (1 + arc.anchor_positions) / 2
    order = assignments[0] if mode == "sequential" else assignments[0][::-1] + assignments[1][1:]
    diagnostics = {"arms": reports, "chroma_scale": grid.chroma_scale}
    if mode == "diverging":
        diagnostics["joint_refinement_accepted"] = balanced
        left_tangent = -arcs[0].spline.derivative()(0)
        right_tangent = arcs[1].spline.derivative()(0)
        denominator = np.linalg.norm(left_tangent) * np.linalg.norm(right_tangent)
        diagnostics["center_join_angle"] = float(np.degrees(np.arccos(np.clip(
            np.dot(left_tangent, right_tangent) / max(denominator, 1e-15), -1, 1
        ))))
        diagnostics["arm_speed_ratio"] = arcs[0].length / arcs[1].length
    return ContinuousColormap(
        arcs, domain, positions[inverse], original_indices[order],
        [original_indices[indices] for indices in assignments], diagnostics,
    )