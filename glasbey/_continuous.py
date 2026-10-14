"""Constrained, seeded curves in CAM02-UCS."""

from dataclasses import dataclass
import warnings

import numpy as np
from colorspacious import cspace_convert, CIECAM02Space
from matplotlib.colors import ListedColormap, to_rgb
from scipy.integrate import trapezoid
from scipy.interpolate import BSpline, PchipInterpolator
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

from ._grids import rgb_grid, jch_grid
from ._converters import palette_to_sRGB1, get_rgb_palette


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


@dataclass
class _Domain:
    lightness_bounds: object = None
    chroma_bounds: object = None
    hue_bounds: tuple = (0.0, 360.0)
    rgb_bounds: tuple = ((0.0, 1.0),) * 3

    def margins(self, colors):
        colors = np.atleast_2d(colors)
        rgb, jch = _color_spaces(colors)
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
        self.colors = np.unique(np.vstack([cam, seeds]), axis=0)
        self.chroma = cspace_convert(self.colors, "CAM02-UCS", "JCh")[:, 1]
        self.chroma_scale = max(1.0, float(self.chroma.max()))
        self.domain = domain

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


def _validate_curve(spline, domain, anchor_parameters):
    parameters = np.unique(np.concatenate([
        np.linspace(0, 1, 129), spline.t, anchor_parameters,
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


class _SplineProblem:
    def __init__(self, anchors, points, parameters, anchor_parameters, count, domain):
        interior = list(anchor_parameters[1:-1])
        while len(interior) < count - 4:
            edges = np.array([0] + sorted(interior) + [1])
            widest = int(np.argmax(np.diff(edges)))
            interior.append((edges[widest] + edges[widest + 1]) / 2)
        self.knots = np.array([0.0] * 4 + sorted(interior) + [1.0] * 4)
        self.basis = BSpline(self.knots, np.eye(count), 3)
        self.anchor_parameters = anchor_parameters
        self.anchors = anchors
        self.domain = domain
        self.direction = 1 if anchors[-1, 0] >= anchors[0, 0] else -1
        anchor_matrix = self.basis(anchor_parameters)
        self.fixed = np.linalg.lstsq(anchor_matrix, anchors, rcond=None)[0]
        self.free = null_space(anchor_matrix)
        self.scale = max(1.0, float(np.linalg.norm(np.diff(anchors, axis=0), axis=1).sum()))
        fitting_parameters = np.unique(np.concatenate([parameters, np.linspace(0, 1, 129)]))
        target = np.column_stack([
            np.interp(fitting_parameters, parameters, points[:, channel])
            for channel in range(3)
        ])
        fitting_basis = self.basis(fitting_parameters)
        self.initial = np.linalg.lstsq(
            fitting_basis @ self.free,
            (target - fitting_basis @ self.fixed) / self.scale,
            rcond=None,
        )[0].ravel()
        self.mesh = np.unique(np.concatenate([
            np.linspace(0, 1, max(129, 16 * count)), self.knots, anchor_parameters,
        ]))
        self._update_basis()

    def _update_basis(self):
        self.values = self.basis(self.mesh)
        # Interior safety margin stops the solver hugging bounds between mesh points.
        to_anchor = np.min(np.abs(self.mesh[:, None] - self.anchor_parameters[None]), axis=1)
        self.safety = 1e-4 * np.clip(to_anchor / 0.02, 0, 1)[:, None]
        self.first = self.basis.derivative()(self.mesh)
        self.second = self.basis.derivative(2)(self.mesh)
        nodes, weights = np.polynomial.legendre.leggauss(8)
        half_width = np.diff(self.mesh) / 2
        locations = (self.mesh[:-1] + half_width)[:, None] + half_width[:, None] * nodes
        self.quadrature_weights = half_width[:, None] * weights
        self.quadrature_first = self.basis.derivative()(locations)
        self.quadrature_second = self.basis.derivative(2)(locations)

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

    def solve(self, initial, max_iter, smoothness=1.0, chroma_weight=0.0,
              chroma_scale=1.0, reference_length=None, arc_length_factor=1.5):
        def objective(variables):
            colors, length, fractions, bending = self.geometry(variables)
            reference = reference_length or self.scale
            value = smoothness * reference * bending / (4 * np.pi**2)
            value += length / reference if reference_length is None else -length / reference
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
            return float(value) if np.isfinite(value) else 1e12

        def constraints(variables, safety=True):
            coefficients = self.coefficients(variables)
            margins = self.domain.margins(self.values @ coefficients)
            result = [
                (margins - self.safety if safety else margins).ravel(),
                self.direction * np.diff(coefficients[:, 0]) / 100,
            ]
            if reference_length is not None:
                result.append(np.array([
                    arc_length_factor - self.geometry(variables)[1] / reference_length
                ]))
            return np.concatenate(result)

        candidates = []

        def retain(variables):
            if constraints(variables, False).min() >= -_FEASIBILITY_TOL:
                candidates.append((objective(variables), variables.copy()))
                candidates.sort(key=lambda item: item[0])
                del candidates[8:]

        current = initial
        retain(current)
        remaining = max_iter
        for refinement in range(4):
            result = minimize(
                objective, current, method="SLSQP",
                constraints={"type": "ineq", "fun": constraints},
                callback=retain,
                options={"maxiter": remaining, "ftol": 1e-9},
            )
            current = result.x
            remaining -= max(1, result.nit)
            valid, extra, depth = _validate_curve(
                self.spline(current), self.domain, self.anchor_parameters
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
                self.spline(candidate), self.domain, self.anchor_parameters
            )
            if valid and constraints(candidate, False).min() >= -_FEASIBILITY_TOL:
                arc = _Arc(self.spline(candidate), self.anchor_parameters)
                if reference_length is None or arc.length <= arc_length_factor * reference_length + 1e-5:
                    result.success = False
                    result.message = "Retained a feasible iterate; " + str(result.message)
                    return candidate, arc, result, depth
        return None


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