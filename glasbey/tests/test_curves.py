import numpy as np
import pytest
from colorspacious import cspace_convert

from glasbey._curves import _Domain, _Grid, _SplineProblem, _cam02ucs_to_jch


def test_fast_domain_matches_default_domain():
    rgb = np.random.default_rng(1).random((500, 3))
    cam = cspace_convert(rgb, "sRGB1", "CAM02-UCS")
    slow = _Domain((20, 80), (10, 60))
    fast = _Domain((20, 80), (10, 60), fast=True)
    np.testing.assert_allclose(slow.margins(cam), fast.margins(cam), atol=1e-6)


def _single_anchor_problem(profile="default", weights=None):
    base = cspace_convert([0.35, 0.5, 0.7], "sRGB1", "CAM02-UCS")
    dark = base - np.array([20.0, 2.0, 4.0])
    light = base + np.array([20.0, -3.0, 2.0])
    points = np.vstack([dark, base, light])
    lengths = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    parameters = lengths / lengths[-1]
    domain = _Domain(fast=True)
    problem = _SplineProblem(
        base[None], points, parameters, parameters[1:2], 6, domain,
        anchor_weights=weights, direction=1, scale=float(lengths[-1]), profile=profile,
    )
    return problem, base, parameters[1]


@pytest.mark.parametrize("profile", ["default", "fast"])
def test_single_interior_anchor_is_pinned(profile):
    problem, base, position = _single_anchor_problem(profile)
    assert problem.free.shape == (6, 5)
    target = 30.0

    def at_least(problem, variables, geometry):
        return [geometry[1] / target - 1]

    fitted = problem.solve(
        problem.initial, 60, length_weight=0.0, extra_constraints=at_least,
    )
    assert fitted is not None
    variables, arc, result, depth = fitted
    np.testing.assert_allclose(arc.spline(position), base, atol=1e-8)
    assert arc.length >= target - 1e-4
    colors = arc(np.linspace(0, 1, 129))
    assert problem.domain.contains(colors).all()
    assert np.all(np.diff(colors[:, 0]) >= -1e-6)
    if profile == "fast":
        assert len(problem.mesh) < 129


def test_soft_anchor_is_penalised_not_pinned():
    problem, base, position = _single_anchor_problem(weights=[100.0])
    assert problem.free.shape == (6, 6)
    fitted = problem.solve(problem.initial, 60)
    assert fitted is not None
    error = np.linalg.norm(fitted[1].spline(position) - base)
    assert error < 2.0


def test_monotone_lightness_optional_and_extra_objective():
    problem, base, position = _single_anchor_problem()
    problem.monotone_lightness = False
    calls = []

    def objective(problem, variables, geometry):
        calls.append(1)
        return 0.0

    assert problem.solve(problem.initial, 20, extra_objective=objective) is not None
    assert calls


def test_grid_reuse_and_restriction():
    domain = _Domain((10, 90), fast=True)
    seeds = cspace_convert([[0.5, 0.5, 0.5]], "sRGB1", "CAM02-UCS")
    grid = _Grid(domain, 12, "RGB", seeds)
    narrow = _Domain((40, 60), fast=True)
    view = grid.restricted(narrow)
    lightness = _cam02ucs_to_jch(view.colors)[:, 0]
    assert len(view.colors) < len(grid.colors)
    assert lightness.min() >= 40 - 1e-6 and lightness.max() <= 60 + 1e-6
    rebuilt = _Grid.from_colors(domain, grid.colors)
    np.testing.assert_array_equal(rebuilt.colors, grid.colors)
    np.testing.assert_allclose(rebuilt.chroma, grid.chroma, atol=1e-6)


def test_grid_from_colors_dedupes_and_keeps_chroma_aligned():
    domain = _Domain((10, 90), (0, 80), fast=True)
    base = _Grid(domain, 12, "RGB", np.zeros((0, 3))).colors
    shuffled = np.random.default_rng(0).permutation(np.vstack([base, base[:40]]))
    grid = _Grid.from_colors(domain, shuffled)
    assert len(grid.colors) == len(base) == len(grid.chroma)
    np.testing.assert_allclose(grid.chroma, _cam02ucs_to_jch(grid.colors)[:, 1])
    grid.restricted(_Domain((40, 60), (0, 80), fast=True))  # used to raise IndexError
