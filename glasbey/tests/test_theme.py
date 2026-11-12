import warnings

import numpy as np
import pytest

from glasbey._curves import _Domain, _Grid, _cam02ucs_to_jch
from glasbey._theme import (
    HUE_WINDOW_PAD, ThemeArc, discounted_distance, fit_theme_arc, gamut_table, hue_offset,
    sample_theme_arc,
)

RGB = ((0.0, 1.0),) * 3


def _setup(bounds):
    domain = _Domain(bounds, (0, 80), fast=True)
    grid = _Grid(domain, 32, "RGB", np.zeros((0, 3)))
    return domain, grid, gamut_table(bounds, (0, 80), RGB)


def _random_base(grid, rng):
    jch = _cam02ucs_to_jch(grid.colors)
    pool = grid.colors[(jch[:, 0] > 45) & (jch[:, 0] < 65) & (jch[:, 1] > 20)]
    return pool[rng.integers(len(pool))]


def _check_block(theme, domain, allowance):
    colors = theme.colors
    assert domain.contains(colors).all()
    assert np.all(np.diff(colors[:, 0]) > 0)
    assert len(np.unique(np.round(colors, 6), axis=0)) == len(colors)
    if len(colors) > 2:
        assert theme.diagnostics["step_cv"] < 0.05
    jch = _cam02ucs_to_jch(colors)
    base_jch = _cam02ucs_to_jch(theme.base)
    if base_jch[1] >= 12:
        offsets = hue_offset(jch[jch[:, 1] > 15, 2], base_jch[2])
        tolerance = 1e-3
        assert np.all(offsets >= -(allowance[0] + HUE_WINDOW_PAD) - tolerance)
        assert np.all(offsets <= allowance[1] + HUE_WINDOW_PAD + tolerance)


@pytest.mark.parametrize("bounds", [(10, 90), (30, 80), (40, 70)])
@pytest.mark.parametrize("n", [2, 3, 6, 10, 16])
def test_theme_arc_invariants(bounds, n):
    domain, grid, gamut = _setup(bounds)
    rng = np.random.default_rng(1000 * bounds[0] + n)
    base = _random_base(grid, rng)
    allowance = (20.0, 20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theme = fit_theme_arc(base, n, step=10.0, domain=domain, gamut=gamut, grid=grid,
                              hue_allowance=allowance)
    assert isinstance(theme, ThemeArc)
    assert theme.colors.shape == (n, 3)
    assert not theme.diagnostics["fallback"]
    _check_block(theme, domain, allowance)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("bounds", [(10, 90), (30, 80), (40, 70)])
def test_theme_arc_invariants_sweep(seed, bounds):
    domain, grid, gamut = _setup(bounds)
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in (2, 3, 6, 10, 16):
            allowance = tuple(rng.uniform(0, 40, 2))
            theme = fit_theme_arc(_random_base(grid, rng), n, step=10.0, domain=domain,
                                  gamut=gamut, grid=grid, hue_allowance=allowance,
                                  twist=int(rng.choice([-1, 1])))
            _check_block(theme, domain, allowance)


def test_theme_arc_contains_base_region_and_twist():
    domain, grid, gamut = _setup((10, 90))
    base = _random_base(grid, np.random.default_rng(5))
    for twist in (1, -1):
        theme = fit_theme_arc(base, 8, step=10.0, domain=domain, gamut=gamut, grid=grid,
                              hue_allowance=(25.0, 25.0), twist=twist)
        jch = _cam02ucs_to_jch(theme.colors)
        offsets = hue_offset(jch[:, 2], _cam02ucs_to_jch(theme.base)[2])
        assert twist * (offsets[-1] - offsets[0]) > 0
        # The base lies on the arc between the block's endpoints.
        assert theme.colors[0, 0] <= theme.base[0] <= theme.colors[-1, 0]


def test_single_colour_and_resampling():
    domain, grid, gamut = _setup((10, 90))
    base = _random_base(grid, np.random.default_rng(2))
    single = fit_theme_arc(base, 1, domain=domain, gamut=gamut, grid=grid)
    np.testing.assert_allclose(single.colors, base[None])
    theme = fit_theme_arc(base, 5, step=8.0, domain=domain, gamut=gamut, grid=grid)
    np.testing.assert_allclose(sample_theme_arc(theme), theme.colors, atol=1e-6)
    shorter = sample_theme_arc(theme, length=theme.step * 4 * 0.85)
    assert np.linalg.norm(np.diff(shorter, axis=0), axis=1).mean() < theme.step


def test_short_step_warns():
    domain, grid, gamut = _setup((40, 70))
    base = _random_base(grid, np.random.default_rng(3))
    with pytest.warns(RuntimeWarning, match="step"):
        fit_theme_arc(base, 16, step=10.0, domain=domain, gamut=gamut, grid=grid)


def test_deterministic():
    domain, grid, gamut = _setup((30, 80))
    base = _random_base(grid, np.random.default_rng(4))
    first = fit_theme_arc(base, 6, domain=domain, gamut=gamut, grid=grid, hue_allowance=(10, 10))
    second = fit_theme_arc(base, 6, domain=domain, gamut=gamut, grid=grid, hue_allowance=(10, 10))
    np.testing.assert_array_equal(first.colors, second.colors)


def test_discounted_metric_reduces_to_euclidean():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(4, 7, 3)), rng.normal(size=(5, 3))
    expected = np.linalg.norm(a[..., None, :] - b, axis=-1)
    np.testing.assert_allclose(discounted_distance(a, b, w_j=1.0), expected)
    assert np.all(discounted_distance(a, b) <= expected + 1e-12)


def test_gamut_table_relaxes_near_extremes():
    table = gamut_table((10, 90), (0, 80), RGB)
    hues = np.arange(0, 360, 30.0)
    middle = table.max_chroma(np.full_like(hues, 60.0), hues)
    dark = table.max_chroma(np.full_like(hues, 20.0), hues)
    assert np.all(middle > 20)
    assert dark.mean() < middle.mean()


@pytest.mark.parametrize("rgb", [(0.2, 0.2, 0.2), (0.5, 0.5, 0.5), (0.26, 0.2, 0.2), (0.61, 0.66, 0.66)])
def test_grey_base_below_chroma_floor(rgb):
    # create_theme_palette's chroma bounds (10, 90) exclude grey bases: the base is
    # pushed out past the bound instead of falling back to a short nearest-colour ramp,
    # and the hue window still applies (the arc cannot pass through neutral).
    from colorspacious import cspace_convert
    from glasbey._blocks import make_context, plan_blocks

    context = make_context((10.0, 90.0), (10.0, 90.0), (0.0, 360.0))
    spec = plan_blocks([5], context, None)[0]
    allowance = (spec.demand / 2, spec.demand / 2)
    theme = fit_theme_arc(cspace_convert(rgb, "sRGB1", "CAM02-UCS"), 5, domain=context.domain,
                          gamut=context.gamut, grid=context.grid, hue_allowance=allowance, twist=1)
    assert not theme.diagnostics["fallback"]
    assert theme.diagnostics["step"] >= 0.9 * spec.step
    assert theme.diagnostics["hue_spread"] <= sum(allowance) + 2 * HUE_WINDOW_PAD
    assert context.domain.contains(theme.colors).all()
    assert np.all(np.diff(theme.colors[:, 0]) > 0)
