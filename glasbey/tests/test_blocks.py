import time
import warnings

import numpy as np
import pytest

from glasbey._blocks import (
    build_block_palette, footprint_separation, make_context, plan_blocks, reserved_sweeps,
    select_bases,
)
from glasbey._curves import _cam02ucs_to_jch

CONFIGS = [[10, 10, 3, 3, 3, 3, 3, 3], [12, 8, 3, 3, 3, 3, 2, 2]]


@pytest.fixture(scope="module", params=range(len(CONFIGS)), ids=["10-10-3x6", "12-8-3x4-2x2"])
def built(request):
    sizes = CONFIGS[request.param]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        colors, diagnostics = build_block_palette(sizes, context=make_context())
    return sizes, colors, diagnostics


def test_fitted_separation(built):
    sizes, colors, diagnostics = built
    assert len(colors) == sum(sizes)
    assert diagnostics["separation"] >= 9.5


def test_short_bases_in_mid_band_and_long_spans(built):
    sizes, colors, diagnostics = built
    for spec, theme, info in zip(diagnostics["specs"], diagnostics["themes"], diagnostics["selection"]):
        lightness = _cam02ucs_to_jch(theme.colors)[:, 0]
        if spec.kind == "long":
            assert lightness.max() - lightness.min() >= 65
        elif not info.get("crowded"):
            base_lightness = _cam02ucs_to_jch(theme.base)[0]
            assert 45 - 1e-6 <= base_lightness <= 65 + 1e-6


@pytest.mark.parametrize("colorblind_safe", [False, True])
def test_selection_speed_twenty_blocks(colorblind_safe):
    context = make_context(colorblind_safe=colorblind_safe)
    specs = plan_blocks([5] * 20, context)
    sweeps = reserved_sweeps(specs, context)
    select_bases(specs, context, sweeps, 1)  # warm caches shared with the timed run
    start = time.perf_counter()
    footprints, info = select_bases(specs, context, sweeps, 1)
    elapsed = time.perf_counter() - start
    assert len(footprints) == 20
    # Target 0.5 s on a CI core (see dev/benchmarks); allow headroom for slow runners.
    assert elapsed < (1.0 if not colorblind_safe else 2.0)


def test_existing_colours_are_avoided():
    context = make_context()
    specs = plan_blocks([3, 3], context)
    sweeps = reserved_sweeps(specs, context)
    free, _ = select_bases(specs, context, sweeps, 1)
    fixed = np.vstack([footprint.points for footprint in free])
    placed, _ = select_bases(specs, context, sweeps, 1, fixed_colors=fixed)
    for footprint in placed:
        assert np.linalg.norm(footprint.base - free[0].base) > 1e-6
    assert footprint_separation(placed) > 0


def test_demand_scales_with_block_size():
    context = make_context()
    specs = plan_blocks([1, 2, 3, 6, 10, 16], context)
    demands = [spec.demand for spec in specs]
    assert demands[0] == 0
    assert all(a <= b for a, b in zip(demands, demands[1:]))
    zero = make_context(hue_sweep=0)
    assert all(spec.demand == 0 for spec in plan_blocks([3, 10], zero))


# ------------------------------------------------------------------ Phase 4

from glasbey._blocks import (  # noqa: E402
    Footprint, allocate_sweeps, choose_twist, rebuild_footprint, refine,
    separation_objective,
)
from glasbey._theme import hue_offset  # noqa: E402


def _chromatic_offsets(diagnostics):
    for spec, theme in zip(diagnostics["specs"], diagnostics["themes"]):
        base = _cam02ucs_to_jch(theme.base)
        if spec.size < 2 or base[1] < 12 or diagnostics["sweeps"][spec.index] <= 0:
            continue
        offsets = hue_offset(_cam02ucs_to_jch(theme.colors)[:, 2], base[2])
        yield spec, offsets


@pytest.fixture(scope="module", params=range(len(CONFIGS)), ids=["10-10-3x6", "12-8-3x4-2x2"])
def twisted(request):
    sizes = CONFIGS[request.param]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        colors, diagnostics = build_block_palette(sizes, context=make_context(), twist=1)
    return sizes, colors, diagnostics


def test_twist_one_light_ends_have_higher_hue(twisted):
    sizes, colors, diagnostics = twisted
    assert diagnostics["twist"] == 1
    checked = 0
    for spec, offsets in _chromatic_offsets(diagnostics):
        assert offsets[-1] > offsets[0]
        checked += 1
    assert checked >= len(sizes) - 1


def test_long_blocks_ab_step_and_separation(twisted):
    sizes, colors, diagnostics = twisted
    steps, counts = [], []
    for spec, theme in zip(diagnostics["specs"], diagnostics["themes"]):
        if spec.kind == "long":
            steps.append(np.linalg.norm(np.diff(theme.colors[:, 1:], axis=0), axis=1).sum())
            counts.append(spec.size - 1)
    assert sum(steps) / sum(counts) >= 4.0
    assert diagnostics["separation"] >= 9.5


def test_zero_hue_sweep_keeps_hue_constant():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        colors, diagnostics = build_block_palette([10, 6, 3, 3, 2], context=make_context(hue_sweep=0))
    assert all(sweep == 0 for sweep in diagnostics["sweeps"])
    for theme in diagnostics["themes"]:
        assert theme.diagnostics["hue_spread"] < 10


def test_equal_gaps_larger_blocks_get_more_sweep():
    context = make_context()
    specs = plan_blocks([3, 4, 5], context)
    hues = np.array([20.0, 140.0, 260.0])
    footprints = []
    for spec, hue in zip(specs, hues):
        centre = np.array([60.0, 22 * np.cos(np.deg2rad(hue)), 22 * np.sin(np.deg2rad(hue))])
        seed = Footprint(centre[None], centre, hue, 0.0, 1, spec.kind)
        footprints.append(rebuild_footprint(spec, seed, context, 0.0, 1, centre))
    reserved = reserved_sweeps(specs, context)
    allocated, sweeps = allocate_sweeps(specs, footprints, context, reserved, 1)
    assert sweeps[0] <= sweeps[1] <= sweeps[2]
    assert sweeps[2] > 0


def test_refinement_never_worsens_footprint_objective():
    context = make_context()
    specs = plan_blocks([12, 8, 3, 3, 3, 3, 2, 2], context)
    reserved = reserved_sweeps(specs, context)
    footprints, info = select_bases(specs, context, reserved, 1)
    allocated, _ = allocate_sweeps(specs, footprints, context, reserved, 1)
    before = separation_objective(specs, allocated, context)
    refined, sweeps, report = refine(specs, footprints, info, context, reserved, 1)
    after = report["objective"]
    assert after[0] >= before[0] - 1e-9
    assert np.all(np.asarray(sweeps) >= 0)


def test_auto_twist_prefers_yellow_light_ends():
    context = make_context()
    spec = plan_blocks([10], context)[0]
    below_yellow = Footprint(np.zeros((1, 3)), np.zeros(3), 40.0, 60.0, 1, "long")
    above_yellow = Footprint(np.zeros((1, 3)), np.zeros(3), 160.0, 60.0, 1, "long")
    assert choose_twist([spec], [below_yellow]) == 1
    assert choose_twist([spec], [above_yellow]) == -1


@pytest.mark.parametrize("lightness_bounds", [(40, 70), (30, 80)])
def test_bases_on_chroma_lower_bound_do_not_fall_back(lightness_bounds):
    # Short-block bases used to sit on the chroma lower bound, where insetting
    # toward neutral is infeasible, forcing the last-resort fallback.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        colors, diagnostics = build_block_palette(
            [5] * 20, context=make_context(lightness_bounds, (30, 80)))
    assert diagnostics["fallbacks"] == 0
    assert max(theme.diagnostics["step_cv"] for theme in diagnostics["themes"]) < 0.05
