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
    assert elapsed < (0.5 if not colorblind_safe else 1.0)


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
