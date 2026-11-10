import itertools
import warnings

import numpy as np
import pytest
from colorspacious import cspace_convert
from matplotlib.colors import to_rgb

import glasbey
from glasbey._glasbey import _build_block_palette_with_diagnostics, create_theme_palette


def _cam(palette):
    return cspace_convert(np.array([to_rgb(color) for color in palette]), "sRGB1", "CAM02-UCS")


def _blocks(palette, sizes, offset=0):
    edges = offset + np.concatenate([[0], np.cumsum(sizes)])
    return [palette[a:b] for a, b in zip(edges[:-1], edges[1:])]


def _assert_block_invariants(palette, sizes, offset=0):
    assert len(palette) == offset + sum(sizes)
    for block in _blocks(palette, sizes, offset):
        cam = _cam(block)
        assert len(set(block)) == len(block)
        assert np.all(np.diff(cam[:, 0]) > 0)


@pytest.mark.parametrize(
    "grid_space,colorblind_safe,optimize_palette",
    list(itertools.product(["RGB", "JCh"], [False, True], [False, True])),
)
def test_create_and_extend_end_to_end(grid_space, colorblind_safe, optimize_palette):
    options = dict(grid_space=grid_space, colorblind_safe=colorblind_safe,
                   optimize_palette=optimize_palette, optimize_palette_n_iter=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        palette = glasbey.create_block_palette([3, 2, 1], **options)
        extended = glasbey.extend_block_palette("tab10", [3, 2], **options)
    _assert_block_invariants(palette, [3, 2, 1])
    _assert_block_invariants(extended, [3, 2], offset=10)
    tab10 = [glasbey._converters.get_rgb_palette(
        cspace_convert(glasbey._converters.palette_to_sRGB1("tab10", 20), "sRGB1", "CAM02-UCS"))][0]
    assert extended[:10] == tab10


def test_as_hex_false_and_ordering():
    palette = glasbey.create_block_palette([4, 3], as_hex=False, optimize_palette=False)
    assert np.asarray(palette).shape == (7, 3)
    assert np.all((np.asarray(palette) >= 0) & (np.asarray(palette) <= 1))


def test_deterministic():
    first = glasbey.create_block_palette([5, 3, 2], optimize_palette_n_iter=3)
    second = glasbey.create_block_palette([5, 3, 2], optimize_palette_n_iter=3)
    assert first == second


def test_theme_color_spacing_deprecated_maps_to_step():
    with pytest.warns(DeprecationWarning, match="theme_step=12"):
        palette = glasbey.create_block_palette([4], theme_color_spacing=0.2, optimize_palette=False)
    steps = np.linalg.norm(np.diff(_cam(palette), axis=0), axis=1)
    assert np.all(np.abs(steps - 12) < 1.5)


def test_theme_step_controls_spacing():
    palette = glasbey.create_block_palette([5], theme_step=6.0, optimize_palette=False)
    steps = np.linalg.norm(np.diff(_cam(palette), axis=0), axis=1)
    assert np.all(np.abs(steps - 6) < 1.0)


def test_min_color_offset_deprecated():
    with pytest.warns(DeprecationWarning):
        palette = create_theme_palette("#4a7ab0", 4, min_color_offset=0.1)
    steps = np.linalg.norm(np.diff(_cam(palette), axis=0), axis=1)
    assert np.all(np.abs(steps - 6) < 1.0)


def test_legacy_method_warns():
    with pytest.warns(FutureWarning, match="legacy"):
        glasbey.create_block_palette([2, 2], theme_method="legacy")


@pytest.mark.parametrize("options", [
    {"grid_space": "fish"}, {"theme_method": "fancy"}, {"twist": 0}, {"twist": "left"},
    {"hue_sweep": -5}, {"hue_sweep": "lots"}, {"theme_step": 0}, {"theme_step": -1.0},
])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        glasbey.create_block_palette([3, 3], **options)


@pytest.mark.parametrize("sizes", [[0], [2, -1], [1.5]])
def test_invalid_block_sizes(sizes):
    with pytest.raises(ValueError):
        glasbey.create_block_palette(sizes)


def test_wrapped_hue_bounds():
    palette = glasbey.create_block_palette([3, 3], hue_bounds=(330, 40), optimize_palette=False)
    hues = cspace_convert(np.array([to_rgb(c) for c in palette]), "sRGB1", "JCh")
    chromatic = hues[:, 1] > 5
    assert np.all((hues[chromatic, 2] >= 330 - 1) | (hues[chromatic, 2] <= 40 + 1))


def test_diagnostics_report_blocks():
    palette, diagnostics = _build_block_palette_with_diagnostics([6, 3, 1], optimize_palette=False)
    assert len(diagnostics["themes"]) == 3
    assert diagnostics["themes"][2].size == 1
    assert {"select", "fit", "polish"} <= set(diagnostics["timings"])
    assert diagnostics["separation"] > 0
