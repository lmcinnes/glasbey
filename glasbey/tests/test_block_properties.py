"""Property tests for block palettes over random block sizes and bounds."""

import warnings

import numpy as np
import pytest
from colorspacious import cspace_convert
from hypothesis import HealthCheck, given, settings, strategies as st
from matplotlib.colors import to_rgb

import glasbey


@st.composite
def bounds(draw):
    low = draw(st.floats(5, 45))
    high = draw(st.floats(low + 30, 95))
    chroma_low = draw(st.sampled_from([0.0, 10.0, 25.0]))
    chroma_high = draw(st.floats(60, 90))
    return (low, high), (chroma_low, chroma_high)


def _check(palette, sizes, lightness_bounds, chroma_bounds):
    assert len(palette) == sum(sizes)
    assert len(set(palette)) == len(palette)
    rgb = np.array([to_rgb(color) for color in palette])
    cam = cspace_convert(rgb, "sRGB1", "CAM02-UCS")
    jch = cspace_convert(rgb, "sRGB1", "JCh")
    assert np.all(jch[:, 0] >= lightness_bounds[0] - 0.5)
    assert np.all(jch[:, 0] <= lightness_bounds[1] + 0.5)
    assert np.all(jch[:, 1] >= chroma_bounds[0] - 0.5)
    assert np.all(jch[:, 1] <= chroma_bounds[1] + 0.5)
    edges = np.concatenate([[0], np.cumsum(sizes)])
    for start, end in zip(edges[:-1], edges[1:]):
        block = cam[start:end]
        assert np.all(np.diff(block[:, 0]) > 0)
        steps = np.linalg.norm(np.diff(block, axis=0), axis=1)
        assert np.all((steps > 2) & (steps < 42))


def _run(sizes, limits):
    lightness_bounds, chroma_bounds = limits
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        palette = glasbey.create_block_palette(
            sizes, lightness_bounds=lightness_bounds, chroma_bounds=chroma_bounds,
            optimize_palette_n_iter=3,
        )
    _check(palette, sizes, lightness_bounds, chroma_bounds)


@settings(max_examples=6, deadline=None, derandomize=True,
          suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.integers(1, 10), min_size=1, max_size=6), bounds())
def test_block_palette_properties(sizes, limits):
    _run(sizes, limits)


@pytest.mark.slow
@settings(max_examples=40, deadline=None, derandomize=True,
          suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.integers(1, 16), min_size=1, max_size=20), bounds())
def test_block_palette_properties_full(sizes, limits):
    _run(sizes, limits)
