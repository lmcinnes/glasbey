import pytest
import numpy as np

from colorspacious import cspace_convert
from matplotlib.colors import to_rgb
from matplotlib.cm import get_cmap
from glasbey._glasbey import (
    create_palette,
    create_theme_palette,
    create_block_palette,
    extend_palette,
)

from typing import *


@pytest.mark.parametrize("grid_size", [32, 64, (32, 32, 128)])
@pytest.mark.parametrize("grid_space", ["RGB", "JCh"])
def test_create_palette_distances(grid_size, grid_space: Literal["RGB", "JCh"]):
    palette = create_palette(10, grid_size=grid_size, grid_space=grid_space)

    rgb_palette = np.asarray(
        [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)] + [to_rgb(color) for color in palette]
    )
    cam_palette = cspace_convert(rgb_palette, "sRGB1", "CAM02-UCS")

    for i in range(2, 10):
        prev_min_dist = np.min(np.linalg.norm(cam_palette[:i] - cam_palette[i], axis=1))
        current_min_dist = np.min(
            np.linalg.norm(cam_palette[: i + 1] - cam_palette[i + 1], axis=1)
        )

        assert prev_min_dist >= current_min_dist


@pytest.mark.parametrize("grid_size", [32, 64, (32, 32, 128)])
@pytest.mark.parametrize("grid_space", ["RGB", "JCh"])
@pytest.mark.parametrize("palette_to_extend", ["tab10", "Accent", "Set1"])
def test_extend_palette_distances(
    grid_size, grid_space: Literal["RGB", "JCh"], palette_to_extend: str
):
    initial_palette = get_cmap(palette_to_extend, 8).colors
    palette = extend_palette(
        initial_palette, 12, grid_size=grid_size, grid_space=grid_space
    )

    rgb_palette = np.asarray([to_rgb(color) for color in palette])
    cam_palette = cspace_convert(rgb_palette, "sRGB1", "CAM02-UCS")

    for i in range(9, 11):
        prev_min_dist = np.min(np.linalg.norm(cam_palette[:i] - cam_palette[i], axis=1))
        current_min_dist = np.min(
            np.linalg.norm(cam_palette[: i + 1] - cam_palette[i + 1], axis=1)
        )

        assert prev_min_dist >= current_min_dist


@pytest.mark.parametrize("grid_size", [32, 64, (32, 32, 128)])
@pytest.mark.parametrize("grid_space", ["RGB"])
@pytest.mark.parametrize("palette_to_extend", ["tab10", "Accent", "Set1"])
def test_extend_palette_inferred_bounds(
    grid_size, grid_space: Literal["RGB", "JCh"], palette_to_extend: str
):
    palette = get_cmap(palette_to_extend, 8).colors
    palette = extend_palette(
        palette, 12, grid_size=grid_size, grid_space=grid_space, as_hex=False
    )
    jch_palette = cspace_convert(palette, "sRGB1", "JCh")

    assert np.all(jch_palette[8:, 0] >= np.min(jch_palette[:8, 0]))
    assert np.all(jch_palette[8:, 0] <= np.max(jch_palette[:8, 0]))
    assert np.all(jch_palette[8:, 1] >= np.min(jch_palette[:8, 1]))
    assert np.all(jch_palette[8:, 1] <= np.max(jch_palette[:8, 1]))
    assert np.all(jch_palette[8:, 2] >= np.min(jch_palette[:8, 2]))
    assert np.all(jch_palette[8:, 2] <= np.max(jch_palette[:8, 2]))


def test_theme_palette_distances_small():
    base_color = np.clip(np.random.random(3), 0.2, 0.8)
    palette = create_theme_palette(base_color)

    rgb_palette = np.asarray([to_rgb(color) for color in palette])
    cam_palette = cspace_convert(rgb_palette, "sRGB1", "CAM02-UCS")

    for i in range(4):
        assert 0.0 < np.linalg.norm(cam_palette[i] - cam_palette[i + 1]) <= 35.0


def test_theme_palette_bounds():
    base_color = np.clip(np.random.random(3), 0.2, 0.8)
    palette = create_theme_palette(base_color)

    rgb_palette = np.asarray([to_rgb(color) for color in palette])
    jch_palette = cspace_convert(rgb_palette, "sRGB1", "JCh")

    assert 40 <= np.abs(jch_palette[0, 0] - jch_palette[4, 0]) <= 80
    assert 8 <= np.abs(jch_palette[0, 1] - jch_palette[4, 1]) <= 80
    assert (
        36 <= np.abs(jch_palette[0, 2] - jch_palette[4, 2]) <= 90
        or 36 <= 360 - np.abs(jch_palette[0, 2] - jch_palette[4, 2]) <= 90
    )


@pytest.mark.parametrize(
    "block_sizes", [[5, 5, 3, 2, 2, 1], [1, 5, 3, 4, 2], [9, 9, 12, 16]]
)
@pytest.mark.parametrize("sort_block_sizes", [True, False])
def test_block_palette_sizing(block_sizes, sort_block_sizes):
    pal = create_block_palette(block_sizes, sort_block_sizes=sort_block_sizes)
    assert len(pal) == sum(block_sizes)

    for start, end in zip(
        np.hstack([[0], np.cumsum(block_sizes)]), np.cumsum(block_sizes)
    ):
        rgb_palette = np.asarray([to_rgb(color) for color in pal[start:end]])
        cam_palette = cspace_convert(rgb_palette, "sRGB1", "CAM02-UCS")

        for i in range(end - start - 1):
            assert 2.0 < np.linalg.norm(cam_palette[i] - cam_palette[i + 1]) <= 42.0