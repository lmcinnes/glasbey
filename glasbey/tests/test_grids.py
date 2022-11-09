import pytest
import numpy as np

from glasbey._grids import rgb_grid, jch_grid


@pytest.mark.parametrize(
    "grid_size",
    [
        64,
        128,
        (32, 64, 128),
        (128, 37, 51),
    ],
)
@pytest.mark.parametrize("color_space", ["RGB", "JCh"])
def test_grid_length(grid_size, color_space):
    if color_space == "RGB":
        grid = rgb_grid(grid_size=grid_size)
    elif color_space == "JCh":
        grid = jch_grid(grid_size=grid_size)
    else:
        raise ValueError("Bad colorspace")

    if type(grid_size) is int:
        assert len(grid) == grid_size**3
    else:
        assert len(grid) == grid_size[0] * grid_size[1] * grid_size[2]


@pytest.mark.parametrize(
    "grid_size",
    [
        64,
        128,
        (32, 64, 128),
        (128, 37, 51),
    ],
)
@pytest.mark.parametrize("color_space", ["RGB", "JCh"])
def test_grid_dim(grid_size, color_space):
    if color_space == "RGB":
        grid = rgb_grid(grid_size=grid_size)
    elif color_space == "JCh":
        grid = jch_grid(grid_size=grid_size)
    else:
        raise ValueError("Bad colorspace")

    if type(grid_size) is int:
        for i in range(3):
            assert len(np.unique(grid.T[i])) == grid_size
    else:
        for i in range(3):
            assert len(np.unique(grid.T[i])) == grid_size[i]


def test_grid_rgb256():
    grid = rgb_grid(256)
    int_grid = (grid * 255).astype(np.uint8)
    for i in range(3):
        assert len(np.unique(int_grid.T[i])) == 256


def test_grid_jch_bounds():
    grid = jch_grid(
        64, lightness_bounds=(23, 79), chroma_bounds=(27, 91), hue_bounds=(51, 351)
    )

    assert np.min(grid.T[0]) >= 23
    assert np.max(grid.T[0]) <= 79
    assert np.min(grid.T[1]) >= 27
    assert np.max(grid.T[1]) <= 91
    assert np.min(grid.T[2]) >= 51
    assert np.max(grid.T[2]) <= 351