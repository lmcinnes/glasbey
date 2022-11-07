import pytest
import numpy as np

from glasbey._internals import (
    get_next_color,
    generate_next_color_cam02ucs,
    generate_palette_cam02ucs,
)
from glasbey._grids import rgb_grid

from colorspacious import cspace_convert


def test_get_next_color_distances():
    colors = np.random.random(size=(10000, 3)).astype(np.float32, order="C")
    distances = np.full(colors.shape[0], 1e12, dtype=np.float32)

    new_color = np.random.random(size=3).astype(np.float32, order="C")

    _ = get_next_color(distances, colors, new_color)

    comparison_distances = np.linalg.norm(colors - new_color, axis=1)

    # get_next_color uses squared euclidean distance
    assert np.allclose(np.sqrt(distances), comparison_distances)


def test_get_next_color_color_choice():
    colors = np.random.random(size=(10000, 3)).astype(np.float32, order="C")
    distances = np.full(colors.shape[0], 1e12, dtype=np.float32)

    new_color = np.random.random(size=3).astype(np.float32, order="C")

    next_color = get_next_color(distances, colors, new_color)

    comparison_distances = np.linalg.norm(colors - new_color, axis=1)
    index = np.argmax(comparison_distances)

    assert np.allclose(next_color, colors[index])


@pytest.mark.parametrize("grid_size", [16, 32, 64, 128])
def test_generate_palette_distance_order(grid_size):
    custom_palette = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    custom_palette = cspace_convert(custom_palette, "sRGB1", "CAM02-UCS").astype(
        np.float32, "C"
    )

    colors = rgb_grid(grid_size)

    palette = generate_palette_cam02ucs(colors, custom_palette, np.uint32(5))

    for i in range(2):
        assert np.linalg.norm(palette[i] - palette[i + 1]) >= np.linalg.norm(
            palette[i + 1] - palette[i + 2]
        )


@pytest.mark.parametrize("grid_size", [16, 32, 64, 128])
def test_generate_next_color_consistency(grid_size):
    custom_palette = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    custom_palette = cspace_convert(custom_palette, "sRGB1", "CAM02-UCS").astype(
        np.float32, "C"
    )

    colors = rgb_grid(grid_size)
    distances = np.full(colors.shape[0], 1e12, dtype=np.float32)
    initial_color = generate_palette_cam02ucs(colors, custom_palette, 3)[-1]

    for i in range(5):
        next_color = generate_next_color_cam02ucs(colors, distances, custom_palette)
        assert np.allclose(next_color, initial_color)
