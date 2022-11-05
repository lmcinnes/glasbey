import numpy as np

from colorspacious import cspace_convert

from typing import Tuple


def rgb_grid(
    grid_size: int | Tuple[int, int, int] = 64,
    red_bounds: Tuple[float, float] = (0, 1),
    green_bounds: Tuple[float, float] = (0, 1),
    blue_bounds: Tuple[float, float] = (0, 1),
    output_colorspace: str = "sRGB1",
):
    if type(grid_size) is int:
        b = np.tile(
            np.linspace(blue_bounds[0], blue_bounds[1], grid_size), grid_size**2
        )
        g = np.tile(
            np.repeat(
                np.linspace(green_bounds[0], green_bounds[1], grid_size), grid_size
            ),
            grid_size,
        )
        r = np.repeat(
            np.linspace(red_bounds[0], red_bounds[1], grid_size), grid_size**2
        )
    elif hasattr(grid_size, "__len__") and len(grid_size) == 3:
        b = np.tile(
            np.linspace(blue_bounds[0], blue_bounds[1], grid_size[2]),
            grid_size[0] * grid_size[1],
        )
        g = np.tile(
            np.repeat(
                np.linspace(green_bounds[0], green_bounds[1], grid_size[1]),
                grid_size[2],
            ),
            grid_size[0],
        )
        r = np.repeat(
            np.linspace(red_bounds[0], red_bounds[1], grid_size[0]),
            grid_size[1] * grid_size[2],
        )
    else:
        raise ValueError(
            "Parameter grid_size should either be an integer or a triple of integers (j_size, c_size, h_size)"
        )

    rgb_colors = np.vstack([r, g, b]).T

    if output_colorspace == "sRGB1":
        return rgb_colors.astype(np.float32, order="C")
    else:
        try:
            return cspace_convert(rgb_colors, "sRGB1", output_colorspace).astype(
                np.float32, order="C"
            )
        except ValueError:
            raise ValueError(f"Invalid output colorspace {output_colorspace}")


def jch_grid(
    grid_size: int | Tuple[int, int, int] = 64,
    lightness_bounds: Tuple[float, float] = (10, 90),
    chroma_bounds: Tuple[float, float] = (10, 90),
    hue_bounds: Tuple[float, float] = (0, 360),
    output_colorspace: str = "JCh",
):
    if type(grid_size) is int:
        c = np.repeat(
            np.linspace(chroma_bounds[0], chroma_bounds[1], grid_size), grid_size**2
        )
        j = np.tile(
            np.repeat(
                np.linspace(lightness_bounds[0], lightness_bounds[1], grid_size),
                grid_size,
            ),
            grid_size,
        )
        h = np.tile(
            np.linspace(hue_bounds[0], hue_bounds[1], grid_size), grid_size**2
        )
    elif hasattr(grid_size, "__len__") and len(grid_size) == 3:
        c = np.repeat(
            np.linspace(chroma_bounds[0], chroma_bounds[1], grid_size[1]),
            grid_size[0] * grid_size[2],
        )
        j = np.tile(
            np.repeat(
                np.linspace(lightness_bounds[0], lightness_bounds[1], grid_size[0]),
                grid_size[2],
            ),
            grid_size[1],
        )
        h = np.tile(
            np.linspace(hue_bounds[0], hue_bounds[1], grid_size[2]),
            grid_size[0] * grid_size[1],
        )
    else:
        raise ValueError(
            "Parameter grid_size should either be an integer or a triple of integers (j_size, c_size, h_size)"
        )

    jch_colors = np.vstack([j, c, h]).T

    if output_colorspace == "JCh":
        return jch_colors.astype(np.float32, order="C")
    else:
        try:
            return cspace_convert(jch_colors, "JCh", output_colorspace).astype(
                np.float32, order="C"
            )
        except ValueError:
            raise ValueError(f"Invalid output colorspace {output_colorspace}")


def constrain_by_lightness_chroma_hue(
    colors,
    current_colorspace,
    output_colorspace="CAM02-UCS",
    lightness_bounds=(10, 90),
    chroma_bounds=(10, 90),
    hue_bounds=(0, 360),
):
    if current_colorspace != "JCh":
        colors = cspace_convert(colors, current_colorspace, "JCh")

    mask = np.ones(colors.shape[0], dtype=np.bool_)

    mask &= (colors[:, 0] >= lightness_bounds[0]) & (
        colors[:, 0] <= lightness_bounds[1]
    )
    mask &= (colors[:, 1] >= chroma_bounds[0]) & (colors[:, 1] <= chroma_bounds[1])
    if hue_bounds[0] > hue_bounds[1]:
        mask &= (colors[:, 2] >= hue_bounds[0]) | (colors[:, 2] <= hue_bounds[1])
    else:
        mask &= (colors[:, 2] >= hue_bounds[0]) & (colors[:, 2] <= hue_bounds[1])

    colors = np.ascontiguousarray(colors[mask])

    if output_colorspace == "JCh":
        return colors.astype(np.float32, order="C")
    else:
        try:
            return cspace_convert(colors, "JCh", output_colorspace).astype(
                np.float32, order="C"
            )
        except ValueError:
            raise ValueError(f"Invalid output colorspace {output_colorspace}")
