# MIT License
# Leland McInnes

import numpy as np

from colorspacious import cspace_convert

from typing import Tuple, Union, Literal


def rgb_grid(
    grid_size: Union[int, Tuple[int, int, int]] = 64,
    red_bounds: Tuple[float, float] = (0, 1),
    green_bounds: Tuple[float, float] = (0, 1),
    blue_bounds: Tuple[float, float] = (0, 1),
    output_colorspace: str = "sRGB1",
):
    """Generate a three-dimensional grid of points regularly sampled from RGB space.

    Parameters
    ----------
    grid_size: int or triple of int (default 64)
        When generating a grid of colors that can be used for the platte this determines
        the size of the grid. If a single int is given this determines the side length of the cube sampling the grid
        color space. A grid_size of 256 in RGB will generate all colors that can be represented in RGB. If a triple of
        ints is given then it is the side lengths of cuboid sampling the grid color space.

    red_bounds: (float, float) (default (0.0, 1.0))
        The upper and lower bounds of red channel values for the colors to be used in the resulting palette if sampling
        the grid from RGB space.

    green_bounds: (float, float) (default (0.0, 1.0))
        The upper and lower bounds of green channel values for the colors to be used in the resulting palette if sampling
        the grid from RGB space.

    blue_bounds: (float, float) (default (0.0, 1.0))
        The upper and lower bounds of blue channel values for the colors to be used in the resulting palette if sampling
        the grid from RGB space.

    output_colorspace: str (default sRGB1)
        The colorspace that the resulting grid should be transformed to for output.

    Returns
    -------
    grid: array
        The grid converted to the specified output colour space.
    """
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
    elif hasattr(grid_size, "__len__") and len(grid_size) == 3:  # type: ignore
        b = np.tile(
            np.linspace(blue_bounds[0], blue_bounds[1], grid_size[2]),  # type: ignore
            grid_size[0] * grid_size[1],  # type: ignore
        )
        g = np.tile(
            np.repeat(
                np.linspace(green_bounds[0], green_bounds[1], grid_size[1]),  # type: ignore
                grid_size[2],  # type: ignore
            ),
            grid_size[0],  # type: ignore
        )
        r = np.repeat(
            np.linspace(red_bounds[0], red_bounds[1], grid_size[0]),  # type: ignore
            grid_size[1] * grid_size[2],  # type: ignore
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
    grid_size: Union[int, Tuple[int, int, int]] = 64,
    lightness_bounds: Tuple[float, float] = (10, 90),
    chroma_bounds: Tuple[float, float] = (10, 90),
    hue_bounds: Tuple[float, float] = (0, 360),
    output_colorspace: str = "JCh",
):
    """Generate a three-dimensional grid of points regularly sampled from JCh (lightness, chroma, hue) space.

    Parameters
    ----------
    grid_size: int or triple of int (default 64)
        When generating a grid of colors that can be used for the platte this determines
        the size of the grid. If a single int is given this determines the side length of the cube sampling the grid
        color space. A grid_size of 256 in RGB will generate all colors that can be represented in RGB. If a triple of
        ints is given then it is the side lengths of cuboid sampling the grid color space.

    lightness_bounds: (float, float) (default (10, 90))
        The upper and lower bounds of lightness values for the colors to be used in the resulting palette.

    chroma_bounds: (float, float) (default (10, 90))
        The upper and lower bounds of chroma values for the colors to be used in the resulting palette.

    hue_bounds: (float, float) (default (0, 360))
        The upper and lower bounds of hue values for the colors to be used in the resulting palette.

    output_colorspace: str (default sRGB1)
        The colorspace that the resulting grid should be transformed to for output.

    Returns
    -------
    grid: array
        The grid converted to the specified output colour space. If the output space is not JCh then all colors
        not representable in a standard sRGB1 color space will be removed.
    """
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
    elif hasattr(grid_size, "__len__") and len(grid_size) == 3:  # type: ignore
        c = np.repeat(
            np.linspace(chroma_bounds[0], chroma_bounds[1], grid_size[1]),  # type: ignore
            grid_size[0] * grid_size[2],  # type: ignore
        )
        j = np.tile(
            np.repeat(
                np.linspace(lightness_bounds[0], lightness_bounds[1], grid_size[0]),  # type: ignore
                grid_size[2],  # type: ignore
            ),
            grid_size[1],  # type: ignore
        )
        h = np.tile(
            np.linspace(hue_bounds[0], hue_bounds[1], grid_size[2]),  # type: ignore
            grid_size[0] * grid_size[1],  # type: ignore
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
            # Drop unrepresentable colors
            rgb_colors = cspace_convert(jch_colors, "JCh", "sRGB1")
            rgb_colors = rgb_colors[
                np.all((rgb_colors >= 0.0) & (rgb_colors <= 1.0), axis=1)
            ]
            # convert to output space
            return cspace_convert(rgb_colors, "sRGB1", output_colorspace).astype(
                np.float32, order="C"
            )
        except ValueError:
            raise ValueError(f"Invalid output colorspace {output_colorspace}")


def constrain_by_lightness_chroma_hue(
    colors,
    current_colorspace: Literal["CAM02-UCS", "sRGB1", "JCh"],
    output_colorspace: Literal["CAM02-UCS", "sRGB1", "JCh"] = "CAM02-UCS",
    lightness_bounds: Tuple[float, float] = (10, 90),
    chroma_bounds: Tuple[float, float] = (10, 90),
    hue_bounds: Tuple[float, float] = (0, 360),
):
    """Given an array of colors in ``current_colorspace``, constrain the colors to fit with a given set of
    lightness, chroma and hue bounds, and return the pruned set of colors in the designated ``output_colorspace``.

    Parameters
    ----------
    colors: array of shape (n_colors, 3)
        The colors to constrain/prune.

    current_colorspace: str
        The colorspace of the ``colors`` array passed in.

    output_colorspace: str
        The colorspace that we want the output colors to be in.

    lightness_bounds: (float, float) (default (10, 90))
        The upper and lower bounds of lightness values for the colors .

    chroma_bounds: (float, float) (default (10, 90))
        The upper and lower bounds of chroma values for the colors

    hue_bounds: (float, float) (default (0, 360))
        The upper and lower bounds of hue values for the colors.


    Returns
    -------
    output_colors: array
        The output colors that fall within the specified lightness, chroma and hue bounds in the ``output_colorspace``.
    """
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
