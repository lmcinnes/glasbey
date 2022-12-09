# MIT License
# Leland McInnes, Sergey Alexandrov

import numpy as np

from colorspacious import cspace_convert
from matplotlib.colors import rgb2hex, to_rgb, LinearSegmentedColormap
from sklearn.neighbors import NearestNeighbors

from ._grids import rgb_grid, jch_grid, constrain_by_lightness_chroma_hue
from ._internals import (
    generate_palette_cam02ucs,
    get_next_color,
    generate_palette_cam02ucs_and_other,
    two_space_get_next_color,
)
from ._converters import get_rgb_palette, palette_to_sRGB1
from ._optimizers import optimize_endpoint_color

from typing import *


def create_palette(
    palette_size: int = 256,
    *,
    grid_size: Union[int, Tuple[int, int, int]] = 64,
    as_hex: bool = True,
    grid_space: Literal["RGB", "JCh"] = "RGB",
    lightness_bounds: Tuple[float, float] = (10, 90),
    chroma_bounds: Tuple[float, float] = (10, 90),
    hue_bounds: Tuple[float, float] = (0, 360),
    red_bounds: Tuple[float, float] = (0, 1),
    green_bounds: Tuple[float, float] = (0, 1),
    blue_bounds: Tuple[float, float] = (0, 1),
    colorblind_safe: bool = False,
    cvd_type: Literal["protanomaly", "deuteranomaly", "tritanomaly"] = "deuteranomaly",
    cvd_severity: float = 50.0,
) -> Union[List[str], np.ndarray]:
    """Create a categorical color palette with ``palette_size`` many colours using the Glasbey algorithm with the
    given bounds on hue, chroma and lightness. This should generate a palette that maximizes the perceptual distances
    between colours in the palette up to the constraints on hue, chroma and lightness, and the granularity of the
    possible colour sampling grid.

    Parameters
    ----------
    palette_size: int (default 256)
        The number of colors the created palette should have.

    grid_size: int or triple of int (default 64)
        When generating a grid of colors that can be used for the platte this determines
        the size of the grid. If a single int is given this determines the side length of the cube sampling the grid
        color space. A grid_size of 256 in RGB will generate all colors that can be represented in RGB. If a triple of
        ints is given then it is the side lengths of cuboid sampling the grid color space. This can be useful if sampling
        JCH space when you may want to sample more hues than chroma values, for example.

    as_hex: bool (default True)
        Whether to return the palette as hex-codes or RGB float triples.

    grid_space: "RGB" or "JCh" (default RGB)
        The color space to sample the grid from. Sampling RGB space is the best option to ensure representable colors,
        however it can be useful to sample JCh (lightness, chroma, hue) space instead if you want to use a smaller
        grid size, but what to maintain sampling density with respect to lightness, chroma or hue constraints.

    lightness_bounds: (float, float) (default (10, 90))
        The upper and lower bounds of lightness values for the colors to be used in the resulting palette.

    chroma_bounds: (float, float) (default (10, 90))
        The upper and lower bounds of chroma values for the colors to be used in the resulting palette.

    hue_bounds: (float, float) (default (0, 360))
        The upper and lower bounds of hue values for the colors to be used in the resulting palette.

    red_bounds: (float, float) (default (0.0, 1.0))
        The upper and lower bounds of red channel values for the colors to be used in the resulting palette if sampling
        the grid from RGB space.

    green_bounds: (float, float) (default (0.0, 1.0))
        The upper and lower bounds of green channel values for the colors to be used in the resulting palette if sampling
        the grid from RGB space.

    blue_bounds: (float, float) (default (0.0, 1.0))
        The upper and lower bounds of blue channel values for the colors to be used in the resulting palette if sampling
        the grid from RGB space.

    colorblind_safe: bool (default False)
        If True the created palette will attempt to select colours in a way that should be more easily distinguishable
        for individuals with color vision deficiency. In particular the palette will be selected using distance in
        CAM02-UCS space of a color vision deficient simulation of the sampling grid, so distances will more closely
        resemble perceptual distances of individuals with color vision deficiency.

    cvd_type: one of "protanomaly", "deuteranomaly", "tritanomaly" (default "deuteranomaly")
        The type of colour vision deficiency to attempt to be robust to if ``colorblind_safe`` is True. The cvd_type will
        be passed to colorspacious to simulate the appropriate colour vision deficiency. Per the colorspacious docs:

        * "protanomaly": A common form of red-green colorblindness; affects ~2% of white men to some degree (less
          common among other ethnicities, much less common among women).
        * "deuteranomaly": The most common form of red-green colorblindness; affects ~6% of white men to some degree
          (less common among other ethnicities, much less common among women).
        * "tritanomaly": A very rare form of colorblindness affecting blue/yellow discrimination – so rare that its
          detailed effects and even rate of occurrence are not well understood. Affects <0.1% of people, possibly much
          less. Also, the name we use here is somewhat misleading because only full tritanopia has been documented,
          and partial tritanomaly likely does not exist. What this means is that while Colorspacious will happily
          allow any severity value to be passed, probably only severity = 100 corresponds to any real people.

    cvd_severity: float between 0 and 100 (default 50.0)
        The severity of colour vision deficiency to attemnpt to be robust to if ``colorblind_safe`` is True. The
        cvd_severity will be passed to colorspacious to similate the appropriate colour vision deficiency. Per the
        colorspacious docs: Severity is any number between 0 (indicating regular vision) and 100 (indicating
        complete dichromacy).


    Returns
    -------
    palette: List of hex-code string or array of shape (palette_size, 3)
        The palette created, either as hex colors, or an array of floats of RGB values -- consumable by
        most plotting libraries.
    """
    if grid_space == "JCh":
        colors = jch_grid(
            grid_size=grid_size,
            lightness_bounds=lightness_bounds,
            chroma_bounds=chroma_bounds,
            hue_bounds=hue_bounds,
            output_colorspace="CAM02-UCS",
        )
    elif grid_space == "RGB":
        colors = rgb_grid(
            grid_size=grid_size,
            red_bounds=red_bounds,
            green_bounds=green_bounds,
            blue_bounds=blue_bounds,
            output_colorspace="JCh",
        )
        colors = constrain_by_lightness_chroma_hue(
            colors,
            "JCh",
            lightness_bounds=lightness_bounds,
            chroma_bounds=chroma_bounds,
            hue_bounds=hue_bounds,
        )
    else:
        raise ValueError(
            f'Parameter grid_space should be one of "JCh" or "RGB" not {grid_space}'
        )

    initial_palette = cspace_convert(
        np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]), "sRGB1", "CAM02-UCS"
    ).astype(np.float32, order="C")

    if not colorblind_safe:
        palette = generate_palette_cam02ucs(
            colors, initial_palette, np.uint32(palette_size + 2)
        )
    else:
        cvd_space = {
            "name": "sRGB1+CVD",
            "cvd_type": cvd_type,
            "severity": cvd_severity,
        }
        cvd_colors = cspace_convert(colors, "CAM02-UCS", "sRGB1")
        cvd_colors = cspace_convert(cvd_colors, cvd_space, "CAM02-UCS").astype(
            np.float32, order="C"
        )

        palette = generate_palette_cam02ucs_and_other(
            colors,
            cvd_colors,
            initial_palette,
            initial_palette,
            np.uint32(palette_size + 2),
            np.float32(0.0),
        )

    palette = get_rgb_palette(palette, as_hex=as_hex)[2:]
    return palette


def extend_palette(
    palette,
    palette_size: int = 256,
    *,
    grid_size: Union[int, Tuple[int, int, int]] = 64,  # type: ignore
    as_hex: bool = True,
    grid_space: Literal["RGB", "JCh"] = "RGB",
    lightness_bounds: Optional[Tuple[float, float]] = None,
    chroma_bounds: Optional[Tuple[float, float]] = None,
    hue_bounds: Optional[Tuple[float, float]] = None,
    red_bounds: Tuple[float, float] = (0, 1),
    green_bounds: Tuple[float, float] = (0, 1),
    blue_bounds: Tuple[float, float] = (0, 1),
    colorblind_safe: bool = False,
    cvd_type: Literal["protanomaly", "deuteranomaly", "tritanomaly"] = "deuteranomaly",
    cvd_severity: float = 50.0,
) -> Union[List[str], np.ndarray]:
    """Extend an existing categorical color palette to have ``palette_size`` many colors using the Glasbey algorithm.
    This should generate a palette that maximizes the perceptual distances between colours in the palette up to the
    constraints on hue, chroma and lightness, and the granularity of the possible colour sampling grid. If the
    existing platte is long enough (at least 4 colors), and no explicit bounds are specified, bounds on the
    lightness, chroma and hue will be inferred from the existing colors so that the extended palette should match
    thematically to some degree.

    Parameters
    ----------
    palette: List of hex colors or array of floats of shape (n_colors, 3)
        The palette to be extended, either specified as a list of hex string colors or an array of RGB float
        triples such that the floats represent red, green and blue channels and are in the range 0 to 1.

    palette_size: int (default 256)
        The number of colors the created palette should have.

    grid_size: int or triple of int (default 64)
        When generating a grid of colors that can be used for the platte this determines
        the size of the grid. If a single int is given this determines the side length of the cube sampling the grid
        color space. A grid_size of 256 in RGB will generate all colors that can be represented in RGB. If a triple of
        ints is given then it is the side lengths of cuboid sampling the grid color space. This can be useful if sampling
        JCH space when you may want to sample more hues than chroma values, for example.

    as_hex: bool (default True)
        Whether to return the palette as hex-codes or RGB float triples.

    grid_space: "RGB" or "JCh" (default RGB)
        The color space to sample the grid from. Sampling RGB space is the best option to ensure representable colors,
        however it can be useful to sample JCh (lightness, chroma, hue) space instead if you want to use a smaller
        grid size, but what to maintain sampling density with respect to lightness, chroma or hue constraints.

    lightness_bounds: None or (float, float) (default None)
        The upper and lower bounds of lightness values for the colors to be used in the resulting palette.

    chroma_bounds: None or (float, float) (default None)
        The upper and lower bounds of chroma values for the colors to be used in the resulting palette.

    hue_bounds: None or (float, float) (default None)
        The upper and lower bounds of hue values for the colors to be used in the resulting palette.

    red_bounds: (float, float) (default (0.0, 1.0))
        The upper and lower bounds of red channel values for the colors to be used in the resulting palette if sampling
        the grid from RGB space.

    green_bounds: (float, float) (default (0.0, 1.0))
        The upper and lower bounds of green channel values for the colors to be used in the resulting palette if sampling
        the grid from RGB space.

    blue_bounds: (float, float) (default (0.0, 1.0))
        The upper and lower bounds of blue channel values for the colors to be used in the resulting palette if sampling
        the grid from RGB space.

    colorblind_safe: bool (default False)
        If True the created palette will attempt to select colours in a way that should be more easily distinguishable
        for individuals with color vision deficiency. In particular the palette will be selected using distance in
        CAM02-UCS space of a color vision deficient simulation of the sampling grid, so distances will more closely
        resemble perceptual distances of individuals with color vision deficiency.

    cvd_type: one of "protanomaly", "deuteranomaly", "tritanomaly" (default "deuteranomaly")
        The type of colour vision deficiency to attempt to be robust to if ``colorblind_safe`` is True. The cvd_type will
        be passed to colorspacious to simulate the appropriate colour vision deficiency. Per the colorspacious docs:

        * "protanomaly": A common form of red-green colorblindness; affects ~2% of white men to some degree (less
          common among other ethnicities, much less common among women).
        * "deuteranomaly": The most common form of red-green colorblindness; affects ~6% of white men to some degree
          (less common among other ethnicities, much less common among women).
        * "tritanomaly": A very rare form of colorblindness affecting blue/yellow discrimination – so rare that its
          detailed effects and even rate of occurrence are not well understood. Affects <0.1% of people, possibly much
          less. Also, the name we use here is somewhat misleading because only full tritanopia has been documented,
          and partial tritanomaly likely does not exist. What this means is that while Colorspacious will happily
          allow any severity value to be passed, probably only severity = 100 corresponds to any real people.

    cvd_severity: float between 0 and 100 (default 50.0)
        The severity of colour vision deficiency to attemnpt to be robust to if ``colorblind_safe`` is True. The
        cvd_severity will be passed to colorspacious to similate the appropriate colour vision deficiency. Per the
        colorspacious docs: Severity is any number between 0 (indicating regular vision) and 100 (indicating
        complete dichromacy).


    Returns
    -------
    palette: List of hex-code string or array of shape (palette_size, 3)
        The palette created, either as hex colors, or an array of floats of RGB values -- consumable by
        most plotting libraries.
    """
    try:
        palette = palette_to_sRGB1(palette)
    except:
        raise ValueError(
            "Failed to parse the palette to be extended. Is it formatted correctly?"
        )

    if any(param is None for param in (lightness_bounds, chroma_bounds, hue_bounds)):
        jch_palette = cspace_convert(palette, "sRGB1", "JCh")

        if lightness_bounds is None:
            if len(palette) > 3:
                lightness_bounds = (np.min(jch_palette.T[0]), np.max(jch_palette.T[0]))
            else:
                lightness_bounds = (10, 90)

        if chroma_bounds is None:
            if len(palette) > 3:
                chroma_bounds = (np.min(jch_palette.T[1]), np.max(jch_palette.T[1]))
            else:
                chroma_bounds = (10, 90)

        if hue_bounds is None:
            if len(palette) > 3:
                hue_bounds = (np.min(jch_palette.T[2]), np.max(jch_palette.T[2]))
            else:
                hue_bounds = (0, 360)

    if grid_space == "JCh":
        colors = jch_grid(
            grid_size=grid_size,
            lightness_bounds=lightness_bounds,  # type: ignore
            chroma_bounds=chroma_bounds,  # type: ignore
            hue_bounds=hue_bounds,  # type: ignore
            output_colorspace="CAM02-UCS",
        )
    elif grid_space == "RGB":
        colors = rgb_grid(
            grid_size=grid_size,
            red_bounds=red_bounds,
            green_bounds=green_bounds,
            blue_bounds=blue_bounds,
            output_colorspace="JCh",
        )
        colors = constrain_by_lightness_chroma_hue(
            colors,
            "JCh",
            lightness_bounds=lightness_bounds,
            chroma_bounds=chroma_bounds,
            hue_bounds=hue_bounds,
        )
    else:
        raise ValueError(
            f'Parameter grid_space should be on of "JCh" or "RGB" not {grid_space}'
        )

    if not colorblind_safe:
        palette = cspace_convert(palette, "sRGB1", "CAM02-UCS").astype(
            np.float32, order="C"
        )
        palette = generate_palette_cam02ucs(colors, palette, np.uint32(palette_size))
    else:
        cvd_space = {
            "name": "sRGB1+CVD",
            "cvd_type": cvd_type,
            "severity": cvd_severity,
        }
        cvd_colors = cspace_convert(colors, "CAM02-UCS", "sRGB1")
        cvd_colors = cspace_convert(cvd_colors, cvd_space, "CAM02-UCS").astype(
            np.float32, order="C"
        )
        cam_palette = cspace_convert(palette, "sRGB1", "CAM02-UCS").astype(
            np.float32, order="C"
        )
        cvd_palette = cspace_convert(palette, "sRGB1", "CAM02-UCS").astype(
            np.float32, order="C"
        )

        palette = generate_palette_cam02ucs_and_other(
            colors,
            cvd_colors,
            cam_palette,
            cvd_palette,
            np.uint32(palette_size),
            np.float32(0.0),
        )

    palette = get_rgb_palette(palette, as_hex=as_hex)
    return palette


def create_theme_palette(
    base_color,
    palette_size: int = 5,
    *,
    color_grid: Optional[np.ndarray] = None,
    color_search_index: Optional[NearestNeighbors] = None,
    lightness_bounds: Tuple[float, float] = (10.0, 90.0),
    chroma_bounds: Tuple[float, float] = (10.0, 90.0),
    hue_bounds: Tuple[float, float] = (0.0, 360),
    other_colors: Optional[np.ndarray] = None,
    min_color_offset: float = 0.2,
    input_colorspace: Literal["RGB", "CAM02-UCS"] = "RGB",
    output_colorspace: Literal["RGB", "CAM02-UCS"] = "RGB",
    as_hex: bool = True,
) -> Union[List[str], List[Tuple[float, float, float]], np.ndarray]:
    """Create a color palette with a range of colors around a central theme color that vary smoothly in a range
    of lightness, chroma and (to less of a degree) hue. The goal is to generate a smooth color palette that
    provides some variation of colors while remaining relatively close to the base color. This is primarily for
    use in the creation of block color palettes, but may be useful in its own right.

    Parameters
    ----------
    base_color: hex str or triple of floats for RGB
        The base color around which to build the theme palette.

    palette_size: int (default 5)
        The number of colors the created palette should have.

    lightness_bounds: (float, float) (default (10, 90))
        The upper and lower bounds of lightness values for the colors to be used in the resulting palette.

    chroma_bounds: (float, float) (default (10, 90))
        The upper and lower bounds of chroma values for the colors to be used in the resulting palette.

    hue_bounds: (float, float) (default (0, 360))
        The upper and lower bounds of hue values for the colors to be used in the resulting palette.

    as_hex: bool (default True)
        Whether to return the palette as hex-codes or RGB float triples.

    Returns
    -------
    palette: List of hex-code string or array of shape (palette_size, 3)
        The resulting theme palette. It may not contain the base color, but should provide a smooth
        palette over a variation of chroma and lightness with a small change in hue based around
        the base color.
    """
    if palette_size == 1:
        if output_colorspace == "RGB":
            if input_colorspace == "RGB":
                return [base_color]
            else:
                return get_rgb_palette([base_color], as_hex=as_hex)
        else:
            if input_colorspace == "RGB":
                return cspace_convert([to_rgb(base_color)], "sRGB1", "CAM02-UCS")
            else:
                return np.asarray(base_color).reshape(1, -1)

    if input_colorspace == "RGB":
        base_color = cspace_convert(to_rgb(base_color), "sRGB1", "CAM02-UCS").astype(
            np.float32
        )
    else:
        base_color = np.asarray(base_color, dtype=np.float32)

    if color_grid is None:
        color_grid = rgb_grid(
            grid_size=64,
            output_colorspace="JCh",
        )
        color_grid = constrain_by_lightness_chroma_hue(
            color_grid,
            "JCh",
            lightness_bounds=lightness_bounds,
            chroma_bounds=chroma_bounds,
            hue_bounds=hue_bounds,
        )

    if color_search_index is None:
        color_search_index = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
            color_grid
        )

    if other_colors is None:
        other_colors = cspace_convert([[1, 1, 1], [0, 0, 0]], "sRGB1", "CAM02-UCS")

    # Specially coded offset vectors based on my aesthetic preferences
    left_offset = np.array(
        (
            min(60, base_color[0] - 10),
            -np.sign(base_color[1]) * max(0, 30 - base_color[0]) / 4.0,
            np.sign(base_color[1]) * max(0, 30 - base_color[0]),
        )
    )
    right_offset = np.array(
        (
            min(80, 110 - base_color[0] + max(base_color[1], 0) / 3.0),
            np.sign(base_color[1]) * 5,
            20,
        )
    )

    left = color_grid[
        np.squeeze(
            color_search_index.kneighbors(
                [base_color - left_offset], return_distance=False
            )
        )
    ]
    right = color_grid[
        np.squeeze(
            color_search_index.kneighbors(
                [base_color + right_offset], return_distance=False
            )
        )
    ]

    left = optimize_endpoint_color(left, other_colors, color_grid, color_search_index, base_color - left)
    right = optimize_endpoint_color(right, other_colors, color_grid, color_search_index, base_color - right)

    left_to_mid = np.sqrt(np.sum((left - base_color) ** 2))
    mid_to_right = np.sqrt(np.sum((right - base_color) ** 2))

    xdata = np.array(
        [-0.05, mid_to_right / (left_to_mid + mid_to_right), 1.05], dtype=np.float32
    )
    ydata = np.vstack([left, base_color, right])
    polynomial = [np.polynomial.Polynomial.fit(xdata, ydata[:, i], 2) for i in range(3)]

    window_width = min(min_color_offset * palette_size, 1.0)
    expected_dist_distribution = np.linspace(
        (1.0 - window_width) / 2.0, (1.0 + window_width) / 2.0, palette_size
    )
    xs = np.linspace(
        (1.0 - window_width) / 2.0, (1.0 + window_width) / 2.0, palette_size
    )

    for anneal_val in np.linspace(1, 0, 10):
        candidate_colors = np.asarray([p(xs) for p in polynomial]).T
        indices = np.squeeze(
            color_search_index.kneighbors(candidate_colors, return_distance=False)
        )
        candidate_colors = color_grid[indices]

        dists = np.hstack(
            [
                [0],
                np.sqrt(
                    np.sum((candidate_colors[:-1] - candidate_colors[1:]) ** 2, axis=1)
                ),
            ]
        )
        dist_distribution = (window_width * np.cumsum(dists) / (dists.sum() + 1e-8)) + (
            1.0 - window_width
        ) / 2.0
        errors = dist_distribution - expected_dist_distribution
        mean_squared_error = np.sum((errors) ** 2)
        if mean_squared_error < 0.005:
            break
        else:
            xs = xs - (errors * anneal_val)

    if output_colorspace == "CAM02-UCS":
        return candidate_colors
    else:
        return get_rgb_palette(candidate_colors, as_hex=as_hex)


def create_block_palette(
    block_sizes: List[int],
    *,
    grid_size: Union[int, Tuple[int, int, int]] = 64,  # type: ignore
    grid_space: Literal["RGB", "JCh"] = "RGB",
    lightness_bounds: Tuple[float, float] = (10, 90),
    chroma_bounds: Tuple[float, float] = (0, 80),
    hue_bounds: Tuple[float, float] = (0, 360),
    colorblind_safe: bool = False,
    cvd_type: Literal["protanomaly", "deuteranomaly", "tritanomaly"] = "deuteranomaly",
    cvd_severity: float = 50.0,
    theme_color_spacing = 0.2,
    as_hex: bool = True,
) -> Union[List[str], List[Tuple[float, float, float]]]:
    """Create a categorical color palette in blocks using the Glasbey algorithm.
    This should generate a palette that maximizes the perceptual distances between blocks in the palette up to the
    constraints on hue, chroma and lightness, and the granularity of the possible colour sampling grid. In turn each
    block should be a range of colors thematically centered around a base colour. This is most useful for two
    layer hierarchical categories, where the top layer categories each get their own block, and the finer grained
    sub-categories get their own colour within the block.

    Parameters
    ----------
    block_sizes: List of ints
        The sizes of the different blocks to generate. The total palette size will be the sum of all the individual
        block sizes.

    grid_size: int or triple of int (default 64)
        When generating a grid of colors that can be used for the platte this determines
        the size of the grid. If a single int is given this determines the side length of the cube sampling the grid
        color space. A grid_size of 256 in RGB will generate all colors that can be represented in RGB. If a triple of
        ints is given then it is the side lengths of cuboid sampling the grid color space. This can be useful if sampling
        JCH space when you may want to sample more hues than chroma values, for example.

    grid_space: "RGB" or "JCh" (default RGB)
        The color space to sample the grid from. Sampling RGB space is the best option to ensure representable colors,
        however it can be useful to sample JCh (lightness, chroma, hue) space instead if you want to use a smaller
        grid size, but what to maintain sampling density with respect to lightness, chroma or hue constraints.

    colorblind_safe: bool (default False)
        If True the created palette will attempt to select colours in a way that should be more easily distinguishable
        for individuals with color vision deficiency. In particular the palette will be selected using distance in
        CAM02-UCS space of a color vision deficient simulation of the sampling grid, so distances will more closely
        resemble perceptual distances of individuals with color vision deficiency.

    cvd_type: one of "protanomaly", "deuteranomaly", "tritanomaly" (default "deuteranomaly")
        The type of colour vision deficiency to attempt to be robust to if ``colorblind_safe`` is True. The cvd_type will
        be passed to colorspacious to simulate the appropriate colour vision deficiency. Per the colorspacious docs:

        * "protanomaly": A common form of red-green colorblindness; affects ~2% of white men to some degree (less
          common among other ethnicities, much less common among women).
        * "deuteranomaly": The most common form of red-green colorblindness; affects ~6% of white men to some degree
          (less common among other ethnicities, much less common among women).
        * "tritanomaly": A very rare form of colorblindness affecting blue/yellow discrimination – so rare that its
          detailed effects and even rate of occurrence are not well understood. Affects <0.1% of people, possibly much
          less. Also, the name we use here is somewhat misleading because only full tritanopia has been documented,
          and partial tritanomaly likely does not exist. What this means is that while Colorspacious will happily
          allow any severity value to be passed, probably only severity = 100 corresponds to any real people.

    cvd_severity: float between 0 and 100 (default 50.0)
        The severity of colour vision deficiency to attemnpt to be robust to if ``colorblind_safe`` is True. The
        cvd_severity will be passed to colorspacious to similate the appropriate colour vision deficiency. Per the
        colorspacious docs: Severity is any number between 0 (indicating regular vision) and 100 (indicating
        complete dichromacy).

    as_hex: bool (default True)
        Whether to return the palette as hex-codes or RGB float triples.

    Returns
    -------
    palette:
        The resulting palette with blocks of theme palette of the specified block sizes.
    """
    palette: Union[List[str], List[Tuple[float, float, float]]] = []  # type: ignore
    white, black = cspace_convert([[1, 1, 1], [0, 0, 0]], "sRGB1", "CAM02-UCS").astype(
        np.float32
    )
    if grid_space == "JCh":
        colors = jch_grid(
            grid_size=grid_size,
            lightness_bounds=lightness_bounds,  # type: ignore
            chroma_bounds=chroma_bounds,  # type: ignore
            hue_bounds=hue_bounds,  # type: ignore
            output_colorspace="CAM02-UCS",
        )
    elif grid_space == "RGB":
        colors = rgb_grid(
            grid_size=grid_size,
            output_colorspace="JCh",
        )
        colors = constrain_by_lightness_chroma_hue(
            colors,
            "JCh",
            lightness_bounds=lightness_bounds,
            chroma_bounds=chroma_bounds,
            hue_bounds=hue_bounds,
        )
    else:
        raise ValueError(
            f'Parameter grid_space should be on of "JCh" or "RGB" not {grid_space}'
        )

    sub_grid = colors[(colors[:, 0] >= 40) & (colors[:, 0] <= 70)]
    color_search_index = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        colors
    )

    if colorblind_safe:
        cvd_space = {
            "name": "sRGB1+CVD",
            "cvd_type": cvd_type,
            "severity": cvd_severity,
        }
        cvd_colors = cspace_convert(sub_grid, "CAM02-UCS", "sRGB1")
        cvd_colors = cspace_convert(cvd_colors, cvd_space, "CAM02-UCS").astype(
            np.float32, order="C"
        )

    distances = np.full(sub_grid.shape[0], 1e12, dtype=np.float32, order="C")

    if not colorblind_safe:
        get_next_color(distances, sub_grid, white)
        get_next_color(distances, sub_grid, black)
    else:
        two_space_get_next_color(distances, sub_grid, cvd_colors, white, white, 0.0)
        two_space_get_next_color(distances, sub_grid, cvd_colors, black, black, 0.0)

    colors_to_avoid = np.vstack([black, white])

    for block_size in block_sizes:
        if not colorblind_safe:
            base_color = get_next_color(distances, sub_grid, white)
        else:
            base_color, _ = two_space_get_next_color(
                distances, sub_grid, cvd_colors, white, white, 0.0
            )

        block = create_theme_palette(
            base_color,
            block_size,
            color_grid=colors,
            color_search_index=color_search_index,
            other_colors=colors_to_avoid,
            input_colorspace="CAM02-UCS",
            output_colorspace="CAM02-UCS",
            min_color_offset=theme_color_spacing,
        )
        colors_to_avoid = np.vstack([colors_to_avoid, block])

        for color in block:
            if not colorblind_safe:
                get_next_color(distances, sub_grid, color)
            else:
                cvd_color = cspace_convert(color, "CAM02-UCS", "sRGB1")
                cvd_color = cspace_convert(cvd_color, cvd_space, "CAM02-UCS").astype(
                    np.float32, order="C"
                )
                two_space_get_next_color(
                    distances, sub_grid, cvd_colors, color, cvd_color, 0.0
                )

        palette.extend(block.tolist())

    palette = get_rgb_palette(palette, as_hex=as_hex)

    return palette
