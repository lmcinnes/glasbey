# MIT License
# Leland McInnes, Sergey Alexandrov

import numpy as np

from colorspacious import cspace_convert
from matplotlib.colors import rgb2hex, to_rgb, LinearSegmentedColormap

from ._grids import rgb_grid, jch_grid, constrain_by_lightness_chroma_hue
from ._internals import (
    generate_palette_cam02ucs,
    generate_next_color_cam02ucs,
    generate_palette_cam02ucs_and_other,
    generate_next_color_cam02ucs_and_other,
)
from ._converters import get_rgb_palette, palette_to_sRGB1

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
    lightness_bounds: Tuple[float, float] = (10.0, 90.0),
    chroma_bounds: Tuple[float, float] = (10.0, 90.0),
    hue_bounds: Tuple[float, float] = (0.0, 360),
    lightness_bend_scale: float = 8.0,
    max_lightness_bend: float = 60.0,
    chroma_bend_scale: float = 6.0,
    max_chroma_bend: float = 60.0,
    hue_bend_scale: float = 6.0,
    max_hue_bend: float = 45.0,
    as_hex: bool = True,
) -> Union[List[str], List[Tuple[float, float, float]]]:
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

    lightness_bend_scale: float (default 8.0)
        Approximately how much to deform the lightness value away from the central base color in
        either direction *per* color in the generated palette.

    max_lightness_bend: float (default 60.0)
        The maximum amount to distort lightness away from the central base color in either direction
        over the whole generated palette.

    chroma_bend_scale: float (default 6.0)
        Approximately how much to deform the chroma value away from the central base color in
        either direction *per* color in the generated palette.

    max_chroma_bend: float (default 60.0)
        The maximum amount to distort chroma away from the central base color in either direction
        over the whole generated palette.

    hue_bend_scale: float (default 6.0)
        Approximately how much to deform the hue value away from the central base color in
        either direction *per* color in the generated palette.

    max_hue_bend: float (default 45.0)
        The maximum amount to distort hue away from the central base color in either direction
        over the whole generated palette.

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
        return [base_color]

    base_rgb = to_rgb(base_color)
    base_jch = cspace_convert(base_rgb, "sRGB1", "JCh")

    if palette_size > 2:
        hue_bend = min(palette_size * hue_bend_scale, max_hue_bend)
        light_bend = min(palette_size * lightness_bend_scale, max_lightness_bend)
        chroma_bend = min(palette_size * chroma_bend_scale, max_chroma_bend)
    else:
        light_bend = max_lightness_bend / 2.0
        chroma_bend = max_chroma_bend / 2.0
        hue_bend = 0.0

    start_jch = [
        max(base_jch[0] - light_bend, lightness_bounds[0]),
        min(base_jch[1] + chroma_bend, chroma_bounds[1]),
        (base_jch[2] - hue_bend) % 360,
    ]
    end_jch = [
        min(base_jch[0] + light_bend, lightness_bounds[1]),
        max(base_jch[1] - chroma_bend, chroma_bounds[0]),
        (base_jch[2] + hue_bend) % 360,
    ]

    start_cam02ucs = cspace_convert(start_jch, "JCh", "CAM02-UCS")
    base_cam02ucs = cspace_convert(base_jch, "JCh", "CAM02-UCS")
    end_cam02ucs = cspace_convert(end_jch, "JCh", "CAM02-UCS")

    # Determine perceptual spacing
    start_to_base_dist = np.sqrt(np.sum((start_cam02ucs - base_cam02ucs) ** 2))
    base_to_end_dist = np.sqrt(np.sum((end_cam02ucs - base_cam02ucs) ** 2))
    path_dist = start_to_base_dist + base_to_end_dist

    start_rgb = np.clip(cspace_convert(start_jch, "JCh", "sRGB1"), 0, 1)
    end_rgb = np.clip(cspace_convert(end_jch, "JCh", "sRGB1"), 0, 1)

    # Create a linear colormap, perceptually spacing the colours
    cmap = LinearSegmentedColormap.from_list(
        "blend",
        [(0.0, start_rgb), (start_to_base_dist / path_dist, base_rgb), (1.0, end_rgb)],
    )
    if palette_size > 2:
        result = cmap(np.linspace(0.0, 1.0, palette_size))[:, :3]
    elif palette_size == 2:
        result = cmap(np.linspace(0.0, 1.0, palette_size + 5))[[1, 5], :3]
    elif palette_size == 1:
        return [base_color]
    else:
        raise ValueError(
            f"Bad palette_size {palette_size} provided; cannot generate a theme palette of that size"
        )

    if as_hex:
        return [rgb2hex(color) for color in result]
    else:
        return result.tolist()


def create_block_palette(
    block_sizes: List[int],
    *,
    sort_block_sizes: bool = True,
    grid_size: Union[int, Tuple[int, int, int]] = 64,  # type: ignore
    grid_space: Literal["RGB", "JCh"] = "RGB",
    generated_color_lightness_bounds: Tuple[float, float] = (30.0, 60.0),
    generated_color_chroma_bounds: Tuple[float, float] = (60.0, 90.0),
    theme_lightness_bounds: Tuple[float, float] = (10.0, 90.0),
    theme_chroma_bounds: Tuple[float, float] = (10.0, 60.0),
    theme_hue_bounds: Tuple[float, float] = (0.0, 360),
    lightness_bend_scale: float = 8.0,
    max_lightness_bend: float = 60.0,
    chroma_bend_scale: float = 6.0,
    max_chroma_bend: float = 60.0,
    hue_bend_scale: float = 6.0,
    max_hue_bend: float = 45.0,
    colorblind_safe: bool = False,
    cvd_type: Literal["protanomaly", "deuteranomaly", "tritanomaly"] = "deuteranomaly",
    cvd_severity: float = 50.0,
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

    sort_block_sizes: bool (default True)
        This approach to palette generation works best when longer blocks are generated first. If this is set to True
        the block sizes will be sorted for the purposes of generation -- the returned palette will retain the same
        block size ordering as input however. If this param is False the block sizes will be generated in the order
        given, providing the greatest distances between the early blocks.

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

    generated_color_lightness_bounds: (float, float) (default (30, 60))
        The upper and lower bounds of lightness values for the block base colors to be used in generating the theme
        palette for the block. These are best kept narrower than a standard Glasbey palette since the theme
        palette will introduce extra lightness/chroma variation.

    generated_color_chroma_bounds: (float, float) (default (60, 90))
        The upper and lower bounds of chroma values for the block base colors to be used in generating the theme
        palette for the block. These are best kept narrower than a standard Glasbey palette since the theme
        palette will introduce extra lightness/chroma variation.

    theme_lightness_bounds: (float, float) (default (10, 90))
        The upper and lower bounds of lightness values for the colors to be used in the theme palette of an individual
        block palette.

    theme_chroma_bounds: (float, float) (default (10, 60))
        The upper and lower bounds of chroma values for the colors to be used in the theme palette of an individual
        block palette.

    theme_hue_bounds: (float, float) (default (0, 360))
        The upper and lower bounds of hue values for the colors to be used in the theme palette of an individual
        block palette.

    lightness_bend_scale: float (default 8.0)
        Approximately how much to deform the lightness value away from the central base color in
        either direction *per* color in each block palette.

    max_lightness_bend: float (default 60.0)
        The maximum amount to distort lightness away from the central base color in either direction
        over a whole block palette.

    chroma_bend_scale: float (default 6.0)
        Approximately how much to deform the chroma value away from the central base color in
        either direction *per* color in each block palette.

    max_chroma_bend: float (default 60.0)
        The maximum amount to distort chroma away from the central base color in either direction
        over a whole block palette.

    hue_bend_scale: float (default 6.0)
        Approximately how much to deform the hue value away from the central base color in
        either direction *per* color in each block palette.

    max_hue_bend: float (default 45.0)
        The maximum amount to distort hue away from the central base color in either direction
        over a whole block palette.


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
    if sort_block_sizes:
        block_order = np.argsort(block_sizes)[::-1]
        block_sizes_for_generation = np.asarray(block_sizes)[block_order]
    else:
        block_sizes_for_generation = block_sizes

    palette: Union[List[str], List[Tuple[float, float, float]]] = []  # type: ignore
    initial_color = create_palette(
        2 if colorblind_safe else 1,
        lightness_bounds=(
            generated_color_lightness_bounds[0],
            generated_color_lightness_bounds[1],
        ),
        chroma_bounds=(
            generated_color_chroma_bounds[0],
            generated_color_chroma_bounds[1],
        ),
        hue_bounds=(0, 360),
    )[-1]
    block = create_theme_palette(
        initial_color,
        block_sizes_for_generation[0],
        lightness_bounds=theme_lightness_bounds,
        chroma_bounds=theme_chroma_bounds,
        hue_bounds=theme_hue_bounds,
        lightness_bend_scale=lightness_bend_scale,
        max_lightness_bend=max_lightness_bend,
        chroma_bend_scale=chroma_bend_scale,
        max_chroma_bend=max_chroma_bend,
        hue_bend_scale=hue_bend_scale,
        max_hue_bend=max_hue_bend,
        as_hex=as_hex,
    )
    palette.extend(block)  # type: ignore

    if grid_space == "JCh":
        colors = jch_grid(
            grid_size=grid_size,
            lightness_bounds=generated_color_lightness_bounds,
            chroma_bounds=generated_color_chroma_bounds,
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
            lightness_bounds=generated_color_lightness_bounds,
            chroma_bounds=generated_color_chroma_bounds,
            hue_bounds=(0, 360),
        )
    else:
        raise ValueError(
            f'Parameter grid_space should be on of "JCh" or "RGB" not {grid_space}'
        )


    if colorblind_safe:
        cvd_space = {
            "name": "sRGB1+CVD",
            "cvd_type": cvd_type,
            "severity": cvd_severity,
        }
        cvd_colors = cspace_convert(colors, "CAM02-UCS", "sRGB1")
        cvd_colors = cspace_convert(cvd_colors, cvd_space, "CAM02-UCS").astype(
            np.float32, order="C"
        )

    distances = np.full(colors.shape[0], 1e9, dtype=np.float32, order="C")

    for block_size in block_sizes_for_generation[1:]:
        block_cam02ucs = cspace_convert(
            np.asarray([to_rgb(color) for color in block]), "sRGB1", "CAM02-UCS"
        ).astype(np.float32)

        if not colorblind_safe:
            next_color = generate_next_color_cam02ucs(colors, distances, block_cam02ucs)
            next_color = np.clip(cspace_convert(next_color, "CAM02-UCS", "sRGB1"), 0, 1)
        else:
            block_cvd_cam02ucs = cspace_convert(
                np.asarray([to_rgb(color) for color in block]), cvd_space, "CAM02-UCS"
            ).astype(np.float32)
            next_color = generate_next_color_cam02ucs_and_other(colors, cvd_colors, distances, block_cam02ucs, block_cvd_cam02ucs, np.float32(0.0))
            next_color = np.clip(cspace_convert(next_color, "CAM02-UCS", "sRGB1"), 0, 1)

        block = create_theme_palette(
            next_color,
            block_size,
            lightness_bounds=theme_lightness_bounds,
            chroma_bounds=theme_chroma_bounds,
            hue_bounds=theme_hue_bounds,
            lightness_bend_scale=lightness_bend_scale,
            max_lightness_bend=max_lightness_bend,
            chroma_bend_scale=chroma_bend_scale,
            max_chroma_bend=max_chroma_bend,
            hue_bend_scale=hue_bend_scale,
            max_hue_bend=max_hue_bend,
            as_hex=as_hex,
        )
        palette.extend(block)  # type: ignore

    if sort_block_sizes:
        block_start_indices = np.hstack(
            ([0], np.cumsum(block_sizes_for_generation)[:-1])
        )
        result: Union[List[str], List[Tuple[float, float, float]]] = []  # type: ignore

        for i in np.argsort(block_order):
            size = block_sizes_for_generation[i]
            block = palette[block_start_indices[i] : block_start_indices[i] + size]
            result.extend(block)  # type: ignore
    else:
        result = palette

    return result
