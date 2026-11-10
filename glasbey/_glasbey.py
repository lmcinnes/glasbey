# MIT License
# Leland McInnes, Sergey Alexandrov

import warnings

import numpy as np

from colorspacious import cspace_convert
from matplotlib.colors import rgb2hex, to_rgb, LinearSegmentedColormap
from sklearn.neighbors import NearestNeighbors

from ._grids import rgb_grid, jch_grid, constrain_by_lightness_chroma_hue
from ._internals import (
    generate_palette_cam02ucs,
    generate_palette_cam02ucs_and_other,
)
from ._converters import get_rgb_palette, palette_to_sRGB1
from ._optimizers import optimize_existing_palette
from . import _legacy_blocks
from ._blocks import (
    Footprint, _BlockState, blocks_to_output, build_block_palette, choose_twist, make_context,
    plan_blocks,
)
from ._curves import _Grid, _cam02ucs_to_jch
from ._theme import DEFAULT_STEP, fit_theme_arc

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
    optimize_palette: bool = True,
    optimize_palette_n_iter: int = 100,
    optimize_palette_search_radius: int = 10,
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

    optimize_palette: bool (default True)
        If True the palette will be optimized to ensure that the colors are as perceptually distinct as possible.
        This is done a greeedy optimization on the palette set to attempt to hill-climb to comething better.

    optimize_palette_n_iter: int (default 100)
        The number of iterations to run the palette optimization for. This is a greedy optimization and may not
        find the global optimum, but it should be fast and effective for small palettes.

    optimize_palette_search_radius: int (default 10)
        The search radius to use for the palette optimization. This is the maximum distance in CAM02-UCS space
        that a color can be moved to find a better color. A larger search radius will result in a more optimized
        palette, but will take longer to compute.

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

    if optimize_palette:
        palette = palette[2:]
        color_search_index = NearestNeighbors().fit(colors)
        if colorblind_safe:
            palette = optimize_existing_palette(
                palette,
                colors,
                cvd_colors,
                color_search_index,
                colorblind_safe=True,
                cvd_type=cvd_type,
                cvd_severity=cvd_severity,
                n_iter=optimize_palette_n_iter,
                search_radius=optimize_palette_search_radius,
            )
        else:
            palette = optimize_existing_palette(
                palette,
                colors,
                colors,
                color_search_index,
                n_iter=optimize_palette_n_iter,
                search_radius=optimize_palette_search_radius,
            )
        return get_rgb_palette(palette, as_hex=as_hex)
    else:

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
    optimize_palette: bool = True,
    optimize_palette_n_iter: int = 100,
    optimize_palette_search_radius: int = 10,
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

    optimize_palette: bool (default True)
        If True the palette will be optimized to ensure that the colors are as perceptually distinct as possible.
        This is done a greeedy optimization on the palette set to attempt to hill-climb to something better.

    optimize_palette_n_iter: int (default 100)
        The number of iterations to run the palette optimization for. This is a greedy optimization and may not
        find the global optimum, but it should be fast and effective for small palettes.

    optimize_palette_search_radius: int (default 10)
        The search radius to use for the palette optimization. This is the maximum distance in CAM02-UCS space
        that a color can be moved to find a better color. A larger search radius will result in a more optimized
        palette, but will take longer to compute.

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
    initial_palette_size = len(palette)

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

    if palette_size <= initial_palette_size:
        # Nothing to add; return the leading colors of the given palette.
        palette = cspace_convert(
            np.asarray(palette[:palette_size]), "sRGB1", "CAM02-UCS"
        )
        return get_rgb_palette(palette, as_hex=as_hex)

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

    if optimize_palette:
        color_search_index = NearestNeighbors().fit(colors)
        if colorblind_safe:
            palette = optimize_existing_palette(
                palette,
                colors,
                cvd_colors,
                color_search_index,
                colorblind_safe=True,
                cvd_type=cvd_type,
                cvd_severity=cvd_severity,
                movable_colors=list(range(initial_palette_size, len(palette))),
                n_iter=optimize_palette_n_iter,
                search_radius=optimize_palette_search_radius,
            )
        else:
            palette = optimize_existing_palette(
                palette,
                colors,
                colors,
                color_search_index,
                movable_colors=list(range(initial_palette_size, len(palette))),
                n_iter=optimize_palette_n_iter,
                search_radius=optimize_palette_search_radius,
            )
        return get_rgb_palette(palette, as_hex=as_hex)
    else:
        palette = get_rgb_palette(palette, as_hex=as_hex)
        return palette


# Maximum hue sweep (degrees) per block when colorblind_safe=True.
COLORBLIND_MAX_SWEEP = 15.0


def _resolve_step(theme_step, legacy_value, legacy_name):
    """Map deprecated spacing parameters onto ``theme_step``."""
    if legacy_value is not None:
        warnings.warn(
            f"{legacy_name} is deprecated and will be removed in a future release; use theme_step "
            f"instead ({legacy_name}={legacy_value} corresponds to theme_step={60 * legacy_value:g}).",
            DeprecationWarning, stacklevel=3,
        )
        if theme_step is None:
            theme_step = 60.0 * float(legacy_value)
    if theme_step is None:
        theme_step = DEFAULT_STEP
    if not np.isscalar(theme_step) or not np.isfinite(theme_step) or theme_step <= 0:
        raise ValueError("theme_step must be a positive finite number")
    return float(theme_step)


def _validate_block_options(grid_space, hue_sweep, twist, theme_method):
    if grid_space not in ("RGB", "JCh"):
        raise ValueError(
            f'Parameter grid_space should be one of "JCh" or "RGB" not {grid_space}'
        )
    if theme_method not in ("arc", "legacy"):
        raise ValueError('theme_method must be "arc" or "legacy"')
    if not (isinstance(hue_sweep, str) and hue_sweep == "auto"):
        if isinstance(hue_sweep, str) or not np.isscalar(hue_sweep) or not np.isfinite(hue_sweep) \
                or hue_sweep < 0:
            raise ValueError('hue_sweep must be "auto" or a non-negative number of degrees')
    if not (isinstance(twist, str) and twist == "auto") and twist not in (1, -1):
        raise ValueError('twist must be "auto", 1 or -1')


def _validate_block_sizes(block_sizes):
    sizes = list(block_sizes)
    for size in sizes:
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("block sizes must be positive integers")
    return [int(size) for size in sizes]


def _routing_grid_size(grid_size):
    # The routing grid only guides curve fitting, so half the palette grid resolution is enough.
    if isinstance(grid_size, (int, np.integer)):
        return max(8, int(grid_size) // 2)
    return tuple(max(8, int(size) // 2) for size in grid_size)


def _legacy_warning():
    warnings.warn(
        'theme_method="legacy" reproduces the 0.4.0 block palettes and will be removed in a future '
        "release.", FutureWarning, stacklevel=3,
    )


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
    min_color_offset: Optional[float] = None,
    input_colorspace: Literal["RGB", "CAM02-UCS"] = "RGB",
    output_colorspace: Literal["RGB", "CAM02-UCS"] = "RGB",
    as_hex: bool = True,
) -> Union[List[str], List[Tuple[float, float, float]], np.ndarray]:
    """Create a color palette with a range of colors around a central theme color. The colors lie on a smooth
    curve through the base color that varies mostly in lightness, with some variation in chroma and hue, and are
    spaced evenly by perceptual distance. This is primarily for use in the creation of block color palettes, but
    may be useful in its own right.

    Parameters
    ----------
    base_color: hex str or triple of floats for RGB
        The base color around which to build the theme palette. If it lies on the edge of the colour gamut it
        may be moved slightly toward neutral so that a smooth curve through it is possible.

    palette_size: int (default 5)
        The number of colors the created palette should have.

    color_grid: array or None (default None)
        Optional CAM02-UCS colours used to guide the curve fit. A grid is built from the bounds when omitted.

    color_search_index: ignored
        Kept for backwards compatibility.

    lightness_bounds: (float, float) (default (10, 90))
        The upper and lower bounds of lightness values for the colors to be used in the resulting palette.

    chroma_bounds: (float, float) (default (10, 90))
        The upper and lower bounds of chroma values for the colors to be used in the resulting palette.

    hue_bounds: (float, float) (default (0, 360))
        The upper and lower bounds of hue values for the colors to be used in the resulting palette.

    other_colors: array or None (default None)
        CAM02-UCS colours to keep away from; the palette's position along its curve is chosen to avoid them.

    min_color_offset: float or None (default None)
        Deprecated; the distance between adjacent colours is 10 in CAM02-UCS by default. If given, the
        distance is ``60 * min_color_offset`` and a ``DeprecationWarning`` is emitted.

    input_colorspace: "RGB" or "CAM02-UCS" (default "RGB")
        The colour space of ``base_color``.

    output_colorspace: "RGB" or "CAM02-UCS" (default "RGB")
        The colour space of the returned palette.

    as_hex: bool (default True)
        Whether to return the palette as hex-codes or RGB float triples.

    Returns
    -------
    palette: List of hex-code string or array of shape (palette_size, 3)
        The resulting theme palette, ordered from dark to light.
    """
    step = _resolve_step(None, min_color_offset, "min_color_offset")
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
        base = cspace_convert(to_rgb(base_color), "sRGB1", "CAM02-UCS")
    else:
        base = np.asarray(base_color, dtype=float)

    context = make_context(lightness_bounds, chroma_bounds, hue_bounds)
    if color_grid is not None and len(color_grid) >= 8:
        context.grid = _Grid.from_colors(context.domain, np.asarray(color_grid, dtype=float))
    spec = plan_blocks([palette_size], context, step)[0]
    hue = float(_cam02ucs_to_jch(base)[2])
    seed = Footprint(base[None], base, hue, spec.demand, 1, spec.kind)
    twist = choose_twist([spec], [seed])
    theme = fit_theme_arc(
        base, palette_size, step=step, domain=context.domain, gamut=context.gamut, grid=context.grid,
        hue_allowance=(spec.demand / 2, spec.demand / 2), twist=twist, other_colors=other_colors,
    )
    state = _BlockState(theme, theme.step * (palette_size - 1),
                        theme.diagnostics.get("window_offset", 0.0))
    if output_colorspace == "CAM02-UCS":
        return state.colors()
    return blocks_to_output([state], as_hex=as_hex)


def _build_block_palette_with_diagnostics(
    block_sizes,
    *,
    grid_size=64,
    grid_space="RGB",
    lightness_bounds=(10, 90),
    chroma_bounds=(0, 80),
    hue_bounds=(0, 360),
    colorblind_safe=False,
    cvd_type="deuteranomaly",
    cvd_severity=50.0,
    theme_step=None,
    theme_color_spacing=None,
    hue_sweep="auto",
    twist="auto",
    as_hex=True,
    optimize_palette=True,
    optimize_palette_n_iter=10,
    optimize_palette_search_radius=5,
    fixed_colors=None,
):
    """Build a block palette with the arc method; return (palette, diagnostics)."""
    _validate_block_options(grid_space, hue_sweep, twist, "arc")
    sizes = _validate_block_sizes(block_sizes)
    step = _resolve_step(theme_step, theme_color_spacing, "theme_color_spacing")
    if not sizes:
        return [], {"specs": []}
    if colorblind_safe:
        # Hue rotation trades against CVD separation; keep blocks narrow in hue.
        cap = COLORBLIND_MAX_SWEEP if hue_sweep == "auto" else min(float(hue_sweep), COLORBLIND_MAX_SWEEP)
        hue_sweep = cap
    context = make_context(
        lightness_bounds, chroma_bounds, hue_bounds, grid_size=_routing_grid_size(grid_size),
        grid_space=grid_space, colorblind_safe=colorblind_safe, cvd_type=cvd_type,
        cvd_severity=cvd_severity, hue_sweep=hue_sweep,
    )
    colors, diagnostics = build_block_palette(
        sizes, context=context, step=step, twist=twist, fixed_colors=fixed_colors,
        optimize=optimize_palette, n_iter=optimize_palette_n_iter,
        search_radius=float(optimize_palette_search_radius),
    )
    palette = blocks_to_output(diagnostics["states"], as_hex=as_hex)
    return palette, diagnostics


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
    theme_color_spacing: Optional[float] = None,
    as_hex: bool = True,
    optimize_palette: bool = True,
    optimize_palette_n_iter: int = 10,
    optimize_palette_search_radius: int = 5,
    theme_step: Optional[float] = None,
    hue_sweep: Union[Literal["auto"], float] = "auto",
    twist: Union[Literal["auto"], int] = "auto",
    theme_method: Literal["arc", "legacy"] = "arc",
) -> Union[List[str], List[Tuple[float, float, float]]]:
    """Create a categorical color palette in blocks.
    Each block is a range of colors on a smooth curve around a base colour, ordered from dark to light and
    spaced evenly by perceptual distance; the base colours are chosen, knowing the block sizes, to keep the
    blocks as distinct from one another as possible within the constraints on hue, chroma and lightness. This is
    most useful for two layer hierarchical categories, where the top layer categories each get their own block,
    and the finer grained sub-categories get their own colour within the block.

    Parameters
    ----------
    block_sizes: List of ints
        The sizes of the different blocks to generate. The total palette size will be the sum of all the individual
        block sizes. Blocks appear in the output in this order.

    grid_size: int or triple of int (default 64)
        The resolution of the colour grid. Half of this resolution is used for the grid that guides curve
        fitting. A triple of ints gives separate side lengths for the grid's three axes.

    grid_space: "RGB" or "JCh" (default RGB)
        The color space to sample the grids from. Sampling RGB space is the best option to ensure representable
        colors; sampling JCh (lightness, chroma, hue) space keeps sampling density uniform with respect to
        lightness, chroma and hue constraints.

    lightness_bounds: (float, float) (default (10, 90))
        The upper and lower bounds of CIECAM02 lightness for the colors in the resulting palette.

    chroma_bounds: (float, float) (default (0, 80))
        The upper and lower bounds of CIECAM02 chroma for the colors in the resulting palette.

    hue_bounds: (float, float) (default (0, 360))
        The upper and lower bounds of hue values for the colors in the resulting palette. Wrapped ranges such
        as (350, 20) are supported.

    colorblind_safe: bool (default False)
        If True the palette will attempt to select colours in a way that should be more easily distinguishable
        for individuals with color vision deficiency. Block placement and block polishing then use the
        minimum of normal and CVD-simulated separations.

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
        The severity of colour vision deficiency to attempt to be robust to if ``colorblind_safe`` is True. The
        cvd_severity will be passed to colorspacious to simulate the appropriate colour vision deficiency. Per the
        colorspacious docs: Severity is any number between 0 (indicating regular vision) and 100 (indicating
        complete dichromacy).

    theme_color_spacing: float or None (default None)
        Deprecated; see ``theme_step``.

    as_hex: bool (default True)
        Whether to return the palette as hex-codes or RGB float triples.

    optimize_palette: bool (default True)
        If True, polish the palette at the block level: repeatedly take the two closest blocks and try sliding
        a block's colours along its curve, shrinking its step slightly, or (for short blocks) refitting it from
        a nudged base colour, keeping whichever move most increases the minimum distance between blocks. The
        order and smoothness of colours within each block are always preserved.

    optimize_palette_n_iter: int (default 10)
        The maximum number of block-polishing rounds.

    optimize_palette_search_radius: float (default 5)
        The maximum distance in CAM02-UCS space that a block may slide along its curve, or that a short
        block's base colour may be nudged, in one polishing move.

    theme_step: float or None (default None)
        Target CAM02-UCS distance between adjacent colours within a block; ``None`` means 10. A block of
        ``n`` colours aims for a perceptual length of ``(n - 1) * theme_step``. If a block cannot reach that
        length inside the colour bounds its step shrinks automatically, with a warning when it falls below 4.

    theme_color_spacing: float or None (default None)
        Deprecated; use ``theme_step``. If given, it is mapped to ``theme_step = 60 * theme_color_spacing``
        (so the old default 0.2 corresponds to a step of 12) and a ``DeprecationWarning`` is emitted.

    hue_sweep: "auto" or float (default "auto")
        How much each block may rotate in hue from its dark end to its light end. ``"auto"`` allocates a
        sweep per block from its size and the number of blocks, so larger blocks and palettes with fewer
        blocks rotate more. A float is the maximum total sweep in degrees for any block; ``0`` disables hue
        rotation. With ``colorblind_safe=True`` the sweep is capped at 15 degrees, since hue rotation
        costs separation under colour vision deficiency.

    twist: "auto", 1 or -1 (default "auto")
        The global direction in which the light ends of all blocks rotate: ``1`` toward increasing hue,
        ``-1`` toward decreasing hue. ``"auto"`` chooses the direction that sends the most light ends toward
        yellow. All blocks share one direction, which packs blocks more tightly.

    theme_method: "arc" or "legacy" (default "arc")
        ``"arc"`` builds each block as a smooth curve sampled evenly by perceptual arc length, with block
        base colours chosen knowing the block sizes. ``"legacy"`` runs the 0.4.0 algorithm unchanged, for
        reproducing old palettes; it emits a ``FutureWarning`` and will be removed in a future release.

    Returns
    -------
    palette:
        The resulting palette with blocks of theme palette of the specified block sizes.
    """
    _validate_block_options(grid_space, hue_sweep, twist, theme_method)
    if theme_method == "legacy":
        _legacy_warning()
        return _legacy_blocks.create_block_palette(
            block_sizes, grid_size=grid_size, grid_space=grid_space,
            lightness_bounds=lightness_bounds, chroma_bounds=chroma_bounds, hue_bounds=hue_bounds,
            colorblind_safe=colorblind_safe, cvd_type=cvd_type, cvd_severity=cvd_severity,
            theme_color_spacing=0.2 if theme_color_spacing is None else theme_color_spacing,
            as_hex=as_hex, optimize_palette=optimize_palette,
            optimize_palette_n_iter=optimize_palette_n_iter,
            optimize_palette_search_radius=optimize_palette_search_radius,
        )
    with warnings.catch_warnings():
        # Re-attribute deprecation warnings to the caller.
        warnings.simplefilter("ignore", DeprecationWarning)
        palette, _ = _build_block_palette_with_diagnostics(
            block_sizes, grid_size=grid_size, grid_space=grid_space,
            lightness_bounds=lightness_bounds, chroma_bounds=chroma_bounds, hue_bounds=hue_bounds,
            colorblind_safe=colorblind_safe, cvd_type=cvd_type, cvd_severity=cvd_severity,
            theme_step=theme_step, theme_color_spacing=theme_color_spacing, hue_sweep=hue_sweep,
            twist=twist, as_hex=as_hex, optimize_palette=optimize_palette,
            optimize_palette_n_iter=optimize_palette_n_iter,
            optimize_palette_search_radius=optimize_palette_search_radius,
        )
    if theme_color_spacing is not None:
        _resolve_step(theme_step, theme_color_spacing, "theme_color_spacing")
    return palette


def extend_block_palette(
    palette: List[Union[str, Tuple[float, float, float]]],
    new_block_sizes: List[int],
    *,
    grid_size: Union[int, Tuple[int, int, int]] = 64,  # type: ignore
    grid_space: Literal["RGB", "JCh"] = "RGB",
    lightness_bounds: Tuple[float, float] = (10, 90),
    chroma_bounds: Tuple[float, float] = (0, 80),
    hue_bounds: Tuple[float, float] = (0, 360),
    colorblind_safe: bool = False,
    cvd_type: Literal["protanomaly", "deuteranomaly", "tritanomaly"] = "deuteranomaly",
    cvd_severity: float = 50.0,
    theme_color_spacing: Optional[float] = None,
    as_hex: bool = True,
    optimize_palette: bool = True,
    optimize_palette_n_iter: int = 5,
    optimize_palette_search_radius: int = 2,
    theme_step: Optional[float] = None,
    hue_sweep: Union[Literal["auto"], float] = "auto",
    twist: Union[Literal["auto"], int] = "auto",
    theme_method: Literal["arc", "legacy"] = "arc",
) -> Union[List[str], List[Tuple[float, float, float]]]:
    """Extend an existing palette with new blocks.
    The existing colors are kept unchanged at the start of the palette, and new blocks are placed to be as distinct
    as possible from them and from each other, within the constraints on hue, chroma and lightness. Each new block
    is a range of colors on a smooth curve around a base colour, ordered from dark to light. This is most useful for
    two layer hierarchical categories, where the top layer categories each get their own block, and the finer
    grained sub-categories get their own colour within the block.

    Parameters
    ----------
    palette: List of hex colors or array of floats of shape (n_colors, 3), or a matplotlib palette name
        The palette to be extended. Its colours are fixed and never move.

    new_block_sizes: List of ints
        The sizes of the new blocks to generate, in output order.

    grid_size: int or triple of int (default 64)
        The resolution of the colour grid. Half of this resolution is used for the grid that guides curve
        fitting. A triple of ints gives separate side lengths for the grid's three axes.

    grid_space: "RGB" or "JCh" (default RGB)
        The color space to sample the grids from.

    lightness_bounds: (float, float) or None (default (10, 90))
        The upper and lower bounds of CIECAM02 lightness for new colors. If None, and the palette has at least
        four colors, the bounds are inferred from the existing palette.

    chroma_bounds: (float, float) or None (default (0, 80))
        The upper and lower bounds of CIECAM02 chroma for new colors, inferred as for lightness when None.

    hue_bounds: (float, float) or None (default (0, 360))
        The upper and lower bounds of hue values for new colors, inferred as for lightness when None.

    colorblind_safe: bool (default False)
        If True the palette will attempt to select colours in a way that should be more easily distinguishable
        for individuals with color vision deficiency. Block placement and block polishing then use the
        minimum of normal and CVD-simulated separations.

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
        The severity of colour vision deficiency to attempt to be robust to if ``colorblind_safe`` is True. The
        cvd_severity will be passed to colorspacious to simulate the appropriate colour vision deficiency. Per the
        colorspacious docs: Severity is any number between 0 (indicating regular vision) and 100 (indicating
        complete dichromacy).

    theme_color_spacing: float or None (default None)
        Deprecated; see ``theme_step``.

    as_hex: bool (default True)
        Whether to return the palette as hex-codes or RGB float triples.

    optimize_palette: bool (default True)
        If True, polish the palette at the block level: repeatedly take the two closest blocks and try sliding
        a block's colours along its curve, shrinking its step slightly, or (for short blocks) refitting it from
        a nudged base colour, keeping whichever move most increases the minimum distance between blocks. The
        order and smoothness of colours within each block are always preserved.

    optimize_palette_n_iter: int (default 5)
        The maximum number of block-polishing rounds.

    optimize_palette_search_radius: float (default 2)
        The maximum distance in CAM02-UCS space that a block may slide along its curve, or that a short
        block's base colour may be nudged, in one polishing move.

    theme_step: float or None (default None)
        Target CAM02-UCS distance between adjacent colours within a block; ``None`` means 10. A block of
        ``n`` colours aims for a perceptual length of ``(n - 1) * theme_step``. If a block cannot reach that
        length inside the colour bounds its step shrinks automatically, with a warning when it falls below 4.

    theme_color_spacing: float or None (default None)
        Deprecated; use ``theme_step``. If given, it is mapped to ``theme_step = 60 * theme_color_spacing``
        (so the old default 0.2 corresponds to a step of 12) and a ``DeprecationWarning`` is emitted.

    hue_sweep: "auto" or float (default "auto")
        How much each block may rotate in hue from its dark end to its light end. ``"auto"`` allocates a
        sweep per block from its size and the number of blocks, so larger blocks and palettes with fewer
        blocks rotate more. A float is the maximum total sweep in degrees for any block; ``0`` disables hue
        rotation. With ``colorblind_safe=True`` the sweep is capped at 15 degrees, since hue rotation
        costs separation under colour vision deficiency.

    twist: "auto", 1 or -1 (default "auto")
        The global direction in which the light ends of all blocks rotate: ``1`` toward increasing hue,
        ``-1`` toward decreasing hue. ``"auto"`` chooses the direction that sends the most light ends toward
        yellow. All blocks share one direction, which packs blocks more tightly.

    theme_method: "arc" or "legacy" (default "arc")
        ``"arc"`` builds each block as a smooth curve sampled evenly by perceptual arc length, with block
        base colours chosen knowing the block sizes. ``"legacy"`` runs the 0.4.0 algorithm unchanged, for
        reproducing old palettes; it emits a ``FutureWarning`` and will be removed in a future release.

    Returns
    -------
    palette:
        The existing palette followed by the new blocks.
    """
    _validate_block_options(grid_space, hue_sweep, twist, theme_method)
    if theme_method == "legacy":
        _legacy_warning()
        return _legacy_blocks.extend_block_palette(
            palette, new_block_sizes, grid_size=grid_size, grid_space=grid_space,
            lightness_bounds=lightness_bounds, chroma_bounds=chroma_bounds, hue_bounds=hue_bounds,
            colorblind_safe=colorblind_safe, cvd_type=cvd_type, cvd_severity=cvd_severity,
            theme_color_spacing=0.2 if theme_color_spacing is None else theme_color_spacing,
            as_hex=as_hex, optimize_palette=optimize_palette,
            optimize_palette_n_iter=optimize_palette_n_iter,
            optimize_palette_search_radius=optimize_palette_search_radius,
        )
    try:
        palette = palette_to_sRGB1(palette, max_colors=20)
    except Exception:
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

    cam_palette = cspace_convert(palette, "sRGB1", "CAM02-UCS")
    existing = get_rgb_palette(cam_palette, as_hex=as_hex)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        new_blocks, _ = _build_block_palette_with_diagnostics(
            new_block_sizes, grid_size=grid_size, grid_space=grid_space,
            lightness_bounds=lightness_bounds, chroma_bounds=chroma_bounds, hue_bounds=hue_bounds,
            colorblind_safe=colorblind_safe, cvd_type=cvd_type, cvd_severity=cvd_severity,
            theme_step=theme_step, theme_color_spacing=theme_color_spacing, hue_sweep=hue_sweep,
            twist=twist, as_hex=as_hex, optimize_palette=optimize_palette,
            optimize_palette_n_iter=optimize_palette_n_iter,
            optimize_palette_search_radius=optimize_palette_search_radius,
            fixed_colors=cam_palette,
        )
    if theme_color_spacing is not None:
        _resolve_step(theme_step, theme_color_spacing, "theme_color_spacing")
    return list(existing) + list(new_blocks)
