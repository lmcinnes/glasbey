import numpy as np

from colorspacious import cspace_convert
from matplotlib.colors import rgb2hex, to_rgb, LinearSegmentedColormap

from ._grids import rgb_grid, jch_grid, constrain_by_lightness_chroma_hue
from ._internals import generate_palette_cam02ucs, generate_next_color_cam02ucs
from ._converters import get_rgb_palette, palette_to_sRGB1

from typing import *


def create_palette(
    palette_size: int = 256,
    *,
    grid_size: int | Tuple[int, int, int] = 64,
    as_hex: bool = True,
    grid_space: Literal["RGB", "JCh"] = "RGB",
    lightness_bounds: Tuple[float, float] = (10, 90),
    chroma_bounds: Tuple[float, float] = (10, 90),
    hue_bounds: Tuple[float, float] = (0, 360),
    red_bounds: Tuple[float, float] = (0, 1),
    green_bounds: Tuple[float, float] = (0, 1),
    blue_bounds: Tuple[float, float] = (0, 1),
):
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
            f'Parameter grid_space should be on of "JCh" or "RGB" not {grid_space}'
        )

    initial_palette = cspace_convert(
        np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]), "sRGB1", "CAM02-UCS"
    ).astype(np.float32, order="C")

    palette = generate_palette_cam02ucs(
        colors, initial_palette, np.uint32(palette_size + 2)
    )
    palette = get_rgb_palette(palette, as_hex=as_hex)[2:]

    return palette


def extend_palette(
    palette,
    palette_size: int = 256,
    *,
    grid_size: int | Tuple[int, int, int] = 64,
    as_hex: bool = True,
    grid_space: Literal["RGB", "JCh"] = "RGB",
    lightness_bounds: Optional[Tuple[float, float]] = None,
    chroma_bounds: Optional[Tuple[float, float]] = None,
    hue_bounds: Optional[Tuple[float, float]] = None,
    red_bounds: Tuple[float, float] = (0, 1),
    green_bounds: Tuple[float, float] = (0, 1),
    blue_bounds: Tuple[float, float] = (0, 1),
):
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
            f'Parameter grid_space should be on of "JCh" or "RGB" not {grid_space}'
        )

    palette = cspace_convert(palette, "sRGB1", "CAM02-UCS").astype(
        np.float32, order="C"
    )

    palette = generate_palette_cam02ucs(colors, palette, np.uint32(palette_size))
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
):
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
        return result.to_list()


def create_block_palette(
    block_sizes: List[int],
    *,
    sort_block_sizes: bool = True,
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
    as_hex: bool = True,
):
    if sort_block_sizes:
        block_order = np.argsort(block_sizes)[::-1]
        block_sizes_for_generation = np.asarray(block_sizes)[block_order]
    else:
        block_sizes_for_generation = block_sizes

    palette = []
    initial_color = create_palette(
        1,
        lightness_bounds=(
            generated_color_lightness_bounds[0],
            generated_color_lightness_bounds[1],
        ),
        chroma_bounds=(
            generated_color_chroma_bounds[0],
            generated_color_chroma_bounds[1],
        ),
        hue_bounds=(0, 360),
    )[0]
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
    )
    palette.extend(block)

    colors = rgb_grid(
        grid_size=64,
        output_colorspace="JCh",
    )
    colors = constrain_by_lightness_chroma_hue(
        colors,
        "JCh",
        lightness_bounds=generated_color_lightness_bounds,
        chroma_bounds=generated_color_chroma_bounds,
        hue_bounds=(0, 360),
    )
    distances = np.full(colors.shape[0], 1e9, dtype=np.float32, order="C")

    for block_size in block_sizes_for_generation[1:]:
        block_cam02ucs = cspace_convert(
            np.asarray([to_rgb(color) for color in block]), "sRGB1", "CAM02-UCS"
        ).astype(np.float32)
        next_color = generate_next_color_cam02ucs(colors, distances, block_cam02ucs)
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
        )
        palette.extend(block)

    if sort_block_sizes:
        block_start_indices = np.hstack(
            ([0], np.cumsum(block_sizes_for_generation)[:-1])
        )
        result = []

        for i in np.argsort(block_order):
            size = block_sizes_for_generation[i]
            block = palette[block_start_indices[i] : block_start_indices[i] + size]
            result.extend(block)
    else:
        result = palette

    if as_hex:
        return [rgb2hex(color) for color in result]
    else:
        return result.to_list()
