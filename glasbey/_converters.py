# MIT License
# Leland McInnes

from numpy import clip, max, linspace, array, asarray, float32, integer
from colorspacious import cspace_convert
from matplotlib.colors import rgb2hex, to_rgb
try:
    from matplotlib.cm import get_cmap
except ImportError:
    from matplotlib.pyplot import get_cmap


def get_rgb_palette(cam02ucs_palette, as_hex: bool=True):
    """Given a CAM02-UCS palette, as generated by the internal routines for Glasbey palette generation, return
    a standard RGB palette suitable for use in most plotting libraries.

    Parameters
    ----------
    cam02ucs_palette: array of float
        The palette to be converted for plotting use.

    as_hex: bool (default True)
        Whether to return the palette as a list of hex strings or an array of triples of RGB channel intensities.

    Returns
    -------
    palette: List of str or array
        The converted palette for use in plotting.
    """
    raw_rgb_palette = cspace_convert(cam02ucs_palette, "CAM02-UCS", "sRGB1")
    rgb_palette = clip(raw_rgb_palette, 0.0, 1.0)

    if as_hex:
        return [rgb2hex(color) for color in rgb_palette]
    else:
        return rgb_palette.tolist()


def palette_to_sRGB1(palette, max_colors=12):
    """Given a palette specified in some manner, return a standard sRGB1 palette for use in internal routines
    of this library.

    Parameters
    ----------
    palette:
        A palette specified in any of a variety of formats.

    max_colors: int (default 12)
        If specifying a palette by matplotlib palette name, the maximum number of colors to use for the palette.
        This is particularly relevant for large listed palettes like Viridis, or other continuous palettes.

    Returns
    -------
    palette: array of float
        An array of float triples between 0 and 1 specifying RGB channel intensities.
    """
    if type(palette) is str:
        if palette.startswith("#"):
            return array([to_rgb(palette)], dtype=float32, order="C")
        else:
            try:
                cmap = get_cmap(palette)
                if hasattr(cmap, "colors") and len(cmap.colors) <= max_colors:
                    result = array([color[:3] for color in cmap.colors], dtype=float32, order="C")
                else:
                    result = cmap(linspace(0, 1, max_colors))[:, :3].astype(float32, order="C")
                return result
            except:
                raise ValueError(f"Unrecognized palette name {palette}")
    else:
        result = []
        for color in palette:
            if type(color) is str:
                result.append(to_rgb(color))
            elif hasattr(color, '__len__') and len(color) == 3 or len(color) == 4:
                if any([isinstance(channel, (int, integer)) for channel in color]) or max(color) > 1.0:
                    color = asarray(color, dtype=float32) / 255

                result.append(to_rgb(color))

        return array(result, dtype=float32, order="C")

