from numpy import clip, max, array, asarray, float32, integer
from colorspacious import cspace_convert
from matplotlib.colors import rgb2hex, to_rgb


def get_rgb_palette(cam02ucs_palette, as_hex: bool=True):
    raw_rgb_palette = cspace_convert(cam02ucs_palette, "CAM02-UCS", "sRGB1")
    rgb_palette = clip(raw_rgb_palette, 0.0, 1.0)

    if as_hex:
        return [rgb2hex(color) for color in rgb_palette]
    else:
        return rgb_palette.tolist()


def palette_to_sRGB1(palette):
    result = []
    for color in palette:
        if type(color) is str:
            result.append(to_rgb(color))
        elif hasattr(color, '__len__') and len(color) == 3 or len(color) == 4:
            if any([isinstance(channel, (int, integer)) for channel in color]) or max(color) > 1.0:
                color = asarray(color, dtype=float32) / 255

            result.append(to_rgb(color))

    return array(result, dtype=float32, order="C")
