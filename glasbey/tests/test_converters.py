import pytest
import numpy as np

from glasbey._converters import get_rgb_palette, palette_to_sRGB1
from matplotlib.cm import get_cmap
from matplotlib.colors import to_rgb
from colorspacious import cspace_convert


TAB10 = get_cmap("tab10", 10).colors[:, :3]
SET1 = get_cmap("Set1", 6).colors[:, :3]
ACCENT = get_cmap("Accent", 6).colors[:, :3]


@pytest.mark.parametrize("palette", ["tab10", "Set1", "Accent"])
def test_palette_conversion(palette):
    pal = get_cmap(palette, 6).colors[:, :3]
    cam_pal = cspace_convert(pal, "sRGB1", "CAM02-UCS")

    hex_pal = get_rgb_palette(cam_pal, as_hex=True)
    converted_pal = np.asarray([to_rgb(color) for color in hex_pal])
    assert np.allclose(pal, converted_pal)
