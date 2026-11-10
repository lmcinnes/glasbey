import numpy as np
from colorspacious import cspace_convert

from glasbey._fastcolor import cam02ucs_to_srgb1


def test_fast_conversion_matches_colorspacious():
    rgb = np.random.default_rng(0).random((5000, 3))
    cam = cspace_convert(rgb, "sRGB1", "CAM02-UCS")
    np.testing.assert_allclose(cam02ucs_to_srgb1(cam), rgb, rtol=0, atol=1e-6)
    expected = cspace_convert(cam[:50], "CAM02-UCS", "sRGB1")
    np.testing.assert_allclose(cam02ucs_to_srgb1(cam[:50]), expected, rtol=0, atol=1e-9)


def test_fast_conversion_handles_neutrals_and_bad_rows():
    colors = np.array([[50.0, 0.0, 0.0], [0.0, 0.0, 0.0], [60.0, 1e4, 0.0]])
    result = cam02ucs_to_srgb1(colors)
    np.testing.assert_allclose(
        result[0], cspace_convert(colors[0], "CAM02-UCS", "sRGB1"), atol=1e-9
    )
    assert np.isfinite(result[:2]).all()
