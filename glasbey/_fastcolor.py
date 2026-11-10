"""Vectorised closed-form CAM02-UCS -> sRGB1 conversion.

This reproduces colorspacious's CAM02-UCS -> JMh -> CIECAM02 -> XYZ100 -> sRGB1
path (sRGB viewing conditions) in plain NumPy, without per-call validation, so
domain margins can be evaluated cheaply inside solvers. Colours with no
CIECAM02 inverse yield NaN in the affected rows rather than failing the batch.
"""

import numpy as np
from colorspacious import CIECAM02Space
from colorspacious.basics import XYZ100_to_sRGB1_matrix
from colorspacious.ciecam02 import M_CAT02_inv, M_CAT02_M_HPE_inv

_VC = CIECAM02Space.sRGB
_C1, _C2 = 0.007, 0.0228  # CAM02-UCS (KL = 1)
_FL4 = _VC.F_L**0.25
_T_SCALE = np.sqrt(1 / 100) * (1.64 - 0.29**_VC.n) ** 0.73
_P1 = (50000.0 / 13) * _VC.N_c * _VC.N_cb
_P3 = 21.0 / 20
_D2 = (2 + _P3) * (220.0 / 1403)
_D3 = (-27.0 / 1403) + _P3 * (6300.0 / 1403)
_NUM_SCALE = (2 + _P3) * (460.0 / 1403)
_RGBA = np.asarray([[460, 451, 288], [460, -891, -261], [460, -220, -6300]], dtype=float) / 1403
# Steps 6-8 and XYZ -> linear sRGB folded into one matrix (D_RGB divides per channel).
_TO_LINEAR = XYZ100_to_sRGB1_matrix @ M_CAT02_inv @ np.diag(1 / _VC.D_RGB) @ M_CAT02_M_HPE_inv / 100


def cam02ucs_to_jmh(colors):
    colors = np.asarray(colors, dtype=float)
    jp, ap, bp = colors[..., 0], colors[..., 1], colors[..., 2]
    lightness = -jp / (_C1 * jp - 100 * _C1 - 1)
    mp = np.hypot(ap, bp)
    colorfulness = np.expm1(_C2 * mp) / _C2
    return lightness, colorfulness, ap, bp, mp


def cam02ucs_to_srgb1(colors):
    """Convert CAM02-UCS colours of shape (..., 3) to (unclipped) sRGB1."""
    colors = np.asarray(colors, dtype=float)
    shape = colors.shape
    flat = colors.reshape(-1, 3)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        lightness, colorfulness, ap, bp, mp = cam02ucs_to_jmh(flat)
        safe = mp > 0
        cos_h = np.where(safe, ap / np.where(safe, mp, 1), 1.0)
        sin_h = np.where(safe, bp / np.where(safe, mp, 1), 0.0)
        chroma = colorfulness / _FL4
        t = (chroma / (np.sqrt(lightness) * _T_SCALE)) ** (1 / 0.9)
        t = np.where(chroma > 0, t, 0.0)
        # cos(h + 2) without recovering the angle.
        e_t = 0.25 * (cos_h * np.cos(2.0) - sin_h * np.sin(2.0) + 3.8)
        achromatic = _VC.A_w * (lightness / 100) ** (1 / (_VC.c * _VC.z))
        p2 = achromatic / _VC.N_bb + 0.305
        numerator = p2 * _NUM_SCALE * t
        denominator = _P1 * e_t + t * (_D2 * cos_h + _D3 * sin_h)
        a = numerator * cos_h / denominator
        b = numerator * sin_h / denominator
        rgb_a = np.column_stack([p2, a, b]) @ _RGBA.T
        offset = rgb_a - 0.1
        magnitude = np.abs(offset)
        rgb_prime = np.sign(offset) * (100 / _VC.F_L) * (
            (27.13 * magnitude) / (400 - magnitude)
        ) ** (1 / 0.42)
        linear = rgb_prime @ _TO_LINEAR.T
        small = linear <= 0.0031308
        result = np.where(small, linear * 12.92, 1.055 * np.abs(linear) ** (1 / 2.4) - 0.055)
    return result.reshape(shape)
