"""Compositing shared by the dense and sparse smoke ray marchers."""

import numpy as np
import taichi as ti


def validate_gamma(gamma):
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")


def apply_rgb_gamma(rgb, gamma):
    validate_gamma(gamma)
    if gamma == 1.0:
        return rgb
    result = np.maximum(rgb, 0.0, dtype=np.float32)
    with np.errstate(over="raise", invalid="raise"):
        try:
            np.power(result, gamma, out=result)
        except FloatingPointError as exc:
            raise ValueError("gamma produced an out-of-range reflectance") from exc
    return result


@ti.func
def composite_sample(color, transmittance, sample_color, light, opacity):
    transmittance *= 1.0 - opacity
    color += opacity * sample_color * light * transmittance
    return color, transmittance
