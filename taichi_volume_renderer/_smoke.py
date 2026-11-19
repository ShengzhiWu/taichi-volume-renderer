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


def directional_light_parameters(light):
    direction, intensity = light
    direction = np.asarray(direction, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float32)
    length = np.linalg.norm(direction)
    if (direction.shape != (3,) or not np.isfinite(length) or length == 0
            or intensity.shape != (3,) or not np.all(np.isfinite(intensity)) or np.any(intensity < 0)):
        raise ValueError("Directional light needs a finite nonzero direction and nonnegative RGB intensity")
    return (direction / length).astype(np.float32), intensity


def ambient_light_parameters(light):
    ambient = np.asarray(light, dtype=np.float32)
    if ambient.shape != (3,) or not np.all(np.isfinite(ambient)) or np.any(ambient < 0):
        raise ValueError("ambient_light must be a finite, nonnegative RGB triple")
    return ambient


@ti.func
def composite_sample(color, transmittance, sample_color, light, opacity):
    transmittance *= 1.0 - opacity
    color += opacity * sample_color * light * transmittance
    return color, transmittance
