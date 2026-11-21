"""Compositing shared by the dense and sparse smoke ray marchers."""

import numpy as np
import taichi as ti


REFRACTIVE_MAX_STEPS = 10000
INTERFACE_ENTRY_OFFSET = .004


@ti.func
def intersect_box(origin, direction, lower, upper, start, end):
    for axis in ti.static(range(3)):
        if direction[axis] == 0:
            if origin[axis] < lower[axis] or origin[axis] >= upper[axis]:
                end = -1.0
        else:
            lo = (lower[axis] - origin[axis]) / direction[axis]
            hi = (upper[axis] - origin[axis]) / direction[axis]
            start = ti.max(start, ti.min(lo, hi))
            end = ti.min(end, ti.max(lo, hi))
    return start, end


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


def light_cache_channels(colors):
    colors = np.asarray(colors)
    return 3 if np.any(colors != colors[:, :1]) else 1


@ti.func
def composite_sample(color, transmittance, sample_color, light, opacity):
    transmittance *= 1.0 - opacity
    color += opacity * sample_color * light * transmittance
    return color, transmittance


@ti.func
def refract_from_gradient(direction, index, gradient, step):
    # Apply the dense renderer's centered-gradient Snell step.
    change = direction.dot(gradient) * step
    stop = False
    if change != 0.:
        normal = gradient.normalized()
        normal_component = direction.dot(normal)
        tangent = direction - normal_component * normal
        ratio = (index + .5 * change) / (index - .5 * change)
        sin_new = tangent.norm() / ratio
        if sin_new > 1.:
            stop = True
        else:
            cos_new = ti.sqrt(1. - sin_new * sin_new)
            sign = 1. if normal_component > 0. else -1.
            direction = (tangent / ratio + sign * cos_new * normal).normalized()
    return direction, stop


@ti.func
def refract_at_interface(direction, normal, before, after):
    # Face the incident medium and return the reflected direction for TIR.
    if direction.dot(normal) > 0.:
        normal = -normal
    cosine = ti.min(1., ti.max(0., -direction.dot(normal)))
    ratio = before / after
    discriminant = 1. - ratio*ratio*(1. - cosine*cosine)
    mirror = direction + 2.*cosine*normal
    bent = mirror
    reflectance = 1.
    if discriminant >= 0.:
        bent = ratio*direction + (ratio*cosine - ti.sqrt(discriminant))*normal
        r0 = ((before-after)/(before+after))**2
        reflectance = r0 + (1.-r0)*(1.-cosine)**5
    return bent.normalized(), reflectance, discriminant < 0., mirror


@ti.func
def crosses_refractive_interface(before, after, ambient, occupied, interfaces):
    # A non-refractive volume receives the ray directly.
    return before != after and (after != ambient or not occupied) and interfaces < 19


@ti.func
def follow_refractive_interface(
        pos, direction, before, after, step, voxel_size, normal, sample_ior: ti.template()):
    if normal.norm_sqr() == 0.:
        point = pos - direction * (.5 * step)
        gradient = ti.Vector.zero(ti.f32, 3)
        for sample in range(6):
            axis = sample // 2
            wide = sample % 2 != 0
            radius = 2.576 if wide else 1.12
            weight = .58 if wide else 1.
            offset = ti.Vector.zero(ti.f32, 3)
            offset[axis] = radius * voxel_size
            gradient[axis] += weight * (sample_ior(point + offset) - sample_ior(point - offset))
        normal = -direction
        if gradient.norm_sqr() > 0.:
            normal = gradient.normalized()
    bent, fresnel, reflected, mirror = refract_at_interface(direction, normal, before, after)
    if reflected:
        pos -= direction * step
        pos += bent * (.35 * step)
    return pos, bent, fresnel, reflected, mirror
