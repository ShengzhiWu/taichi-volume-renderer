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
    bent = direction + 2.*cosine*normal
    reflectance = 1.
    if discriminant >= 0.:
        bent = ratio*direction + (ratio*cosine - ti.sqrt(discriminant))*normal
        r0 = ((before-after)/(before+after))**2
        reflectance = r0 + (1.-r0)*(1.-cosine)**5
    return bent.normalized(), reflectance, discriminant < 0.


@ti.func
def follow_refractive_interface(
        incident_pos, transmitted_pos, direction, before, after, voxel_size, sample_ior: ti.template()):
    # Bracket the change in medium and keep each point on its own side.
    lo, hi = incident_pos, transmitted_pos
    for iteration in range(8):
        middle = (lo + hi) * .5
        if sample_ior(middle) == before:
            lo = middle
        else:
            hi = middle
    point = (lo + hi) * .5
    gradient = ti.Vector.zero(ti.f32, 3)
    for axis in range(3):
        offset = ti.Vector.zero(ti.f32, 3)
        offset[axis] = voxel_size
        gradient[axis] = sample_ior(point + offset) - sample_ior(point - offset)
    normal = -direction
    if gradient.norm_sqr() > 0.:
        normal = gradient.normalized()
    bent, fresnel, reflected = refract_at_interface(direction, normal, before, after)
    # The smoothed normal can send the ray back across the discrete voxel face
    # on the wrong side. Verify the outgoing side before restarting traversal.
    distance = voxel_size / 64.
    anchor = lo if reflected else hi
    expected = before if reflected else after
    next_pos = anchor + bent * distance
    if sample_ior(next_pos) != expected:
        face = ti.Vector.zero(ti.f32, 3)
        for axis in range(3):
            across = lo
            across[axis] = hi[axis]
            if sample_ior(across) != before and ti.abs(direction[axis]) > ti.abs(direction.dot(face)):
                face = ti.Vector.zero(ti.f32, 3)
                face[axis] = 1.
        if face.norm_sqr() > 0.:
            bent, fresnel, reflected = refract_at_interface(direction, face, before, after)
            anchor = lo if reflected else hi
            expected = before if reflected else after
        next_pos = anchor + bent * distance
        if sample_ior(next_pos) != expected:
            # At a corner there may be no single separating axis. Continue
            # along the verified incident segment on the required side.
            bent = -direction if reflected else direction
            next_pos = anchor + bent * distance
            if sample_ior(next_pos) != expected:
                next_pos = anchor
    return next_pos, bent, fresnel, reflected
