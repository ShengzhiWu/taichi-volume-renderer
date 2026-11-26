"""Trace directional-light caustics for the sparse radiance cache."""

from itertools import product
import logging
import math

import numpy as np
import taichi as ti

from ._smoke import (INTERFACE_ENTRY_OFFSET, REFRACTIVE_MAX_STEPS,
                     crosses_refractive_interface,
                     follow_refractive_interface, intersect_box)


DEFAULT_PHOTON_DENSITY = 6


@ti.kernel
def _trace(scene: ti.template(), origins: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
           direction: ti.math.vec3, lower: ti.math.vec3, upper: ti.math.vec3,
           dispersion: ti.f32,
           hits: ti.types.ndarray(dtype=ti.math.ivec2, ndim=1),
           weights: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    for photon in origins:
        hits[photon] = ti.Vector([-1, -1])
        ray = direction
        pos = origins[photon]
        entry, end = intersect_box(pos, ray, lower, upper, 0., 1e30)
        if entry < end:
            pos += ray * (entry + INTERFACE_ENTRY_OFFSET)
            ambient = scene._ior[0]
            ambient += dispersion * (ambient - 1.)
            previous = ambient
            previous_code, interfaces = 0, 0
            power = 1.
            refracted = False
            for _ in range(REFRACTIVE_MAX_STEPS):
                if not ((pos >= lower).all() and (pos < upper).all()):
                    break
                distance, brick, voxel, _coords = scene._lookup(pos, ray, 0., 1e30, 0, exact_voxel=True)
                code = 0
                if brick >= 0:
                    code = scene._block_code(brick, voxel)
                medium = scene._ior[code]
                medium += dispersion * (medium - 1.)
                occupied = code != 0 and scene._materials[code].w > 0.
                if occupied and medium == ambient:
                    if refracted:
                        hits[photon] = ti.Vector([brick, voxel])
                        weights[photon] = power
                    break
                if crosses_refractive_interface(previous, medium, ambient, occupied, interfaces):
                    normal = ti.Vector.zero(ti.f32, 3)
                    if ti.static(scene._has_interface_normals):
                        if medium != ambient:
                            normal = scene._interface_normals[code]
                        if normal.norm_sqr() == 0. and previous != ambient:
                            normal = scene._interface_normals[previous_code]
                    pos, ray, fresnel, reflected, _mirror = follow_refractive_interface(
                        pos, ray, previous, medium, .5, 1., normal, scene._ior_at,
                    )
                    interfaces += 1
                    if reflected:
                        continue
                    power *= 1. - fresnel
                    refracted = True
                    distance = 0.
                previous, previous_code = medium, code
                pos += ray * ti.max(.5, distance + .003)


def _stratified_samples(count, seed):
    # Make equal-area cells even when the requested count is not a square.
    rows = max(1, int(np.sqrt(count)))
    per_row, extra = divmod(count, rows)
    sizes = np.full(rows, per_row)
    sizes[:extra] += 1
    starts = np.cumsum(sizes) - sizes
    row_start = np.repeat(starts, sizes)
    row_size = np.repeat(sizes, sizes)
    samples = np.random.default_rng(seed).random((count, 2))
    samples[:, 0] = (np.arange(count) - row_start + samples[:, 0]) / row_size
    samples[:, 1] = (row_start + samples[:, 1] * row_size) / count
    return samples


def _gather_hits(scene, entity, direction, photon_density, dispersion):
    corners = np.array(list(product(*zip(entity.lower, entity.upper))))
    corners = corners @ entity.rotation.T + entity.position
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    if not np.all(upper > lower):
        raise ValueError("Caustic baking requires nonempty scene bounds")
    axis = np.eye(3)[np.abs(direction).argmin()]
    u = np.cross(direction, axis)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    projection = corners @ np.stack((u, v)).T
    start, end = projection.min(axis=0), projection.max(axis=0)
    area = float(np.prod(end - start))
    estimate = area * photon_density
    if not math.isfinite(estimate) or estimate > np.iinfo(np.int32).max:
        raise ValueError("Projected caustic photon count exceeds the supported range")
    count = max(1, math.ceil(estimate))
    logging.getLogger(__name__).info("Tracing %d photons across %.2f voxel²", count, area)
    plane = (corners @ direction).min() - 1.
    samples = start + _stratified_samples(count, 0) * (end - start)
    origins = (samples @ np.stack((u, v)) + plane * direction).astype(np.float32)
    hits = np.empty((count, 2), dtype=np.int32)
    weights = np.zeros(count, dtype=np.float32)

    _trace(scene, origins, direction, lower, upper, dispersion, hits, weights)
    valid = hits[:, 0] >= 0
    return hits[valid], weights[valid], area / count


def _caustic_power(scene, entity, uniform_bricks, direction, photon_density, dispersion):
    hits, weights, area = _gather_hits(scene, entity, direction, photon_density, dispersion)
    volumes = np.ones(scene.brick_count, dtype=np.float64)
    for shift, _x, _y, _z, row in uniform_bricks:
        volumes[row] = 2. ** (3 * shift)
    hits[volumes[hits[:, 0]] > 1, 1] = 0
    voxels, grouping = np.unique(hits.astype(np.int64) @ [512, 1], return_inverse=True)
    bricks, slots = np.divmod(voxels, 512)
    power = np.bincount(grouping, weights=weights) * (area / np.pi) / volumes[bricks]
    return np.column_stack((bricks, slots)).astype(np.int32), power.astype(np.float32)
