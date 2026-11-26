"""Trace directional-light caustics for the sparse radiance cache."""

from itertools import product
import logging
import math

import numpy as np
import taichi as ti

from ._smoke import (INTERFACE_ENTRY_OFFSET, REFRACTIVE_MAX_STEPS,
                     crosses_refractive_interface, directional_light_parameters,
                     follow_refractive_interface, intersect_box)
from .sparse_smoke import SparseSmokeScene, _upload_lookup_field


DEFAULT_PHOTON_DENSITY = 6


@ti.kernel
def _trace(scene: ti.template(), origins: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
           direction: ti.math.vec3, hits: ti.types.ndarray(dtype=ti.math.ivec2, ndim=1),
           weights: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    for photon in origins:
        hits[photon] = ti.Vector([-1, -1])
        ray = direction
        pos = origins[photon]
        lower, upper = scene._bake_lower[0], scene._bake_upper[0]
        entry, end = intersect_box(pos, ray, lower, upper, 0., 1e30)
        if entry < end:
            pos += ray * (entry + INTERFACE_ENTRY_OFFSET)
            ambient = scene._ior[0]
            previous = ambient
            previous_code, interfaces = 0, 0
            power = 1.
            refracted = False
            for _ in range(REFRACTIVE_MAX_STEPS):
                if not ((pos >= lower).all() and (pos < upper).all()):
                    break
                distance, brick, voxel, _coords = scene._lookup(pos, ray, 0., 1e30, 0)
                code = 0
                if brick >= 0:
                    code = scene._block_code(brick, voxel)
                medium = scene._ior[code]
                occupied = code != 0 and scene._materials[code].w > 0.
                if occupied and medium == ambient:
                    if refracted:
                        hits[photon] = ti.Vector([brick, voxel])
                        weights[photon] = power
                    break
                if crosses_refractive_interface(previous, medium, ambient, occupied, interfaces):
                    normal = ti.Vector.zero(ti.f32, 3)
                    if ti.static(scene._use_interface_normals):
                        if medium != ambient:
                            normal = scene._interface_normals[code]
                        if normal.norm_sqr() == 0. and previous != ambient:
                            normal = scene._interface_normals[previous_code]
                    pos, ray, fresnel, reflected, _mirror = follow_refractive_interface(
                        pos, ray, previous, medium, scene._step, 1., normal, scene._ior_at,
                    )
                    interfaces += 1
                    if reflected:
                        continue
                    power *= 1. - fresnel
                    refracted = True
                    distance = 0.
                previous, previous_code = medium, code
                pos += ray * ti.max(scene._step, distance + .003)


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


def _gather_hits(data, materials, ior, normals, direction, photon_density, step, seed):
    entity, = data.entities
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
    samples = start + _stratified_samples(count, seed) * (end - start)
    origins = (samples @ np.stack((u, v)) + plane * direction).astype(np.float32)
    hits = np.empty((count, 2), dtype=np.int32)
    weights = np.zeros(count, dtype=np.float32)

    # Scope the temporary GPU lookup buffers to the bake.
    builder = ti.FieldsBuilder()
    uploads = []

    def upload(values, components=None):
        field, array = _upload_lookup_field(values, components, builder)
        uploads.append((field, array))
        return field

    scene = SparseSmokeScene.__new__(SparseSmokeScene)
    scene._init_lookup(data, upload)
    scene._interface = True
    scene._step = step
    scene._use_interface_normals = normals is not None
    scene._bake_lower = upload(lower[None], 3)
    scene._bake_upper = upload(upper[None], 3)
    # Use the same two 16-bit codes per word as the renderer.
    scene._blocks = upload(np.ascontiguousarray(data.blocks).view(np.int32))
    scene._materials = upload(materials, 4)
    scene._ior = upload(ior)
    if normals is not None:
        scene._interface_normals = upload(normals, 3)
    tree = builder.finalize()
    try:
        for field, array in uploads:
            field.from_numpy(array)
        _trace(scene, origins, direction, hits, weights)
        ti.sync()
    finally:
        tree.destroy()
    valid = hits[:, 0] >= 0
    return hits[valid], weights[valid], area / count


def _runtime_caustics(data, materials, ior, normals, light, photon_density):
    if not math.isfinite(photon_density) or photon_density <= 0:
        raise ValueError("photon_density must be finite and positive")
    if ior.shape != (65536,) or not np.all(np.isfinite(ior) & (ior > 0)):
        raise ValueError("Expected a positive finite index of refraction for each material code")
    if normals is not None:
        if normals.shape != (65536, 3) or not np.all(np.isfinite(normals)):
            raise ValueError("Expected one finite interface normal per material code")
        lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
        normals = (normals / np.where(lengths > 0, lengths, 1.)[:, None]).astype(np.float32)
    toward_light, _ = directional_light_parameters(light)
    hits, weights, area = _gather_hits(data, materials, ior, normals, -toward_light, photon_density, .5, 0)

    volumes = np.ones(len(data.blocks), dtype=np.float64)
    for shift, _x, _y, _z, row in data.uniform_bricks:
        volumes[row] = 2. ** (3 * shift)
    hits[volumes[hits[:, 0]] > 1, 1] = 0
    voxels, grouping = np.unique(hits.astype(np.int64) @ [512, 1], return_inverse=True)
    bricks, slots = np.divmod(voxels, 512)
    power = np.bincount(grouping, weights=weights) * (area / np.pi) / volumes[bricks]
    return bricks, slots, power
