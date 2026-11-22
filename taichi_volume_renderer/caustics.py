"""Bake directional-light caustics into indexed sparse scene materials."""

from dataclasses import replace
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


def bake_caustics(data, materials, index_of_refraction, directional_light, *,
                  interface_normals=None, emission=None, photon_density=DEFAULT_PHOTON_DENSITY,
                  step_length=0.5, seed=0, material_clusters=None):
    """Return renderer-ready data, materials, IOR, normals and emission after baking.

    Initialize Taichi first. The scene must have one indexed entity. Each photon
    deposits at its first non-refractive occupied voxel after refraction.
    Compact uniform bricks can refract light but cannot receive this bake.
    The photon count is derived from the projected area and photon_density,
    which defaults to six photons per square voxel.
    Set material_clusters to cluster baked RGB reflectance and emission.
    """
    if (len(data.entities) != 1 or data.blocks.dtype != np.uint16
            or data.blocks.ndim != 2 or data.blocks.shape[1] != 512 or not len(data.blocks)):
        raise ValueError("Caustic baking requires a single indexed sparse entity")
    if not math.isfinite(photon_density) or photon_density <= 0:
        raise ValueError("photon_density must be finite and positive")
    if not np.isfinite(step_length) or step_length <= 0:
        raise ValueError("Caustic baking requires a finite positive step length")
    if material_clusters is not None and (
        isinstance(material_clusters, (bool, np.bool_))
        or not isinstance(material_clusters, (int, np.integer)) or not 1 <= material_clusters <= 65536
    ):
        raise ValueError("material_clusters must be an integer between 1 and 65536")
    toward_light, intensity = directional_light_parameters(directional_light)
    materials = _material_table(materials, (65536, 4))
    ior = _material_table(index_of_refraction, (65536,))
    if np.any(ior <= 0):
        raise ValueError("Caustic baking requires positive refractive indices")
    normals = None
    if interface_normals is not None:
        normals = _material_table(interface_normals, (65536, 3), nonnegative=False)
        lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
        normals = (normals / np.where(lengths > 0, lengths, 1.)[:, None]).astype(np.float32)
    emission = (np.zeros((65536, 3), dtype=np.float32) if emission is None
                else _material_table(emission, (65536, 3)))
    hits, weights, area = _gather_hits(
        data, materials, ior, normals, -toward_light, photon_density, float(step_length), seed,
    )
    if np.any(hits[:, 0] >= len(data.blocks) - len(data.uniform_bricks)):
        raise ValueError("Caustic receivers must use ordinary sparse bricks")

    voxels, grouping = np.unique(hits.astype(np.int64) @ [512, 1], return_inverse=True)
    bricks, slots = np.divmod(voxels, 512)
    codes = data.blocks[bricks, slots]
    power = np.bincount(grouping, weights=weights) * (area / np.pi)
    rgb = (emission[codes] + power[:, None] * intensity * materials[codes, :3]).astype(np.float32)
    if not np.isfinite(rgb).all():
        raise ValueError("Caustic emission exceeds the float32 range")
    changed = np.any(rgb != emission[codes], axis=1)
    if not changed.any():
        return data, materials, ior, normals, emission

    sources = codes[changed]
    baked_materials = materials[sources]
    baked_emission = rgb[changed]
    if material_clusters is not None and len(sources) > material_clusters:
        points = np.column_stack((baked_materials[:, :3], baked_emission))
        representatives = _cluster_materials(points, int(material_clusters))
        baked_materials[:, :3] = representatives[:, :3]
        baked_emission = representatives[:, 3:]
    columns = [baked_materials, baked_emission, ior[sources]]
    if normals is not None:
        columns.append(normals[sources])
    rows, mapping = np.unique(np.column_stack(columns), axis=0, return_inverse=True)
    used = np.zeros(65536, dtype=bool)
    for start in range(0, len(data.blocks), 8192):
        used[data.blocks[start:start + 8192]] = True
    used[0] = True
    available = np.flatnonzero(~used)
    if len(rows) > len(available):
        raise ValueError("Baked caustics exceed the 16-bit sparse material limit")
    new_codes = available[:len(rows)]
    materials, ior, emission = materials.copy(), ior.copy(), emission.copy()
    materials[new_codes], emission[new_codes], ior[new_codes] = rows[:, :4], rows[:, 4:7], rows[:, 7]
    if normals is not None:
        normals[new_codes] = rows[:, 8:]
    blocks = data.blocks.copy()
    blocks[bricks[changed], slots[changed]] = new_codes[mapping]
    return replace(data, blocks=blocks), materials, ior, normals, emission


def _cluster_materials(points, count):
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import pairwise_distances_argmin
    except ImportError as exc:
        raise ImportError("Caustic material clustering requires scikit-learn") from exc

    # Each point is one voxel, so common reflectance/emission pairs carry more weight.
    model = MiniBatchKMeans(n_clusters=count, batch_size=4096, n_init=1, random_state=21).fit(points)
    labels = model.labels_
    errors = np.sum((points - model.cluster_centers_[labels])**2, axis=1)
    order = np.argsort(errors, kind="stable")
    _, first = np.unique(labels[order], return_index=True)
    representatives = points[order[first]]
    return representatives[pairwise_distances_argmin(points, representatives)]


def _material_table(values, shape, nonnegative=True):
    table = np.asarray(values, dtype=np.float32)
    if table.shape != shape or not np.all(np.isfinite(table)) or (nonnegative and np.any(table < 0.)):
        raise ValueError(f"Invalid caustic material table, expected shape {shape}")
    return table
