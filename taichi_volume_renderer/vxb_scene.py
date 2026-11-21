"""Adapt VXB files to the sparse renderer's existing 128/8 storage."""

from itertools import product

import numpy as np

from .math import rotation_quaternion_to_matrix
from .sparse_data import Entity, SparseVolumeData
from .vxb import load_vxb


_NORMAL_FIELDS = ("normal_x", "normal_y", "normal_z")
_EMISSION_FIELDS = ("emission_r", "emission_g", "emission_b")


def _leaves(brick, origin, sizes, level, indexed):
    size = sizes[level]
    if isinstance(brick, dict):
        child_size = sizes[level + 1]
        n = size // child_size
        for slot, child in brick.items():
            coord = (slot % n, slot // n % n, slot // (n * n))
            start = tuple(a + b * child_size for a, b in zip(origin, coord))
            yield from _leaves(child, start, sizes, level + 1, indexed)
    else:
        uniform = isinstance(brick, int) if indexed else not isinstance(brick[0], tuple)
        ranges = (range(o // 8, (o + size) // 8) for o in origin)
        for coord in product(*ranges):
            yield coord, brick, uniform


def _optical_columns(scene):
    fields = {field.name: (index, field.type) for index, field in enumerate(scene.fields)}
    columns = {}
    for name in ("r", "g", "b", "density", "ior") + _NORMAL_FIELDS + _EMISSION_FIELDS:
        if name in fields:
            index, kind = fields[name]
            if kind != "f32":
                raise ValueError(f"VXB rendering field {name!r} must use f32")
            columns[name] = index
    return columns


def _copy_rgba(target, rows, columns, density):
    target[...] = (1., 1., 1., density)
    for channel, name in enumerate(("r", "g", "b", "density")):
        if name in columns:
            target[..., channel] = rows[..., columns[name]]


def load_vxb_volume(path, density):
    scene = load_vxb(path)
    if not scene.bricks:
        raise ValueError("VXB world has no occupied bricks")
    indexed = scene.materials is not None
    columns = _optical_columns(scene)
    normal_columns = [columns[name] for name in _NORMAL_FIELDS if name in columns]
    emission_columns = [columns[name] for name in _EMISSION_FIELDS if name in columns]
    if normal_columns and len(normal_columns) != 3:
        raise ValueError("VXB interface normals require normal_x, normal_y, and normal_z")
    if emission_columns and len(emission_columns) != 3:
        raise ValueError("VXB emission requires emission_r, emission_g, and emission_b")
    if indexed and len(scene.materials) > 65536:
        raise ValueError("The sparse renderer supports at most 65,536 indexed VXB materials")
    if not indexed and ("ior" in columns or normal_columns or emission_columns):
        raise ValueError("VXB refraction, interface normals, and emission currently require indexed material rows")

    size = scene.brick_sizes[0]
    coordinates = np.array(tuple(scene.bricks), dtype=np.int64)
    lower = coordinates.min(axis=0) * size
    upper = (coordinates.max(axis=0) + 1) * size
    limits = np.iinfo(np.int32)
    if np.any(lower < limits.min) or np.any(upper > limits.max - 127):
        raise ValueError("VXB world is outside the sparse renderer's signed voxel coordinates")

    tables, leaves = {}, []
    for coord, brick in scene.bricks.items():
        origin = tuple(axis * size for axis in coord)
        for (x, y, z), values, uniform in _leaves(brick, origin, scene.brick_sizes, 0, indexed):
            sector = (x // 16, y // 16, z // 16)
            table = tables.get(sector)
            if table is None:
                table = tables[sector] = np.full(4096, -1, dtype=np.int32)
            table[x % 16 + 16 * (y % 16 + 16 * (z % 16))] = len(leaves)
            leaves.append((values, uniform))

    palette = indices = normals = emission = None
    if indexed:
        blocks = np.empty((len(leaves), 512), dtype=np.uint16)
        for index, (values, _) in enumerate(leaves):
            blocks[index] = values
        rows = np.asarray(scene.materials, dtype=np.float32)
        palette = np.zeros((65536, 4), dtype=np.float32)
        _copy_rgba(palette[1:len(rows)], rows[1:], columns, density)
        if "ior" in columns:
            indices = np.ones(65536, dtype=np.float32)
            indices[1:len(rows)] = rows[1:, columns["ior"]]
            if np.all(indices == 1.):
                indices = None
        if normal_columns:
            vectors = rows[1:, normal_columns]
            if np.any(vectors):
                normals = np.zeros((65536, 3), dtype=np.float32)
                normals[1:len(rows)] = vectors
        if emission_columns:
            values = rows[1:, emission_columns]
            if np.any(values):
                emission = np.zeros((65536, 3), dtype=np.float32)
                emission[1:len(rows)] = values
    else:
        blocks = np.empty((len(leaves), 512, 4), dtype=np.float32)
        for index, (values, uniform) in enumerate(leaves):
            rows = np.asarray(values, dtype=np.float32)
            _copy_rgba(blocks[index], rows, columns, density)
            if not uniform:
                blocks[index, ~np.any(rows, axis=1)] = 0.

    pose = None
    if scene.camera is not None:
        quaternion = np.asarray(scene.camera.rotation)
        quaternion /= np.max(np.abs(quaternion))
        rotation = rotation_quaternion_to_matrix(quaternion[[3, 0, 1, 2]] / np.linalg.norm(quaternion))
        rotation[:, 2] *= -1  # The renderer's saved pose stores its forward vector in column 2.
        pose = (np.asarray(scene.camera.position), rotation)
    sectors = sorted(tables)
    entity = Entity(np.zeros(3), np.eye(3), 0, len(sectors), lower, upper)
    data = SparseVolumeData([entity], np.asarray(sectors, dtype=np.int32),
                            np.stack([tables[coord] for coord in sectors]), blocks, pose)
    return data, palette, indices, normals, emission
