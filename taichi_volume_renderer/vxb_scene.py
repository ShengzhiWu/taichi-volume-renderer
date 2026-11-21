"""Adapt VXB files to the sparse renderer's existing 128/8 storage."""

from itertools import product

import numpy as np

from .math import rotation_quaternion_to_matrix
from .sparse_data import Entity, SparseVolumeData
from .vxb import _open_vxb, _read_exact


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


def _optical_columns(schema):
    fields = {field.name: (index, field.type) for index, field in enumerate(schema)}
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


def _skip_mixed(stream, size):
    _read_exact(stream, size)


def _leaf_count(brick, sizes, level=0):
    if not isinstance(brick, dict):
        return (sizes[level] // 8)**3
    return sum(_leaf_count(child, sizes, level + 1) for child in brick.values())


def _write_mixed_indices(blocks, mixed, width, material_count):
    indices, packed = zip(*mixed)
    row_bytes = 512 * width // 8
    # Each index spans at most three bytes. Padding permits the final gather.
    raw = np.frombuffer(b"".join(packed) + b"\x00\x00", dtype=np.uint8)
    bit = np.arange(512, dtype=np.uint32) * width
    offsets = np.arange(len(packed), dtype=np.intp)[:, None] * row_bytes + (bit // 8)
    bits = raw[offsets].astype(np.uint32)
    bits |= raw[offsets + 1].astype(np.uint32) << 8
    bits |= raw[offsets + 2].astype(np.uint32) << 16
    values = (bits >> (bit % 8)) & ((1 << width) - 1)
    maximum = values.max(axis=1)
    if np.any(maximum >= material_count) or np.any(values.min(axis=1) == maximum):
        raise ValueError("Invalid VXB mixed brick indices")
    blocks[list(indices)] = values


def _load_bricks(read_roots, sizes, materials, columns, density):
    # Count expanded leaves first so the voxel pool is allocated only once.
    roots = []
    brick_count = 0
    for coord, brick in read_roots(_skip_mixed):
        roots.append(coord)
        brick_count += _leaf_count(brick, sizes)
    if not roots:
        raise ValueError("VXB world has no occupied bricks")

    coordinates = np.array(roots, dtype=np.int64)
    lower = coordinates.min(axis=0) * sizes[0]
    upper = (coordinates.max(axis=0) + 1) * sizes[0]
    limits = np.iinfo(np.int32)
    if np.any(lower < limits.min) or np.any(upper > limits.max - 127):
        raise ValueError("VXB world is outside the sparse renderer's signed voxel coordinates")

    indexed = materials is not None
    blocks = (np.empty((brick_count, 512), dtype=np.uint16) if indexed
              else np.empty((brick_count, 512, 4), dtype=np.float32))
    width = (len(materials) - 1).bit_length() if indexed else 0
    tables = {}
    index = 0
    for coord, brick in read_roots(_read_exact if indexed else None):
        origin = tuple(axis * sizes[0] for axis in coord)
        mixed = []
        for (x, y, z), values, uniform in _leaves(brick, origin, sizes, 0, indexed):
            sector = (x // 16, y // 16, z // 16)
            table = tables.get(sector)
            if table is None:
                table = np.full(4096, -1, dtype=np.int32)
                tables[sector] = table
            table[x % 16 + 16 * (y % 16 + 16 * (z % 16))] = index
            if indexed:
                if uniform:
                    blocks[index] = values
                else:
                    mixed.append((index, values))
                    if len(mixed) == 1024:
                        _write_mixed_indices(blocks, mixed, width, len(materials))
                        mixed.clear()
            else:
                rows = np.asarray(values, dtype=np.float32)
                _copy_rgba(blocks[index], rows, columns, density)
                if not uniform:
                    blocks[index, ~np.any(rows, axis=1)] = 0.
            index += 1
        if mixed:
            _write_mixed_indices(blocks, mixed, width, len(materials))

    sectors = sorted(tables)
    entity = Entity(np.zeros(3), np.eye(3), 0, len(sectors), lower, upper)
    return SparseVolumeData([entity], np.asarray(sectors, dtype=np.int32),
                            np.stack([tables[coord] for coord in sectors]), blocks, None)


def load_vxb_volume(path, density):
    with _open_vxb(path) as (fields, materials, sizes, camera, read_roots):
        columns = _optical_columns(fields)
        indexed = materials is not None
        normal_columns = [columns[name] for name in _NORMAL_FIELDS if name in columns]
        emission_columns = [columns[name] for name in _EMISSION_FIELDS if name in columns]
        if normal_columns and len(normal_columns) != 3:
            raise ValueError("VXB interface normals require normal_x, normal_y, and normal_z")
        if emission_columns and len(emission_columns) != 3:
            raise ValueError("VXB emission requires emission_r, emission_g, and emission_b")
        if indexed and len(materials) > 65536:
            raise ValueError("The sparse renderer supports at most 65536 indexed VXB materials")
        if not indexed and ("ior" in columns or normal_columns or emission_columns):
            raise ValueError("VXB refraction, interface normals, and emission currently require indexed material rows")
        data = _load_bricks(read_roots, sizes, materials, columns, density)
    palette = indices = normals = emission = None
    if indexed:
        rows = np.asarray(materials, dtype=np.float32)
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
    if camera is not None:
        quaternion = np.asarray(camera.rotation)
        quaternion /= np.max(np.abs(quaternion))
        rotation = rotation_quaternion_to_matrix(quaternion[[3, 0, 1, 2]] / np.linalg.norm(quaternion))
        rotation[:, 2] *= -1  # The renderer's saved pose stores its forward vector in column 2.
        data.player_pose = (np.asarray(camera.position), rotation)
    return data, palette, indices, normals, emission
