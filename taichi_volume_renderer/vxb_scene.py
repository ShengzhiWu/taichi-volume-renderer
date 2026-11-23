"""Load native VXB brick hierarchies into renderer buffers."""

from dataclasses import replace

import numpy as np

from .math import rotation_quaternion_to_matrix
from .sparse_data import BrickHierarchy, Entity, SparseVolumeData
from .vxb import _open_vxb, _read_exact


_NORMAL_FIELDS = ("normal_x", "normal_y", "normal_z")
_EMISSION_FIELDS = ("emission_r", "emission_g", "emission_b")


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


def _tree_counts(brick, indexed, counts):
    counts[0] += 1
    if isinstance(brick, dict):
        for child in brick.values():
            _tree_counts(child, indexed, counts)
    else:
        uniform = isinstance(brick, int) if indexed else not isinstance(brick[0], tuple)
        counts[2 if uniform else 1] += 1


def _write_tree(nodes, index, next_index, slot, brick, origin, sizes, write_leaf):
    if not isinstance(brick, dict):
        nodes[index] = slot, ~write_leaf(brick, origin, sizes[0]), 0
        return next_index
    count = len(brick)
    nodes[index] = slot, next_index, count
    child_start = next_index
    next_index += count
    width = sizes[1]
    ratio = sizes[0] // width
    for row, (child_slot, child) in enumerate(sorted(brick.items()), child_start):
        x, y, z = child_slot % ratio, child_slot // ratio % ratio, child_slot // (ratio * ratio)
        child_origin = (origin[0] + width * x, origin[1] + width * y, origin[2] + width * z)
        next_index = _write_tree(nodes, row, next_index, child_slot, child, child_origin,
                                 sizes[1:], write_leaf)
    return next_index


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
    indexed = materials is not None
    roots = []
    counts = [0, 0, 0]
    for coord, brick in read_roots(_skip_mixed if indexed else None):
        roots.append(coord)
        _tree_counts(brick, indexed, counts)
    if not roots:
        raise ValueError("VXB world has no occupied bricks")
    node_count, mixed_count, uniform_count = counts
    coordinates = np.asarray(roots, dtype=np.int64)
    lower = coordinates.min(axis=0) * sizes[0]
    upper = (coordinates.max(axis=0) + 1) * sizes[0]
    limits = np.iinfo(np.int32)
    if np.any(lower < limits.min) or np.any(upper > limits.max):
        raise ValueError("VXB world is outside the sparse renderer's signed voxel coordinates")

    brick_count = mixed_count + uniform_count
    blocks = (np.empty((brick_count, 512), dtype=np.uint16) if indexed
              else np.empty((brick_count, 512, 4), dtype=np.float32))
    width = (len(materials) - 1).bit_length() if indexed else 0
    nodes = np.empty((node_count, 3), dtype=np.int32)
    mixed_origins = np.empty((mixed_count, 3), dtype=np.int32)
    uniform_bricks = []
    mixed = []
    mixed_row = 0

    def write_leaf(values, origin, size):
        nonlocal mixed_row
        uniform = isinstance(values, int) if indexed else not isinstance(values[0], tuple)
        if uniform:
            row = mixed_count + len(uniform_bricks)
            uniform_bricks.append((size.bit_length() - 1, *(axis // size for axis in origin), row))
        else:
            row = mixed_row
            mixed_row += 1
            mixed_origins[row] = origin
        if indexed:
            if uniform:
                blocks[row] = values
            else:
                mixed.append((row, values))
                if len(mixed) == 1024:
                    _write_mixed_indices(blocks, mixed, width, len(materials))
                    mixed.clear()
        else:
            rows = np.asarray(values, dtype=np.float32)
            _copy_rgba(blocks[row], rows, columns, density)
            if not uniform:
                blocks[row, ~np.any(rows, axis=1)] = 0.
        return row

    next_index = len(roots)
    for index, (coord, brick) in enumerate(read_roots(_read_exact if indexed else None)):
        origin = tuple(axis * sizes[0] for axis in coord)
        next_index = _write_tree(nodes, index, next_index, 0, brick, origin, sizes, write_leaf)
    if mixed:
        _write_mixed_indices(blocks, mixed, width, len(materials))
    entity = Entity(np.zeros(3), np.eye(3), 0, len(roots), lower, upper)
    hierarchy = BrickHierarchy(sizes, nodes, mixed_origins)
    return SparseVolumeData([entity], coordinates.astype(np.int32), None,
                            blocks, None, tuple(uniform_bricks), hierarchy)


def restore_edited_hierarchy(data, sizes):
    """Index explicitly edited ground voxels at the original file levels."""
    origins = np.empty((len(data.blocks) - len(data.uniform_bricks), 3), dtype=np.int32)
    for sector, table in zip(data.sectors, data.brick_ids):
        slots = np.flatnonzero(table >= 0)
        local = np.column_stack((slots & 15, (slots >> 4) & 15, slots >> 8))
        origins[table[slots]] = sector * 128 + local * 8

    trees = {}

    def insert(origin, size, row):
        root = tuple(int(axis) // sizes[0] for axis in origin)
        tree = trees
        key = root
        for parent, child in zip(sizes, sizes[1:]):
            if size == parent:
                break
            tree = tree.setdefault(key, {})
            if not isinstance(tree, dict):
                raise ValueError("Ground edit overlaps a uniform VXB brick")
            ratio = parent // child
            x, y, z = (int(axis) // child % ratio for axis in origin)
            key = x + ratio * (y + ratio * z)
        if key in tree:
            raise ValueError("Ground edit overlaps a uniform VXB brick")
        tree[key] = row

    for level, x, y, z, row in data.uniform_bricks:
        size = 1 << level
        insert((x * size, y * size, z * size), size, row)
    for row, origin in enumerate(origins):
        insert(origin, 8, row)

    roots = sorted(trees)
    counts = [0, 0, 0]
    for root in roots:
        _tree_counts(trees[root], True, counts)
    nodes = np.empty((counts[0], 3), dtype=np.int32)
    next_index = len(roots)
    for index, root in enumerate(roots):
        origin = tuple(axis * sizes[0] for axis in root)
        next_index = _write_tree(nodes, index, next_index, 0, trees[root], origin, sizes,
                                 lambda row, _origin, _size: row)
    entity = replace(data.entities[0], first_sector=0, end_sector=len(roots))
    return replace(data, entities=[entity], sectors=np.asarray(roots, dtype=np.int32), brick_ids=None,
                   hierarchy=BrickHierarchy(sizes, nodes, origins))


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
