"""Extend the lowest occupied voxel in each world-vertical column."""

from collections import defaultdict
from dataclasses import replace
from itertools import product

import numpy as np

from .sparse_data import SparseVolumeData


def _columns(data, entity, axes, upward):
    horizontal = axes[:2]
    vertical = axes[2]
    tiles = defaultdict(list)
    for sector, table in zip(data.sectors[entity.first_sector:entity.end_sector],
                             data.brick_ids[entity.first_sector:entity.end_sector]):
        slots = np.flatnonzero(table >= 0)
        coords = np.column_stack((slots & 15, (slots >> 4) & 15, slots >> 8)) + sector * 16
        for coord, row in zip(coords.tolist(), table[slots].tolist()):
            tiles[tuple(coord[axis] for axis in horizontal)].append((upward * coord[vertical], row))
    for level, x, y, z, row in data.uniform_bricks:
        width = 1 << (level - 3)
        origin = (x * width, y * width, z * width)
        bottom = origin[vertical] + (width - 1 if upward < 0 else 0)
        if not np.any(_non_air(data.blocks[row])):
            continue
        for tile in product(*(range(origin[axis], origin[axis] + width) for axis in horizontal)):
            tiles[tile].append((upward * bottom, row))

    for tile, bricks in tiles.items():
        bricks.sort()
        heights, rows = np.asarray(bricks, dtype=np.int64).T
        # Brick slots are x-fastest. Arrange samples as (brick, up, horizontal, horizontal).
        samples = data.blocks[rows]
        samples = samples.reshape((len(rows), 8, 8, 8) + samples.shape[2:])
        samples = samples.transpose((0,) + tuple(3 - axis for axis in axes[::-1])
                                    + tuple(range(4, samples.ndim)))
        if upward < 0:
            samples = samples[:, ::-1]
        samples = samples.reshape((-1, 64) + data.blocks.shape[2:])
        occupied = _non_air(samples)
        columns = np.flatnonzero(occupied.any(axis=0))
        first = occupied[:, columns].argmax(axis=0)
        z = (8 * heights[first // 8] + first % 8) * upward + (7 if upward < 0 else 0)
        yield tile, columns, z, samples[first, columns]


def _non_air(samples):
    return samples[..., 3] != 0 if samples.ndim and samples.shape[-1] == 4 else samples != 0


def _thicken_ground(data, depth):
    """Add ground voxels to entities with vertical voxel columns while preserving entity transforms."""
    if (isinstance(depth, (bool, np.bool_))
            or not isinstance(depth, (int, np.integer)) or depth < 0):
        raise ValueError("thicken_ground must be a nonnegative integer")
    if depth == 0:
        return data
    depth = int(depth)
    if depth >= 2**32:
        raise ValueError("Ground thickness extends outside signed voxel coordinates")
    if data.hierarchy is not None:
        from .vxb_scene import restore_edited_hierarchy

        # The ground editor works by 8³ columns. Build its temporary index
        # only for mixed leaves and keep all native uniform leaves intact.
        coordinates = data.hierarchy.mixed_origins // 8
        sectors, inverse = np.unique(coordinates // 16, axis=0, return_inverse=True)
        local = coordinates % 16
        slots = local[:, 0] + 16 * (local[:, 1] + 16 * local[:, 2])
        tables = np.full((len(sectors), 4096), -1, dtype=np.int32)
        tables[inverse, slots] = np.arange(len(coordinates), dtype=np.int32)
        entity = replace(data.entities[0], first_sector=0, end_sector=len(sectors))
        edit_data = replace(data, entities=[entity], sectors=sectors, brick_ids=tables, hierarchy=None)
        return restore_edited_hierarchy(_thicken_ground(edit_data, depth), data.hierarchy.sizes)

    regular_count = len(data.blocks) - len(data.uniform_bricks)
    additions = []
    entities, sectors, tables = [], [], []
    orientations = []
    for entity in data.entities:
        vertical = int(np.abs(entity.rotation[2]).argmax())
        upward = 1 if entity.rotation[2, vertical] > 0 else -1
        if not np.allclose(entity.rotation[2], np.eye(3)[vertical] * upward, atol=1e-5):
            raise ValueError("Ground thickness requires a voxel axis parallel to world Z")
        axes = tuple(axis for axis in range(3) if axis != vertical) + (vertical,)
        orientations.append((axes, upward, entity.rotation))

    multiple_entities = len(data.entities) > 1
    lowest = {}
    if multiple_entities:
        for index, (entity, (axes, upward, rotation)) in enumerate(zip(data.entities, orientations)):
            for tile, columns, z, _ in _columns(data, entity, axes, upward):
                centers = _world_centers(entity, tile, columns, z, axes, rotation)
                for x, y, height in centers.tolist():
                    key = (x, y)
                    candidate = (height, index)
                    if key not in lowest or candidate < lowest[key]:
                        lowest[key] = candidate

    for index, (entity, (axes, upward, rotation)) in enumerate(zip(data.entities, orientations)):
        sector_tables = dict(zip(map(tuple, data.sectors[entity.first_sector:entity.end_sector]),
                                 data.brick_ids[entity.first_sector:entity.end_sector]))
        lower, upper = entity.lower.copy(), entity.upper.copy()
        vertical = axes[2]
        for tile, columns, z, values in _columns(data, entity, axes, upward):
            if multiple_entities:
                centers = _world_centers(entity, tile, columns, z, axes, rotation)
                keep = [lowest[x, y] == (height, index) for x, y, height in centers.tolist()]
                columns, z, values = columns[keep], z[keep], values[keep]
            if not len(columns):
                continue

            end = z - int(upward) * depth
            if end.min() < -(2**31) or end.max() >= 2**31 - 128:
                raise ValueError("Ground thickness extends outside signed voxel coordinates")
            lower[vertical] = min(lower[vertical], end.min())
            upper[vertical] = max(upper[vertical], end.max() + 1)
            horizontal_slots = (columns % 8) * 8**int(axes[0]) + (columns // 8) * 8**int(axes[1])
            coord = [0, 0, 0]
            coord[axes[0]], coord[axes[1]] = tile
            starts = np.minimum(z - upward, end) // 8
            stops = np.maximum(z - upward, end) // 8
            for height in _brick_layers(starts, stops):
                coord[vertical] = height
                layers = 8 * height + np.arange(8)
                distance = int(upward) * (z[:, None] - layers)
                column, layer = np.nonzero((distance > 0) & (distance <= depth))
                if not len(column):
                    continue
                sector = tuple(axis // 16 for axis in coord)
                table = sector_tables.get(sector)
                if table is None:
                    table = np.full(4096, -1, dtype=np.int32)
                    sector_tables[sector] = table
                slot = sum((axis % 16) * 16**i for i, axis in enumerate(coord))
                row = table[slot]
                if row < 0:
                    row = regular_count + len(additions)
                    table[slot] = row
                    additions.append(np.zeros(data.blocks.shape[1:], dtype=data.blocks.dtype))
                block = data.blocks[row] if row < regular_count else additions[row - regular_count]
                block[horizontal_slots[column] + layer * 8**int(vertical)] = values[column]

        first = len(sectors)
        for coord, table in sorted(sector_tables.items()):
            sectors.append(coord)
            tables.append(table)
        entities.append(replace(entity, lower=lower, upper=upper, first_sector=first, end_sector=len(sectors)))

    blocks = data.blocks
    if additions:
        blocks = np.concatenate((blocks[:regular_count], np.asarray(additions), blocks[regular_count:]))
    uniform = tuple((*brick[:-1], brick[-1] + len(additions)) for brick in data.uniform_bricks)
    brick_ids = np.stack(tables) if tables else data.brick_ids
    return SparseVolumeData(entities, np.asarray(sectors, dtype=np.int32).reshape(-1, 3),
                            brick_ids, blocks, data.player_pose, uniform)


def _world_centers(entity, tile, columns, z, axes, rotation):
    coords = np.empty((len(columns), 3))
    coords[:, axes[0]] = 8 * tile[0] + columns % 8
    coords[:, axes[1]] = 8 * tile[1] + columns // 8
    coords[:, axes[2]] = z
    return (coords + 0.5) @ rotation.T + entity.position


def _brick_layers(starts, stops):
    previous = None
    for first, last in sorted(zip(starts.tolist(), stops.tolist())):
        start = first if previous is None else max(first, previous + 1)
        yield from range(start, last + 1)
        previous = last if previous is None else max(previous, last)
