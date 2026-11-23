"""Optional material simplification for VXB scenes before encoding."""

from dataclasses import replace
from itertools import product
import math
from numbers import Integral

import numpy as np
from scipy.ndimage import maximum_filter

from .vxb import VxbScene, merge_uniform_children


def _child_origin(origin, slot, ratio, size):
    # Array coordinates are Z, Y, X. VXB stores X as the fastest axis.
    offset = (slot // (ratio * ratio), slot // ratio % ratio, slot % ratio)
    return tuple(start + index * size for start, index in zip(origin, offset))


def _tile_transmission(coord, exits, size):
    padded = np.ones((size + 2,) * 3)
    for offset in product((-1, 0, 1), repeat=3):
        neighbor = tuple(a + b for a, b in zip(coord, offset))
        if neighbor in exits:
            target = tuple((0, slice(1, -1), -1)[axis + 1] for axis in offset)
            source = tuple((-1, slice(None), 0)[axis + 1] for axis in offset)
            padded[target] = exits[neighbor][source]
    return maximum_filter(padded, size=3)[1:-1, 1:-1, 1:-1]


def merge_hidden_materials(
    scene: VxbScene, target_materials, *, enabled=True, transmission_threshold=0.01,
    default_density=0.22,
) -> VxbScene:
    """Unify hidden target voxels and merge the resulting uniform bricks.

    Supply palette indices for an indexed scene or material rows for a direct
    scene. Space outside the target materials admits full light, including at
    internal cavities. The estimate propagates the strongest transmission
    through 26 neighbors with a unit voxel step, using 1 - density per step
    (clamped to [0, 1]). A voxel's own density does not hide its surface.
    The most frequent material strictly below the threshold wins, with ties
    resolved by target order. The input scene is left intact.

    The working grid expands only tiles containing target materials, so
    large solid regions can require substantial temporary memory.
    """
    if not enabled:
        return scene
    if not math.isfinite(transmission_threshold) or not 0 < transmission_threshold <= 1:
        raise ValueError("transmission_threshold must be in (0, 1]")
    indexed = scene.materials is not None
    targets = tuple(dict.fromkeys(target_materials))
    if indexed:
        if any(not isinstance(index, Integral) or isinstance(index, bool)
               or not 0 < index < len(scene.materials) for index in targets):
            raise ValueError("Target materials must be non-air VXB palette indices")
        rows = [scene.materials[index] for index in targets]
    else:
        if any(not isinstance(row, tuple) or len(row) != len(scene.fields) or not any(row)
               for row in targets):
            raise ValueError("Target materials must be non-air VXB material rows")
        rows = targets
    if len(targets) < 2:
        return scene

    density_column = next((i for i, field in enumerate(scene.fields) if field.name == "density"), None)
    densities = np.asarray([default_density if density_column is None else row[density_column]
                            for row in rows], dtype=np.float64)
    if not np.isfinite(densities).all() or (densities < 0).any():
        raise ValueError("Target densities must be finite and nonnegative")
    if not densities.any():
        return scene
    factors = np.r_[1., 1. - np.clip(densities, 0., 1.)]
    ids = {material: index for index, material in enumerate(targets, 1)}
    dtype = np.min_scalar_type(len(targets))
    if indexed:
        lookup = np.zeros(len(scene.materials), dtype=dtype)
        lookup[list(targets)] = range(1, len(targets) + 1)

    sizes = scene.brick_sizes
    tile_size = min(sizes[0], 32)
    tiles = {}

    def uniform(brick):
        return isinstance(brick, Integral) if indexed else not isinstance(brick[0], tuple)

    def rasterize(brick, origin, level):
        size = sizes[level]
        if isinstance(brick, dict):
            child_size = sizes[level + 1]
            ratio = size // child_size
            for slot, child in brick.items():
                rasterize(child, _child_origin(origin, slot, ratio, child_size), level + 1)
            return
        if uniform(brick):
            values = ids.get(brick, 0)
            if not values:
                return
        else:
            values = (lookup[np.asarray(brick, dtype=np.intp)] if indexed else
                      np.fromiter((ids.get(row, 0) for row in brick), dtype=dtype, count=512))
            if not values.any():
                return
            values = values.reshape((8,) * 3)
        ranges = (range(start // tile_size, (start + size - 1) // tile_size + 1) for start in origin)
        for coord in product(*ranges):
            tile = tiles.get(coord)
            if tile is None:
                tiles[coord] = tile = np.zeros((tile_size,) * 3, dtype=dtype)
            start = tuple(a % tile_size for a in origin)
            region = tuple(slice(a, a + size) for a in start)
            tile[region] = values

    for coord, brick in scene.bricks.items():
        rasterize(brick, tuple(axis * sizes[0] for axis in reversed(coord)), 0)

    # Zero is a lower bound for exit transmission inside a target. Values below
    # the cutoff cannot propagate back above it, so discard them as we converge.
    exits = {coord: (tile == 0).astype(float) for coord, tile in tiles.items()}
    tiles_factors = {coord: factors[tile] for coord, tile in tiles.items()}
    while True:
        changed = False
        for coord, attenuation in tiles_factors.items():
            incoming = _tile_transmission(coord, exits, tile_size)
            incoming[tiles[coord] == 0] = 1.
            incoming *= attenuation
            incoming[incoming < transmission_threshold] = 0.
            if not np.array_equal(incoming, exits[coord]):
                exits[coord] = incoming.copy()
                changed = True
        if not changed:
            break

    hidden = {}
    counts = np.zeros(len(targets) + 1, dtype=np.int64)
    for coord, tile in tiles.items():
        mask = (tile != 0) & (_tile_transmission(coord, exits, tile_size) < transmission_threshold)
        if mask.any():
            hidden[coord] = mask
            counts += np.bincount(tile[mask].astype(np.intp), minlength=len(counts))
    if not hidden:
        return scene
    winner = int(counts.argmax())
    representative = targets[winner - 1]
    for coord in list(hidden):
        mask = hidden[coord]
        mask &= tiles[coord] != winner
        if not mask.any():
            del hidden[coord]
    if not hidden:
        return scene

    changed_bricks = {
        size: {tuple(axis * tile_size // size for axis in coord) for coord in hidden}
        for size in sizes if size > tile_size
    }

    def replace_brick(brick, origin, level):
        size = sizes[level]
        if size > tile_size:
            if tuple(axis // size for axis in origin) not in changed_bricks[size]:
                return brick
        else:
            mask = hidden.get(tuple(axis // tile_size for axis in origin))
            if mask is None:
                return brick
            region = tuple(slice(axis % tile_size, axis % tile_size + size) for axis in origin)
            mask = mask[region]
            if not mask.any():
                return brick
            if mask.all():
                return representative
        if not isinstance(brick, dict) and uniform(brick) and brick not in ids:
            return brick
        if level == len(sizes) - 1:
            values = [brick] * 512 if uniform(brick) else list(brick)
            for index in np.flatnonzero(mask):
                values[index] = representative
            first = values[0]
            return first if all(value == first for value in values) else tuple(values)
        child_size = sizes[level + 1]
        ratio = size // child_size
        children = brick if isinstance(brick, dict) else dict.fromkeys(range(ratio**3), brick)
        return merge_uniform_children({
            slot: replace_brick(child, _child_origin(origin, slot, ratio, child_size), level + 1)
            for slot, child in children.items()
        }, ratio)

    return replace(scene, bricks={
        coord: replace_brick(brick, tuple(axis * sizes[0] for axis in reversed(coord)), 0)
        for coord, brick in scene.bricks.items()
    })
