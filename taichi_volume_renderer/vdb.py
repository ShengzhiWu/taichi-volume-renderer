"""Import an OpenVDB density grid directly into sparse float bricks."""

from itertools import product
from os import fspath

import numpy as np

from .sparse_data import Entity, SparseVolumeData


def load_vdb(path, *, grid_name=None):
    """Read one scalar density grid using the optional OpenVDB Python binding."""
    try:
        import openvdb as vdb
    except ModuleNotFoundError as exc:
        if exc.name != "openvdb":
            raise
        try:
            import pyopenvdb as vdb
        except ModuleNotFoundError as exc:
            if exc.name != "pyopenvdb":
                raise
            raise ImportError("VDB loading requires the OpenVDB Python binding (openvdb or pyopenvdb)") from exc

    path = fspath(path)
    if grid_name is None:
        grids = [grid for grid in vdb.readAllGridMetadata(path)
                 if isinstance(grid, vdb.FloatGrid) and grid.gridClass != vdb.GridClass.LEVEL_SET]
        if not grids:
            raise ValueError("The VDB file has no float density grid")
        grid_name = next((grid.name for grid in grids if grid.name == "density"), grids[0].name)
    grid = vdb.read(path, grid_name)
    if not isinstance(grid, vdb.FloatGrid) or grid.gridClass == vdb.GridClass.LEVEL_SET:
        raise ValueError("The selected VDB grid must be a FloatGrid containing density")
    return _density_bricks(grid)


def _density_bricks(grid):
    if grid.background != 0.0:
        raise ValueError("Sparse VDB density grids must have a zero background")
    transform = grid.transform
    linear = transform.isLinear
    if not (linear() if callable(linear) else linear):
        raise ValueError("Sparse VDB density grids require a linear transform")

    offset = np.asarray(transform.indexToWorld((0, 0, 0)))
    axes = np.eye(3, dtype=int)
    basis = np.column_stack([np.asarray(transform.indexToWorld(tuple(axis.tolist()))) - offset
                             for axis in axes])
    size = np.linalg.norm(basis[:, 0])
    if not np.isfinite(size) or size == 0:
        raise ValueError("The VDB voxel size must be finite and positive")
    rotation = basis / size
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("Sparse VDB density grids require uniform voxel sizes")
    position = offset / size - rotation @ np.full(3, 0.5)

    bricks = {}
    lower = np.full(3, np.iinfo(np.int64).max, dtype=np.int64)
    upper = np.full(3, np.iinfo(np.int64).min, dtype=np.int64)
    for item in grid.iterOnValues():
        density = item.value
        if density == 0:
            continue
        if density < 0 or not np.isfinite(density):
            raise ValueError("VDB densities must be finite and nonnegative")
        lo, hi = np.asarray(item.min), np.asarray(item.max) + 1
        lower = np.minimum(lower, lo)
        upper = np.maximum(upper, hi)
        ranges = (range(int(a // 8), int((b - 1) // 8) + 1) for a, b in zip(lo, hi))
        for brick in product(*ranges):
            block = bricks.get(brick)
            if block is None:
                block = bricks[brick] = np.zeros((8, 8, 8), dtype=np.float32)
            start = np.asarray(brick) * 8
            x, y, z = (slice(a, b) for a, b in zip(np.maximum(lo - start, 0), np.minimum(hi - start, 8)))
            block[z, y, x] = density

    if not bricks:
        raise ValueError("The VDB density grid has no nonzero active voxels")
    sectors = {}
    pool = np.empty((len(bricks), 512), dtype=np.float32)
    for index, (brick, block) in enumerate(bricks.items()):
        sector = tuple(axis // 16 for axis in brick)
        table = sectors.get(sector)
        if table is None:
            table = sectors[sector] = np.full(4096, -1, dtype=np.int32)
        x, y, z = (axis % 16 for axis in brick)
        table[x + 16 * y + 256 * z] = index
        pool[index] = block.ravel()

    coords = sorted(sectors)
    entity = Entity(position, rotation, 0, len(coords), lower, upper)
    return SparseVolumeData([entity], np.asarray(coords, dtype=np.int32),
                            np.stack([sectors[coord] for coord in coords]), pool, None)
