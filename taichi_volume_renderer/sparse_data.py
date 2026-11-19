"""In-memory volume data shared by sparse rendering and file importers."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Entity:
    """An entity-local volume with a world transform and a range of sector rows."""

    position: np.ndarray
    rotation: np.ndarray
    first_sector: int
    end_sector: int
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class SparseVolumeData:
    """Sector coordinates, brick indices, and a compact pool of occupied bricks.

    Each brick has 512 slots. The blocks array contains uint16 material codes,
    float densities, or float RGB-density tuples (last dimension of length 4).
    """

    entities: list
    sectors: np.ndarray
    brick_ids: np.ndarray
    blocks: np.ndarray
    player_pose: object


def index_brick_values(blocks):
    """Build an exact material table from occupied values in the brick pool."""
    densities = blocks if blocks.ndim == 2 else blocks[..., 3]
    active = densities != 0
    values, inverse = np.unique(blocks[active], axis=0, return_inverse=True)
    if len(values) >= 2**16:
        raise ValueError(
            f"Indexed storage needs {len(values):,} non-air values, but 16 bits allow "
            "at most 65,535 plus air. Use storage='direct' or 'auto'."
        )
    indices = np.zeros(densities.shape, dtype=np.uint16)
    indices[active] = inverse.reshape(-1) + 1
    materials = np.zeros((2**16, 4), dtype=np.float32)
    if blocks.ndim == 2:
        materials[1:len(values) + 1, :3] = 1.0
        materials[1:len(values) + 1, 3] = values
    else:
        materials[1:len(values) + 1] = values
    return indices, materials
