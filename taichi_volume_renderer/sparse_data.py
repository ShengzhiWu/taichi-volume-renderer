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
    """Sector coordinates, brick indices, and 16-bit material codes for occupied bricks."""

    entities: list
    sectors: np.ndarray
    brick_ids: np.ndarray
    blocks: np.ndarray
    player_pose: object
