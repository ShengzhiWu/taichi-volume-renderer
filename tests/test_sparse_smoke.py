import unittest

import numpy as np
import taichi as ti

from taichi_volume_renderer.cxw import CXWData, Entity
from taichi_volume_renderer.sparse_smoke import SparseSmokeScene


class SparseSmokeTest(unittest.TestCase):
    def test_translucent_voxels_across_empty_sectors(self):
        ti.init(arch=ti.cpu)
        self.addCleanup(ti.reset)
        blocks = np.zeros((2, 512), dtype=np.uint16)
        blocks[0, 7] = 0xfc00  # Red voxel at (-1, 0, 0).
        blocks[1, 0] = 0x83e0  # Green voxel at (256, 0, 0).
        table = np.full((2, 4096), -1, dtype=np.int32)
        table[0, 15] = 0
        table[1, 0] = 1
        sectors = np.array([[-1, 0, 0], [2, 0, 0]], dtype=np.int32)
        entity = Entity(np.zeros(3), np.eye(3), 0, 2,
                        np.array([-128, 0, 0]), np.array([384, 128, 128]))
        data = CXWData([entity], sectors, table, blocks, None)
        scene = SparseSmokeScene(data, density=0.25, background=(0, 0, 1))
        scene._target[None] = [-1.5, 0.5, 0.5]
        scene._view_basis[None] = np.eye(3, dtype=np.float32)
        output = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def trace():
            output[None] = scene.ray_tracing(ti.Vector([0.0, 0.0, 0.0]), ti.Vector([1.0, 0.0, 0.0]))

        trace()
        # Two half-voxel samples in each occupied voxel.
        opacity = 0.25 * 0.5
        remaining = 1.0 - opacity
        red = opacity * (remaining + remaining**2)
        np.testing.assert_allclose(output[None].to_numpy(), [red, red * remaining**2, remaining**4])
