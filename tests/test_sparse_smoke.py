import unittest

import numpy as np
import taichi as ti

from taichi_volume_renderer import DisplayWindow
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
        scene = SparseSmokeScene(data, density=0.25, step_length=0.5,
                                 lighting=False, background=(0, 0, 1))
        scene._target[None] = [-1.5, 0.5, 0.5]
        scene._view_basis[None] = np.eye(3, dtype=np.float32)
        scene.camera_distance = 0.5
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

    def test_point_lit_sparse_volume_matches_padded_dense_renderer(self):
        ti.init(arch=ti.cpu)
        self.addCleanup(ti.reset)
        extent = np.array([32, 16, 8])
        density = 0.18
        blocks = np.zeros((4, 512), dtype=np.uint16)
        table = np.full((1, 4096), -1, dtype=np.int32)
        # Pad the dense grid with air to retain the sparse volume's aspect ratio.
        dense_density = np.zeros((32, 32, 32), dtype=np.float32)
        dense_color = np.zeros((*dense_density.shape, 3), dtype=np.float32)
        for x in range(12, 20):
            for y in range(4, 12):
                for z in range(2, 6):
                    brick = x // 8 - 1 + (y // 8) * 2
                    table[0, x // 8 + (y // 8) * 16] = brick
                    index = x % 8 + (y % 8) * 8 + z * 64
                    blocks[brick, index] = 0xfc00 if x < 16 else 0x83e0
                    dense_density[x, z + 12, y + 8] = density * extent.max()
                    dense_color[x, z + 12, y + 8] = [1, 0, 0] if x < 16 else [0, 1, 0]
        entity = Entity(np.zeros(3), np.eye(3), 0, 1, np.zeros(3, dtype=np.int64), extent)
        data = CXWData([entity], np.zeros((1, 3), dtype=np.int32), table, blocks, None)
        for intensity in (None, np.array([[35, 50, 65]], dtype=np.float32)):
            with self.subTest(intensity=intensity):
                sparse = SparseSmokeScene(data, density=density, step_length=0.5,
                                          point_lights_intensity=intensity)
                dense = DisplayWindow(dense_density, dense_color, init_taichi=False,
                                      resolution=(24, 20), ray_tracing_step_size_factor=0.5,
                                      point_lights_intensity=intensity)
                sparse_pixels = ti.Vector.field(3, dtype=ti.f32, shape=dense.resolution)
                dense.scene.update_light()
                sparse.update_light()
                dense.scene.render(dense.pixels)
                sparse.render(sparse_pixels)
                np.testing.assert_allclose(sparse_pixels.to_numpy(), dense.pixels.to_numpy(), atol=2e-5)
