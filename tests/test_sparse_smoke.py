from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import taichi as ti

from taichi_volume_renderer import DisplayWindow
from taichi_volume_renderer.sparse_data import Entity, SparseVolumeData, index_brick_values
from taichi_volume_renderer.sparse_smoke import SparseSmokeScene
from taichi_volume_renderer.vdb import _density_bricks
from taichi_volume_renderer.vxb import MaterialField, VxbScene, save_vxb


class SparseSmokeTest(unittest.TestCase):
    def test_uniform_bricks_share_center_lighting(self):
        ti.init(arch=ti.cpu)
        self.addCleanup(ti.reset)
        fields = tuple(MaterialField(name, "f32") for name in ("r", "g", "b", "density"))
        palette = ((0., 0., 0., 0.), (1., 1., 1., .01))
        mixed = (1,) + (0,) * 511
        world = VxbScene(fields, palette, (128, 32, 8),
                         {(-1, 0, 0): 1, (0, 0, 0): {0: 1, 1: {0: 1, 1: mixed}}})
        with TemporaryDirectory() as folder:
            path = Path(folder) / "world.vxb"
            save_vxb(path, world)
            scene = SparseSmokeScene.from_file(
                path, directional_light=((1., 0., 0.), (1., .5, .25)), light_step_length=1.,
            )
        scene.update_light()

        self.assertEqual(scene._uniform_light_cache.shape, (3,))
        self.assertEqual(scene._light_cache.shape, (1, 512))
        output = ti.Vector.field(3, dtype=ti.f32, shape=(4, 2))

        @ti.kernel
        def sample():
            output[0, 0] = scene._voxel_light(0, 0)
            output[0, 1] = scene._sample_light(ti.Vector([16., 16., 16.]))
            output[1, 0] = scene._voxel_light(63, 511)
            output[1, 1] = output[0, 1]
            output[2, 0] = scene._voxel_light(64, 511)
            output[2, 1] = scene._sample_light(ti.Vector([36., 4., 4.]))
            output[3, 0] = scene._voxel_light(66, 0)
            output[3, 1] = scene._sample_light(ti.Vector([-64., 64., 64.]))

        sample()
        values = output.to_numpy()
        np.testing.assert_allclose(values[:, 0], values[:, 1])
        self.assertFalse(np.allclose(values[0, 0], values[3, 0]))

    def test_translucent_voxels_across_empty_sectors(self):
        ti.init(arch=ti.cpu)
        self.addCleanup(ti.reset)
        blocks = np.zeros((2, 512), dtype=np.uint16)
        blocks[0, 7] = 0xfc00  # Red voxel at (-1, 0, 0).
        blocks[1, 0] = 0x83e0  # Green voxel at (6144, 0, 0), beyond the old step limit.
        table = np.full((2, 4096), -1, dtype=np.int32)
        table[0, 15] = 0
        table[1, 0] = 1
        sectors = np.array([[-1, 0, 0], [48, 0, 0]], dtype=np.int32)
        entity = Entity(np.zeros(3), np.eye(3), 0, 2,
                        np.array([-128, 0, 0]), np.array([6272, 128, 128]))
        data = SparseVolumeData([entity], sectors, table, blocks, None)
        # Two half-voxel samples in each occupied voxel.
        opacity = 0.25 * 0.5
        remaining = 1.0 - opacity
        red = opacity * (remaining + remaining**2)

        for options, expected in (
            ({}, [red, red * remaining**2, remaining**4]),
            ({"storage": "direct"}, [red, red * remaining**2, remaining**4]),
            ({"max_steps": 10000}, [red, 0, remaining**2]),
        ):
            with self.subTest(options=options):
                scene = SparseSmokeScene(data, density=0.25, step_length=0.5,
                                         lighting=False, background=(0, 0, 1), **options)
                scene._target[None] = [-1.5, 0.5, 0.5]
                output = ti.Vector.field(3, dtype=ti.f32, shape=2)

                @ti.kernel
                def trace():
                    origin = ti.Vector([0.0, 0.0, 0.0])
                    output[0] = scene.ray_tracing(origin, ti.Vector([1.0, 0.0, 0.0]))
                    output[1] = scene.ray_tracing(origin + ti.Vector([0.0, 0.0, 1.0]),
                                                  ti.Vector([1.0, 0.0, 0.0]))

                trace()
                np.testing.assert_allclose(output.to_numpy(), [expected, [0, 0, 1]])

    def test_density_only_bricks_keep_floating_point_values(self):
        ti.init(arch=ti.cpu)
        self.addCleanup(ti.reset)
        blocks = np.zeros((1, 512), dtype=np.float32)
        blocks[0, :2] = [0.123456, 0.234567]
        table = np.full((1, 4096), -1, dtype=np.int32)
        table[0, 0] = 0
        entity = Entity(np.zeros(3), np.eye(3), 0, 1, np.zeros(3, dtype=np.int64), np.array([8, 8, 8]))
        data = SparseVolumeData([entity], np.zeros((1, 3), dtype=np.int32), table, blocks, None)
        first, second = blocks[0, :2]
        expected = (1 - first) * (first + second * (1 - second))
        for storage in ("auto", "indexed"):
            with self.subTest(storage=storage):
                with patch("taichi_volume_renderer.vdb.load_vdb", return_value=data):
                    scene = SparseSmokeScene.from_file("test.vdb", lighting=False,
                                                      background=(0, 0, 0), storage=storage)
                self.assertEqual(scene._direct, storage == "auto")
                scene._target[None] = [0.5, 0.5, 0.5]
                output = ti.Vector.field(3, dtype=ti.f32, shape=())

                @ti.kernel
                def trace():
                    output[None] = scene.ray_tracing(ti.Vector([0.0, 0.0, 0.0]), ti.Vector([1.0, 0.0, 0.0]))

                trace()
                np.testing.assert_allclose(output[None].to_numpy(), [expected] * 3, atol=1e-7)

    def test_indexed_storage_rejects_too_many_distinct_values(self):
        blocks = np.arange(1, 2**16 + 1, dtype=np.float32).reshape(-1, 512)
        with self.assertRaisesRegex(ValueError, "65,535 plus air"):
            index_brick_values(blocks)
        blocks[-1, -1] = 0
        codes, materials = index_brick_values(blocks)
        np.testing.assert_array_equal(materials[codes, 3], blocks)

    def test_vdb_import_preserves_active_tile_and_negative_coordinates(self):
        tile = SimpleNamespace(min=(-1, 0, 0), max=(1, 0, 0), value=0.125)
        transform = SimpleNamespace(isLinear=True, indexToWorld=lambda xyz: np.asarray(xyz) * 0.5)
        grid = SimpleNamespace(background=0.0, transform=transform, iterOnValues=lambda: iter([tile]))
        data = _density_bricks(grid)

        np.testing.assert_array_equal(data.sectors, [[-1, 0, 0], [0, 0, 0]])
        self.assertEqual(data.brick_ids[0, 15], 0)
        self.assertEqual(data.brick_ids[1, 0], 1)
        np.testing.assert_array_equal(data.blocks[0, :8], [0, 0, 0, 0, 0, 0, 0, 0.125])
        np.testing.assert_array_equal(data.blocks[1, :3], [0.125, 0.125, 0])

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
                    dense_density[x, y + 8, z + 12] = density * extent.max()
                    dense_color[x, y + 8, z + 12] = [1, 0, 0] if x < 16 else [0, 1, 0]
        entity = Entity(np.zeros(3), np.eye(3), 0, 1, np.zeros(3, dtype=np.int64), extent)
        data = SparseVolumeData([entity], np.zeros((1, 3), dtype=np.int32), table, blocks, None)
        for storage, intensity in (("indexed", None), ("direct", np.array([[35, 50, 65]], dtype=np.float32))):
            with self.subTest(storage=storage, intensity=intensity):
                sparse = SparseSmokeScene(data, density=density, step_length=0.5,
                                          point_lights_intensity=intensity, storage=storage)
                dense = DisplayWindow(dense_density, dense_color, init_taichi=False,
                                      resolution=(24, 20), ray_tracing_step_size_factor=0.5,
                                      point_lights_intensity=intensity)
                sparse_pixels = ti.Vector.field(3, dtype=ti.f32, shape=dense.resolution)
                dense.scene.update_light()
                sparse.update_light()
                dense.scene.render(dense.pixels)
                sparse.render(sparse_pixels)
                np.testing.assert_allclose(sparse_pixels.to_numpy(), dense.pixels.to_numpy(), atol=2e-5)
