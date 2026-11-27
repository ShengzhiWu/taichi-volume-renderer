from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import taichi as ti

from taichi_volume_renderer import DisplayWindow
from taichi_volume_renderer.caustics import _spectral_rgb
from taichi_volume_renderer._ground import _thicken_ground
from taichi_volume_renderer.sparse_data import Entity, SparseVolumeData, index_brick_values
from taichi_volume_renderer.sparse_smoke import SparseSmokeScene
from taichi_volume_renderer.vdb import _density_bricks
from taichi_volume_renderer.vxb import MaterialField, VxbScene, save_vxb
from taichi_volume_renderer.vxb_scene import load_vxb_volume


class SparseSmokeTest(unittest.TestCase):
    def test_caustic_spectral_colors_interpolate_between_balanced_anchors(self):
        hues = np.array([-1/6, -1/12, 0., 1/12, 1/6, 1/3, 1/2, 2/3, 5/6, 11/12, 1.])
        np.testing.assert_allclose(_spectral_rgb(hues), [
            [0., 0., 0.], [.5, 0., 0.], [1., 0., 0.], [.75, .25, 0.],
            [.5, .5, 0.], [0., 1., 0.], [0., .5, .5], [0., 0., 1.],
            [.5, 0., .5], [.25, 0., .25], [0., 0., 0.],
        ], atol=1e-7)

    def test_shadow_samples_full_steps_at_their_midpoints(self):
        ti.init(arch=ti.cpu)
        self.addCleanup(ti.reset)
        fields = (MaterialField("density", "f32"),)
        voxels = [0] * 512
        for index in (0, 2, 4, 10):
            voxels[index] = 1
        world = VxbScene(fields, ((0.,), (.1,)), (8,), {(0, 0, 0): tuple(voxels)})
        with TemporaryDirectory() as folder:
            path = Path(folder) / "world.vxb"
            save_vxb(path, world)
            scene = SparseSmokeScene.from_file(path, light_step_length=2.)
        output = ti.field(dtype=ti.f32, shape=(2, 2))

        @ti.kernel
        def sample():
            for row, sign in output:
                output[row, sign] = scene._light_transmittance(
                    ti.Vector([2.5, row + .5, .5]), ti.Vector([2. * sign - 1., 0., 0.]),
                )

        sample()
        np.testing.assert_allclose(output.to_numpy(), [[.95 * .8, .95 * .8], [.95, .95]])

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
            data, materials, _, _, _ = load_vxb_volume(path, density=.22)
        self.assertEqual(data.hierarchy.sizes, (128, 32, 8))
        self.assertEqual(data.hierarchy.nodes.shape, (6, 3))
        self.assertEqual(len(data.blocks), 4)
        self.assertEqual([brick[0] for brick in data.uniform_bricks], [7, 5, 3])
        scene = SparseSmokeScene(data, materials=materials,
                                 directional_light=((1., 0., 0.), (1., .5, .25)), light_step_length=1.)
        scene.update_light()

        self.assertEqual(scene._uniform_light_cache.shape, (3,))
        self.assertEqual(scene._light_cache.shape, (1, 512))
        output = ti.Vector.field(3, dtype=ti.f32, shape=(4, 2))

        @ti.kernel
        def sample():
            output[0, 0] = scene._voxel_radiance(2, 0)
            output[0, 1] = scene._sample_light(ti.Vector([16., 16., 16.]))
            output[1, 0] = scene._voxel_radiance(2, 511)
            output[1, 1] = output[0, 1]
            output[2, 0] = scene._voxel_radiance(3, 511)
            output[2, 1] = scene._sample_light(ti.Vector([36., 4., 4.]))
            output[3, 0] = scene._voxel_radiance(1, 0)
            output[3, 1] = scene._sample_light(ti.Vector([-64., 64., 64.]))

        sample()
        values = output.to_numpy()
        np.testing.assert_allclose(values[:, 0], values[:, 1])
        self.assertFalse(np.allclose(values[0, 0], values[3, 0]))

    def test_caustics_add_to_outgoing_radiance_without_accumulating(self):
        ti.init(arch=ti.cpu)
        self.addCleanup(ti.reset)
        fields = tuple(MaterialField(name, "f32") for name in ("r", "g", "b", "density"))
        world = VxbScene(fields, ((0., 0., 0., 0.), (.5, .25, 1., .1)),
                         (32, 8), {(0, 0, 0): 1})
        with TemporaryDirectory() as folder:
            path = Path(folder) / "world.vxb"
            save_vxb(path, world)
            data, materials, _, _, _ = load_vxb_volume(path, density=.22)
        emission = np.zeros((65536, 3), dtype=np.float32)
        emission[1] = (.1, .2, .3)
        intensity = (2., 3., 4.)
        sun = ((0, 0, 1), intensity)
        scene = SparseSmokeScene(
            data, materials=materials, emission=emission, directional_light=sun,
            index_of_refraction=np.ones(65536), caustic_photon_density=6,
        )
        scene.update_light()
        baseline = scene._uniform_light_cache.to_numpy()
        with patch("taichi_volume_renderer.caustics._gather_hits",
                   return_value=[(np.array([[0, 12], [0, 13]]), np.array([1., 3.]), np.pi)]) as trace:
            scene.update_caustics()
            first = scene._uniform_light_cache.to_numpy()
            scene.update_light()
            np.testing.assert_array_equal(scene._uniform_light_cache.to_numpy(), baseline)
            scene.update_caustics()
            np.testing.assert_array_equal(scene._uniform_light_cache.to_numpy(), first)
            self.assertEqual(trace.call_count, 2)
        self.assertEqual(scene._mixed_light_count, 0)
        np.testing.assert_allclose(first[0] - baseline[0],
                                   4. / 32**3 * np.asarray(intensity) * materials[1, :3], atol=1e-7)

    def test_native_lookup_uses_file_levels_for_leaves_and_gaps(self):
        ti.init(arch=ti.cpu)
        self.addCleanup(ti.reset)
        fields = (MaterialField("density", "f32"),)
        world = VxbScene(fields, ((0.,), (.1,)), (64, 16, 8),
                         {(-1, 0, 0): {0: 1, 1: {0: (0,) * 511 + (1,)}}})
        with TemporaryDirectory() as folder:
            path = Path(folder) / "world.vxb"
            save_vxb(path, world)
            scene = SparseSmokeScene.from_file(path, lighting=False)
        output = ti.Vector.field(3, dtype=ti.f32, shape=4)

        @ti.kernel
        def sample():
            for ray in range(4):
                x = ti.Vector([-63.5, -40.5, -39.5, -31.5])[ray]
                advance, row, index, _ = scene._lookup(ti.Vector([x, 7.5, 7.5]),
                                                      ti.Vector([1., 0., 0.]), 0., 1e30, 0)
                output[ray] = ti.Vector([advance, ti.cast(row, ti.f32), ti.cast(index, ti.f32)])

        sample()
        np.testing.assert_array_equal(output.to_numpy(), [[0, 1, 0], [0, 0, 511],
                                                          [7.5, -1, 0], [15.5, -1, 0]])

    def test_ground_edit_preserves_direct_vxb_uniform_leaf_size(self):
        fields = (MaterialField("density", "f32"),)
        world = VxbScene(fields, None, (32, 8), {(-1, 0, 0): (.1,)})
        with TemporaryDirectory() as folder:
            path = Path(folder) / "world.vxb"
            save_vxb(path, world)
            data, _, _, _, _ = load_vxb_volume(path, density=.22)
        edited = _thicken_ground(data, 1)
        self.assertEqual(edited.hierarchy.sizes, (32, 8))
        self.assertEqual(len(edited.uniform_bricks), 1)
        self.assertEqual(edited.uniform_bricks[0][:4], (5, -1, 0, 0))
        self.assertEqual(len(edited.hierarchy.mixed_origins), 16)
        np.testing.assert_array_equal(edited.hierarchy.mixed_origins[:, 2], -8)

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
