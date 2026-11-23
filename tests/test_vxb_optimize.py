import unittest

from taichi_volume_renderer.vxb import MaterialField, VxbScene
from taichi_volume_renderer.vxb_optimize import merge_hidden_materials


class HiddenMaterialsTest(unittest.TestCase):
    def test_hidden_voxel_frequency_and_merging_across_brick_boundaries(self):
        fields = (MaterialField("density", "f32"), MaterialField("color", "i32"))
        materials = ((0., 0), (1., 1), (1., 2))
        center = (1, 2, 2, 2) * 128
        # Surround the center on faces, edges, and corners with opaque stone.
        bricks = {(x, y, z): 1 for x in range(-1, 2) for y in range(-1, 2) for z in range(-1, 2)}
        bricks[(0, 0, 0)] = center
        scene = VxbScene(fields, materials, (8,), bricks)
        result = merge_hidden_materials(scene, [1, 2])
        self.assertEqual(result.bricks[(0, 0, 0)], 1)
        self.assertEqual(scene.bricks[(0, 0, 0)], center)
        self.assertIs(merge_hidden_materials(scene, [1, 2], enabled=False), scene)

        # Let a ray reach the corner through a missing diagonal brick.
        del bricks[(-1, -1, -1)]
        center = (2,) + (1,) * 511
        bricks[(0, 0, 0)] = center
        self.assertEqual(merge_hidden_materials(scene, [1, 2]).bricks[(0, 0, 0)], center)

    def test_actual_density_and_strict_threshold_in_direct_mode(self):
        fields = (MaterialField("density", "f32"), MaterialField("color", "i32"))
        plain, fleck, other = (.5, 1), (.5, 2), (1., 3)
        values = [plain] * 512
        depth_one = 1 + 8 * (4 + 8 * 4)
        depth_two = 2 + 8 * (4 + 8 * 4)
        values[depth_one] = values[depth_two] = fleck
        values[0] = other
        scene = VxbScene(fields, None, (8,), {(0, 0, 0): tuple(values)})
        result = merge_hidden_materials(scene, [plain, fleck], transmission_threshold=.5)
        updated = result.bricks[(0, 0, 0)]
        self.assertEqual(updated[depth_one], fleck)
        self.assertEqual(updated[depth_two], plain)
        self.assertEqual(updated[0], other)
        self.assertIs(merge_hidden_materials(scene, [plain, fleck]), scene)
