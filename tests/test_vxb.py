import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib


# Load the same single-file codec a standalone consumer would use.
_CODEC_PATH = Path(__file__).resolve().parents[1] / "taichi_volume_renderer" / "vxb.py"
_SPEC = importlib.util.spec_from_file_location("vxb", _CODEC_PATH)
vxb = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = vxb
_SPEC.loader.exec_module(vxb)
_FIXTURE = Path(__file__).with_name("fixtures") / "vxb-v1.vxb"


class VxbTest(unittest.TestCase):
    def test_reference_file_and_round_trip(self):
        scene = vxb.load_vxb(_FIXTURE)
        self.assertEqual(scene.brick_sizes, (128, 32, 8))
        self.assertEqual(scene.camera, vxb.CameraPose((1.25, -2., 3.), (0., 0., 0., 1.)))
        self.assertEqual(scene.fields[-1], vxb.MaterialField("\u6807\u8bb0", "i32"))
        self.assertEqual(scene.materials[2], (.25, .5, .75, .125, 1.25, True, 12))
        self.assertIs(scene.materials[4][5], False)
        self.assertEqual(scene.bricks, {
            (-1, 0, 0): 1,
            (0, 0, 0): {0: 2, 1: {0: (1, 2, 3, 4) + (0,) * 508}},
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "round-trip.vxb"
            vxb.save_vxb(path, scene)
            self.assertEqual(vxb.load_vxb(path), scene)

    def test_variable_index_width_and_no_camera(self):
        fields = (vxb.MaterialField("density", "f32"),)
        materials = ((0.,),) + ((.125,),) * 299
        scene = vxb.VxbScene(fields, materials, (128, 8), {
            (1, -1, 0): {4095: (0, 8, 299, 256) * 128},
            (1, 2, 3): 299,
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wide.vxb"
            vxb.save_vxb(path, scene)
            self.assertEqual(vxb.load_vxb(path), scene)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                vxb.load_vxb(path)

    def test_air_only_and_failed_replacement(self):
        fields = (vxb.MaterialField("density", "f32"),)
        scene = vxb.VxbScene(fields, ((0.,),), (8,), {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.vxb"
            vxb.save_vxb(path, scene)
            self.assertEqual(vxb.load_vxb(path), scene)
            invalid = vxb.VxbScene(fields, ((0.,), (1.,)), (8,), {(0, 0, 0): (99, 0) * 256})
            with self.assertRaisesRegex(ValueError, "index"):
                vxb.save_vxb(path, invalid)
            self.assertEqual(vxb.load_vxb(path), scene)

    def test_direct_material_rows_without_a_table(self):
        fields = tuple(vxb.MaterialField(name, kind) for name, kind in (
            ("density", "f32"), ("use_voxel_face_normal", "bool"), ("region", "i32"),
        ))
        uniform = (.25, True, -2)
        rows = tuple((index / 512, bool(index % 2), index) for index in range(512))
        scene = vxb.VxbScene(fields, None, (128, 8), {(-1, 0, 0): uniform, (0, 0, 0): {0: rows}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "direct.vxb"
            vxb.save_vxb(path, scene)
            self.assertEqual(vxb.load_vxb(path), scene)
            encoded = path.read_bytes()
            self.assertEqual(struct.unpack_from("<4sHBBIHI", encoded), (b"VXB\0", 1, 2, 2, 0, 3, 2))
            first_entry = 18 + 2 + sum(3 + len(field.name) for field in fields)
            payloads = []
            for entry in (first_entry, first_entry + 28):
                _, _, _, offset, size, raw_size = struct.unpack_from("<3iQII", encoded, entry)
                raw = zlib.decompress(encoded[offset:offset + size], wbits=-15)
                self.assertEqual(len(raw), raw_size)
                payloads.append(raw)
            self.assertEqual(payloads[0], b"\x00" + struct.pack("<fBi", *uniform))
            self.assertEqual(payloads[1], b"\x01" + struct.pack("<II", 1, 0) + b"\x02"
                             + b"".join(struct.pack("<fBi", *row) for row in rows))


if __name__ == "__main__":
    unittest.main()
