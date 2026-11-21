"""Convert a single untransformed packed-color CXW entity to a VXB world grid."""

import argparse
from pathlib import Path

import numpy as np

from taichi_volume_renderer.cxw import load_cxw, packed_color_materials
from taichi_volume_renderer.math import rotation_matrix_to_quaternion
from taichi_volume_renderer.vxb import CameraPose, MaterialField, VxbScene, save_vxb


def to_vxb(data):
    # The CXW importer represents the Y-up to Z-up conversion on the entity transform.
    if (len(data.entities) != 1
            or np.any(data.entities[0].position != 0.)
            or not np.array_equal(data.entities[0].rotation, np.eye(3)[[0, 2, 1]])):
        raise ValueError("VXB conversion requires one untransformed occupied CXW entity")

    counts = np.zeros(65536, dtype=np.int64)
    for first in range(0, len(data.blocks), 8192):
        counts += np.bincount(data.blocks[first:first + 8192].ravel(), minlength=len(counts))
    codes = np.union1d(np.flatnonzero(counts), [0])
    rgba = packed_color_materials(.22)
    fields = tuple(MaterialField(name, "f32") for name in
                   ("r", "g", "b", "density", "emission_r", "emission_g", "emission_b"))
    fields += (MaterialField("cxw_code", "i32"),)
    materials = []
    indices = [0] * 65536
    for index, code in enumerate(codes.tolist()):
        indices[code] = index
        emission = (0., 0., 0.)
        if code & 0xc000 == 0x4000:
            emission = rgba[code, :3] * (.1 * 8**(code & 3))
        materials.append((*rgba[code], *emission, code))

    bricks = {}
    for (x, y, z), table in zip(data.sectors.tolist(), data.brick_ids):
        children = {}
        for slot in np.flatnonzero(table >= 0).tolist():
            local_x, local_y, local_z = slot % 16, slot // 16 % 16, slot // 256
            world_slot = local_x + 16 * (local_z + 16 * local_y)
            block = data.blocks[table[slot]]
            if np.all(block == block[0]):
                children[world_slot] = indices[block[0]]
            else:
                world_order = block.reshape(8, 8, 8).transpose(1, 0, 2).ravel()
                children[world_slot] = tuple(map(indices.__getitem__, world_order.tolist()))
        bricks[(x, z, y)] = children

    camera = None
    if data.player_pose is not None:
        position, rotation = data.player_pose
        # The CXW camera looks along +Z; the VXB camera looks along -Z.
        rotation = rotation.copy()
        rotation[:, 2] *= -1
        quaternion = rotation_matrix_to_quaternion(rotation)
        camera = CameraPose(tuple(position), tuple(quaternion[[1, 2, 3, 0]]))
    return VxbScene(fields, tuple(materials), (128, 8), bricks, camera)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Packed-color CXW v7/v8 file")
    parser.add_argument("output", type=Path, help="Destination VXB file")
    args = parser.parse_args()
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("Input and output must be different files")
        data = load_cxw(args.input)
        scene = to_vxb(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_vxb(args.output, scene)
        before, after = args.input.stat().st_size, args.output.stat().st_size
    except (ValueError, OSError) as error:
        parser.error(str(error))

    print(f"Converted {len(data.blocks):,} bricks and {len(scene.materials):,} material rows.")
    print(f"CXW {before:,} bytes, VXB {after:,} bytes ({(after / before - 1):+.1%}).")


if __name__ == "__main__":
    main()
