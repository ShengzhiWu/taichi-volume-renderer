"""Render a vertical ascent looking straight down.

Install Pillow alongside the project.
Run from the repository root. Use --help for all settings.

    python -m scripts.animation_templates.render_flight_ascent path/to/scene.vxb path/to/output

"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import taichi as ti

from taichi_volume_renderer.math import rotation_matrix_to_quaternion
from taichi_volume_renderer.sparse_smoke import SparseSmokeScene
from taichi_volume_renderer.vxb_scene import load_vxb_volume


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", type=Path, help="Indexed VXB scene")
    parser.add_argument("output_root", type=Path, help="Parent of the numbered JPEG directory")
    parser.add_argument("--frames", type=int, default=600, help="Number of frames")
    parser.add_argument("--resolution", type=int, nargs=2, default=(2560, 1440), metavar=("WIDTH", "HEIGHT"), help="Image size in pixels")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality from 1 to 100")
    parser.add_argument("--arch", choices=("cuda", "vulkan", "cpu", "gpu"), default="cuda", help="Taichi backend")
    parser.add_argument("--light-direction", type=float, nargs=3, default=(.5, -.5, np.sqrt(.5)), metavar=("X", "Y", "Z"), help="World-space direction toward the sun")
    parser.add_argument("--light-rgb", type=float, nargs=3, default=(8, 8, 8), metavar=("R", "G", "B"), help="Sun intensity in linear RGB")
    parser.add_argument("--ambient-rgb", type=float, nargs=3, default=(.16, .21, .30), metavar=("R", "G", "B"), help="Ambient intensity in linear RGB")
    parser.add_argument("--start-z", type=float, help="Absolute start Z in voxels, otherwise use the longest occupied center layer")
    parser.add_argument("--end-z", type=float, help="Absolute end Z in voxels, otherwise fit scene bounds")
    parser.add_argument("--ground-clearance", type=float, default=.5, help="Start height above detected ground in voxels")
    parser.add_argument("--framing-margin", type=float, default=.05, help="Extra framing fraction for automatic end Z")
    args = parser.parse_args(argv)
    if args.frames < 2 or min(args.resolution) < 1 or not 1 <= args.quality <= 100:
        parser.error("Specify at least two frames, positive dimensions, and JPEG quality from 1 to 100")
    if not np.isfinite([args.ground_clearance, args.framing_margin]).all() or min(args.ground_clearance, args.framing_margin) < 0:
        parser.error("Ground clearance and framing margin must be finite and nonnegative")
    for altitude in (args.start_z, args.end_z):
        if altitude is not None and not np.isfinite(altitude):
            parser.error("Start and end Z must be finite")
    return args


def center_ground(data, palette, x, y):
    sector_x, local_x = divmod(x, 128)
    sector_y, local_y = divmod(y, 128)
    brick_x, voxel_x = divmod(local_x, 8)
    brick_y, voxel_y = divmod(local_y, 8)
    levels = []
    for coord, table in zip(data.sectors, data.brick_ids):
        if (coord[0], coord[1]) != (sector_x, sector_y):
            continue
        for brick_z in range(16):
            brick = table[brick_x + 16 * (brick_y + 16 * brick_z)]
            if brick >= 0:
                codes = data.blocks[brick].reshape(8, 8, 8)[:, voxel_y, voxel_x]
                levels.extend(coord[2] * 128 + brick_z * 8 + np.flatnonzero(palette[codes, 3] > 0))
    occupied = np.sort(levels)
    if not occupied.size:
        raise ValueError("No occupied voxels under the scene center")
    layers = np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1)
    return float(max(layers, key=len)[-1] + 1)


def numbered_directory(root):
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name) for path in root.iterdir()
               if path.name.isascii() and path.name.isdecimal()]
    output = root / str(max(numbers, default=0) + 1)
    output.mkdir()
    return output


def main():
    args = parse_args()
    count = args.frames
    width, height = args.resolution

    ti.init(arch=getattr(ti, args.arch))
    data, palette, ior, normals, emission = load_vxb_volume(args.model, density=.22)
    if palette is None:
        raise ValueError("Expected an indexed VXB scene")
    entity = data.entities[0]
    center = (entity.lower + entity.upper) / 2
    extent = entity.upper - entity.lower
    start_z = args.start_z
    if start_z is None:
        ground = center_ground(data, palette, *np.floor(center[:2]).astype(int))
        start_z = ground + args.ground_clearance
    scene = SparseSmokeScene(
        data, materials=palette, index_of_refraction=ior, interface_normals=normals, emission=emission,
        directional_light=(args.light_direction, args.light_rgb),
        ambient_light=args.ambient_rgb,
    )
    del data
    pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
    scene.update_light()

    end_z = args.end_z
    if end_z is None:
        half_view = np.tan(np.deg2rad(scene.get_vertical_field_of_view()) / 2)
        half_extent = max(extent[0] / 2, extent[1] * height / (2 * width))
        end_z = entity.upper[2] + (1 + args.framing_margin) * half_extent / half_view
    if end_z <= start_z:
        raise ValueError("End Z must exceed start Z for an ascent")
    positions = np.tile(center, (count, 1))
    positions[:, 2] = np.linspace(start_z, end_z, count)
    # With +Y to camera right, world X falls along the vertical image axis.
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    quaternion = rotation_matrix_to_quaternion(rotation)[[1, 2, 3, 0]]
    rotations = np.tile(quaternion, (count, 1))

    output = numbered_directory(args.output_root)
    for index, (position, rotation) in enumerate(zip(positions, rotations), start=1):
        scene.set_camera_pose({"position": position.tolist(), "rotation": rotation.tolist()})
        scene.render(pixels)
        rgb = pixels.to_numpy().transpose(1, 0, 2)[::-1]
        Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255 + .5)).save(
            output / f"frame_{index:03d}.jpg", quality=args.quality, subsampling=0,
        )
        if index % 200 == 0:
            print(f"Saved {index}/{count} frames.", flush=True)
    print(f"Saved {count} JPEG frames to {output.resolve()}")


if __name__ == "__main__":
    main()
