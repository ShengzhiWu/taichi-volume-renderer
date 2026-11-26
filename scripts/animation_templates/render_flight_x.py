"""Render a straight flight along the positive X axis.

Install Pillow and SciPy alongside the project.
Run from the repository root. Use --help for all settings.

    python -m scripts.animation_templates.render_flight_x path/to/scene.vxb path/to/output

"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
import taichi as ti

from taichi_volume_renderer.math import rotation_matrix_to_quaternion
from taichi_volume_renderer import DisplayWindow


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
    parser.add_argument("--z-range", type=float, nargs=2, default=(470., 520.), metavar=("MIN", "MAX"), help="Absolute camera Z range in voxel units")
    parser.add_argument("--pitch", type=float, default=18., help="Baseline downward pitch in degrees")
    parser.add_argument("--yaw-offset", type=float, default=0.0, help="Clockwise yaw from the forward tangent in degrees")
    parser.add_argument("--position-jitter", type=float, default=15., help="Maximum XY control-point offset per axis in voxels")
    parser.add_argument("--yaw-jitter", type=float, default=2., help="Maximum yaw control-point offset in degrees")
    parser.add_argument("--pitch-jitter", type=float, default=1., help="Maximum pitch control-point offset in degrees")
    parser.add_argument("--height-controls", type=int, default=6, help="Number of height control points")
    parser.add_argument("--motion-controls", type=int, default=20, help="Number of position and angle control points")
    parser.add_argument("--filter-radius", type=float, default=.1, help="Gaussian radius as a fraction of frames, 0 disables it")
    parser.add_argument("--seed", type=int, default=20261121, help="Random seed for control points")
    parser.add_argument("--start-back", type=float, default=1/3, help="Fraction of X span to subtract from the minimum bound")
    parser.add_argument("--end-back", type=float, default=1/6, help="Fraction of X span to subtract from the maximum bound")
    args = parser.parse_args(argv)
    if args.frames < 2 or min(args.resolution) < 1 or not 1 <= args.quality <= 100:
        parser.error("Specify at least two frames, positive dimensions, and JPEG quality from 1 to 100")
    controls = (args.height_controls, args.motion_controls)
    jitter = (args.position_jitter, args.yaw_jitter, args.pitch_jitter)
    if not np.isfinite([*args.z_range, *jitter, args.pitch, args.yaw_offset, args.filter_radius]).all():
        parser.error("Camera settings must be finite")
    if args.z_range[0] > args.z_range[1] or min(jitter) < 0 or not 0 <= args.filter_radius <= 1:
        parser.error("Use an ordered Z range, nonnegative jitter, and a filter radius from 0 to 1")
    if not -90 <= args.pitch - args.pitch_jitter <= args.pitch + args.pitch_jitter <= 90:
        parser.error("Pitch and its jitter must stay within -90 to 90 degrees")
    if controls[0] < 2 or controls[1] < 2 or args.seed < 0:
        parser.error("Specify at least two height controls, two motion controls, and a nonnegative seed")
    if not np.isfinite([args.start_back, args.end_back]).all() or args.end_back >= 1 + args.start_back:
        parser.error("Start and end offsets must describe a forward flight")
    return args


def smooth_controls(values, phases, radius_fraction):
    knots = np.linspace(0, 1, len(values))
    frames = np.column_stack([
        np.interp(phases, knots, axis) for axis in values.reshape(len(values), -1).T
    ]).reshape((len(phases), *values.shape[1:]))
    radius = int(len(phases) * radius_fraction)
    if radius == 0:
        return frames
    return gaussian_filter1d(
        frames, sigma=radius / 3, axis=0, mode="nearest", truncate=3,
    )


def flight(args, center, extent):
    count = args.frames
    phases = np.linspace(0, 1, count)
    rng = np.random.default_rng(args.seed)
    radius = args.filter_radius
    height_count, motion_count = args.height_controls, args.motion_controls
    heights = smooth_controls(rng.uniform(.85, 1.15, size=height_count) - 1, phases, radius)
    low_z, high_z = args.z_range
    height_span = np.ptp(heights)
    z = np.full(count, low_z)
    if height_span > 0:
        z += (high_z - low_z) * (heights - heights.min()) / height_span
    offsets = rng.uniform(-args.position_jitter, args.position_jitter, size=(motion_count, 2))
    yaw_offsets = np.deg2rad(rng.uniform(-args.yaw_jitter, args.yaw_jitter, size=motion_count))
    pitch_offsets = rng.uniform(-args.pitch_jitter, args.pitch_jitter, size=motion_count)
    pitch = smooth_controls(np.deg2rad(args.pitch + pitch_offsets), phases, radius)

    along, across = smooth_controls(offsets, phases, radius).T
    yaw = smooth_controls(yaw_offsets, phases, radius) - np.deg2rad(args.yaw_offset)
    x = center[0] + np.linspace(-.5 - args.start_back, .5 - args.end_back, count) * extent[0] + along
    y = center[1] + across
    xy = np.column_stack((x, y))
    return np.column_stack((xy, z)), yaw, pitch


def orientation(yaw, pitch):
    backward = np.array([-np.cos(pitch) * np.cos(yaw), -np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.])
    return rotation_matrix_to_quaternion(
        np.column_stack((right, np.cross(backward, right), backward)),
    )[[1, 2, 3, 0]]


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

    window = DisplayWindow.from_file(
        args.model, resolution=(width, height), taichi_arch=getattr(ti, args.arch),
        directional_light=(args.light_direction, args.light_rgb),
        ambient_light=args.ambient_rgb,
    )
    scene = window.scene
    scene.update_light()
    scene.update_caustics()
    pixels = window.pixels
    center = scene._center
    extent = scene._world_extent[None].to_numpy()
    positions, yaw, pitch = flight(args, center, extent)
    rotations = [orientation(heading, tilt) for heading, tilt in zip(yaw, pitch)]

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
