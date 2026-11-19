"""Render a packed-color CXW world with the sparse smoke ray marcher."""

import argparse

import taichi as ti

from taichi_volume_renderer import DisplayWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a packed-color CXW v7/v8 file")
    parser.add_argument("--camera", choices=("overview", "saved"), default="overview")
    parser.add_argument("--arch", choices=("gpu", "cuda", "vulkan", "cpu"), default="gpu")
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720),
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--density", type=float, default=0.22, help="Density per occupied voxel")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Apply x^gamma to each loaded RGB component (default: 1)")
    parser.add_argument("--step", type=float, default=1.0, help="Camera ray step in voxel units")
    parser.add_argument("--max-steps", type=int, help="Optional camera ray step limit (default: unlimited)")
    lighting = parser.add_mutually_exclusive_group()
    lighting.add_argument("--lighting", dest="lighting", action="store_true", help="Enable lighting (default)")
    lighting.add_argument("--no-lighting", dest="lighting", action="store_false", help="Use uniform lighting")
    parser.set_defaults(lighting=True)
    parser.add_argument("--light-type", choices=("directional", "point"), default="directional",
                        help="Light source used by the viewer (default: directional)")
    parser.add_argument("--ambient-light", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                        metavar=("R", "G", "B"), help="Unshadowed ambient RGB intensity (default: 0 0 0)")
    parser.add_argument("--fly", action="store_true", help="Start with first-person mouse controls")
    args = parser.parse_args()
    window = DisplayWindow.from_cxw(
        args.path, resolution=args.resolution, taichi_arch=getattr(ti, args.arch),
        camera=args.camera, density=args.density, gamma=args.gamma,
        step_length=args.step, max_steps=args.max_steps,
        lighting=args.lighting, ambient_light=args.ambient_light,
        # Match a point light of intensity 200 at a distance of 5 from the scene center.
        directional_light=((0, 1, 0), (8, 8, 8)) if args.light_type == "directional" else None,
    )
    window.show(title="Sparse smoke | Drag to look | WASD move | Q/E height | Tab orbit/fly | R reset",
                resize=True, fly=args.fly)


if __name__ == "__main__":
    main()
