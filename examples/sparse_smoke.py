"""Render a CXW or OpenVDB scene with the sparse smoke ray marcher."""

import argparse
from pathlib import Path

import numpy as np
import taichi as ti

from taichi_volume_renderer import DisplayWindow
from taichi_volume_renderer.cxw import packed_color_materials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a packed-color CXW v7/v8 or OpenVDB file")
    parser.add_argument("--grid", help="Density grid name for VDB input")
    parser.add_argument("--camera", choices=("overview", "saved"), default="overview")
    parser.add_argument("--arch", choices=("gpu", "cuda", "vulkan", "cpu"))
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--ggui", dest="ggui", action="store_true",
                         help="Display GPU textures in a Vulkan GGUI window (default)")
    display.add_argument("--no-ggui", dest="ggui", action="store_false",
                         help="Use the standard window (Tk on Windows)")
    parser.set_defaults(ggui=None)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720),
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--density", type=float, default=0.22, help="Density assigned to occupied CXW voxels")
    parser.add_argument("--storage", choices=("auto", "indexed", "direct"), default="auto",
                        help="Choose storage from the file (default), or force indexed or direct values")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Apply x^gamma to each loaded RGB component (default: 1)")
    parser.add_argument("--refraction-mode", choices=("gradient", "interface"),
                        help="Enable refraction for packed CXW water (palette ID 3)")
    parser.add_argument("--step", type=float, default=1.0, help="Camera ray step in voxel units")
    parser.add_argument("--max-steps", type=int,
                        help="Camera ray step limit (default: 4600 for interface, 10000 for gradient)")
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
    ggui = args.ggui if args.ggui is not None else args.arch in (None, "vulkan")
    if ggui and args.arch not in (None, "vulkan"):
        parser.error("--ggui requires --arch vulkan")
    arch = args.arch or ("vulkan" if ggui else "gpu")
    optics = {}
    if args.refraction_mode is not None:
        if Path(args.path).suffix.lower() != ".cxw":
            parser.error("--refraction-mode is for packed CXW water")
        indices = np.ones(65536, dtype=np.float32)
        indices[3] = 1.33
        materials = packed_color_materials(args.density)
        materials[3, 3] = 0.0024
        optics = dict(materials=materials, index_of_refraction=indices, refraction_mode=args.refraction_mode)
    try:
        window = DisplayWindow.from_file(
            args.path, resolution=args.resolution, taichi_arch=getattr(ti, arch),
            camera=args.camera, density=args.density, gamma=args.gamma, storage=args.storage,
            grid_name=args.grid, step_length=args.step, max_steps=args.max_steps,
            lighting=args.lighting, ambient_light=args.ambient_light,
            # Match a point light of intensity 200 at a distance of 5 from the scene center.
            directional_light=((0, 1, 0), (8, 8, 8)) if args.light_type == "directional" else None,
            **optics,
        )
    except (ValueError, ImportError, OSError) as exc:
        parser.error(str(exc))
    window.show(title="Sparse smoke | Drag to look | WASD move | Q/E height | Tab orbit/fly | R reset",
                resize=True, fly=args.fly, ggui=ggui)


if __name__ == "__main__":
    main()
