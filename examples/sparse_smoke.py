"""Render a CXW, VXB, or OpenVDB scene with the sparse smoke ray marcher."""

import argparse
import logging
from pathlib import Path

import numpy as np
import taichi as ti

from taichi_volume_renderer import DisplayWindow
from taichi_volume_renderer.caustics import DEFAULT_PHOTON_DENSITY
from taichi_volume_renderer.cxw import packed_color_materials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a packed-color CXW v7/v8, VXB, or OpenVDB file")
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
    parser.add_argument("--thicken-ground", type=int, default=0, metavar="N",
                        help="Extend the lowest non-air voxel in each vertical column downward by N voxels")
    caustics = parser.add_mutually_exclusive_group()
    caustics.add_argument("--caustic-photon-density", type=float, nargs="?", const=DEFAULT_PHOTON_DENSITY, metavar="N",
                          help=f"Photons per projected square voxel for VXB caustics (default {DEFAULT_PHOTON_DENSITY})")
    caustics.add_argument("--no-caustics", action="store_true", help="Skip load-time VXB caustic baking")
    parser.add_argument("--refractive-shadow-mix", default="auto", metavar="auto|N",
                        help="Blend opaque refractive shadows (0) with ordinary shadows (1). "
                             "Auto uses 0 if caustics are baked at load time and 1 otherwise.")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Apply x^gamma to each loaded RGB component (default: 1)")
    parser.add_argument("--refraction-mode", choices=("gradient", "interface"),
                        help="Select VXB refraction mode or enable refraction for packed CXW water (ID 3)")
    parser.add_argument("--step", type=float, default=1.0, help="Camera ray step in voxel units")
    parser.add_argument("--max-steps", type=int,
                        help="Camera ray step limit (default: 10000 with refraction, unlimited otherwise)")
    lighting = parser.add_mutually_exclusive_group()
    lighting.add_argument("--lighting", dest="lighting", action="store_true", help="Enable lighting (default)")
    lighting.add_argument("--no-lighting", dest="lighting", action="store_false", help="Use uniform lighting")
    parser.set_defaults(lighting=True)
    parser.add_argument("--light-type", choices=("directional", "point"), default="directional",
                        help="Light source used by the viewer (default: directional)")
    parser.add_argument("--ambient-light", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                        metavar=("R", "G", "B"), help="Unshadowed ambient RGB intensity (default: 0 0 0)")
    parser.add_argument("--fly", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.no_caustics:
        logging.basicConfig(level=logging.INFO)
    ggui = args.ggui if args.ggui is not None else args.arch in (None, "vulkan")
    if ggui and args.arch not in (None, "vulkan"):
        parser.error("--ggui requires --arch vulkan")
    arch = args.arch or ("vulkan" if ggui else "gpu")
    optics = {}
    if args.refraction_mode is not None:
        optics["refraction_mode"] = args.refraction_mode
        suffix = Path(args.path).suffix.lower()
        if suffix == ".cxw":
            indices = np.ones(65536, dtype=np.float32)
            indices[3] = 1.33
            materials = packed_color_materials(args.density)
            materials[3, 3] = 0.0024
            optics.update(materials=materials, index_of_refraction=indices)
        elif suffix != ".vxb":
            parser.error("--refraction-mode applies to CXW and VXB files")
    try:
        window = DisplayWindow.from_file(
            args.path, resolution=args.resolution, taichi_arch=getattr(ti, arch),
            camera=args.camera, density=args.density, gamma=args.gamma, storage=args.storage,
            grid_name=args.grid, step_length=args.step, max_steps=args.max_steps,
            caustics=not args.no_caustics, caustic_photon_density=args.caustic_photon_density,
            refractive_shadow_mix=args.refractive_shadow_mix,
            thicken_ground=args.thicken_ground, lighting=args.lighting, ambient_light=args.ambient_light,
            # Match a point light of intensity 200 at a distance of 5 from the scene center.
            directional_light=((0, 0, 1), (8, 8, 8)) if args.light_type == "directional" else None,
            **optics,
        )
    except (ValueError, ImportError, OSError) as exc:
        parser.error(str(exc))
    window.show(title="Sparse smoke | Drag to look | WASD move | Q/E height | R reset",
                resize=True, ggui=ggui)


if __name__ == "__main__":
    main()
