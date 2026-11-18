"""Render a packed-color CXW world with the sparse smoke ray marcher."""

import argparse
import json
from time import perf_counter

import taichi as ti

from taichi_volume_renderer import DisplayWindow


class CXWViewer(DisplayWindow):
    def show_controls(self, camera, fly):
        self.fly = fly
        self.initial_camera = camera
        self._tab_pressed = False
        self._reset_pressed = False
        self._last_frame = perf_counter()
        self.show(title="Sparse smoke | Drag to look | WASD move | Q/E height | Tab orbit/fly | R reset",
                  resize=True, context_menu=(("复制视角", self._copy_view), ("粘贴视角", self._paste_view)))

    def _copy_view(self, window):
        window.copy_text(json.dumps(self.scene.get_camera_pose()))

    def _paste_view(self, window):
        try:
            self.scene.set_camera_pose(json.loads(window.read_text()))
        except (ValueError, TypeError, OverflowError):
            window.show_error("粘贴视角", "剪贴板中没有有效的视角 JSON。")

    def mouse_drag_event(self, pos, pos_delta):
        if self.fly:
            self.scene.turn_camera(-pos_delta[0] * self.camera_rotation_speed,
                                   -pos_delta[1] * self.camera_rotation_speed)
        else:
            super().mouse_drag_event(pos, pos_delta)

    def mouse_wheel_event(self, delta):
        scroll = 2 * delta[1]
        if self.fly:
            self.scene.move_camera(0.025 * scroll, 0, 0)
        else:
            super().mouse_wheel_event((delta[0], scroll))

    def _handle_keyboard(self, gui):
        now = perf_counter()
        duration = min(now - self._last_frame, 0.1)
        self._last_frame = now
        if gui.is_pressed(ti.GUI.ESCAPE):
            gui.running = False
            return
        tab = gui.is_pressed(ti.GUI.TAB)
        if tab and not self._tab_pressed:
            self.fly = not self.fly
        self._tab_pressed = tab
        reset = gui.is_pressed("r")
        if reset and not self._reset_pressed:
            self.scene.set_view(self.initial_camera)
        self._reset_pressed = reset

        forward = int(gui.is_pressed("w")) - int(gui.is_pressed("s"))
        right = int(gui.is_pressed("d")) - int(gui.is_pressed("a"))
        up = int(gui.is_pressed("e")) - int(gui.is_pressed("q"))
        if forward or right or up:
            speed = 640.0 if gui.is_pressed(ti.GUI.SHIFT) else 120.0
            distance = speed * duration
            self.scene.move_camera(2.5 * forward * distance, right * distance, up * distance)


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
    lighting = parser.add_mutually_exclusive_group()
    lighting.add_argument("--lighting", dest="lighting", action="store_true", help="Enable lighting (default)")
    lighting.add_argument("--no-lighting", dest="lighting", action="store_false", help="Use uniform lighting")
    parser.set_defaults(lighting=True)
    parser.add_argument("--light-type", choices=("directional", "point"), default="directional",
                        help="Light source used by the viewer (default: directional)")
    parser.add_argument("--fly", action="store_true", help="Start with first-person mouse controls")
    args = parser.parse_args()
    window = CXWViewer.from_cxw(
        args.path, resolution=args.resolution, taichi_arch=getattr(ti, args.arch),
        camera=args.camera, density=args.density, gamma=args.gamma,
        step_length=args.step, lighting=args.lighting,
        # Match a point light of intensity 200 at a distance of 5 from the scene center.
        directional_light=((0, 1, 0), (8, 8, 8)) if args.light_type == "directional" else None,
    )
    window.show_controls(args.camera, args.fly)


if __name__ == "__main__":
    main()
