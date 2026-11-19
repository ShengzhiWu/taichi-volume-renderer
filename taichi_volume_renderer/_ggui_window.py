from time import perf_counter

import taichi as ti


@ti.kernel
def _write_texture(pixels: ti.types.ndarray(dtype=ti.math.vec3, ndim=2),
                   image: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8, lod=0)):
    for i, j in pixels:
        rgb = ti.math.clamp(pixels[i, j], 0.0, 1.0)
        image.store(ti.Vector([i, j]), ti.Vector([rgb.x, rgb.y, rgb.z, 1.0]))


class Window:
    """Display a Taichi image in a Vulkan window without CPU readback."""

    def __init__(self, title, resolution):
        self._window = ti.ui.Window(title, resolution, vsync=False, fps_limit=120)
        self._canvas = self._window.get_canvas()
        self._texture = None
        self._texture_size = None
        self._window_size = tuple(resolution)
        self._resize_deadline = 0.0
        self.is_pressed = self._window.is_pressed
        self.get_cursor_pos = self._window.get_cursor_pos
        self.show = self._window.show
        self.close = self._window.destroy

    @property
    def running(self):
        return self._window.running

    @running.setter
    def running(self, value):
        self._window.running = value

    def poll_events(self):
        self._window.get_events()
        size = tuple(self._window.get_window_shape())
        if min(size) > 1 and size != self._window_size:
            self._window_size = size
            self._resize_deadline = perf_counter() + 0.15

    def take_resize(self):
        if perf_counter() >= self._resize_deadline:
            return self._window_size
        return None

    def get_events(self, key):
        # GGUI's Python event API does not expose wheel events.
        return ()

    def set_image(self, pixels):
        size = tuple(pixels.shape)
        if size != self._texture_size:
            self._texture = ti.Texture(ti.Format.rgba8, size)
            self._texture_size = size
        _write_texture(pixels, self._texture)
        self._canvas.set_image(self._texture)
