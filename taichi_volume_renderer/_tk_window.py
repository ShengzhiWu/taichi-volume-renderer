from time import perf_counter, sleep
from types import SimpleNamespace
import tkinter as tk

import numpy as np
import taichi as ti


class Window:
    """Present Taichi pixels using Tk's native Windows window and input handling."""

    _KEYS = {"Escape": ti.GUI.ESCAPE, "Tab": ti.GUI.TAB, "ISO_Left_Tab": ti.GUI.TAB}

    def __init__(self, title, resolution, resize=False):
        self._root = tk.Tk()
        self._root.title(title)
        self._title = title
        width, height = resolution
        self._header = f"P6\n{width} {height}\n255\n".encode("ascii")
        self._image = tk.PhotoImage(width=width, height=height, master=self._root)
        self._view = tk.Label(self._root, image=self._image, borderwidth=0, highlightthickness=0)
        self._size = tuple(resolution)
        self._window_size = self._size
        self._resize_deadline = 0.0
        if resize:
            self._root.geometry(f"{width}x{height}")
            self._view.place(relx=0.5, rely=0.5, anchor="center")
            self._root.bind("<Configure>", self._configure)
        else:
            self._view.pack(expand=True)
        self._root.configure(background="black")
        self._root.protocol("WM_DELETE_WINDOW", self._stop)
        self._root.bind("<KeyPress>", self._key_press)
        self._root.bind("<KeyRelease>", self._key_release)
        self._root.bind("<FocusOut>", self._clear_input)
        self._view.bind("<Motion>", self._move_pointer)
        self._view.bind("<ButtonPress-1>", self._mouse_press)
        self._view.bind("<ButtonRelease-1>", self._mouse_release)
        self._root.bind("<MouseWheel>", self._scroll)

        self.running = True
        self._keys = set()
        self._cursor = (0.0, 0.0)
        self._wheel = []
        self._frame_start = self._fps_start = perf_counter()
        self._frames = 0

    def _stop(self):
        self.running = False

    def _clear_input(self, event):
        self._keys.clear()
        self._wheel.clear()

    def _key_press(self, event):
        self._keys.add(self._KEYS.get(event.keysym, event.keysym.lower()))

    def _key_release(self, event):
        self._keys.discard(self._KEYS.get(event.keysym, event.keysym.lower()))

    def _move_pointer(self, event):
        width, height = self._size
        self._cursor = (event.x / width, 1.0 - event.y / height)

    def _mouse_press(self, event):
        self._root.focus_set()
        self._move_pointer(event)
        self._keys.add(ti.GUI.LMB)

    def _mouse_release(self, event):
        self._move_pointer(event)
        self._keys.discard(ti.GUI.LMB)

    def _scroll(self, event):
        self._wheel.append(SimpleNamespace(delta=(0, event.delta)))

    def poll_events(self):
        self._frame_start = perf_counter()
        self._root.update()

    def _configure(self, event):
        size = (event.width, event.height)
        if event.widget == self._root and min(size) > 1 and size != self._window_size:
            self._window_size = size
            self._resize_deadline = perf_counter() + 0.15

    def take_resize(self):
        if self._window_size != self._size and perf_counter() >= self._resize_deadline:
            return self._window_size
        return None

    def get_events(self, key):
        if key == ti.GUI.WHEEL:
            events, self._wheel = self._wheel, []
            yield from events

    def is_pressed(self, key):
        if key == ti.GUI.SHIFT:
            return "shift_l" in self._keys or "shift_r" in self._keys
        return key in self._keys

    def get_cursor_pos(self):
        return self._cursor

    def set_image(self, pixels):
        size = tuple(pixels.shape)
        if size != self._size:
            self._size = size
            width, height = size
            self._header = f"P6\n{width} {height}\n255\n".encode("ascii")
            self._image.configure(width=width, height=height)
        rgb = pixels.to_numpy().transpose(1, 0, 2)[::-1]
        np.clip(rgb, 0.0, 1.0, out=rgb)
        rgb *= 255.0
        self._image.configure(format="PPM", data=self._header + rgb.astype(np.uint8).tobytes())

    def show(self):
        self._root.update_idletasks()
        now = perf_counter()
        self._frames += 1
        elapsed = now - self._fps_start
        if elapsed >= 1.0:
            self._root.title(f"{self._title} ({self._frames / elapsed:.1f} FPS)")
            self._fps_start = now
            self._frames = 0
        sleep(max(0.0, 1.0 / 60.0 - (now - self._frame_start)))

    def close(self):
        self._image = None
        self._root.destroy()
