import numpy as np
import taichi as ti
from taichi_volume_renderer import plot_volume
import taichi_volume_renderer.canvas as canvas

ti.init(arch=ti.gpu)

# print(ti.Vector(ti.Vector([1, 2, 3])))

N = 100
smoke, smoke_color=canvas.empty_canvas(N)

canvas.fill_disk(smoke, smoke_color, [70, 70, 70], 20, 5, [1, 0, 0])
canvas.fill_platonic_solid(smoke, smoke_color, [50, 50, 50], 30, 12, 10, [0, 0.5, 1])
canvas.fill_rectangle(smoke, smoke_color, [10, 10, 60], [30, 30, 30], 1, [1, 1, 0])
canvas.draw_line_simple(smoke, smoke_color, [90, 90, 90], [10, 10, 10], 80, [0, 0, 0])
canvas.draw_helix(smoke, smoke_color, [50, 50, 0], [50, 50, 100], 20, 5, 80, [0, 0, 0])

plot_volume(
    smoke_density=smoke,
    smoke_color=smoke_color,
    init_taichi=False,
    light_ray_tracing_step_size_factor=1)
