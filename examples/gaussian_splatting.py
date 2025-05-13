from plyfile import PlyData
import numpy as np
import taichi as ti
import taichi_volume_renderer
import taichi_volume_renderer.canvas as canvas
from taichi_volume_renderer.io import parse_gaussian_splatting_data

ti.init(arch=ti.gpu)

file_path = "D:/Gaussian_spot_test/your_gaussian_splatting_data_file.ply"
ply_data = PlyData.read(file_path)
data = parse_gaussian_splatting_data(ply_data)

N = 400
smoke, smoke_color = canvas.empty_canvas(N)

ranges = np.array([[np.percentile(e, 10), np.percentile(e, 90)] for e in data['positions'].T])
center = np.mean(ranges, axis=-1)
scaling = N / np.max(ranges[:, 1] - ranges[:, 0]) * 1.

data['positions'] -= center
canvas.gaussian_splatting(
    smoke,
    smoke_color,
    data,
    offset=0.5 * np.array(smoke.shape),
    y_upward=True,
    scaling=scaling)

canvas.clip(smoke, max=0.5)
canvas.gamma(smoke_color, 1.3)
canvas.multiply(smoke_color, 1.5)
canvas.clip(smoke_color, max=1)

taichi_volume_renderer.plot_volume(
    smoke,
    smoke_color,
    lighting=False,
    init_taichi=False,
    smoke_density_factor=600,
    ray_tracing_step_size_factor=0.5,
    light_ray_tracing_step_size_factor=0.5)
