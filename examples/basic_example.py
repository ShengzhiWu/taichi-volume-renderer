import numpy as np
import taichi_volume_renderer

# Volume
x, y, z = np.mgrid[-0.5:0.5:100j, -0.5:0.5:100j, -0.5:0.5:100j]
smoke_density_numpy = np.zeros_like(x)
# volume_numpy[x ** 2 + y ** 2 + z ** 2 < 0.5 ** 2] = 5  # Single sphere
for x_0 in [-0.25, 0.25]:  # 8 spheres
    for y_0 in [-0.25, 0.25]:
        for z_0 in [-0.25, 0.25]:
            if x_0 > 0 and y_0 < 0 and z_0 > 0:
                continue
            smoke_density_numpy[(x - x_0) ** 2 + (y - y_0) ** 2 + (z - z_0) ** 2 < 0.25 ** 2] = 6
smoke_density_numpy += np.maximum(0, 1 - ((x - 0.25) ** 2 + (y - -0.25) ** 2 + (z - 0.25) ** 2) ** 0.5 / 0.25) * 10
smoke_color_numpy = np.ones(list(x.shape) + [3])
smoke_color_numpy[np.logical_and(x > 0, np.logical_and(y > 0, z > 0))] = 0

# Light
point_lights_pos_numpy = np.array([
    [0, 4, 7],
    [0, 0, 8]], dtype=float)
point_lights_intensity_numpy = np.array([
    [100, 50, 0],
    [0, 0, 100]], dtype=float)

taichi_volume_renderer.plot_volume(
    smoke_density=smoke_density_numpy,
    smoke_color=smoke_color_numpy,
    point_lights_pos=point_lights_pos_numpy,
    point_lights_intensity=point_lights_intensity_numpy,
    resolution=(720, 720)
)
