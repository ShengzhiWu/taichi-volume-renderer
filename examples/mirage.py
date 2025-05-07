import numpy as np
import taichi_volume_renderer

# Mirages are caused by the non-uniform distribution of index of refraction (IOR) of air.
# The IOR of air decreases with increasing temperature, by the Edlén formula.

x, y, z = np.mgrid[-0.5:0.5:150j, -0.5:0.5:150j, -0.5:0.5:150j]
smoke_density_numpy = np.zeros_like(x)

# Ground
ground_z = 0
ground_thickness = 0.1
smoke_density_numpy[np.logical_and(z <= ground_z, z > ground_z - ground_thickness)] = 40  # Ground
smoke_color_numpy = np.ones(list(x.shape) + [3])
smoke_color_numpy[np.logical_and(smoke_density_numpy > 0, (np.array(np.round(x * 10), dtype=int) + np.array(np.round(y * 10), dtype=int)) % 2 == 0)] = 0

# Buildings
np.random.seed(0)
for _ in range(40):
    building_x = np.random.rand() - 0.5
    building_y = np.random.rand() - 0.5
    building_width = np.random.rand() * 0.1 + 0.01
    building_depth = np.random.rand() * 0.1 + 0.01
    building_height = np.random.rand() * 0.1 + 0.01
    mask = np.logical_and(np.abs(x - building_x) < building_width / 2, np.logical_and(np.abs(y - building_y) < building_depth / 2, np.logical_and(z > ground_z, z - ground_z < building_height)))
    smoke_density_numpy[mask] = 80
    smoke_color_numpy[mask] = np.random.rand(3)

# Air
normal_layer = 0.95 * 2 ** z  # Normal lapse layer (hotter below, colder above)
inversion_layer = 0.8 ** z  # Inversion layer (colder below, hotter above)
index_of_refraction_numpy = np.minimum(normal_layer, inversion_layer)
index_of_refraction_numpy[z < 0] = 1

# Light
point_lights_pos_numpy = np.array([
    [0, 0, 5]], dtype=float)
point_lights_intensity_numpy = np.array([
    np.ones(3) * 60], dtype=float)

taichi_volume_renderer.plot_volume(
    smoke_density=smoke_density_numpy,
    smoke_color=smoke_color_numpy,
    index_of_refraction=index_of_refraction_numpy,
    point_lights_pos=point_lights_pos_numpy,
    point_lights_intensity=point_lights_intensity_numpy,
    resolution=(720, 720),
    ray_tracing_step_size_factor=0.5,
    light_ray_tracing_step_size_factor=1,
    camera_phi=0,
    camera_theta=7
)
