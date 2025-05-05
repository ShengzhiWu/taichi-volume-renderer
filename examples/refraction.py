import math
import numpy as np
# from scipy.ndimage import gaussian_filter
import taichi_volume_renderer

# Volume
x, y, z = np.mgrid[-0.5:0.5:150j, -0.5:0.5:150j, -0.5:0.5:150j]
smoke_density_numpy = np.zeros_like(x)
ball_radius = 0.3
smoke_density_numpy[z <= -ball_radius] = 20  # Ground
smoke_color_numpy = np.ones(list(x.shape) + [3])
smoke_color_numpy[np.logical_and(z <= -ball_radius, (np.array(np.round(x * 10), dtype=int) + np.array(np.round(y * 10), dtype=int)) % 2 == 0)] = 0
index_of_refraction_numpy = np.ones_like(x)
# An easier way to construct the IOR distribution:
# index_of_refraction_numpy[x ** 2 + y ** 2 + z ** 2 < 0.25 ** 2] = 1.5  # Glass ball
# index_of_refraction_numpy = gaussian_filter(index_of_refraction_numpy, sigma=1.5, mode='constant', cval=1)  # Blur the index of refraction distribution
# A complexer way to construct the IOR distribution with higher quality:
sharpness = 7.
index_of_refraction_numpy[:, :, :] = np.clip(1.25 - ((x ** 2 + y ** 2 + z ** 2) ** 0.5 - ball_radius) * sharpness, 1, 1.5)

# Light
point_lights_pos_numpy = np.array([
    [0, 0, 5]], dtype=float)
point_lights_intensity_numpy = np.array([
    [80, 80, 80]], dtype=float)

taichi_volume_renderer.plot_volume(
    smoke_density=smoke_density_numpy,
    smoke_color=smoke_color_numpy,
    index_of_refraction=index_of_refraction_numpy,
    point_lights_pos=point_lights_pos_numpy,
    point_lights_intensity=point_lights_intensity_numpy,
    resolution=(720, 720),
    camera_phi=45,
    camera_theta=math.acos(2 / (2 ** 0.5 * 3 ** 0.5)) * 180 / math.pi  # The constant here is the angle between (1, 1, 0) and (1, 1, 1). degrees=False means using the radian system.
)
