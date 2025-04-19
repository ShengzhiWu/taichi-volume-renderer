# In most cases, you don't need to care the code in this script, which has been encapsulated in higher-level functions. See example.py.

import numpy as np
import taichi as ti
from taichi_volume_renderer import Scene

ti.init(arch=ti.gpu)

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
smoke_density_taichi = ti.field(dtype=ti.f32, shape=smoke_density_numpy.shape)
smoke_density_taichi.from_numpy(smoke_density_numpy)
smoke_color_numpy = np.ones(list(x.shape) + [3])
smoke_color_numpy[np.logical_and(x > 0, np.logical_and(y > 0, z > 0))] = 0
smoke_color_taichi = ti.Vector.field(3, dtype=ti.f32, shape=smoke_color_numpy.shape[:-1])
smoke_color_taichi.from_numpy(smoke_color_numpy)

# Light
point_lights_pos_numpy = np.array([
    [0, 4, 7],
    [0, 0, 8]], dtype=float)
point_lights_pos_taichi = ti.Vector.field(3, dtype=ti.f32, shape=point_lights_pos_numpy.shape[:-1])
point_lights_pos_taichi.from_numpy(point_lights_pos_numpy)
point_lights_intensity_numpy = np.array([
    [100, 50, 0],
    [0, 0, 100]], dtype=float)
point_lights_intensity_taichi = ti.Vector.field(3, dtype=ti.f32, shape=point_lights_intensity_numpy.shape[:-1])
point_lights_intensity_taichi.from_numpy(point_lights_intensity_numpy)

scene = Scene(
    smoke_density_taichi=smoke_density_taichi,
    smoke_color_taichi=smoke_color_taichi,
    point_lights_pos_taichi=point_lights_pos_taichi,
    point_lights_intensity_taichi=point_lights_intensity_taichi)

print(scene.get_vertical_field_of_view())

# Window
res = (720, 720)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)

# Interaction
mouse_pressed = False
cursor_start_pos = (-1, -1)
camera_rotation_speed = 230.
    
def mouse_pressed_event(pos):
    pass

def mouse_drag_event(pos, pos_delta):
    scene.set_camera_phi(scene.get_camera_phi() - pos_delta[0] * camera_rotation_speed)
    scene.set_camera_theta(scene.get_camera_theta() - pos_delta[1] * camera_rotation_speed)

scene.update_light()  # Calculate light and shadow

gui = ti.GUI("Render", res=res)
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    # scene.update_light()  # If need recalculate light and shadow, uncomment this line.
    scene.render(pixels)
    gui.set_image(pixels)
    gui.show()

    # Manage mouse events
    if gui.is_pressed(ti.GUI.LMB):
        cursor_pos = gui.get_cursor_pos()
        if not mouse_pressed:
            mouse_pressed_event(cursor_pos)
            mouse_pressed = True
            cursor_start_pos = cursor_pos
        if cursor_start_pos[0] != cursor_pos[0] or cursor_start_pos[1] != cursor_pos[1]:
            mouse_drag_event(cursor_pos, (cursor_pos[0] - cursor_start_pos[0], cursor_pos[1] - cursor_start_pos[1]))
            cursor_start_pos = cursor_pos
    else:
        mouse_pressed = False
