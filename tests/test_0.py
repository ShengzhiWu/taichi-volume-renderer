import numpy as np
import taichi as ti
from taichi_volume_renderer import Scene

ti.init(arch=ti.gpu)

# Volume
x, y, z = np.mgrid[-0.5:0.5:100j, -0.5:0.5:100j, -0.5:0.5:100j]
smoke_numpy = np.zeros_like(x)
# volume_numpy[x ** 2 + y ** 2 + z ** 2 < 0.5 ** 2] = 5  # Single sphere
for x_0 in [-0.25, 0.25]:  # 8 spheres
    for y_0 in [-0.25, 0.25]:
        for z_0 in [-0.25, 0.25]:
            if x_0 > 0 and y_0 < 0 and z_0 > 0:
                continue
            smoke_numpy[(x - x_0) ** 2 + (y - y_0) ** 2 + (z - z_0) ** 2 < 0.25 ** 2] = 6
smoke_numpy += np.maximum(0, 1 - ((x - 0.25) ** 2 + (y - -0.25) ** 2 + (z - 0.25) ** 2) ** 0.5 / 0.25) * 10
smoke_color_numpy = np.ones(list(x.shape) + [3])
smoke_color_numpy[np.logical_and(x > 0, np.logical_and(y > 0, z > 0))] = 0

# Light
point_lights_pos_numpy = np.array([
    [0, 4, 7],
    [0, 0, 8]], dtype=float)
point_lights_intensity_numpy = np.array([
    [100, 50, 0],
    [0, 0, 100]], dtype=float)

scene = Scene(
    smoke_numpy=smoke_numpy,
    smoke_color_numpy=smoke_color_numpy,
    point_lights_pos_numpy=point_lights_pos_numpy,
    point_lights_intensity_numpy=point_lights_intensity_numpy)


# Window
res = 720, 720
pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)

# Interaction
mouse_pressed = False
cursor_start_pos = (-1, -1)

scene.update_light()  # Calculate light and shadow

gui = ti.GUI("Render", res=res)
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    # scene.update_light()  # If need recalculate light and shadow, uncomment this line.
    scene.render(pixels)
    gui.set_image(pixels)
    gui.show()

    # 处理鼠标事件
    if gui.is_pressed(ti.GUI.LMB):
        cursor_pos = gui.get_cursor_pos()
        if not mouse_pressed:
            scene.mouse_pressed_event(cursor_pos)
            mouse_pressed = True
            cursor_start_pos = cursor_pos
        if cursor_start_pos[0] != cursor_pos[0] or cursor_start_pos[1] != cursor_pos[1]:
            scene.mouse_drag_event(cursor_pos, (cursor_pos[0] - cursor_start_pos[0], cursor_pos[1] - cursor_start_pos[1]))
            cursor_start_pos = cursor_pos
    else:
        mouse_pressed = False
        