# Gray-Scott System
 
import math
import numpy as np
import taichi as ti
from taichi_volume_renderer import DisplayWindow
import imageio

ti.init(arch=ti.cuda)

# PDE parameters
# Diffusion strength
D_u, D_v = 0.2, 0.1
# Feed and kill rates of Gray-Scott system. Different combinations impart different behaviors to the system.
# F, k = 0.01, 0.053  # 呼吸的会复制和死亡的点
# F, k = 0.02, 0.055  # Chaotic foams
# F, k = 0.027, 0.065  # Self-replicating spots （也有呼吸行为和条状在不改变拓扑的情况下形成皱褶的行为）
# F, k = 0.03, 0.065  # 点和条混合
# F, k = 0.03, 0.0661  # 点和条混合，点主导
F, k = 0.045, 0.067  # 条状（蛇形行进，并在几乎不改变拓扑的情况下伸展并产生褶皱）（演化缓慢，substeps=50）
# F, k = 0.052, 0.06  # 片状（生长、融合）（演化缓慢，substeps=50）
# F, k = 0.062, 0.055  # 片状（生长、融合）（演化缓慢，substeps=50）
# F, k = 0.06, 0.055  # 片状（在几乎不改变拓扑的情况下伸展并产生褶皱）（演化缓慢，substeps=500）
# F, k = 0.065, 0.054  # 条状（蛇形行进，在几乎不改变拓扑的情况下伸展并产生褶皱）（演化缓慢，substeps=500）
# F, k = 0.07, 0.05  # 条状（蛇形行进）（演化缓慢，substeps=500）
# F, k = 0.07, 0.051  # 条状（蛇形行进，不擅长拐弯）（演化缓慢，substeps=500）
# F, k = 0.075, 0.042  # 螺旋波（旋转的带，会破裂）
# F, k = 0.085, 0.005  # 片状（迷宫，墙壁平行程度很高）（演化缓慢，substeps=500）
# F, k = 0.085, 0.012  # 条状（蛇形行进，会分叉）（演化缓慢，substeps=50）
# F, k = 0.085, 0.018  # 螺旋波（旋转的带，会破裂）（演化缓慢，substeps=50）
# F, k = 0.09, 0.008  # 条状（蛇形行进，会分叉）（演化缓慢，substeps=50）

N = 128  # Grid resolution
u = ti.field(dtype=ti.f32, shape=(N, N, N))
v = ti.field(dtype=ti.f32, shape=(N, N, N))
u_new = ti.field(dtype=ti.f32, shape=(N, N, N))
v_new = ti.field(dtype=ti.f32, shape=(N, N, N))

dt = 0.8  # Time step

@ti.kernel
def initialize(random_radius: int):
    # Initial conditions: uniform background + random perturbations.
    for i, j, k in u:
        u[i, j, k] = 1.0
        v[i, j, k] = 0.0
    # Add initial perturbations in the central region.
    for i, j, k in ti.ndrange((N//2 - random_radius, N//2 + random_radius), (N//2 - random_radius, N//2 + random_radius), (N//2 - random_radius, N//2 + random_radius)):
        v[i, j, k] = 0.25 + ti.random() * 0.1

@ti.kernel
def update():
    # 3D Laplacian operator (finite difference)
    for i, j, l in ti.ndrange((1, N-1), (1, N-1), (1, N-1)):
        laplace_u = (u[i+1, j, l] + u[i-1, j, l] + u[i, j+1, l] + 
                    u[i, j-1, l] + u[i, j, l+1] + u[i, j, l-1] - 6 * u[i, j, l])
        laplace_v = (v[i+1, j, l] + v[i-1, j, l] + v[i, j+1, l] + 
                    v[i, j-1, l] + v[i, j, l+1] + v[i, j, l-1] - 6 * v[i, j, l])

        uv_sq = u[i, j, l] * v[i, j, l] ** 2  # Gray-Scott reaction term
        u_new[i, j, l] = u[i, j, l] + dt * (D_u * laplace_u - uv_sq + F * (1 - u[i, j, l]))
        v_new[i, j, l] = v[i, j, l] + dt * (D_v * laplace_v + uv_sq - (F + k) * v[i, j, l])
    
    # Update fields
    for i, j, l in u:
        u[i, j, l] = u_new[i, j, l]
        v[i, j, l] = v_new[i, j, l]

initialize(random_radius=5)

# Lights
point_lights_pos_numpy = np.array([
    [0, 4, 7],
    [0, 0, 8]], dtype=float)
point_lights_intensity_numpy = np.array([
    [100, 50, 0],
    [0, 0, 100]], dtype=float)

window = DisplayWindow(
    v,
    point_lights_pos=point_lights_pos_numpy,
    point_lights_intensity=point_lights_intensity_numpy,
    init_taichi=False,  # Taichi already initiated. So must set init_taichi=False here!
    smoke_density_factor=30)
window.scene.set_camera_phi(45)
window.scene.set_camera_theta(math.acos(2 / (2 ** 0.5 * 3 ** 0.5)), degrees=False)  # The constant here is the angle between (1, 1, 0) and (1, 1, 1). degrees=False means using the radian system.

def one_step(iteration, scene):
    for substep in range(50):  # 5
        update()

animation = []
def image_process(iteration, image):
    global animation
    if iteration in range(0, 1050, 50):
        image_numpy = np.clip(np.transpose(image.to_numpy()[:, ::-1], [1, 0, 2]) * 256, 0, 255.9)
        animation.append(image_numpy)
    if iteration == 1050:
        animation = np.array(animation, dtype=np.uint8)
        imageio.mimsave('output.gif', animation, duration=0.3, loop=0)  # Save animation.
        print("Animation saved")
        del animation

window.show(
    callback=one_step,
    image_process=image_process,
    update_light_each_step=True,
    title=f"Gray-Scott Model, F={F}, k={k}")
