# Adapted from Ye Kuang's Taichi demo pbf2d.py (https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/pbf2d.py).

# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)

import math
import numpy as np
import taichi as ti
from taichi_volume_renderer import DisplayWindow
import imageio

ti.init(arch=ti.gpu)

screen_size = (800, 800, 400)
screen_to_world_ratio = 10.0
boundary = (screen_size[0] / screen_to_world_ratio,
            screen_size[1] / screen_to_world_ratio,
            screen_size[2] / screen_to_world_ratio)
cell_size = 2.51
cell_recpr = 1.0 / cell_size

def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s

grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1), round_up(boundary[2], 1))

dim = 3
g = ti.Vector([0.0, 0.0, -9.8])
num_particles_x = 60
num_particles = num_particles_x * 2000
max_num_particles_per_cell = 200
max_num_neighbors = 100
time_delta = 1.0 / 20.0
epsilon = 1e-5
particle_radius = 1.0  # 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

# PBF params
h_ = 1.1  # 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 5
corr_deltaQ_coeff = 0.3
corrK = 0.001
# Need ti.pow()
# corrN = 4.0
neighbor_radius = h_ * 1.05

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
lambdas = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)

ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.l, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas)
ti.root.place(board_states)

N = 50  # Grid resolution
index_of_refraction_taichi = ti.field(dtype=ti.f32, shape=[N, N, N])

@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result

@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result

@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h_) / poly6_value(corr_deltaQ_coeff * h_,
                                                     h_)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x

@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)

@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] \
        and 0 <= c[1] and c[1] < grid_size[1] \
        and 0 <= c[2] and c[2] < grid_size[2]

@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1], boundary[2]
                      ]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p

@ti.kernel
def move_board():
    # probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    period = 90
    vel_strength = 8.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    board_states[None] = b

@ti.kernel
def prologue():
    # save old positions
    for i in positions:
        old_positions[i] = positions[i]
    # apply gravity within boundary
    for i in positions:
        pos, vel = positions[i], velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos)

    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1

    # update grid
    for p_i in positions:
        cell = get_cell(positions[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i

@ti.kernel
def substep():
    pass
    # compute lambdas
    # Eq (8) ~ (11)
    for p_i in positions:
        pos_i = positions[p_i]

        grad_i = ti.Vector([0.0, 0.0, 0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            grad_j = spiky_gradient(pos_ji, h_)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), h_)

        # Eq(1)
        density_constraint = (mass * density_constraint / rho0) - 1.0

        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr +
                                                lambda_epsilon)
    # compute position deltas
    # Eq(12), (14)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - positions[p_j]
            scorr_ij = compute_scorr(pos_ji)
            pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                spiky_gradient(pos_ji, h_)

        pos_delta_i /= rho0
        position_deltas[p_i] = pos_delta_i
    # apply position deltas
    for i in positions:
        positions[i] += position_deltas[i]

@ti.kernel
def epilogue():
    # confine to boundary
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
    # no vorticity/xsph because we cannot do cross product in 2D...

def run_pbf():
    prologue()
    for _ in range(pbf_num_iters):
        substep()
    epilogue()

@ti.kernel
def init_particles():
    for i in range(num_particles):
        delta = h_ * 0.8
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5,
                          boundary[1] * 0.02,
                          boundary[2] * 0.02])
        positions[i] = ti.Vector([i % num_particles_x, (i // num_particles_x) % num_particles_x, i // (num_particles_x*num_particles_x)]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])

def print_stats():
    print('PBF stats:')
    num = grid_num_particles.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f'  #particles per cell: avg={avg:.2f} max={max_}')
    num = particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f'  #neighbors per particle: avg={avg:.2f} max={max_}')

@ti.kernel
def update_index_of_refraction():
    for i, j, k in index_of_refraction_taichi:
        index_of_refraction_taichi[i, j, k] = 1.
    for p_i in positions:
        cell_float = (positions[p_i] / 80. + ti.Vector([0, 0, 0.5])) * N
        cell = int(cell_float)
        cell_fraction = cell_float - cell
        # cell_fraction = ti.Vector([0.5, 0.5, 0.5])
        # cell_fraction = ti.Vector([0, 0, 0])
        if cell.x >= 0 and cell.x < index_of_refraction_taichi.shape[0] - 1 and cell.y >= 0 and cell.y < index_of_refraction_taichi.shape[1] - 1 and cell.z >= 0 and cell.z < index_of_refraction_taichi.shape[2] - 1:
            index_of_refraction_taichi[cell.x, cell.y, cell.z] += (1 - cell_fraction.x) * (1 - cell_fraction.y) * (1 - cell_fraction.z) * 0.1
            index_of_refraction_taichi[cell.x, cell.y, cell.z + 1] += (1 - cell_fraction.x) * (1 - cell_fraction.y) * cell_fraction.z * 0.1
            index_of_refraction_taichi[cell.x, cell.y + 1, cell.z] += (1 - cell_fraction.x) * cell_fraction.y * (1 - cell_fraction.z) * 0.1
            index_of_refraction_taichi[cell.x, cell.y + 1, cell.z + 1] += (1 - cell_fraction.x) * cell_fraction.y * cell_fraction.z * 0.1
            index_of_refraction_taichi[cell.x + 1, cell.y, cell.z] += cell_fraction.x * (1 - cell_fraction.y) * (1 - cell_fraction.z) * 0.1
            index_of_refraction_taichi[cell.x + 1, cell.y, cell.z + 1] += cell_fraction.x * (1 - cell_fraction.y) * cell_fraction.z * 0.1
            index_of_refraction_taichi[cell.x + 1, cell.y + 1, cell.z] += cell_fraction.x * cell_fraction.y * (1 - cell_fraction.z) * 0.1
            index_of_refraction_taichi[cell.x + 1, cell.y + 1, cell.z + 1] += cell_fraction.x * cell_fraction.y * cell_fraction.z * 0.1
    # for i, j, k in index_of_refraction_taichi:
    #     index_of_refraction_taichi[i, j, k] = min(1.33, index_of_refraction_taichi[i, j, k])

init_particles()
print(f'boundary={boundary} grid={grid_size} cell_size={cell_size}')
print(f'num_particles={num_particles}')

update_index_of_refraction()

x, y, z = np.mgrid[-0.5:0.5:N * 1j, -0.5:0.5:N * 1j, -0.5:0.5:N * 1j]
smoke_density_numpy = np.zeros_like(x)
ground_z = 0
ground_thickness = 0.1
smoke_density_numpy[np.logical_and(z <= ground_z, z > ground_z - ground_thickness)] = 20  # Ground
smoke_color_numpy = np.ones(list(x.shape) + [3])
smoke_color_numpy[np.logical_and(smoke_density_numpy > 0, (np.array(np.round(x * 10), dtype=int) + np.array(np.round(y * 10), dtype=int)) % 2 == 0)] = 0

# Lights
point_lights_pos_numpy = np.array([
    [0, 0, 5]], dtype=float)
point_lights_intensity_numpy = np.array([
    np.ones(3) * 80], dtype=float)

window = DisplayWindow(
    smoke_density=smoke_density_numpy,
    smoke_color=smoke_color_numpy,
    index_of_refraction=index_of_refraction_taichi,
    point_lights_pos=point_lights_pos_numpy,
    point_lights_intensity=point_lights_intensity_numpy,
    resolution=[720, 500],
    init_taichi=False,  # Taichi already initiated. So must set init_taichi=False here!
    ray_tracing_step_size_factor=0.5,
    light_ray_tracing_step_size_factor=1)
window.scene.set_camera_phi(45)
window.scene.set_camera_theta(math.acos(2 / (2 ** 0.5 * 3 ** 0.5)), degrees=False)  # The constant here is the angle between (1, 1, 0) and (1, 1, 1). degrees=False means using the radian system.
window.scene.camera_distance = 2.4

def one_step(iteration, scene):
    for substep in range(1):  # 5
        move_board()
        run_pbf()
    if iteration % 50 == 1:
        print_stats()
    update_index_of_refraction()

animation = []
def image_process(iteration, image):
    pass
    global animation
    if iteration in range(200, 340, 6):
        image_numpy = np.clip(np.transpose(image.to_numpy()[:, ::-1], [1, 0, 2]) * 256, 0, 255.9)
        animation.append(image_numpy)
    if iteration == 340:
        animation = np.array(animation, dtype=np.uint8)
        imageio.mimsave('output.gif', animation, duration=0.3, loop=0)  # Save animation.
        print("Animation saved")
        del animation

window.show(
    callback=one_step,
    image_process=image_process,
    update_light_each_step=False,
    title=f"PBF 3D")
