import numpy as np
import taichi as ti
from .math import compute_covariance_inv, rotation_matrix_to_quaternion, rotation_quaternion_to_matrix, quaternion_multiply

def construct_frame(x, y=None):
    x /= np.linalg.norm(x)
    for e in [y, [1, 0, 0], [0, 1, 0]]:
        if e is None:
            continue
        z = np.cross(x, e)
        z_norm = np.linalg.norm(z)
        if z_norm != 0:
            z /= z_norm
            y = np.cross(z, x)
            return x, y, z

def empty_canvas(resolution):
    if type(resolution) == int:
        smoke = ti.field(dtype=float, shape=[resolution, resolution, resolution])
        smoke_color = ti.Vector.field(3, dtype=float, shape=smoke.shape)
        smoke_color.from_numpy(np.ones(list(smoke_color.shape) + [3]))
        return smoke, smoke_color
    raise TypeError("Unsupported type of resolution: " + str(type(resolution)))@ti.kernel

@ti.kernel
def clean(
    smoke_density_taichi: ti.template(),  # type: ignore
    smoke_color_taichi: ti.template()  # type: ignore
    ):
    for I in ti.grouped(smoke_density_taichi):
        smoke_density_taichi[I] = 0
        smoke_color_taichi[I] = ti.Vector([1, 1, 1])

@ti.kernel
def multiply(
    taichi_field: ti.template(),  # type: ignore
    k: float
):
    for I in ti.grouped(taichi_field):
        taichi_field[I] *= k

@ti.kernel
def clip_kernel(
    taichi_field: ti.template(),  # type: ignore
    min: float,
    max: float
):
    for I in ti.grouped(taichi_field):
        taichi_field[I] = ti.math.clamp(taichi_field[I], min, max)

def clip(
    taichi_field,
    min=0,
    max=1
):
    clip_kernel(taichi_field, min, max)

@ti.kernel
def gamma(
    taichi_field: ti.template(),  # type: ignore
    power: float
):
    for I in ti.grouped(taichi_field):
        taichi_field[I] = taichi_field[I] ** power

@ti.func
def mix(color_1, density_1, color_2, density_2):
    return (color_1 * density_1 + color_2 * density_2) / (density_1 + density_2)

@ti.kernel
def fill_rectangle(  # Fill rectangle
    smoke_density_taichi: ti.template(),  # type: ignore
    smoke_color_taichi: ti.template(),  # type: ignore
    start: ti.math.vec3,  # type: ignore
    scale: ti.math.vec3,  # type: ignore
    density: float,
    color: ti.math.vec3  # type: ignore
    ):
    end_int = int(ti.round(start + scale))
    start_int = int(ti.round(start))
    start_int = ti.math.max(0, start_int)
    end_int = ti.math.min(smoke_density_taichi.shape, end_int)
    for I in ti.grouped(ti.ndrange([start_int.x, end_int.x], [start_int.y, end_int.y], [start_int.z, end_int.z])):
        smoke_color_taichi[I] = mix(smoke_color_taichi[I], smoke_density_taichi[I], color, density)
        smoke_density_taichi[I] += density

@ti.kernel
def fill_disk(  # Fill disk (No anti-aliasing)
    smoke_density_taichi: ti.template(),  # type: ignore
    smoke_color_taichi: ti.template(),  # type: ignore
    center: ti.math.vec3,  # type: ignore
    radius: float,
    density: float,
    color: ti.math.vec3  # type: ignore
    ):
    start = int(ti.floor(center - radius))
    end = int(ti.ceil(center + radius))
    start = ti.math.max(0, start)
    end = ti.math.min(smoke_density_taichi.shape, end)

    for I in ti.grouped(ti.ndrange([start.x, end.x], [start.y, end.y], [start.z, end.z])):
        if (I - center).norm() <= radius:
            smoke_color_taichi[I] = mix(smoke_color_taichi[I], smoke_density_taichi[I], color, density)
            smoke_density_taichi[I] += density

@ti.func
def _draw_point_scalar(
    smoke_density_taichi,
    point,
    density
):
    point_int = int(point)
    point_fraction = point - point_int
    if point_int.x >= 0 and point_int.x < smoke_density_taichi.shape[0] - 1 and point_int.y >= 0 and point_int.y < smoke_density_taichi.shape[1] - 1 and point_int.z >= 0 and point_int.z < smoke_density_taichi.shape[2] - 1:
        strength = density * (1 - point_fraction.x) * (1 - point_fraction.y) * (1 - point_fraction.z)
        if strength > 0:
            I_000 = point_int.x, point_int.y, point_int.z
            smoke_density_taichi[I_000] += strength
        strength = density * (1 - point_fraction.x) * (1 - point_fraction.y) * point_fraction.z
        if strength > 0:
            I_001 = point_int.x, point_int.y, point_int.z + 1
            smoke_density_taichi[I_001] += strength
        strength = density * (1 - point_fraction.x) * point_fraction.y * (1 - point_fraction.z)
        if strength > 0:
            I_010 = point_int.x, point_int.y + 1, point_int.z
            smoke_density_taichi[I_010] += strength
        strength = density * (1 - point_fraction.x) * point_fraction.y * point_fraction.z
        if strength > 0:
            I_011 = point_int.x, point_int.y + 1, point_int.z + 1
            smoke_density_taichi[I_011] += strength
        strength = density * point_fraction.x * (1 - point_fraction.y) * (1 - point_fraction.z)
        if strength > 0:
            I_100 = point_int.x + 1, point_int.y, point_int.z
            smoke_density_taichi[I_100] += strength
        strength = density * point_fraction.x * (1 - point_fraction.y) * point_fraction.z
        if strength > 0:
            I_101 = point_int.x + 1, point_int.y, point_int.z + 1
            smoke_density_taichi[I_101] += strength
        strength = density * point_fraction.x * point_fraction.y * (1 - point_fraction.z)
        if strength > 0:
            I_110 = point_int.x + 1, point_int.y + 1, point_int.z
            smoke_density_taichi[I_110] += strength
        strength = density * point_fraction.x * point_fraction.y * point_fraction.z
        if strength > 0:
            I_111 = point_int.x + 1, point_int.y + 1, point_int.z + 1
            smoke_density_taichi[I_111] += strength

@ti.func
def _draw_point(
    smoke_density_taichi,
    smoke_color_taichi,
    point,
    density,
    color
):
    point_int = int(point)
    point_fraction = point - point_int
    if point_int.x >= 0 and point_int.x < smoke_density_taichi.shape[0] - 1 and point_int.y >= 0 and point_int.y < smoke_density_taichi.shape[1] - 1 and point_int.z >= 0 and point_int.z < smoke_density_taichi.shape[2] - 1:
        strength = density * (1 - point_fraction.x) * (1 - point_fraction.y) * (1 - point_fraction.z)
        if strength > 0:
            I_000 = point_int.x, point_int.y, point_int.z
            smoke_color_taichi[I_000] = mix(smoke_color_taichi[I_000], smoke_density_taichi[I_000], color, strength)
            smoke_density_taichi[I_000] += strength
        strength = density * (1 - point_fraction.x) * (1 - point_fraction.y) * point_fraction.z
        if strength > 0:
            I_001 = point_int.x, point_int.y, point_int.z + 1
            smoke_color_taichi[I_001] = mix(smoke_color_taichi[I_001], smoke_density_taichi[I_001], color, strength)
            smoke_density_taichi[I_001] += strength
        strength = density * (1 - point_fraction.x) * point_fraction.y * (1 - point_fraction.z)
        if strength > 0:
            I_010 = point_int.x, point_int.y + 1, point_int.z
            smoke_color_taichi[I_010] = mix(smoke_color_taichi[I_010], smoke_density_taichi[I_010], color, strength)
            smoke_density_taichi[I_010] += strength
        strength = density * (1 - point_fraction.x) * point_fraction.y * point_fraction.z
        if strength > 0:
            I_011 = point_int.x, point_int.y + 1, point_int.z + 1
            smoke_color_taichi[I_011] = mix(smoke_color_taichi[I_011], smoke_density_taichi[I_011], color, strength)
            smoke_density_taichi[I_011] += strength
        strength = density * point_fraction.x * (1 - point_fraction.y) * (1 - point_fraction.z)
        if strength > 0:
            I_100 = point_int.x + 1, point_int.y, point_int.z
            smoke_color_taichi[I_100] = mix(smoke_color_taichi[I_100], smoke_density_taichi[I_100], color, strength)
            smoke_density_taichi[I_100] += strength
        strength = density * point_fraction.x * (1 - point_fraction.y) * point_fraction.z
        if strength > 0:
            I_101 = point_int.x + 1, point_int.y, point_int.z + 1
            smoke_color_taichi[I_101] = mix(smoke_color_taichi[I_101], smoke_density_taichi[I_101], color, strength)
            smoke_density_taichi[I_101] += strength
        strength = density * point_fraction.x * point_fraction.y * (1 - point_fraction.z)
        if strength > 0:
            I_110 = point_int.x + 1, point_int.y + 1, point_int.z
            smoke_color_taichi[I_110] = mix(smoke_color_taichi[I_110], smoke_density_taichi[I_110], color, strength)
            smoke_density_taichi[I_110] += strength
        strength = density * point_fraction.x * point_fraction.y * point_fraction.z
        if strength > 0:
            I_111 = point_int.x + 1, point_int.y + 1, point_int.z + 1
            smoke_color_taichi[I_111] = mix(smoke_color_taichi[I_111], smoke_density_taichi[I_111], color, strength)
            smoke_density_taichi[I_111] += strength

@ti.kernel
def _draw_line_simple_kernel(  # Draw single-pixel-wide line (Anti-aliasing)
    smoke_density_taichi: ti.template(),  # type: ignore
    smoke_color_taichi: ti.template(),  # type: ignore
    start: ti.math.vec3,  # type: ignore
    end: ti.math.vec3,  # type: ignore
    density: float,
    color: ti.math.vec3,  # type: ignore
    end_point: bool,
    step: float
    ):
    length = (end - start).norm()
    point_num = int(ti.ceil(length / step))
    point_num = max(2, point_num)
    point_strength = density * length / (point_num - 1)
    for i in ti.ndrange(point_num if end_point else point_num - 1):
        t = float(i) / (point_num - 1)
        _draw_point(smoke_density_taichi, smoke_color_taichi, start * (1 - t) + end * t, point_strength, color)

def draw_line_simple(  # Draw single-pixel-wide line (Anti-aliasing)
    smoke_density_taichi,
    smoke_color_taichi,
    start,
    end,
    density,
    color,
    step=0.5
    ):
    _draw_line_simple_kernel(smoke_density_taichi, smoke_color_taichi, start, end, density, color, True, step)

def draw_polyline_simple(  # Draw single-pixel-wide line (Anti-aliasing)
    smoke_density_taichi,
    smoke_color_taichi,
    polyline,
    density,
    color,
    step=0.5
):
    if len(polyline) < 2:
        return
    for i in range(len(polyline) - 2):
        _draw_line_simple_kernel(smoke_density_taichi, smoke_color_taichi, polyline[i], polyline[i + 1], density, color, False, step)
    _draw_line_simple_kernel(smoke_density_taichi, smoke_color_taichi, polyline[-2], polyline[-1], density, color, True, step)

# TODO: fill_cylinder

def draw_line(  # Draw line (Anti-aliasing)
    smoke_density_taichi,
    smoke_color_taichi,
    start,
    end,
    radius,
    density,
    color
):
    assert radius == 1  # TODO
    draw_line_simple(smoke_density_taichi, smoke_color_taichi, start, end, density, color)

# TODO: fill_cone

# TODO: draw_arrow

# TODO: draw_circle

@ti.kernel
def _draw_helix_kernel(
    smoke_density_taichi: ti.template(),  # type: ignore
    smoke_color_taichi: ti.template(),  # type: ignore
    start: ti.math.vec3,  # type: ignore
    height: float,
    radius: float,
    rounds: float,
    x: ti.math.vec3,  # type: ignore
    y: ti.math.vec3,  # type: ignore
    z: ti.math.vec3,  # type: ignore
    density: float,
    color: ti.math.vec3,  # type: ignore
    step: float
):
    length = (height ** 2 + (2 * ti.math.pi * radius * rounds) ** 2) ** 0.5
    point_num = int(ti.ceil(length / step))
    point_num = max(2, point_num)
    point_strength = density * length / (point_num - 1)
    for i in ti.ndrange(point_num):
        t = float(i) / (point_num - 1)
        _draw_point(smoke_density_taichi, smoke_color_taichi, start + x * (t * height) + y * (radius * ti.cos(2 * ti.math.pi * rounds * t)) + z * (radius * ti.sin(2 * ti.math.pi * rounds * t)), point_strength, color)

def draw_helix(  # Draw helix (Anti-aliasing)
    smoke_density_taichi: ti.template(),  # type: ignore
    smoke_color_taichi: ti.template(),  # type: ignore
    start,
    end,
    radius,
    rounds,
    density,
    color,
    initial_direction=None,
    step=0.5
):
    # start = np.array(start, dtype=float)
    # end = np.array(end, dtype=float)
    start = ti.Vector(start)
    end = ti.Vector(end)
    x, y, z = construct_frame(end - start, initial_direction)
    _draw_helix_kernel(
        smoke_density_taichi,
        smoke_color_taichi,
        start,
        (end - start).norm(),
        radius,
        rounds,
        x,
        y,
        z,
        density,
        color,
        step)


# TODO: draw_cubic_bezier_curve

# TODO: draw_cubic_bezier_surface

# TODO: draw_spline

@ti.kernel
def fill_convex(
    smoke_density_taichi: ti.template(),  # type: ignore
    smoke_color_taichi: ti.template(),  # type: ignore
    center: ti.math.vec3,  # type: ignore
    face_vectors: ti.template(),  # type: ignore
    radius_consider: float,
    density: float,
    color: ti.math.vec3  # type: ignore
):
    start = int(ti.floor(center - radius_consider))
    end = int(ti.ceil(center + radius_consider))
    start = ti.math.max(0, start)
    end = ti.math.min(smoke_density_taichi.shape, end)

    for I in ti.grouped(ti.ndrange([start.x, end.x], [start.y, end.y], [start.z, end.z])):
        inside = True
        relative_location = I - center
        for i in range(face_vectors.shape[0]):
            if ti.math.dot(relative_location, face_vectors[i]) > 1:
                inside = False
        if inside:
            smoke_color_taichi[I] = mix(smoke_color_taichi[I], smoke_density_taichi[I], color, density)
            smoke_density_taichi[I] += density

def fill_platonic_solid(
    smoke_density_taichi,
    smoke_color_taichi,
    center,
    radius,
    face_num,
    density,
    color,
    transform=None
):
    if transform is None:
        transform = np.eye(3)

    phi = (1 + np.sqrt(5)) / 2
    face_vectors = {
        4: [
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1]
        ],
        6:[
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ],
        8:[
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1]
        ],
        12:[
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ],
        20:[
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, 1 / phi, phi], [0, 1 / phi, -phi], [0, -1 / phi, phi], [0, -1 / phi, -phi],
            [phi, 0, 1 / phi], [phi, 0, -1 / phi], [-phi, 0, 1 / phi], [-phi, 0, -1 / phi],
            [1 / phi, phi, 0], [1 / phi, -phi, 0], [-1 / phi, phi, 0], [-1 / phi, -phi, 0]
        ]
    }[face_num]
    face_vectors = np.array(face_vectors, dtype=float)
    face_vectors /= np.sum(face_vectors ** 2, axis=-1)[:, np.newaxis] ** 0.5  # Normalize
    face_vectors /= radius
    face_vectors @= np.linalg.inv(transform)
    face_vectors_taichi = ti.Vector.field(3, ti.f32, shape=[len(face_vectors)])
    face_vectors_taichi.from_numpy(face_vectors)

    fill_convex(
        smoke_density_taichi,
        smoke_color_taichi,
        center,
        face_vectors_taichi,
        radius * np.max(np.linalg.eigvals(transform)) * {  # Vertex-to-center to face-center-to-center distance ratio in Platonic solids
            4: 3,
            6: 3 ** 0.5,
            8: 3 ** 0.5,
            12: 3 ** 0.5 / phi * ((5 - 5 ** 0.5) / 2) ** 0.5,
            20: 3 ** 0.5 / phi
        }[face_num],
        density,
        color)

# TODO: Plot function

@ti.kernel
def _draw_identical_particles_scalar_kernel(
    field_taichi: ti.template(),  # type: ignore
    particles_taichi: ti.template(),  # type: ignore
    density: float
):
    for point in particles_taichi:
        _draw_point_scalar(field_taichi, point, density)

@ti.kernel
def _draw_particles_scalar_kernel(
    field_taichi: ti.template(),  # type: ignore
    particles_taichi: ti.template(),  # type: ignore
    densities_taichi: ti.template(),  # type: ignore
):
    for i in ti.ndrange(particles_taichi.shape[0]):
        _draw_point_scalar(field_taichi, particles_taichi[i], densities_taichi[i])

@ti.kernel
def _draw_identical_particles_kernel(  # Draw particles with single color and density (Anti-aliasing)
    smoke_density_taichi: ti.template(),  # type: ignore
    smoke_color_taichi: ti.template(),  # type: ignore
    particles_taichi: ti.template(),  # type: ignore
    density: float,
    color: ti.math.vec3  # type: ignore
):
    for point in particles_taichi:
        _draw_point(smoke_density_taichi, smoke_color_taichi, point, density, color)

@ti.kernel
def _draw_particles_kernel(  # Draw particles (Anti-aliasing)
    smoke_density_taichi: ti.template(),  # type: ignore
    smoke_color_taichi: ti.template(),  # type: ignore
    particles_taichi: ti.template(),  # type: ignore
    densities_taichi: ti.template(),  # type: ignore
    colors_taichi: ti.template(),  # type: ignore
):
    for i in ti.ndrange(particles_taichi.shape[0]):
        _draw_point(smoke_density_taichi, smoke_color_taichi, particles_taichi[i], densities_taichi[i], colors_taichi[i])

def is_vector(a):
    if isinstance(a, list) and len(a) == 3:  # List
        return True
    if isinstance(a, ti.Vector):  # Taichi vector
        return True
    if isinstance(a, ti.Field):  # Taichi fiels or taichi vector field
        return False
    if hasattr(a, '__len__'):
        if len(a) != 3:
            return False
        if hasattr(a, 'shape'):
            if len(a.shape) == 1:  # Array with shape [3]
                return True
    return False

def draw_particles(  # Draw particles (Anti-aliasing)
    smoke_density_taichi,
    particles,
    smoke_color_taichi=None,
    densities=1.,
    colors=[1, 1, 1],
):
    if not isinstance(particles, ti.Field):
        particles_numpy = particles
        particles = ti.Vector.field(3, dtype=ti.f32, shape=particles.shape[:-1])
        particles.from_numpy(particles_numpy)
    if not type(densities) in [int, float]:
        if not isinstance(densities, ti.Field):  # Numpy array
            densities_numpy = densities
            densities = ti.field(dtype=ti.f32, shape=densities.shape)
            densities.from_numpy(densities_numpy)
    if not is_vector(colors):
        if not isinstance(colors, ti.Field):  # Numpy array
            colors_numpy = colors
            colors = ti.Vector.field(3, dtype=ti.f32, shape=colors.shape[:-1])
            colors.from_numpy(colors_numpy)
    if not smoke_color_taichi is None:
        if isinstance(densities, ti.Field) ^ isinstance(colors, ti.Field):
            if isinstance(densities, ti.Field):  # colors is a value. Need to convert to a constant taichi field.
                colors_numpy = np.zeros([particles.shape[0], 3])
                for i in range(3):
                    colors_numpy[:, i] = colors[i]
                colors = ti.Vector.field(3, dtype=ti.f32, shape=colors_numpy.shape[:-1])
                colors.from_numpy(colors_numpy)
            else:  # densities is a value. Need to convert to a constant taichi field.
                densities_numpy = np.zeros(particles.shape[0])
                densities_numpy.fill(densities)
                densities = ti.field(dtype=ti.f32, shape=densities_numpy.shape)
                densities.from_numpy(densities_numpy)

    if smoke_color_taichi is None:
        if type(densities) in [int, float]:
            _draw_identical_particles_scalar_kernel(smoke_density_taichi, particles, densities)
        else:
            _draw_particles_scalar_kernel(smoke_density_taichi, particles, densities)
    else:
        if type(densities) in [int, float]:
            _draw_identical_particles_kernel(smoke_density_taichi, smoke_color_taichi, particles, densities, colors)
        else:
            _draw_particles_kernel(smoke_density_taichi, smoke_color_taichi, particles, densities, colors)

# TODO: def gaussian_blur(smoke_density_taichi, smoke_color_taichi, radius)

# TODO: def draw_volume(smoke_density_taichi, smoke_color_taichi, smoke_density_taichi_to_draw, smoke_color_taichi_to_draw)

@ti.kernel
def _gaussian_splatting_kernel(
    smoke: ti.template(),  # type: ignore
    smoke_color: ti.template(),  # type: ignore
    positions: ti.template(),  # type: ignore
    rotations: ti.template(),  # type: ignore
    scales: ti.template(),  # type: ignore
    colors: ti.template(),  # type: ignore
    opacities: ti.template()  # type: ignore
):    
    # Render each particle
    for i in ti.ndrange(positions.shape[0]):
        pos = positions[i]
        scale = scales[i]
        
        # Calculate inversed covariance matrix
        inv_cov3d = compute_covariance_inv(rotations[i], scale)
        
        radius = ti.max(scale.x, scale.y, scale.z) * 3.0
        min_pos = ti.max(ti.cast(ti.floor(pos - radius), ti.i32), 0)
        max_pos = ti.min(ti.cast(ti.ceil(pos + radius), ti.i32), ti.Vector(smoke.shape) - 1)
        if max_pos.x < min_pos.x or max_pos.y < min_pos.y or max_pos.z < min_pos.z:
            continue
        
        for I in ti.grouped(ti.ndrange(
            (min_pos.x, max_pos.x + 1),
            (min_pos.y, max_pos.y + 1),
            (min_pos.z, max_pos.z + 1)
        )):
            delta = float(I) - pos
            density = opacities[i] * ti.exp(-0.5 * delta.dot(inv_cov3d @ delta))
            
            # Mix
            _draw_point(smoke, smoke_color, I, density, colors[i])

def gaussian_splatting(  # Render Gaussian splatting data (Tested with data exported from Jawset Postshot)
    smoke,
    smoke_color,
    data,
    offset=[0, 0, 0],
    rotation_matrix=None,
    rotation_quaternion=None,
    y_upward=False,
    scaling=1
):
    if not rotation_matrix is None:
        rotation_matrix = np.array(rotation_matrix, dtype=float)
        rotation_quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    elif not rotation_quaternion is None:
        rotation_matrix = rotation_quaternion_to_matrix(rotation_quaternion)
    else:
        if y_upward:
            rotation_matrix = np.array([
                [1,  0, 0],
                [0,  0, 1],
                [0, -1, 0]
            ], dtype=float)
        else:
            rotation_matrix = np.eye(3)
        rotation_quaternion = rotation_matrix_to_quaternion(rotation_matrix)

    particle_num = len(data['positions'])
    positions = ti.Vector.field(3, ti.f32, shape=[particle_num])
    rotations = ti.Vector.field(4, ti.f32, shape=[particle_num])
    scales = ti.Vector.field(3, ti.f32, shape=[particle_num])
    colors = ti.Vector.field(3, ti.f32, shape=[particle_num])
    opacities = ti.field(ti.f32, shape=[particle_num])

    positions.from_numpy((data['positions'] @ rotation_matrix.T * scaling + offset))
    rotations.from_numpy(quaternion_multiply(rotation_quaternion, data['rotations'].T).T)
    scales.from_numpy(data['scales'] * scaling)
    colors.from_numpy(data['sh_coeffs'][:, 0])
    opacities.from_numpy(data['opacities'])

    _gaussian_splatting_kernel(
        smoke,
        smoke_color,
        positions,
        rotations,
        scales,
        colors,
        opacities)
