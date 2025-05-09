import numpy as np
import taichi as ti

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
def draw_point(
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
        draw_point(smoke_density_taichi, smoke_color_taichi, start * (1 - t) + end * t, point_strength, color)

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

def draw_line(
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
        draw_point(smoke_density_taichi, smoke_color_taichi, start + x * (t * height) + y * (radius * ti.cos(2 * ti.math.pi * rounds * t)) + z * (radius * ti.sin(2 * ti.math.pi * rounds * t)), point_strength, color)

def draw_helix(
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

# TODO: Particles to volume
