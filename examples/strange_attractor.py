import taichi as ti
import numpy as np
import taichi_volume_renderer

# Initialize Taichi (using GPU acceleration)
ti.init(arch=ti.gpu)

# Parameters for the Lorenz system
steps = 3000  # Total number of points to generate
dt = 0.00025          # Time step size
sigma = 10.0        # Prandtl number
rho = 28.0          # Rayleigh number
beta = 8.0 / 3.0    # Geometric parameter

num_particles = 500

N = 150
density = ti.field(dtype=ti.f32, shape=[N, N, N])
plot_range = ti.Vector.field(3, dtype=ti.f32, shape=[2])
plot_range.from_numpy(np.array([
    [-35, -35, -10],
    [35, 35, 60]
], dtype=float))

@ti.kernel
def compute_lorenz():
    # Euler method integration of the Lorenz system
    for _ in ti.ndrange(num_particles):
        p = ti.Vector([ti.random(), ti.random(), ti.random()]) * (plot_range[1] - plot_range[0]) + plot_range[0]  # Initial conditions
        for step in range(steps):
            # Calculate derivatives
            dp = ti.Vector([
                sigma * (p.y - p.x),
                p.x * (rho - p.z) - p.y,
                p.x * p.y - beta * p.z
            ])
        
            # Update positions
            p = p + dp * dt

            p_maped = (p - plot_range[0]) / (plot_range[1] - plot_range[0]) * N
            x_int = int(p_maped.x)
            y_int = int(p_maped.y)
            z_int = int(p_maped.z)
            if x_int >= 0 and x_int < N and y_int >= 0 and y_int < N and z_int >= 0 and z_int < N:
                density[x_int, y_int, z_int] += 1

# Run the computation
compute_lorenz()

# Lights
point_lights_pos_numpy = np.array([
    [0, 4, 7],
    [0, 0, 8]], dtype=float)
point_lights_intensity_numpy = np.array([
    [100, 50, 0],
    [0, 0, 100]], dtype=float)

taichi_volume_renderer.plot_volume(
    density,
    point_lights_pos=point_lights_pos_numpy,
    point_lights_intensity=point_lights_intensity_numpy,
    init_taichi=False,
    title="Lorenz Attractor",
    smoke_density_factor=1.5,
    light_ray_tracing_step_size_factor=0.1  # This scene requires high quality shadow. So use small steps.
)
