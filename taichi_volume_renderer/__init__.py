import json
import sys
from time import perf_counter

import numpy as np
import taichi as ti

from ._smoke import (ambient_light_parameters, apply_rgb_gamma, composite_sample,
                     follow_refractive_interface, refract_from_gradient,
                     directional_light_parameters, light_cache_channels, validate_gamma)
from .math import rotation_matrix_to_quaternion, rotation_quaternion_to_matrix

__version__ = "1.6.0"

class _OrbitScene:
    def __init__(self, background, y_up=False):
        self._y_up = y_up
        self._scale = self._navigation_unit = 1.0
        self._target = ti.Vector.field(3, dtype=ti.f32, shape=())
        # Camera
        self._fov = ti.field(dtype=ti.f32, shape=())
        self._fov[None] = 0.5924  # 2 * tan(vertical field of view / 2)
        self._long_edge_fov = False
        self._viewport_fov_ratio = 1.0
        self._custom_fov = False
        self._camera_distance = ti.field(dtype=ti.f32, shape=())
        self._camera_distance[None] = 3
        self._camera_phi = ti.field(dtype=ti.f32, shape=())
        self._camera_theta = ti.field(dtype=ti.f32, shape=())
        self._background = ti.Vector.field(3, dtype=ti.f32, shape=())
        self._background[None] = ti.Vector(background)

    def _make_render(self, pixel_type=ti.template(), long_edge_fov=None):
        if long_edge_fov is None:
            long_edge_fov = self._long_edge_fov
        @ti.kernel
        def render(pixels: pixel_type):
            world_up = ti.Vector([0., 0., 1.])
            horizontal_axis = ti.Vector([0., 1., 0.])
            if ti.static(self._y_up):
                world_up, horizontal_axis = horizontal_axis, world_up
            cos_phi, sin_phi = ti.cos(self._camera_phi[None]), ti.sin(self._camera_phi[None])
            cos_theta, sin_theta = ti.cos(self._camera_theta[None]), ti.sin(self._camera_theta[None])
            horizontal = ti.Vector([cos_phi, 0., 0.]) + sin_phi * horizontal_axis
            camera_pos = self._camera_distance[None] * (cos_theta * horizontal + sin_theta * world_up)
            camera_u_vector = ti.Vector([-sin_phi, 0., 0.]) + cos_phi * horizontal_axis
            camera_v_vector = cos_theta * world_up - sin_theta * horizontal
            camera_direction = -camera_pos / self._camera_distance[None]

            width, height = pixels.shape[0], pixels.shape[1]
            fov_edge = height
            if ti.static(long_edge_fov):
                fov_edge = ti.max(width, height)
            pixel_angle = self._fov[None] / fov_edge
            for i, j in ti.ndrange(width, height):
                pos = camera_pos
                d = (camera_direction + camera_u_vector * (pixel_angle * (i - width / 2))
                     + camera_v_vector * (pixel_angle * (j - height / 2)))
                d = d.normalized()

                pixels[i, j] = self.ray_tracing(pos, d)
        return render

    def get_vertical_field_of_view(self, degrees=True):  # Get vertical field of view. Default is 33°.
        return np.arctan(self._fov[None] * self._viewport_fov_ratio / 2) * 2 * (180 / np.pi if degrees else 1)

    def set_vertical_field_of_view(self, angle, degrees=True):  # Set vertical field of view. Default is 33°.
        self._fov[None] = 2 * np.tan(angle * (np.pi / 180 if degrees else 1) / 2) / self._viewport_fov_ratio
        self._custom_fov = True

    def get_camera_phi(self, degrees=True):
        return self._camera_phi[None] * (180 / np.pi if degrees else 1)

    def set_camera_phi(self, angle, degrees=True):
        self._camera_phi[None] = angle * (np.pi / 180 if degrees else 1)

    def get_camera_theta(self, degrees=True):
        return self._camera_theta[None] * (180 / np.pi if degrees else 1)

    def set_camera_theta(self, angle, degrees=True):
        self._camera_theta[None] = angle * (np.pi / 180 if degrees else 1)
        if self._camera_theta[None] < np.pi * -0.5:
            self._camera_theta[None] = np.pi * -0.5
        if self._camera_theta[None] > np.pi * 0.5:
            self._camera_theta[None] = np.pi * 0.5

    @property
    def camera_distance(self):
        return self._camera_distance[None]

    @camera_distance.setter
    def camera_distance(self, value):
        self._camera_distance[None] = value

    @property
    def background(self):
        return list(self._background[None])

    @background.setter
    def background(self, value):
        self._background[None] = ti.Vector(value)


    def _pose_axes(self, vector):
        # Clipboard poses use Y-up voxel coordinates for both renderers.
        return vector if self._y_up else vector[[0, 2, 1]]

    def _camera_axes(self):
        phi = self.get_camera_phi(degrees=False)
        theta = self.get_camera_theta(degrees=False)
        backward = np.array([np.cos(phi) * np.cos(theta), np.sin(phi) * np.cos(theta), np.sin(theta)])
        right = np.array([-np.sin(phi), np.cos(phi), 0.0])
        if self._y_up:
            backward, right = backward[[0, 2, 1]], right[[0, 2, 1]]
        return backward, right

    def _set_camera_direction(self, backward):
        x, y, z = backward[[0, 2, 1]] if self._y_up else backward
        horizontal = np.hypot(x, y)
        self.set_camera_phi(np.arctan2(y, x) if horizontal else 0.0, degrees=False)
        self.set_camera_theta(np.arctan2(z, horizontal), degrees=False)

    def get_camera_pose(self):
        """Return the world position, XYZW rotation, and orbit distance in voxel units."""
        backward, right = self._camera_axes()
        backward, right = self._pose_axes(backward), self._pose_axes(right)
        forward = -backward
        rotation = np.column_stack((right, np.cross(forward, right), forward))
        distance = self.camera_distance * self._scale / self._navigation_unit
        position = self._pose_axes(self._target[None].to_numpy()) / self._navigation_unit + distance * backward
        quaternion = rotation_matrix_to_quaternion(rotation)
        return {
            "position": position.tolist(),
            "rotation": quaternion[[1, 2, 3, 0]].tolist(),
            "orbit_distance": distance,
        }

    def set_camera_pose(self, pose, *, preserve_orbit_distance=False):
        """Restore position and view direction from a Y-up voxel pose."""
        if not isinstance(pose, dict) or "position" not in pose or "rotation" not in pose:
            raise ValueError("Expected a camera position and rotation")
        position = np.asarray(pose["position"], dtype=np.float64)
        quaternion = np.asarray(pose["rotation"], dtype=np.float64)
        distance = (self.camera_distance * self._scale / self._navigation_unit
                    if preserve_orbit_distance else float(pose.get("orbit_distance", 64.0)))
        if (position.shape != (3,) or quaternion.shape != (4,)
                or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
                or not np.isfinite(distance) or distance <= 0):
            raise ValueError("Invalid camera pose")
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Invalid camera rotation")
        rotation = rotation_quaternion_to_matrix(quaternion[[3, 0, 1, 2]] / norm)
        forward = self._pose_axes(rotation[:, 2])
        target = (self._pose_axes(position) + distance * forward) * self._navigation_unit
        camera_distance = distance * self._navigation_unit / self._scale
        limit = np.finfo(np.float32).max
        if (distance > limit or np.any(np.abs(position) > limit) or np.any(np.abs(target) > limit)
                or not np.finfo(np.float32).tiny <= camera_distance <= limit):
            raise ValueError("Camera pose is outside the supported range")

        self._target[None] = target
        if not preserve_orbit_distance:
            self.camera_distance = camera_distance
        self._set_camera_direction(-forward)

    def move_camera(self, forward, right, up):
        """Translate the camera by distances in voxel units."""
        backward_axis, right_axis = self._camera_axes()
        offset = -forward * backward_axis + right * right_axis + self._pose_axes(np.array([0.0, up, 0.0]))
        self._target[None] = self._target[None].to_numpy() + offset * self._navigation_unit

    def turn_camera(self, delta_phi, delta_theta):
        """Change the view direction while keeping the camera in place."""
        backward, _ = self._camera_axes()
        self.orbit_camera(delta_phi, delta_theta)
        new_backward, _ = self._camera_axes()
        offset = (backward - new_backward) * self.camera_distance * self._scale
        self._target[None] = self._target[None].to_numpy() + offset

    def orbit_camera(self, delta_phi, delta_theta):
        """Rotate around the target using the fixed world up axis."""
        self.set_camera_phi(self.get_camera_phi() + delta_phi)
        self.set_camera_theta(self.get_camera_theta() + delta_theta)

class Scene(_OrbitScene):
    def __init__(
        self,
        smoke_density_taichi,
        smoke_color_taichi,
        point_lights_pos_taichi=None,
        point_lights_intensity_taichi=None,
        lighting=True,
        index_of_refraction_taichi=None,
        ray_tracing_stop_threshold=0.01,  # 0 ~ 1
        background=[0.2, 0.2, 0.2],
        smoke_density_factor=1.,
        ray_tracing_step_size_factor=1.,  # The smaller the value here, the higher the ray tracing quality.
        light_ray_tracing_step_size_factor=3.,  # The smaller the value here, the higher the shadow quality.
        ray_tracing_max_steps=10000,
        *, gamma=1.0, directional_light=None, ambient_light=(0.0, 0.0, 0.0),
        refraction_mode="interface",
    ):
        if refraction_mode not in ("gradient", "interface"):
            raise ValueError("refraction_mode must be 'gradient' or 'interface'")
        validate_gamma(gamma)
        ambient = tuple(ambient_light_parameters(ambient_light))
        light_direction = light_intensity = None
        has_directional_light = directional_light is not None
        if directional_light is not None:
            if point_lights_pos_taichi is not None or point_lights_intensity_taichi is not None:
                raise ValueError("Specify a directional light or point lights")
            light_direction, light_intensity = map(tuple, directional_light_parameters(directional_light))

        if directional_light is None:
            if point_lights_pos_taichi is None:
                point_lights_pos_taichi = np.array([[0, 0, 5]], dtype=float)
            if not isinstance(point_lights_pos_taichi, ti.Field):
                point_lights_pos_taichi_numpy = point_lights_pos_taichi
                point_lights_pos_taichi = ti.Vector.field(3, dtype=ti.f32, shape=point_lights_pos_taichi_numpy.shape[:-1])
                point_lights_pos_taichi.from_numpy(point_lights_pos_taichi_numpy)

            if point_lights_intensity_taichi is None:
                point_lights_intensity_taichi = np.array([[50, 50, 50]], dtype=float)
            if not isinstance(point_lights_intensity_taichi, ti.Field):
                point_lights_intensity_taichi_numpy = point_lights_intensity_taichi
                point_lights_intensity_taichi = ti.Vector.field(3, dtype=ti.f32, shape=point_lights_intensity_taichi_numpy.shape[:-1])
                point_lights_intensity_taichi.from_numpy(point_lights_intensity_taichi_numpy)

        # Volume data
        self.smoke_density = smoke_density_taichi  # Smoke density
        self._smoke_density_factor = ti.field(dtype=ti.f32, shape=())
        self._smoke_density_factor[None] = smoke_density_factor
        self.smoke_color = smoke_color_taichi  # Smoke color
        self.index_of_refraction = index_of_refraction_taichi

        # Light
        self.point_lights_pos = point_lights_pos_taichi
        self.point_lights_intensity = point_lights_intensity_taichi
    
        super().__init__(background)

        # Ray tracing
        self._step_length = ti.field(dtype=ti.f32, shape=())
        grid_shape = smoke_density_taichi.shape
        has_ior = self.index_of_refraction is not None
        use_interface = has_ior and refraction_mode == "interface"
        use_gradient = has_ior and refraction_mode == "gradient"
        if has_ior and self.index_of_refraction.shape != grid_shape:
            raise ValueError("The IOR field must have the same shape as the density field")
        grid_size = max(grid_shape)
        pixel_size = 1 / grid_size
        self._navigation_unit = pixel_size
        half_extent = tuple(n * pixel_size / 2 for n in grid_shape)

        @ti.func
        def past_volume(pos, direction, padding):
            half = ti.Vector(half_extent) + padding
            return ((pos > half) & (direction > 0)).any() or ((pos < -half) & (direction < 0)).any()
        self._step_length[None] = pixel_size * ray_tracing_step_size_factor
        self._step_length_light = ti.field(dtype=ti.f32, shape=())
        self._step_length_light[None] = pixel_size * light_ray_tracing_step_size_factor
        self._stop_threshold = ti.field(dtype=ti.f32, shape=())
        self._stop_threshold[None] = ray_tracing_stop_threshold  # Terminate ray tracing when the accumulated transparency of the view ray falls below this value.

        # Light density in volume
        # Point-light fields can change color between updates.
        channels = 3 if directional_light is None else light_cache_channels((ambient, light_intensity))
        self.light_density = ti.Vector.field(channels, dtype=ti.f32, shape=grid_shape)
        self.light_density.fill(1.0)

        @ti.func
        def light_at(i, j, k):
            light = ti.Vector([1.0, 1.0, 1.0])
            if ti.static(channels == 1):
                light *= self.light_density[i, j, k][0]
            else:
                light = self.light_density[i, j, k]
            return light

        if lighting:
            @ti.func
            def light_transmittance(pos, direction):
                transmittance = 1.0
                while not past_volume(pos, direction, 0.):
                    mapped = (pos + ti.Vector(half_extent)) * grid_size
                    voxel = ti.cast(mapped, ti.i32)
                    if (voxel >= 0).all() and (voxel < ti.Vector(grid_shape)).all():
                        transmittance *= (1 - self._smoke_density_factor[None] * self.smoke_density[voxel]
                                          * self._step_length_light[None])
                    pos += direction * self._step_length_light[None]
                return transmittance

            @ti.kernel
            def update_light():
                for i, j, k in self.light_density:
                    pos = (ti.Vector([i, j, k]) + 0.5) * pixel_size - ti.Vector(half_extent)
                    light = ti.Vector(ambient)
                    if ti.static(has_directional_light):
                        light += ti.Vector(light_intensity) * light_transmittance(pos, ti.Vector(light_direction))
                    else:
                        for index in range(self.point_lights_pos.shape[0]):
                            offset = self.point_lights_pos[index] - pos
                            distance_squared = offset.dot(offset)
                            if distance_squared > 0.0:
                                transmittance = light_transmittance(pos, offset.normalized())
                                light += self.point_lights_intensity[index] * (transmittance / distance_squared)
                    if ti.static(channels == 1):
                        self.light_density[i, j, k][0] = light.x
                    else:
                        self.light_density[i, j, k] = light
            self.update_light = update_light
        else:
            @ti.kernel
            def update_light():  # Update shadow.
                pass
            self.update_light = update_light

        @ti.func
        def color_at(i, j, k):
            color = self.smoke_color[i, j, k]
            if ti.static(gamma != 1.0):
                color = ti.pow(ti.max(color, 0.0), gamma)
            return color

        @ti.func
        def ior_at(pos):
            voxel = ti.cast(ti.floor((pos + ti.Vector(half_extent)) * grid_size), ti.i32)
            index = 1.
            if (voxel >= 0).all() and (voxel < ti.Vector(grid_shape)).all():
                index = self.index_of_refraction[voxel]
            return index

        @ti.func
        def ray_tracing_one_step(pos, d, pixels_color, transmittance):
            mapped = (pos + ti.Vector(half_extent)) * grid_size
            if ti.static(use_interface):
                mapped = ti.floor(mapped)
            voxel = ti.cast(mapped, ti.i32)
            to_break = False
            if (voxel >= 0).all() and (voxel < ti.Vector(grid_shape)).all():
                x, y, z = voxel
                step = self._step_length[None]
                opacity = self._smoke_density_factor[None] * self.smoke_density[voxel] * step
                pixels_color, transmittance = composite_sample(
                    pixels_color, transmittance, color_at(x, y, z), light_at(x, y, z), opacity,
                )
                if ti.static(use_gradient):
                    if (voxel >= 1).all() and (voxel < ti.Vector(grid_shape) - 1).all():
                        indices = ti.static(self.index_of_refraction)
                        gradient = ti.Vector([
                            indices[x+1, y, z] - indices[x-1, y, z],
                            indices[x, y+1, z] - indices[x, y-1, z],
                            indices[x, y, z+1] - indices[x, y, z-1],
                        ]) / (pixel_size * 2.)
                        d, to_break = refract_from_gradient(d, indices[voxel], gradient, step)
            pos += d * self._step_length[None]
            return pos, d, pixels_color, transmittance, to_break
        self.ray_tracing_one_step = ray_tracing_one_step

        @ti.func
        def ray_tracing(pos, d):
            pos = self._target[None] + pos
            pixels_color = ti.Vector([0., 0., 0.])
            transmittance = 1.
            distance_to_sphere = pos.norm() - 0.866025  # Radius of the sphere enclosing the volume.
            if distance_to_sphere > 0:
                pos += d * distance_to_sphere
            previous_pos = pos
            previous_ior = 1.
            padding = 0.
            if ti.static(use_interface):
                previous_ior = ior_at(pos)
                padding = self._step_length[None]
            i = ray_tracing_max_steps
            while i > 0:
                if past_volume(pos, d, padding):
                    break
                if transmittance < self._stop_threshold[None]:
                    break

                if ti.static(use_interface):
                    ior = ior_at(pos)
                    if ior != previous_ior:
                        pos, d, fresnel, reflected = follow_refractive_interface(
                            previous_pos, pos, d, previous_ior, ior, pixel_size, ior_at,
                        )
                        if not reflected:
                            pixels_color += transmittance * fresnel * self._background[None]
                            transmittance *= 1. - fresnel
                            previous_ior = ior
                    previous_pos = pos
                pos, d, pixels_color, transmittance, to_break = self.ray_tracing_one_step(pos, d, pixels_color, transmittance)
                if to_break:
                    break

                i -= 1
            
            pixels_color += self._background[None] * transmittance
            return pixels_color
        self.ray_tracing = ray_tracing

        self.render = self._make_render()

    @property
    def smoke_density_factor(self):
        return self._smoke_density_factor[None]

    @smoke_density_factor.setter
    def smoke_density_factor(self, value):
        self._smoke_density_factor[None] = value

    @property
    def step_length(self):
        return self._step_length[None]

    @step_length.setter
    def step_length(self, value):
        self._step_length[None] = value
    
    @property
    def step_length_light(self):
        return self._step_length_light[None]

    @step_length_light.setter
    def step_length_light(self, value):
        self._step_length_light[None] = value
    
    @property
    def stop_threshold(self):
        return self._stop_threshold[None]

    @stop_threshold.setter
    def stop_threshold(self, value):
        self._stop_threshold[None] = value

class DisplayWindow():
    def __init__(
        self,
        smoke_density,  # Can be NumPy array or Taichi field.
        smoke_color=None,  # Can be None, NumPy array or Taichi vector field. If left None, uniform white applied.
        index_of_refraction=None,  # Can be None, NumPy array or Taichi vector field.
        point_lights_pos=None,  # Can be None, NumPy array or Taichi vector field. If left None, default lights applied.
        point_lights_intensity=None,  # Can be None, NumPy array or Taichi vector field. If left None, default lights applied.
        lighting=True,
        resolution=(1280, 720),
        ray_tracing_stop_threshold=0.01,  # 0 ~ 1
        background=[0.2, 0.2, 0.2],
        init_taichi=True,
        taichi_arch=ti.gpu,
        smoke_density_factor=1.,
        ray_tracing_step_size_factor=1.,  # The smaller the value here, the higher the ray tracing quality.
        light_ray_tracing_step_size_factor=3.,  # The smaller the value here, the higher the shadow quality.
        ray_tracing_max_steps=10000,
        *, gamma=1.0, directional_light=None, ambient_light=(0.0, 0.0, 0.0),
        refraction_mode="interface",
    ):
        if init_taichi:
            ti.init(arch=taichi_arch)
        
        if not isinstance(smoke_density, ti.Field):
            smoke_density_numpy = smoke_density
            smoke_density = ti.field(dtype=ti.f32, shape=smoke_density_numpy.shape)
            smoke_density.from_numpy(smoke_density_numpy)

        if smoke_color is None:
            smoke_color = np.ones(list(smoke_density.shape) + [3])
        if not isinstance(smoke_color, ti.Field):
            smoke_color_numpy = apply_rgb_gamma(smoke_color, gamma)
            gamma = 1.0
            smoke_color = ti.Vector.field(3, dtype=ti.f32, shape=smoke_color_numpy.shape[:-1])
            smoke_color.from_numpy(smoke_color_numpy)

        if not index_of_refraction is None:
            if not isinstance(index_of_refraction, ti.Field):
                index_of_refraction_numpy = index_of_refraction
                index_of_refraction = ti.field(dtype=ti.f32, shape=index_of_refraction_numpy.shape)
                index_of_refraction.from_numpy(index_of_refraction_numpy)

        self.scene = Scene(
            smoke_density_taichi=smoke_density,
            smoke_color_taichi=smoke_color,
            index_of_refraction_taichi=index_of_refraction,
            point_lights_pos_taichi=point_lights_pos,
            point_lights_intensity_taichi=point_lights_intensity,
            lighting=lighting,
            ray_tracing_stop_threshold=ray_tracing_stop_threshold,
            background=background,
            smoke_density_factor=smoke_density_factor,
            ray_tracing_step_size_factor=ray_tracing_step_size_factor,
            light_ray_tracing_step_size_factor=light_ray_tracing_step_size_factor,
            ray_tracing_max_steps=ray_tracing_max_steps,
            gamma=gamma, directional_light=directional_light, ambient_light=ambient_light,
            refraction_mode=refraction_mode)

        self._init_display(resolution)

    @classmethod
    def from_cxw(cls, path, resolution=(1280, 720), init_taichi=True,
                 taichi_arch=ti.gpu, *, gamma=1.0, **scene_options):
        """Create a CXW smoke window with an optional RGB power-law exponent."""
        from .sparse_smoke import SparseSmokeScene

        return cls._from_sparse(SparseSmokeScene.from_cxw, path, resolution, init_taichi,
                                taichi_arch, gamma=gamma, **scene_options)

    @classmethod
    def from_file(cls, path, resolution=(1280, 720), init_taichi=True,
                  taichi_arch=ti.gpu, **scene_options):
        """Create a sparse smoke window for a CXW or VDB file."""
        from .sparse_smoke import SparseSmokeScene

        return cls._from_sparse(SparseSmokeScene.from_file, path, resolution, init_taichi,
                                taichi_arch, **scene_options)

    @classmethod
    def _from_sparse(cls, loader, path, resolution, init_taichi, taichi_arch, **scene_options):
        if init_taichi:
            ti.init(arch=taichi_arch)
        window = cls.__new__(cls)
        window.scene = loader(path, **scene_options)
        window._init_display(resolution, orbit_controls=False)
        return window

    def _init_display(self, resolution, orbit_controls=True):
        self.resolution = tuple(resolution)
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=resolution)

        # Interaction
        self._orbit_controls = orbit_controls
        self.fly = False
        self.mouse_pressed = False
        self.cursor_start_pos = (-1, -1)
        self.camera_rotation_speed = 230.  # Unit: degree pre image width or height
        self.camera_zooming_speed = 0.0007
    
    def mouse_pressed_event(self, pos):
        pass

    def mouse_drag_event(self, pos, pos_delta):
        delta_phi = -pos_delta[0] * self.camera_rotation_speed
        delta_theta = -pos_delta[1] * self.camera_rotation_speed
        if self.fly or not self._orbit_controls:
            self.scene.turn_camera(delta_phi, delta_theta)
        else:
            self.scene.orbit_camera(delta_phi, delta_theta)

    def mouse_wheel_event(self, delta):
        if self.fly or not self._orbit_controls:
            self.scene.move_camera(0.05 * delta[1], 0, 0)
        else:
            self.scene.camera_distance *= (1 - self.camera_zooming_speed) ** (2 * delta[1])

    def _copy_view(self, window):
        window.copy_text(json.dumps(self.scene.get_camera_pose()))

    def _paste_view(self, window):
        try:
            self.scene.set_camera_pose(json.loads(window.read_text()),
                                       preserve_orbit_distance=not self._orbit_controls)
        except (ValueError, TypeError, OverflowError):
            window.show_error("粘贴视角", "剪贴板中没有有效的视角 JSON。")

    def _handle_keyboard(self, gui):
        now = perf_counter()
        duration = min(now - self._last_frame, 0.1)
        self._last_frame = now
        if gui.is_pressed(ti.GUI.ESCAPE):
            gui.running = False
            return
        tab = gui.is_pressed(ti.GUI.TAB)
        if self._orbit_controls and tab and not self._tab_pressed:
            self.fly = not self.fly
        self._tab_pressed = tab
        reset = gui.is_pressed("r")
        if reset and not self._reset_pressed:
            self.scene.set_camera_pose(self._initial_pose,
                                       preserve_orbit_distance=not self._orbit_controls)
        self._reset_pressed = reset

        forward = int(gui.is_pressed("w")) - int(gui.is_pressed("s"))
        right = int(gui.is_pressed("d")) - int(gui.is_pressed("a"))
        up = int(gui.is_pressed("e")) - int(gui.is_pressed("q"))
        if forward or right or up:
            speed = 640.0 if gui.is_pressed(ti.GUI.SHIFT) else 120.0
            distance = speed * duration
            self.scene.move_camera(2.5 * forward * distance, right * distance, up * distance)
    
    def show(
            self,
            title="Render",
            update_light_each_step=False,
            callback=None,  # Users can update smoke density, rotate camera etc. each step by assigning this callback function.
            image_process=None,  # Users can edit the rendering result before it displayed in the window each step by assigning this callback function.
            enable_mouse_rotating=True,
            resize=None,
            context_menu=None,
            fly=False,
            ggui=False,
        ):
        if ggui and ti.lang.impl.current_cfg().arch != ti.vulkan:
            raise ValueError("GGUI textures require the Taichi Vulkan backend")
        self.fly = fly
        self._initial_pose = self.scene.get_camera_pose()
        self._tab_pressed = self._reset_pressed = False
        self.scene.update_light()  # Calculate light and shadow
        self._last_frame = perf_counter()

        native_window = ggui or sys.platform == "win32"
        resize = native_window and (image_process is None if resize is None else resize)
        render = self.scene.render
        if resize or ggui:
            render = self.scene._make_render(
                ti.types.ndarray(dtype=ti.math.vec3, ndim=2),
                long_edge_fov=resize or self.scene._long_edge_fov,
            )
            self.pixels = ti.Vector.ndarray(3, dtype=ti.f32, shape=self.resolution)
        elif not isinstance(self.pixels, ti.Field):
            self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=self.resolution)
        if ggui:
            from ._ggui_window import Window

            gui = Window(title, self.resolution)
        elif native_window:
            from ._tk_window import Window

            if context_menu is None:
                context_menu = (("复制视角", self._copy_view), ("粘贴视角", self._paste_view))
            gui = Window(title, self.resolution, resize=resize, context_menu=context_menu)
        else:
            gui = ti.GUI(title, res=self.resolution)
        if resize or self.scene._long_edge_fov:
            self.scene._viewport_fov_ratio = self.resolution[1] / max(self.resolution)
            if self.scene._custom_fov:
                self.scene._fov[None] /= self.scene._viewport_fov_ratio
        try:
            self._show_loop(gui, native_window, resize, render, update_light_each_step, callback,
                            image_process, enable_mouse_rotating)
        finally:
            if self.scene._custom_fov:
                self.scene._fov[None] *= self.scene._viewport_fov_ratio
            self.scene._viewport_fov_ratio = 1.0
            if native_window:
                gui.close()

    def _show_loop(self, gui, native_window, resize, render, update_light_each_step, callback,
                   image_process, enable_mouse_rotating):
        iteration = 0
        while gui.running:
            if native_window:
                gui.poll_events()
                if not gui.running:
                    break
                if resize:
                    resolution = gui.take_resize()
                    if resolution is not None and resolution != self.resolution:
                        self.pixels = ti.Vector.ndarray(3, dtype=ti.f32, shape=resolution)
                        self.resolution = resolution
                        self.scene._viewport_fov_ratio = resolution[1] / max(resolution)
                        self.mouse_pressed = False

            for event in gui.get_events(ti.GUI.WHEEL):
                if enable_mouse_rotating:
                    self.mouse_wheel_event(event.delta)

            # Manage mouse events
            if enable_mouse_rotating:
                if gui.is_pressed(ti.GUI.LMB):
                    cursor_pos = gui.get_cursor_pos()
                    if not self.mouse_pressed:
                        self.mouse_pressed_event(cursor_pos)
                        self.mouse_pressed = True
                        self.cursor_start_pos = cursor_pos
                    if self.cursor_start_pos[0] != cursor_pos[0] or self.cursor_start_pos[1] != cursor_pos[1]:
                        self.mouse_drag_event(cursor_pos, (cursor_pos[0] - self.cursor_start_pos[0], cursor_pos[1] - self.cursor_start_pos[1]))
                        self.cursor_start_pos = cursor_pos
                else:
                    self.mouse_pressed = False

            self._handle_keyboard(gui)
            if not gui.running:
                break
            if update_light_each_step:
                self.scene.update_light()
            render(self.pixels)
            if image_process is not None:
                image_process(iteration, self.pixels)
            gui.set_image(self.pixels)
            gui.show()

            if not callback is None:
                callback(iteration, self.scene)
            iteration += 1

def plot_volume(
    smoke_density=None,  # Can be NumPy array or Taichi field.
    smoke_color=None,  # Can be None, NumPy array or Taichi vector field. If left None, uniform white applied.
    index_of_refraction=None,  # Can be None, NumPy array or Taichi vector field.
    point_lights_pos=None,  # Can be None, NumPy array or Taichi vector field. If left None, default lights applied.
    point_lights_intensity=None,  # Can be None, NumPy array or Taichi vector field. If left None, default lights applied.
    lighting=True,
    resolution=(1280, 720),
    ray_tracing_stop_threshold=0.01,  # 0 ~ 1
    background=[0.2, 0.2, 0.2],
    init_taichi=True,
    taichi_arch=ti.gpu,
    smoke_density_factor=1.,
    ray_tracing_step_size_factor=1.,  # The smaller the value here, the higher the ray tracing quality.
    light_ray_tracing_step_size_factor=3.,  # The smaller the value here, the higher the shadow quality.
    ray_tracing_max_steps=10000,  # This only takes effect in scenes where light rays may bend, such as those containing refractive materials.
    camera_phi=0,
    camera_theta=0,
    camera_distance=3,

    title="Render",
    update_light_each_step=False,
    callback=None,  # Users can update smoke density, rotate camera etc. each step by assigning this callback function.
    image_process=None,  # Users can edit the rendering result before it displayed in the window each step by assigning this callback function.
    enable_mouse_rotating=True,
    *, gamma=1.0, directional_light=None, ambient_light=(0.0, 0.0, 0.0),
    refraction_mode="interface",
):
    window = DisplayWindow(
        smoke_density=smoke_density,
        smoke_color=smoke_color,
        index_of_refraction=index_of_refraction,
        point_lights_pos=point_lights_pos,
        point_lights_intensity=point_lights_intensity,
        lighting=lighting,
        resolution=resolution,
        ray_tracing_stop_threshold=ray_tracing_stop_threshold,
        background=background,
        init_taichi=init_taichi,
        taichi_arch=taichi_arch,
        smoke_density_factor=smoke_density_factor,
        ray_tracing_step_size_factor=ray_tracing_step_size_factor,
        light_ray_tracing_step_size_factor=light_ray_tracing_step_size_factor,
        ray_tracing_max_steps=ray_tracing_max_steps,
        gamma=gamma, directional_light=directional_light, ambient_light=ambient_light,
        refraction_mode=refraction_mode,
    )
    window.scene.set_camera_phi(camera_phi)
    window.scene.set_camera_theta(camera_theta)
    window.scene.camera_distance = camera_distance

    window.show(
        title=title,
        update_light_each_step=update_light_each_step,
        callback=callback,
        image_process=image_process,
        enable_mouse_rotating=enable_mouse_rotating
    )
