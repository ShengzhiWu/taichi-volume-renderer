import json
import sys
from time import perf_counter

import numpy as np
import taichi as ti

from ._smoke import composite_sample
from .math import rotation_matrix_to_quaternion, rotation_quaternion_to_matrix

__version__ = "1.6.0"

class _OrbitScene:
    def __init__(self, background, y_up=False):
        self._y_up = y_up
        self._scale = self._navigation_unit = 1.0
        self._target = ti.Vector.field(3, dtype=ti.f32, shape=())
        self._view_basis = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self._view_basis[None] = np.eye(3, dtype=np.float32)
        # Camera
        self._fov = ti.field(dtype=ti.f32, shape=())
        self._fov[None] = 0.5924  # 2 * tan(vertical field of view / 2)
        self._camera_distance = ti.field(dtype=ti.f32, shape=())
        self._camera_distance[None] = 3
        self._camera_phi = ti.field(dtype=ti.f32, shape=())
        self._camera_theta = ti.field(dtype=ti.f32, shape=())
        self._background = ti.Vector.field(3, dtype=ti.f32, shape=())
        self._background[None] = ti.Vector(background)

    def _make_render(self, pixel_type=ti.template(), long_edge_fov=False):
        @ti.kernel
        def render(pixels: pixel_type):
            camera_pos = self._camera_distance[None] * ti.Vector([
                ti.cos(self._camera_phi[None]) * ti.cos(self._camera_theta[None]),
                ti.sin(self._camera_phi[None]) * ti.cos(self._camera_theta[None]),
                ti.sin(self._camera_theta[None])
            ])
            camera_u_vector = ti.Vector([
                -ti.sin(self._camera_phi[None]),
                ti.cos(self._camera_phi[None]),
                0
            ])
            camera_v_vector = ti.Vector([
                -ti.cos(self._camera_phi[None]) * ti.sin(self._camera_theta[None]),
                -ti.sin(self._camera_phi[None]) * ti.sin(self._camera_theta[None]),
                ti.cos(self._camera_theta[None])
            ])
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
        return np.arctan(self._fov[None] / 2) * 2 * (180 / np.pi if degrees else 1)

    def set_vertical_field_of_view(self, angle, degrees=True):  # Set vertical field of view. Default is 33°.
        self._fov[None] = 2 * np.tan(angle * (np.pi / 180 if degrees else 1) / 2)

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
        basis = self._view_basis[None].to_numpy()
        backward = basis @ [np.cos(phi) * np.cos(theta), np.sin(phi) * np.cos(theta), np.sin(theta)]
        right = basis @ [-np.sin(phi), np.cos(phi), 0.0]
        return backward, right

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

    def set_camera_pose(self, pose):
        """Restore a camera pose in Y-up voxel coordinates."""
        if not isinstance(pose, dict) or "position" not in pose or "rotation" not in pose:
            raise ValueError("Expected a camera position and rotation")
        position = np.asarray(pose["position"], dtype=np.float64)
        quaternion = np.asarray(pose["rotation"], dtype=np.float64)
        distance = float(pose.get("orbit_distance", 64.0))
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

        self._view_basis[None] = np.column_stack((-forward, self._pose_axes(rotation[:, 0]), self._pose_axes(rotation[:, 1])))
        self._target[None] = target
        self.camera_distance = camera_distance
        self.set_camera_phi(0)
        self.set_camera_theta(0)

    def move_camera(self, forward, right, up):
        """Translate the camera by distances in voxel units."""
        backward_axis, right_axis = self._camera_axes()
        offset = -forward * backward_axis + right * right_axis + self._pose_axes(np.array([0.0, up, 0.0]))
        self._target[None] = self._target[None].to_numpy() + offset * self._navigation_unit

    def turn_camera(self, delta_phi, delta_theta):
        """Change the view direction while keeping the camera in place."""
        backward, _ = self._camera_axes()
        self.set_camera_phi(self.get_camera_phi() + delta_phi)
        self.set_camera_theta(self.get_camera_theta() + delta_theta)
        new_backward, _ = self._camera_axes()
        offset = (backward - new_backward) * self.camera_distance * self._scale
        self._target[None] = self._target[None].to_numpy() + offset

class Scene(_OrbitScene):
    def __init__(
        self,
        smoke_density_taichi,
        smoke_color_taichi,
        point_lights_pos_taichi,
        point_lights_intensity_taichi,
        lighting=True,
        index_of_refraction_taichi=None,
        ray_tracing_stop_threshold=0.01,  # 0 ~ 1
        background=[0.2, 0.2, 0.2],
        smoke_density_factor=1.,
        ray_tracing_step_size_factor=1.,  # The smaller the value here, the higher the ray tracing quality.
        light_ray_tracing_step_size_factor=3.,  # The smaller the value here, the higher the shadow quality.
        ray_tracing_max_steps=10000  # This only takes effect in scenes where light rays may bend, such as those containing refractive materials.
    ):
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
        grid_size = max(grid_shape)
        pixel_size = 1 / grid_size
        self._navigation_unit = pixel_size
        half_extent = tuple(n * pixel_size / 2 for n in grid_shape)

        @ti.func
        def past_volume(pos, direction):
            half = ti.Vector(half_extent)
            return ((pos > half) & (direction > 0)).any() or ((pos < -half) & (direction < 0)).any()
        self._step_length[None] = pixel_size * ray_tracing_step_size_factor
        self._step_length_light = ti.field(dtype=ti.f32, shape=())
        self._step_length_light[None] = pixel_size * light_ray_tracing_step_size_factor
        self._stop_threshold = ti.field(dtype=ti.f32, shape=())
        self._stop_threshold[None] = ray_tracing_stop_threshold  # Terminate ray tracing when the accumulated transparency of the view ray falls below this value.

        # Light density in volume
        self.light_density = ti.Vector.field(3, dtype=ti.f32, shape=smoke_density_taichi.shape)
        self.light_density.from_numpy(np.ones(list(smoke_density_taichi.shape) + [3]))

        if lighting:
            @ti.kernel
            def update_light():  # Update shadow.
                for i, j, k in self.light_density:
                    self.light_density[i, j, k] = ti.Vector([0., 0., 0.])
                    pos = ti.Vector([i + 0.5, j + 0.5, k + 0.5]) * pixel_size - ti.Vector(half_extent)
                    for l in ti.ndrange(self.point_lights_pos.shape[0]):
                        d = self.point_lights_pos[l] - pos
                        distance_squared = ti.math.dot(d, d)
                        transmittance = 1.
                        d = d.normalized()
                        pos_2 = pos
                        # pos_2 += d * (pixel_size * 0.5)
                        while True:
                            if past_volume(pos_2, d):
                                break
                            pos_maped = (pos_2 + ti.Vector(half_extent)) * grid_size
                            x_int = int(pos_maped.x)
                            y_int = int(pos_maped.y)
                            z_int = int(pos_maped.z)
                            if x_int >= 0 and x_int < self.smoke_density.shape[0] and y_int >= 0 and y_int < self.smoke_density.shape[1] and z_int >= 0 and z_int < self.smoke_density.shape[2]:
                                transmittance *= 1 - self._smoke_density_factor[None] * self.smoke_density[x_int, y_int, z_int] * self._step_length_light[None]
                            pos_2 += d * self._step_length_light[None]
                        self.light_density[i, j, k] += self.point_lights_intensity[l] * (transmittance / distance_squared)
            self.update_light = update_light
        else:
            @ti.kernel
            def update_light():  # Update shadow.
                pass
            self.update_light = update_light

        if self.index_of_refraction is None:
            @ti.func
            def ray_tracing_one_step(pos, d, pixels_color, transmittance):
                pos_maped = (pos + ti.Vector(half_extent)) * grid_size
                x_int = int(pos_maped.x)
                y_int = int(pos_maped.y)
                z_int = int(pos_maped.z)
                if x_int >= 0 and x_int < self.smoke_density.shape[0] and y_int >= 0 and y_int < self.smoke_density.shape[1] and z_int >= 0 and z_int < self.smoke_density.shape[2]:
                    opacity = self._smoke_density_factor[None] * self.smoke_density[x_int, y_int, z_int] * self._step_length[None]
                    pixels_color, transmittance = composite_sample(
                        pixels_color, transmittance, self.smoke_color[x_int, y_int, z_int],
                        self.light_density[x_int, y_int, z_int], opacity,
                    )
                pos += d * self._step_length[None]
                return pos, d, pixels_color, transmittance, False
            self.ray_tracing_one_step = ray_tracing_one_step
        else:
            @ti.func
            def ray_tracing_one_step(pos, d, pixels_color, transmittance):
                pos_maped = (pos + ti.Vector(half_extent)) * grid_size
                x_int = int(pos_maped.x)
                y_int = int(pos_maped.y)
                z_int = int(pos_maped.z)
                to_break = False
                if x_int >= 0 and x_int < self.smoke_density.shape[0] and y_int >= 0 and y_int < self.smoke_density.shape[1] and z_int >= 0 and z_int < self.smoke_density.shape[2]:
                    opacity = self._smoke_density_factor[None] * self.smoke_density[x_int, y_int, z_int] * self._step_length[None]
                    pixels_color, transmittance = composite_sample(
                        pixels_color, transmittance, self.smoke_color[x_int, y_int, z_int],
                        self.light_density[x_int, y_int, z_int], opacity,
                    )

                    if x_int >= 1 and x_int < self.smoke_density.shape[0] - 1 and y_int >= 1 and y_int < self.smoke_density.shape[1] - 1 and z_int >= 1 and z_int < self.smoke_density.shape[2] - 1:
                        index_of_refraction = self.index_of_refraction[x_int, y_int, z_int]
                        index_of_refraction_grad = ti.Vector([
                            (self.index_of_refraction[x_int + 1, y_int, z_int] - self.index_of_refraction[x_int - 1, y_int, z_int]),
                            (self.index_of_refraction[x_int, y_int + 1, z_int] - self.index_of_refraction[x_int, y_int - 1, z_int]),
                            (self.index_of_refraction[x_int, y_int, z_int + 1] - self.index_of_refraction[x_int, y_int, z_int - 1])
                        ]) / (pixel_size * 2.)
                        index_of_refraction_change = ti.math.dot(d, index_of_refraction_grad) * self._step_length[None]
                        if index_of_refraction_change != 0.:
                            normal = index_of_refraction_grad.normalized()
                            relative_index_of_refraction = (index_of_refraction + index_of_refraction_change * 0.5) / (index_of_refraction - index_of_refraction_change * 0.5)
                            d_n = ti.math.dot(d, normal) * normal
                            d_t = d - d_n
                            cos_old = ti.abs(ti.math.dot(d, normal))
                            sin_old = (1 - cos_old ** 2) ** 0.5
                            sin_new = sin_old / relative_index_of_refraction
                            if sin_new > 1.:
                                to_break = True
                            else:
                                cos_new = (1 - sin_new ** 2) ** 0.5
                                d_n *= cos_new / cos_old
                                d_t *= sin_new / sin_old
                                d = d_n + d_t
                                d = d.normalized()

                pos += d * self._step_length[None]
                return pos, d, pixels_color, transmittance, to_break
            self.ray_tracing_one_step = ray_tracing_one_step

        @ti.func
        def ray_tracing(pos, d):
            basis = self._view_basis[None]
            pos = self._target[None] + basis @ pos
            d = basis @ d
            pixels_color = ti.Vector([0., 0., 0.])
            transmittance = 1.
            distance_to_sphere = pos.norm() - 0.866025  # Radius of the sphere enclosing the volume.
            if distance_to_sphere > 0:
                pos += d * distance_to_sphere
            i = ray_tracing_max_steps
            while i > 0:
                if past_volume(pos, d):
                    break
                if transmittance < self._stop_threshold[None]:
                    break

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
        resolution=(720, 720),
        ray_tracing_stop_threshold=0.01,  # 0 ~ 1
        background=[0.2, 0.2, 0.2],
        init_taichi=True,
        taichi_arch=ti.gpu,
        smoke_density_factor=1.,
        ray_tracing_step_size_factor=1.,  # The smaller the value here, the higher the ray tracing quality.
        light_ray_tracing_step_size_factor=3.,  # The smaller the value here, the higher the shadow quality.
        ray_tracing_max_steps=10000  # This only takes effect in scenes where light rays may bend, such as those containing refractive materials.
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
            smoke_color_numpy = smoke_color
            smoke_color = ti.Vector.field(3, dtype=ti.f32, shape=smoke_color_numpy.shape[:-1])
            smoke_color.from_numpy(smoke_color_numpy)

        if not index_of_refraction is None:
            if not isinstance(index_of_refraction, ti.Field):
                index_of_refraction_numpy = index_of_refraction
                index_of_refraction = ti.field(dtype=ti.f32, shape=index_of_refraction_numpy.shape)
                index_of_refraction.from_numpy(index_of_refraction_numpy)

        if point_lights_pos is None:
            point_lights_pos = np.array([[0, 0, 5]], dtype=float)
        if not isinstance(point_lights_pos, ti.Field):
            point_lights_pos_numpy = point_lights_pos
            point_lights_pos = ti.Vector.field(3, dtype=ti.f32, shape=point_lights_pos_numpy.shape[:-1])
            point_lights_pos.from_numpy(point_lights_pos_numpy)

        if point_lights_intensity is None:
            point_lights_intensity = np.array([[50, 50, 50]], dtype=float)
        if not isinstance(point_lights_intensity, ti.Field):
            point_lights_intensity_numpy = point_lights_intensity
            point_lights_intensity = ti.Vector.field(3, dtype=ti.f32, shape=point_lights_intensity_numpy.shape[:-1])
            point_lights_intensity.from_numpy(point_lights_intensity_numpy)

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
            ray_tracing_max_steps=ray_tracing_max_steps)

        self._init_display(resolution)

    @classmethod
    def from_cxw(cls, path, resolution=(1280, 720), init_taichi=True,
                 taichi_arch=ti.gpu, *, gamma=1.0, **scene_options):
        """Create a CXW smoke window with an optional RGB power-law exponent."""
        from .sparse_smoke import SparseSmokeScene

        if init_taichi:
            ti.init(arch=taichi_arch)
        window = cls.__new__(cls)
        window.scene = SparseSmokeScene.from_cxw(path, gamma=gamma, **scene_options)
        window._init_display(resolution)
        return window

    def _init_display(self, resolution):
        self.resolution = tuple(resolution)
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=resolution)

        # Interaction
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
        if self.fly:
            self.scene.turn_camera(delta_phi, delta_theta)
        else:
            self.scene.set_camera_phi(self.scene.get_camera_phi() + delta_phi)
            self.scene.set_camera_theta(self.scene.get_camera_theta() + delta_theta)

    def mouse_wheel_event(self, delta):
        if self.fly:
            self.scene.move_camera(0.05 * delta[1], 0, 0)
        else:
            self.scene.camera_distance *= (1 - self.camera_zooming_speed) ** (2 * delta[1])

    def _copy_view(self, window):
        window.copy_text(json.dumps(self.scene.get_camera_pose()))

    def _paste_view(self, window):
        try:
            self.scene.set_camera_pose(json.loads(window.read_text()))
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
        if tab and not self._tab_pressed:
            self.fly = not self.fly
        self._tab_pressed = tab
        reset = gui.is_pressed("r")
        if reset and not self._reset_pressed:
            self.scene.set_camera_pose(self._initial_pose)
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
            resize=False,
            context_menu=None,
            fly=False,
        ):
        self.fly = fly
        self._initial_pose = self.scene.get_camera_pose()
        self._tab_pressed = self._reset_pressed = False
        self.scene.update_light()  # Calculate light and shadow
        self._last_frame = perf_counter()

        windows = sys.platform == "win32"
        resize = windows and resize
        render = self.scene.render
        if resize:
            render = self.scene._make_render(ti.types.ndarray(dtype=ti.math.vec3, ndim=2), long_edge_fov=True)
            self.pixels = ti.Vector.ndarray(3, dtype=ti.f32, shape=self.resolution)
        if windows:
            from ._tk_window import Window

            if context_menu is None:
                context_menu = (("复制视角", self._copy_view), ("粘贴视角", self._paste_view))
            gui = Window(title, self.resolution, resize=resize, context_menu=context_menu)
        else:
            gui = ti.GUI(title, res=self.resolution)
        try:
            self._show_loop(gui, windows, resize, render, update_light_each_step, callback,
                            image_process, enable_mouse_rotating)
        finally:
            if windows:
                gui.close()

    def _show_loop(self, gui, windows, resize, render, update_light_each_step, callback,
                   image_process, enable_mouse_rotating):
        iteration = 0
        while gui.running:
            if windows:
                gui.poll_events()
                if not gui.running:
                    break
                if resize:
                    resolution = gui.take_resize()
                    if resolution is not None and resolution != self.resolution:
                        self.pixels = ti.Vector.ndarray(3, dtype=ti.f32, shape=resolution)
                        self.resolution = resolution
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
    resolution=(720, 720),
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
    enable_mouse_rotating=True
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
        ray_tracing_max_steps=ray_tracing_max_steps
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
