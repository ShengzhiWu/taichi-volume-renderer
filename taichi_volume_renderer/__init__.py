import math
import numpy as np
import taichi as ti

__version__ = "1.3.0"

class Scene():
    def __init__(
        self,
        smoke_density_taichi,
        smoke_color_taichi,
        point_lights_pos_taichi,
        point_lights_intensity_taichi,
        index_of_refraction_taichi=None,
        ray_tracing_stop_threshold=0.01,  # 0 ~ 1
        background=[0.2, 0.2, 0.2],
        smoke_density_factor=1.,
        ray_tracing_step_size_factor=1.,  # The smaller the value here, the higher the ray tracing quality.
        light_ray_tracing_step_size_factor=3.  # The smaller the value here, the higher the shadow quality.
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
    
        # Camera
        self._fov = ti.field(dtype=ti.f32, shape=())
        self._fov[None] = 0.5924  # 2 * tan(vertical field of view / 2)
        self._camera_distance = ti.field(dtype=ti.f32, shape=())
        self._camera_distance[None] = 3
        self._camera_phi = ti.field(dtype=ti.f32, shape=())
        self._camera_theta = ti.field(dtype=ti.f32, shape=())
        self._background = ti.Vector.field(3, dtype=ti.f32, shape=())
        self._background[None] = ti.Vector(background)

        # Ray tracing
        self._step_length = ti.field(dtype=ti.f32, shape=())
        pixel_size = 1 / np.max(smoke_density_taichi.shape)
        self._step_length[None] = pixel_size * ray_tracing_step_size_factor
        self._step_length_light = ti.field(dtype=ti.f32, shape=())
        self._step_length_light[None] = pixel_size * light_ray_tracing_step_size_factor
        self._stop_threshold = ti.field(dtype=ti.f32, shape=())
        self._stop_threshold[None] = ray_tracing_stop_threshold  # Terminate ray tracing when the accumulated transparency of the view ray falls below this value.

        # Light density in volume
        self.light_density = ti.Vector.field(3, dtype=ti.f32, shape=smoke_density_taichi.shape)

        @ti.kernel
        def update_light():  # Update shadow.
            for i, j, k in self.light_density:
                self.light_density[i, j, k] = ti.Vector([0., 0., 0.])
                pos = ti.Vector([i + 0.5, j + 0.5, k + 0.5]) / self.smoke_density.shape - 0.5
                for l in ti.ndrange(self.point_lights_pos.shape[0]):
                    d = self.point_lights_pos[l] - pos
                    distance_squared = ti.math.dot(d, d)
                    transmittance = 1.
                    d = d.normalized()
                    pos_2 = pos
                    # pos_2 += d * (pixel_size * 0.5)
                    while True:
                        if pos_2.x > 0.5 and d.x > 0 or pos_2.x < -0.5 and d.x < 0:
                            break
                        if pos_2.y > 0.5 and d.y > 0 or pos_2.y < -0.5 and d.y < 0:
                            break
                        if pos_2.z > 0.5 and d.z > 0 or pos_2.z < -0.5 and d.z < 0:
                            break
                        pos_maped = (pos_2 + 0.5) * self.smoke_density.shape
                        x_int = int(pos_maped.x)
                        y_int = int(pos_maped.y)
                        z_int = int(pos_maped.z)
                        if x_int >= 0 and x_int < self.smoke_density.shape[0] and y_int >= 0 and y_int < self.smoke_density.shape[1] and z_int >= 0 and z_int < self.smoke_density.shape[2]:
                            transmittance *= 1 - self._smoke_density_factor[None] * self.smoke_density[x_int, y_int, z_int] * self._step_length_light[None]
                        pos_2 += d * self._step_length_light[None]
                    self.light_density[i, j, k] += self.point_lights_intensity[l] * (transmittance / distance_squared)
        self.update_light = update_light

        if self.index_of_refraction is None:
            @ti.func
            def ray_tracing_one_step(pos, d, pixels_color, transmittance):
                pos_maped = (pos + 0.5) * self.smoke_density.shape
                x_int = int(pos_maped.x)
                y_int = int(pos_maped.y)
                z_int = int(pos_maped.z)
                if x_int >= 0 and x_int < self.smoke_density.shape[0] and y_int >= 0 and y_int < self.smoke_density.shape[1] and z_int >= 0 and z_int < self.smoke_density.shape[2]:
                    transmittance *= 1 - self._smoke_density_factor[None] * self.smoke_density[x_int, y_int, z_int] * self._step_length[None]
                    pixels_color += self._smoke_density_factor[None] * self.smoke_density[x_int, y_int, z_int] * self.smoke_color[x_int, y_int, z_int] * self._step_length[None] * self.light_density[x_int, y_int, z_int] * transmittance
                pos += d * self._step_length[None]
                return pos, d, pixels_color, transmittance, False
            self.ray_tracing_one_step = ray_tracing_one_step
        else:
            @ti.func
            def ray_tracing_one_step(pos, d, pixels_color, transmittance):
                pos_maped = (pos + 0.5) * self.smoke_density.shape
                x_int = int(pos_maped.x)
                y_int = int(pos_maped.y)
                z_int = int(pos_maped.z)
                to_break = False
                if x_int >= 0 and x_int < self.smoke_density.shape[0] and y_int >= 0 and y_int < self.smoke_density.shape[1] and z_int >= 0 and z_int < self.smoke_density.shape[2]:
                    transmittance *= 1 - self._smoke_density_factor[None] * self.smoke_density[x_int, y_int, z_int] * self._step_length[None]
                    pixels_color += self._smoke_density_factor[None] * self.smoke_density[x_int, y_int, z_int] * self.smoke_color[x_int, y_int, z_int] * self._step_length[None] * self.light_density[x_int, y_int, z_int] * transmittance

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
            pixels_color = ti.Vector([0., 0., 0.])
            transmittance = 1.
            distance_to_sphere = self._camera_distance[None] - 0.866025  # The constant here is 0.5 * math.sqrt(2)
            if distance_to_sphere > 0:
                pos += d * distance_to_sphere
            while True:
                if pos.x > 0.5 and d.x > 0 or pos.x < -0.5 and d.x < 0:
                    break
                if pos.y > 0.5 and d.y > 0 or pos.y < -0.5 and d.y < 0:
                    break
                if pos.z > 0.5 and d.z > 0 or pos.z < -0.5 and d.z < 0:
                    break
                if transmittance < self._stop_threshold[None]:
                    break

                pos, d, pixels_color, transmittance, to_break = self.ray_tracing_one_step(pos, d, pixels_color, transmittance)
                if to_break:
                    break
            pixels_color += self._background[None] * transmittance
            return pixels_color
        self.ray_tracing = ray_tracing

        @ti.kernel
        def render(pixels: ti.template()):
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

            for i, j in pixels:
                pos = camera_pos
                d = camera_direction + camera_u_vector * (self._fov[None] * (i - pixels.shape[0] / 2) / pixels.shape[1]) + camera_v_vector * (self._fov[None] * (j / pixels.shape[1] - 0.5))
                d = d.normalized()
                
                pixels[i, j] = self.ray_tracing(pos, d)
        self.render = render
    
    @property
    def smoke_density_factor(self):
        return self._smoke_density_factor[None]

    @smoke_density_factor.setter
    def smoke_density_factor(self, value):
        self._smoke_density_factor[None] = value

    def get_vertical_field_of_view(self, degrees=True):  # Get vertical field of view. Default is 33°.
        return math.atan(self._fov[None] / 2) * 2 * (180 / math.pi if degrees else 1)

    def set_vertical_field_of_view(self, angle, degrees=True):  # Set vertical field of view. Default is 33°.
        self._fov[None] = 2 * math.tan(angle * (math.pi / 180 if degrees else 1) / 2)

    def get_camera_phi(self, degrees=True):
        return self._camera_phi[None] * (180 / math.pi if degrees else 1)

    def set_camera_phi(self, angle, degrees=True):
        self._camera_phi[None] = angle * (math.pi / 180 if degrees else 1)

    def get_camera_theta(self, degrees=True):
        return self._camera_theta[None] * (180 / math.pi if degrees else 1)

    def set_camera_theta(self, angle, degrees=True):
        self._camera_theta[None] = angle * (math.pi / 180 if degrees else 1)
        if self._camera_theta[None] < math.pi * -0.5:
            self._camera_theta[None] = math.pi * -0.5
        if self._camera_theta[None] > math.pi * 0.5:
            self._camera_theta[None] = math.pi * 0.5
    
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
        resolution=(720, 720),
        ray_tracing_stop_threshold=0.01,  # 0 ~ 1
        background=[0.2, 0.2, 0.2],
        init_taichi=True,
        taichi_arch=ti.gpu,
        smoke_density_factor=1.,
        ray_tracing_step_size_factor=1.,  # The smaller the value here, the higher the ray tracing quality.
        light_ray_tracing_step_size_factor=3.  # The smaller the value here, the higher the shadow quality.
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
            ray_tracing_stop_threshold=ray_tracing_stop_threshold,
            background=background,
            smoke_density_factor=smoke_density_factor,
            ray_tracing_step_size_factor=ray_tracing_step_size_factor,
            light_ray_tracing_step_size_factor=light_ray_tracing_step_size_factor)

        # Window
        self.resolution = resolution
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=resolution)

        # Interaction
        self.mouse_pressed = False
        self.cursor_start_pos = (-1, -1)
        self.camera_rotation_speed = 230.  # Unit: degree pre image width or height
    
    def mouse_pressed_event(self, pos):
        pass

    def mouse_drag_event(self, pos, pos_delta):
        self.scene.set_camera_phi(self.scene.get_camera_phi() - pos_delta[0] * self.camera_rotation_speed)
        self.scene.set_camera_theta(self.scene.get_camera_theta() - pos_delta[1] * self.camera_rotation_speed)
    
    def show(
            self,
            title="Render",
            update_light_each_step=False,
            callback=None,  # Users can update smoke density, rotate camera etc. each step by assigning this callback function.
            image_process=None,  # Users can edit the rendering result before it displayed in the window each step by assigning this callback function.
            enable_mouse_rotating=True
        ):
        self.scene.update_light()  # Calculate light and shadow

        gui = ti.GUI(title, res=self.resolution)
        iteration = 0
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            if update_light_each_step:
                self.scene.update_light()
            self.scene.render(self.pixels)
            if not image_process is None:
                image_process(iteration, self.pixels)
            gui.set_image(self.pixels)
            gui.show()

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

            if not callback is None:
                callback(iteration, self.scene)
            iteration += 1

def plot_volume(
    smoke_density=None,  # Can be NumPy array or Taichi field.
    smoke_color=None,  # Can be None, NumPy array or Taichi vector field. If left None, uniform white applied.
    index_of_refraction=None,  # Can be None, NumPy array or Taichi vector field.
    point_lights_pos=None,  # Can be None, NumPy array or Taichi vector field. If left None, default lights applied.
    point_lights_intensity=None,  # Can be None, NumPy array or Taichi vector field. If left None, default lights applied.
    resolution=(720, 720),
    ray_tracing_stop_threshold=0.01,  # 0 ~ 1
    background=[0.2, 0.2, 0.2],
    init_taichi=True,
    taichi_arch=ti.gpu,
    smoke_density_factor=1.,
    ray_tracing_step_size_factor=1.,  # The smaller the value here, the higher the ray tracing quality.
    light_ray_tracing_step_size_factor=3.,  # The smaller the value here, the higher the shadow quality.

    title="Render",
    update_light_each_step=False,
    callback=None,  # Users can update smoke density, rotate camera etc. each step by assigning this callback function.
    image_process=None,  # Users can edit the rendering result before it displayed in the window each step by assigning this callback function.
    enable_mouse_rotating=True,
    camera_phi=0,
    camera_theta=0
):
    window = DisplayWindow(
        smoke_density=smoke_density,
        smoke_color=smoke_color,
        index_of_refraction=index_of_refraction,
        point_lights_pos=point_lights_pos,
        point_lights_intensity=point_lights_intensity,
        resolution=resolution,
        ray_tracing_stop_threshold=ray_tracing_stop_threshold,
        background=background,
        init_taichi=init_taichi,
        taichi_arch=taichi_arch,
        smoke_density_factor=smoke_density_factor,
        ray_tracing_step_size_factor=ray_tracing_step_size_factor,
        light_ray_tracing_step_size_factor=light_ray_tracing_step_size_factor
    )
    window.scene.set_camera_phi(camera_phi)
    window.scene.set_camera_theta(camera_theta)

    window.show(
        title=title,
        update_light_each_step=update_light_each_step,
        callback=callback,
        image_process=image_process,
        enable_mouse_rotating=enable_mouse_rotating
    )
