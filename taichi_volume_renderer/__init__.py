import math
import numpy as np
import taichi as ti

__version__ = "0.0.0"

class Scene():
    def __init__(
        self,
        smoke_numpy,
        smoke_color_numpy,
        point_lights_pos_numpy,
        point_lights_intensity_numpy
    ):
        # ti.init(arch=ti.gpu)

        # 体积数据
        self.smoke = ti.field(dtype=ti.f32, shape=smoke_numpy.shape)  # 烟浓度
        self.smoke.from_numpy(smoke_numpy)
        self.smoke_color = ti.Vector.field(3, dtype=ti.f32, shape=self.smoke.shape)  # 烟雾颜色
        self.smoke_color.from_numpy(smoke_color_numpy)

        # 光源
        self.point_lights_pos = ti.Vector.field(3, dtype=ti.f32, shape=[len(point_lights_pos_numpy)])
        self.point_lights_pos.from_numpy(point_lights_pos_numpy)
        self.point_lights_intensity = ti.Vector.field(3, dtype=ti.f32, shape=self.point_lights_pos.shape)
        self.point_lights_intensity.from_numpy(point_lights_intensity_numpy)
    
        # 摄像机
        self.fov = 0.6  # 2 * tan(纵向视角 / 2)
        self.dist_limit = 100
        self.camera_distance = ti.field(dtype=ti.f32, shape=())
        self.camera_distance[None] = 3
        self.camera_phi = ti.field(dtype=ti.f32, shape=())
        self.camera_theta = ti.field(dtype=ti.f32, shape=())
        self.camera_rotate_speed_factor = 4.
        self.background = ti.Vector([0.2, 0.2, 0.2])

        # 光追
        self.step_length = ti.field(dtype=ti.f32, shape=())
        self.step_length[None] = 1 / np.max(smoke_numpy.shape) * 1.  # 这里的数越小，光线追踪越仔细
        self.step_length_light = ti.field(dtype=ti.f32, shape=())
        self.step_length_light[None] = 1 / np.max(smoke_numpy.shape) * 3.  # 这里的数越小，阴影计算越仔细
        self.stop_threshold = ti.field(dtype=ti.f32, shape=())
        self.stop_threshold[None] = 0.01  # 当视线累积透明度低于此值时终止光线追踪

        # 体积中的光能量密度
        self.light_density = ti.Vector.field(3, dtype=ti.f32, shape=smoke_numpy.shape)

        @ti.kernel
        def update_light():  # Update light_density
            for i, j, k in self.light_density:
                self.light_density[i, j, k] = ti.Vector([0., 0., 0.])
                pos = ti.Vector([float(i), float(j), float(k)]) / self.smoke.shape - 0.5
                for l in ti.ndrange(self.point_lights_pos.shape[0]):
                    d = self.point_lights_pos[l] - pos
                    distance_squared = ti.math.dot(d, d)
                    transmittance = 1.
                    d = d.normalized()
                    pos_2 = pos
                    while True:
                        if pos_2.x > 0.5 and d.x > 0 or pos_2.x < -0.5 and d.x < 0:
                            break
                        if pos_2.y > 0.5 and d.y > 0 or pos_2.y < -0.5 and d.y < 0:
                            break
                        if pos_2.z > 0.5 and d.z > 0 or pos_2.z < -0.5 and d.z < 0:
                            break
                        pos_maped = (pos_2 + 0.5) * self.smoke.shape
                        x_int = int(pos_maped.x)
                        y_int = int(pos_maped.y)
                        z_int = int(pos_maped.z)
                        if x_int >= 0 and x_int < self.smoke.shape[0] and y_int >= 0 and y_int < self.smoke.shape[1] and z_int >= 0 and z_int < self.smoke.shape[2]:
                            transmittance *= 1 - self.smoke[x_int, y_int, z_int] * self.step_length_light[None]
                        pos_2 += d * self.step_length_light[None]
                    self.light_density[i, j, k] += self.point_lights_intensity[l] * (transmittance / distance_squared)
        self.update_light = update_light

        @ti.kernel
        def render(pixels: ti.template()):
            camera_pos = self.camera_distance[None] * ti.Vector([
                ti.cos(self.camera_phi[None]) * ti.cos(self.camera_theta[None]),
                ti.sin(self.camera_phi[None]) * ti.cos(self.camera_theta[None]),
                ti.sin(self.camera_theta[None])
            ])
            camera_u_vector = ti.Vector([
                -ti.sin(self.camera_phi[None]),
                ti.cos(self.camera_phi[None]),
                0
            ])
            camera_v_vector = ti.Vector([
                -ti.cos(self.camera_phi[None]) * ti.sin(self.camera_theta[None]),
                -ti.sin(self.camera_phi[None]) * ti.sin(self.camera_theta[None]),
                ti.cos(self.camera_theta[None])
            ])
            camera_direction = -camera_pos / self.camera_distance[None]

            for i, j in pixels:
                pos = camera_pos
                d = camera_direction + camera_u_vector * (self.fov * (i - pixels.shape[0] / 2) / pixels.shape[1]) + camera_v_vector * (self.fov * (j / pixels.shape[1] - 0.5))
                d = d.normalized()
                pixels[i, j] = ti.Vector([0., 0., 0.])
                transmittance = 1.
                distance_to_sphere = self.camera_distance[None] - 0.866025  # 这里的常数是 0.5 * sqrt(2)
                if distance_to_sphere > 0:
                    pos += d * distance_to_sphere
                while True:
                    if pos.x > 0.5 and d.x > 0 or pos.x < -0.5 and d.x < 0:
                        break
                    if pos.y > 0.5 and d.y > 0 or pos.y < -0.5 and d.y < 0:
                        break
                    if pos.z > 0.5 and d.z > 0 or pos.z < -0.5 and d.z < 0:
                        break
                    if transmittance < self.stop_threshold[None]:
                        break

                    pos_maped = (pos + 0.5) * self.smoke.shape
                    x_int = int(pos_maped.x)
                    y_int = int(pos_maped.y)
                    z_int = int(pos_maped.z)
                    if x_int >= 0 and x_int < self.smoke.shape[0] and y_int >= 0 and y_int < self.smoke.shape[1] and z_int >= 0 and z_int < self.smoke.shape[2]:
                        transmittance *= 1 - self.smoke[x_int, y_int, z_int] * self.step_length[None]
                        pixels[i, j] += self.smoke[x_int, y_int, z_int] * self.smoke_color[x_int, y_int, z_int] * self.step_length[None] * self.light_density[x_int, y_int, z_int] * transmittance
                    pos += d * self.step_length[None]
                pixels[i, j] += self.background * transmittance
        self.render = render

    def mouse_pressed_event(self, pos):
        pass

    def mouse_drag_event(self, pos, pos_delta):
        self.camera_phi[None] -= pos_delta[0] * self.camera_rotate_speed_factor
        self.camera_theta[None] -= pos_delta[1] * self.camera_rotate_speed_factor
        if self.camera_theta[None] < math.pi * -0.5:
            self.camera_theta[None] = math.pi * -0.5
        if self.camera_theta[None] > math.pi * 0.5:
            self.camera_theta[None] = math.pi * 0.5
