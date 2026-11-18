"""Taichi smoke ray marching over sparse 8³ voxel bricks."""

import numpy as np
import taichi as ti

from . import _OrbitScene
from .cxw import load_cxw
from ._smoke import composite_sample


@ti.func
def _intersect_box(origin, direction, lower, upper, start, end):
    for axis in ti.static(range(3)):
        if direction[axis] == 0:
            if origin[axis] < lower[axis] or origin[axis] >= upper[axis]:
                end = -1.0
        else:
            lo = (lower[axis] - origin[axis]) / direction[axis]
            hi = (upper[axis] - origin[axis]) / direction[axis]
            start = ti.max(start, ti.min(lo, hi))
            end = ti.min(end, ti.max(lo, hi))
    return start, end


@ti.func
def _distance_to_cell_exit(point, direction, cell, width):
    distance = 1e30
    for axis in ti.static(range(3)):
        if direction[axis] != 0:
            edge = (cell[axis] + int(direction[axis] > 0)) * width
            distance = ti.min(distance, (edge - point[axis]) / direction[axis])
    return ti.max(0.0, distance)


def _packed_materials(density):
    codes = np.arange(65536, dtype=np.uint32)
    rgba = np.zeros((len(codes), 4), dtype=np.float32)
    rgba[1:, 3] = density
    opaque = codes >= 0x8000
    emissive = (codes & 0xc000) == 0x4000
    for channel, shift in enumerate((10, 5, 0)):
        rgba[opaque, channel] = ((codes[opaque] >> shift) & 31) / 31
    for channel, shift in enumerate((10, 6, 2)):
        rgba[emissive, channel] = ((codes[emissive] >> shift) & 15) / 15
    rgba[1:256, :3] = (0.72, 0.83, 0.9)
    return rgba


@ti.data_oriented
class SparseSmokeScene(_OrbitScene):
    """Static sparse volume. Initialize Taichi before construction."""

    @classmethod
    def from_cxw(cls, path, **options):
        return cls(load_cxw(path), **options)

    def __init__(
        self, data, background=(0.56, 0.72, 0.90), camera="overview",
        density=0.22, step_length=0.5, light_step_length=3.0,
        lighting=False, point_lights_pos=None, point_lights_intensity=None,
        materials=None, stop_threshold=0.01,
    ):
        """Use voxel units for distances and a (65536, 4) RGB-density material table."""
        for name, value in (("step_length", step_length), ("light_step_length", light_step_length)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(density) or density < 0:
            raise ValueError("density must be finite and nonnegative")
        if not 0 < stop_threshold < 1:
            raise ValueError("stop_threshold must be between 0 and 1")
        material_array = _packed_materials(density) if materials is None else np.asarray(materials, dtype=np.float32)
        if material_array.shape != (65536, 4) or not np.all(np.isfinite(material_array)) or np.any(material_array < 0):
            raise ValueError("materials must be a finite, nonnegative (65536, 4) RGB-density array")
        self._step = float(step_length)
        self._light_step = float(light_step_length)
        self._stop_threshold = float(stop_threshold)
        self._lighting = bool(lighting)
        super().__init__(background)
        self._target = ti.Vector.field(3, dtype=ti.f32, shape=())
        self._view_basis = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self._materials = ti.Vector.field(4, dtype=ti.f32, shape=65536)
        self._materials.from_numpy(material_array)

        self.entity_count = len(data.entities)
        self.sector_count = len(data.sectors)
        self.brick_count = len(data.blocks)
        self._sectors = ti.Vector.field(3, dtype=ti.i32, shape=self.sector_count)
        self._brick_ids = ti.field(ti.i32, shape=data.brick_ids.shape)
        # Pair two codes in one u32 to avoid requiring 16-bit GPU storage features.
        packed = data.blocks.view(np.uint32)
        self._blocks = ti.field(ti.u32, shape=packed.shape)
        self._sectors.from_numpy(data.sectors)
        self._brick_ids.from_numpy(data.brick_ids)
        self._blocks.from_numpy(packed)

        self._positions = ti.Vector.field(3, dtype=ti.f32, shape=self.entity_count)
        self._rotations = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.entity_count)
        self._lower = ti.Vector.field(3, dtype=ti.i32, shape=self.entity_count)
        self._upper = ti.Vector.field(3, dtype=ti.i32, shape=self.entity_count)
        self._ranges = ti.Vector.field(2, dtype=ti.i32, shape=self.entity_count)
        self._positions.from_numpy(np.array([entity.position for entity in data.entities], dtype=np.float32))
        self._rotations.from_numpy(np.array([entity.rotation for entity in data.entities], dtype=np.float32))
        self._lower.from_numpy(np.array([entity.lower // 128 for entity in data.entities], dtype=np.int32))
        self._upper.from_numpy(np.array([entity.upper // 128 for entity in data.entities], dtype=np.int32))
        self._ranges.from_numpy(np.array([
            [entity.first_sector, entity.end_sector] for entity in data.entities
        ], dtype=np.int32))

        bounds = []
        for entity in data.entities:
            center = (entity.lower + entity.upper) * 0.5
            half = np.abs(entity.rotation) @ ((entity.upper - entity.lower) * 0.5)
            center = entity.position + entity.rotation @ center
            bounds.append((center - half, center + half))
        lower, upper = np.min([pair[0] for pair in bounds], axis=0), np.max([pair[1] for pair in bounds], axis=0)
        self._center = (lower + upper) * 0.5
        self._scale = float(np.linalg.norm(upper - lower))
        if self._lighting:
            if point_lights_pos is None:
                point_lights_pos = [self._center + [0.25 * self._scale, 0.6 * self._scale, 0.15 * self._scale]]
            positions = np.asarray(point_lights_pos, dtype=np.float32)
            if point_lights_intensity is None:
                point_lights_intensity = np.full_like(positions, 0.45 * self._scale ** 2)
            intensities = np.asarray(point_lights_intensity, dtype=np.float32)
            if (positions.ndim != 2 or positions.shape[1] != 3 or not len(positions)
                    or positions.shape != intensities.shape or not np.all(np.isfinite(positions))
                    or not np.all(np.isfinite(intensities)) or np.any(intensities < 0)):
                raise ValueError("point lights need matching, finite Nx3 positions and nonnegative intensities")
            self._lights_pos = ti.Vector.field(3, dtype=ti.f32, shape=len(positions))
            self._lights_intensity = ti.Vector.field(3, dtype=ti.f32, shape=len(positions))
            self._lights_pos.from_numpy(positions)
            self._lights_intensity.from_numpy(intensities)
        self._player_pose = data.player_pose
        self.set_view(camera)
        self._init_render()

    def set_view(self, camera):
        """Reset to the overview or to the saved player view, when provided."""
        if camera == "overview":
            self._view_basis[None] = ti.Matrix([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
            self._target[None] = self._center
            self.camera_distance = 1.65
            self.set_camera_phi(45)
            self.set_camera_theta(32)
        elif camera == "saved" and self._player_pose is not None:
            position, rotation = self._player_pose
            forward = rotation[:, 2]
            # The orbit camera with zero angles looks along -X, with +Y right and +Z up.
            self._view_basis[None] = np.column_stack((-forward, rotation[:, 0], rotation[:, 1]))
            self.camera_distance = 64.0 / self._scale
            self._target[None] = position + 64.0 * forward
            self.set_camera_phi(0)
            self.set_camera_theta(0)
        else:
            raise ValueError("Expected 'overview' or an available 'saved' camera pose")

    def _camera_axes(self):
        phi = self.get_camera_phi(degrees=False)
        theta = self.get_camera_theta(degrees=False)
        basis = self._view_basis[None].to_numpy()
        backward = basis @ [np.cos(phi) * np.cos(theta), np.sin(phi) * np.cos(theta), np.sin(theta)]
        right = basis @ [-np.sin(phi), np.cos(phi), 0.0]
        return backward, right

    def move_camera(self, forward, right, up):
        """Translate the camera by distances in voxel units."""
        backward_axis, right_axis = self._camera_axes()
        offset = -forward * backward_axis + right * right_axis + np.array([0.0, up, 0.0])
        self._target[None] = self._target[None].to_numpy() + offset

    def turn_camera(self, delta_phi, delta_theta):
        """Change the view direction while keeping the camera in place."""
        backward, _ = self._camera_axes()
        self.set_camera_phi(self.get_camera_phi() + delta_phi)
        self.set_camera_theta(self.get_camera_theta() + delta_theta)
        new_backward, _ = self._camera_axes()
        offset = (backward - new_backward) * (self.camera_distance * self._scale)
        self._target[None] = self._target[None].to_numpy() + offset

    def update_light(self):
        """Lighting is computed during tracing."""

    @ti.func
    def _sector_id(self, coord, bounds):
        lo, hi = bounds[0], bounds[1]
        found = -1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            other = self._sectors[mid]
            if (coord == other).all():
                found = mid
                break
            if (other.x < coord.x or
                (other.x == coord.x and (other.y < coord.y or
                 (other.y == coord.y and other.z < coord.z)))):
                lo = mid + 1
            else:
                hi = mid
        return found


    @ti.func
    def _lookup(self, origin, direction, t, end, entity):
        rotation = self._rotations[entity].transpose()
        direction = rotation @ direction
        origin = rotation @ (origin - self._positions[entity])
        p = origin + direction * t
        voxel = ti.cast(ti.floor(p), ti.i32)
        sector_cell = voxel >> 7
        lower, upper = self._lower[entity], self._upper[entity]
        code, next_sample = 0, t
        if (sector_cell >= lower).all() and (sector_cell < upper).all():
            sector = self._sector_id(sector_cell, self._ranges[entity])
            if sector < 0:
                next_sample += _distance_to_cell_exit(p, direction, sector_cell, 128)
            else:
                brick_cell = voxel >> 3
                local_brick = brick_cell - sector_cell * 16
                brick = self._brick_ids[sector, local_brick.x + 16 * local_brick.y + 256 * local_brick.z]
                if brick < 0:
                    next_sample += _distance_to_cell_exit(p, direction, brick_cell, 8)
                else:
                    local_voxel = voxel - brick_cell * 8
                    index = local_voxel.x + 8 * local_voxel.y + 64 * local_voxel.z
                    pair = self._blocks[brick, index >> 1]
                    code = int((pair >> ((index & 1) * 16)) & ti.u32(0xffff))
        else:
            entry, exit = _intersect_box(origin, direction, lower * 128.0, upper * 128.0, t, end)
            next_sample = end
            if entry < exit:
                next_sample = entry
        return code, next_sample

    @ti.func
    def _sample(self, origin, direction, t, end, use_color: ti.template()):
        density, next_sample = 0.0, end
        weighted_color = ti.Vector.zero(ti.f32, 3)
        for entity in range(self.entity_count):
            code, next_entity_sample = self._lookup(origin, direction, t, end, entity)
            next_sample = ti.min(next_sample, next_entity_sample)
            if code != 0:
                material = self._materials[code]
                density += material.w
                if ti.static(use_color):
                    weighted_color += material.xyz * material.w
        return density, weighted_color, next_sample

    @ti.func
    def _advance(self, index, t, next_sample, end, step):
        # Round down so a skipped cell cannot take us past the first valid sample.
        advance = 1
        if next_sample < end:
            advance = ti.max(1, int((next_sample - t) / step))
        return index + advance

    @ti.func
    def _light_transmittance(self, origin, direction, end):
        transmittance, index = 1.0, 0
        while transmittance > self._stop_threshold:
            t = index * self._light_step
            if t >= end:
                break
            density, _, next_sample = self._sample(origin, direction, t, end, False)
            if next_sample >= end:
                break
            transmittance *= 1.0 - ti.min(1.0, density * ti.min(self._light_step, end - t))
            index = self._advance(index, t, next_sample, end, self._light_step)
        return transmittance

    @ti.func
    def _sample_light(self, pos):
        light = ti.Vector([1.0, 1.0, 1.0])
        if ti.static(self._lighting):
            light = ti.Vector.zero(ti.f32, 3)
            for i in range(self._lights_pos.shape[0]):
                offset = self._lights_pos[i] - pos
                distance = ti.max(offset.norm(), 0.001)
                visibility = self._light_transmittance(pos, offset / distance, distance)
                light += self._lights_intensity[i] * (visibility / (distance * distance))
        return light

    @ti.func
    def ray_tracing(self, pos, direction):
        basis = self._view_basis[None]
        origin = self._target[None] + basis @ (pos * self._scale)
        direction = basis @ direction
        color = ti.Vector.zero(ti.f32, 3)
        transmittance, index = 1.0, 0
        end = 1e30
        while transmittance > self._stop_threshold:
            t = index * self._step
            density, weighted_color, next_sample = self._sample(origin, direction, t, end, True)
            if next_sample >= end:
                break
            if density > 0:
                opacity = ti.min(1.0, density * self._step)
                light = self._sample_light(origin + direction * t)
                color, transmittance = composite_sample(
                    color, transmittance, weighted_color / density, light, opacity,
                )
            index = self._advance(index, t, next_sample, end, self._step)
        return color + self._background[None] * transmittance
