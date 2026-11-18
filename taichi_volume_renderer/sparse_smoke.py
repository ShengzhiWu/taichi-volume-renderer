"""Taichi smoke ray marching over sparse 8³ voxel bricks."""

import numpy as np
import taichi as ti

from . import _OrbitScene
from .cxw import load_cxw
from .math import rotation_matrix_to_quaternion, rotation_quaternion_to_matrix
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
        self, data, background=(0.2, 0.2, 0.2), camera="overview",
        density=0.22, step_length=1.0, light_step_length=3.0,
        lighting=True, point_lights_pos=None, point_lights_intensity=None,
        materials=None, stop_threshold=0.01, max_steps=10000,
    ):
        """Use voxel units for ray steps and the dense scene's units for point lights."""
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
        if not isinstance(max_steps, (int, np.integer)) or not 0 < max_steps <= 2**31 - 1:
            raise ValueError("max_steps must be a positive 32-bit integer")
        self._max_steps = int(max_steps)
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
        self._upper.from_numpy(np.array([(entity.upper + 127) // 128 for entity in data.entities], dtype=np.int32))
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
        extent = upper - lower
        self._scale = float(extent.max())
        self._world_extent = ti.Vector.field(3, dtype=ti.f32, shape=())
        self._world_extent[None] = extent
        self._world_center = ti.Vector.field(3, dtype=ti.f32, shape=())
        self._world_center[None] = self._center
        if self._lighting:
            if point_lights_pos is None:
                point_lights_pos = [[0, 0, 5]]
            positions = np.asarray(point_lights_pos, dtype=np.float32)
            if point_lights_intensity is None:
                point_lights_intensity = [[50, 50, 50]]
            intensities = np.asarray(point_lights_intensity, dtype=np.float32)
            if (positions.ndim != 2 or positions.shape[1] != 3 or not len(positions)
                    or positions.shape != intensities.shape or not np.all(np.isfinite(positions))
                    or not np.all(np.isfinite(intensities)) or np.any(intensities < 0)):
                raise ValueError("point lights need matching, finite Nx3 positions and nonnegative intensities")
            self._lights_pos = ti.Vector.field(3, dtype=ti.f32, shape=len(positions))
            self._lights_intensity = ti.Vector.field(3, dtype=ti.f32, shape=len(positions))
            # The dense renderer is Z-up; the CXW world and lookup are Y-up.
            self._lights_pos.from_numpy(np.ascontiguousarray(positions[:, [0, 2, 1]]))
            self._lights_intensity.from_numpy(intensities)
            self._init_light_cache(data, intensities)
        self._player_pose = data.player_pose
        self.set_view(camera)
        self.render = self._make_render(long_edge_fov=True)

    def set_view(self, camera):
        """Reset to the overview or to the saved player view, when provided."""
        if camera == "overview":
            self._view_basis[None] = ti.Matrix([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
            self._target[None] = self._center
            self.camera_distance = 3
            self.set_camera_phi(0)
            self.set_camera_theta(0)
        elif camera == "saved" and self._player_pose is not None:
            position, rotation = self._player_pose
            forward = rotation[:, 2]
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

    def get_camera_pose(self):
        """Return the world position, XYZW rotation, and orbit distance in voxel units."""
        backward, right = self._camera_axes()
        forward = -backward
        rotation = np.column_stack((right, np.cross(forward, right), forward))
        distance = self.camera_distance * self._scale
        position = self._target[None].to_numpy() + distance * backward
        quaternion = rotation_matrix_to_quaternion(rotation)
        return {
            "position": position.tolist(),
            "rotation": quaternion[[1, 2, 3, 0]].tolist(),
            "orbit_distance": distance,
        }

    def set_camera_pose(self, pose):
        """Restore a camera pose in CXW world coordinates."""
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
        forward = rotation[:, 2]
        target = position + distance * forward
        camera_distance = distance / self._scale
        limit = np.finfo(np.float32).max
        if (distance > limit or np.any(np.abs(position) > limit) or np.any(np.abs(target) > limit)
                or not np.finfo(np.float32).tiny <= camera_distance <= limit):
            raise ValueError("Camera pose is outside the supported range")

        self._view_basis[None] = np.column_stack((-forward, rotation[:, 0], rotation[:, 1]))
        self._target[None] = target
        self.camera_distance = camera_distance
        self.set_camera_phi(0)
        self.set_camera_theta(0)

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
        offset = (backward - new_backward) * self.camera_distance * self._scale
        self._target[None] = self._target[None].to_numpy() + offset

    def _init_light_cache(self, data, intensities):
        # Store exactly the same light value once when all sources are achromatic.
        self._light_channels = 3 if np.any(intensities != intensities[:, :1]) else 1
        self._light_cache = ti.Vector.field(self._light_channels, dtype=ti.f32)
        for channel in range(self._light_channels):
            # Separate buffers keep each channel within the Vulkan buffer limit.
            fields = ti.FieldsBuilder()
            fields.dense(ti.ij, (self.brick_count, 512)).place(self._light_cache.get_scalar_field(channel))
            fields.finalize()

        locations = np.full((self.brick_count, 4), -1, dtype=np.int32)
        for entity_id, entity in enumerate(data.entities):
            rows = slice(entity.first_sector, entity.end_sector)
            sector_ids, absolute_bricks = np.nonzero(data.brick_ids[rows] >= 0)
            brick_ids = data.brick_ids[rows][sector_ids, absolute_bricks]
            local = np.stack(
                (absolute_bricks & 15, (absolute_bricks >> 4) & 15, absolute_bricks >> 8), axis=1
            )
            locations[brick_ids, :3] = data.sectors[rows][sector_ids] * 128 + local * 8
            locations[brick_ids, 3] = entity_id
        self._brick_locations = ti.Vector.field(4, dtype=ti.i32, shape=self.brick_count)
        self._brick_locations.from_numpy(locations)

    def update_light(self):
        """Precompute lighting for occupied voxels before rendering."""
        if self._lighting:
            # Use short launches for desktop GPUs that enforce a kernel timeout.
            batch_size = 512
            for first in range(0, self.brick_count, batch_size):
                self._cache_lighting(first, min(first + batch_size, self.brick_count))
            ti.sync()

    @ti.kernel
    def _cache_lighting(self, first: ti.i32, end: ti.i32):
        for brick, voxel in ti.ndrange((first, end), 512):
            code = self._block_code(brick, voxel)
            location = self._brick_locations[brick]
            if location.w >= 0 and code != 0 and self._materials[code].w != 0:
                coordinates = location.xyz + ti.Vector([voxel & 7, (voxel >> 3) & 7, voxel >> 6])
                entity = location.w
                center = self._positions[entity] + self._rotations[entity] @ (ti.cast(coordinates, ti.f32) + 0.5)
                light = self._sample_light(center)
                for channel in ti.static(range(self._light_channels)):
                    self._light_cache[brick, voxel][channel] = light[channel]

    @ti.func
    def _block_code(self, brick, voxel):
        pair = self._blocks[brick, voxel >> 1]
        return int((pair >> ((voxel & 1) * 16)) & ti.u32(0xffff))

    @ti.func
    def _cached_light(self, brick, voxel):
        light = ti.Vector([1.0, 1.0, 1.0])
        if ti.static(self._lighting):
            if ti.static(self._light_channels == 1):
                light *= self._light_cache[brick, voxel][0]
            else:
                light = self._light_cache[brick, voxel]
        return light

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
        rotation = self._rotations[entity]
        position = self._positions[entity]
        local_direction = rotation.transpose() @ direction
        local_origin = rotation.transpose() @ (origin - position)
        p = local_origin + local_direction * t
        lower, upper = self._lower[entity], self._upper[entity]
        voxel_lower = lower * 128
        # Match the dense sampler's cast from coordinates relative to the grid origin.
        voxel = voxel_lower + ti.cast(p - voxel_lower, ti.i32)
        sector_cell = voxel >> 7
        code, next_sample = 0, t
        brick, index = -1, 0
        if (sector_cell >= lower).all() and (sector_cell < upper).all():
            sector = self._sector_id(sector_cell, self._ranges[entity])
            if sector < 0:
                next_sample += _distance_to_cell_exit(p, local_direction, sector_cell, 128)
            else:
                brick_cell = voxel >> 3
                local_brick = brick_cell - sector_cell * 16
                brick = self._brick_ids[sector, local_brick.x + 16 * local_brick.y + 256 * local_brick.z]
                if brick < 0:
                    next_sample += _distance_to_cell_exit(p, local_direction, brick_cell, 8)
                else:
                    local_voxel = voxel - brick_cell * 8
                    index = local_voxel.x + 8 * local_voxel.y + 64 * local_voxel.z
                    code = self._block_code(brick, index)
        else:
            entry, exit = _intersect_box(local_origin, local_direction, voxel_lower - 1.0,
                                         upper * 128.0, t, end)
            next_sample = end
            if entry < exit:
                next_sample = entry
        return code, next_sample, brick, index

    @ti.func
    def _advance(self, t, next_sample, step, remaining):
        # Keep the original sample lattice. Empty space still consumes the step budget.
        steps = ti.min(float(remaining), (next_sample - t) / step)
        return ti.max(1, int(steps))

    @ti.func
    def _volume_exit(self, origin, direction):
        half_extent = self._world_extent[None] * 0.5
        center = self._world_center[None]
        _, end = _intersect_box(origin, direction, center - half_extent, center + half_extent,
                                -1e30, 1e30)
        return end

    @ti.func
    def _light_transmittance(self, origin, direction):
        transmittance, t = 1.0, 0.0
        end = self._volume_exit(origin, direction)
        while t <= end and transmittance != 0.0:
            density, next_sample = 0.0, 1e30
            for entity in range(self.entity_count):
                code, next_entity_sample, _, _ = self._lookup(origin, direction, t, 1e30, entity)
                next_sample = ti.min(next_sample, next_entity_sample)
                if code != 0:
                    density += self._materials[code].w
            if next_sample > end:
                break
            transmittance *= 1.0 - density * self._light_step
            next_t = t + float(self._advance(t, next_sample, self._light_step, 2**24)) * self._light_step
            if next_t <= t:
                break
            t = next_t
        return transmittance

    @ti.func
    def _sample_light(self, voxel_center):
        light = ti.Vector([1.0, 1.0, 1.0])
        if ti.static(self._lighting):
            light = ti.Vector.zero(ti.f32, 3)
            pos = (voxel_center - self._world_center[None]) / self._scale
            for i in range(self._lights_pos.shape[0]):
                offset = self._lights_pos[i] - pos
                distance_squared = offset.dot(offset)
                if distance_squared > 0.0:
                    visibility = self._light_transmittance(voxel_center, offset.normalized())
                    light += self._lights_intensity[i] * (visibility / distance_squared)
        return light

    @ti.func
    def ray_tracing(self, pos, direction):
        basis = self._view_basis[None]
        origin = self._target[None] + basis @ (pos * self._scale)
        direction = basis @ direction
        origin += direction * (ti.max(0.0, self._camera_distance[None] - 0.866025) * self._scale)
        color = ti.Vector.zero(ti.f32, 3)
        transmittance, index = 1.0, 0
        end = self._volume_exit(origin, direction)
        while index < self._max_steps and transmittance >= self._stop_threshold:
            t = index * self._step
            if t > end:
                break
            density, next_sample = 0.0, 1e30
            lit_color = ti.Vector.zero(ti.f32, 3)
            for entity in range(self.entity_count):
                code, next_entity_sample, brick, voxel = self._lookup(origin, direction, t, 1e30, entity)
                next_sample = ti.min(next_sample, next_entity_sample)
                if code != 0:
                    material = self._materials[code]
                    density += material.w
                    if material.w != 0:
                        lit_color += material.xyz * self._cached_light(brick, voxel) * material.w
            if next_sample > end:
                break
            if density != 0:
                color, transmittance = composite_sample(
                    color, transmittance, lit_color / density, ti.Vector([1.0, 1.0, 1.0]),
                    density * self._step,
                )
            index += self._advance(t, next_sample, self._step, self._max_steps - index)
        return color + self._background[None] * transmittance
