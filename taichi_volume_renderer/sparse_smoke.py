"""Taichi smoke ray marching over sparse 8³ voxel bricks."""

from pathlib import Path

import numpy as np
import taichi as ti

from . import _OrbitScene
from .cxw import load_cxw, packed_color_materials
from .sparse_data import SparseVolumeData, index_brick_values
from ._smoke import (ambient_light_parameters, apply_rgb_gamma, composite_sample,
                     directional_light_parameters, light_cache_channels,
                     refract_at_interface, refract_from_gradient, validate_gamma)


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


@ti.data_oriented
class SparseSmokeScene(_OrbitScene):
    """Static sparse volume. Initialize Taichi before construction."""

    @classmethod
    def from_cxw(cls, path, *, gamma=1.0, **options):
        return cls(load_cxw(path), gamma=gamma, **options)

    @classmethod
    def from_file(cls, path, *, grid_name=None, **options):
        suffix = Path(path).suffix.lower()
        if suffix == ".cxw":
            if grid_name is not None:
                raise ValueError("grid_name applies only to VDB files")
            return cls.from_cxw(path, **options)
        if suffix == ".vdb":
            from .vdb import load_vdb

            return cls(load_vdb(path, grid_name=grid_name), **options)
        raise ValueError("Expected a .cxw or .vdb file")

    def __init__(
        self, data: SparseVolumeData, background=(0.2, 0.2, 0.2), camera="overview",
        density=0.22, step_length=1.0, light_step_length=3.0,
        lighting=True, point_lights_pos=None, point_lights_intensity=None,
        materials=None, stop_threshold=0.01, max_steps=None, gamma=1.0,
        directional_light=None, ambient_light=(0.0, 0.0, 0.0), storage="auto",
        index_of_refraction=None, refraction_mode="interface",
    ):
        """Use voxel units for ray steps and the dense scene's units for point lights.

        A directional light is (direction towards the source in Y-up world axes, RGB intensity).
        Ambient light is a constant RGB intensity added at every occupied voxel.
        index_of_refraction is an optional 65,536-entry lookup table for indexed storage.
        Entry 0 is the ambient medium, including air outside allocated bricks.
        refraction_mode selects the dense gradient step or discrete interface optics.
        """
        for name, value in (("step_length", step_length), ("light_step_length", light_step_length)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(density) or density < 0:
            raise ValueError("density must be finite and nonnegative")
        validate_gamma(gamma)
        ambient = ambient_light_parameters(ambient_light)
        if not 0 < stop_threshold < 1:
            raise ValueError("stop_threshold must be between 0 and 1")
        self._step = float(step_length)
        self._light_step = float(light_step_length)
        self._stop_threshold = float(stop_threshold)
        self._lighting = bool(lighting)
        self._directional = directional_light is not None
        if max_steps is not None and (
            not isinstance(max_steps, (int, np.integer)) or not 0 < max_steps <= 2**31 - 1
        ):
            raise ValueError("max_steps must be None or a positive 32-bit integer")
        if refraction_mode not in ("gradient", "interface"):
            raise ValueError("refraction_mode must be 'gradient' or 'interface'")
        self._refractive = index_of_refraction is not None
        self._interface = self._refractive and refraction_mode == "interface"
        default_max_steps = 4600 if self._interface else (10000 if self._refractive else 0)
        self._max_steps = default_max_steps if max_steps is None else int(max_steps)
        super().__init__(background, y_up=True)
        self.entity_count = len(data.entities)
        self.sector_count = len(data.sectors)
        self.brick_count = len(data.blocks)
        self._sectors = ti.Vector.field(3, dtype=ti.i32, shape=self.sector_count)
        self._brick_ids = ti.field(ti.i32, shape=data.brick_ids.shape)
        self._sectors.from_numpy(data.sectors)
        self._brick_ids.from_numpy(data.brick_ids)
        self._load_voxels(data.blocks, materials, density, gamma, storage)
        if self._refractive:
            if self._direct or data.blocks.dtype != np.uint16:
                raise ValueError("Refraction requires indexed uint16 material codes")
            indices = np.asarray(index_of_refraction,dtype=np.float32)
            if indices.shape != (65536,) or not np.all(np.isfinite(indices) & (indices > 0)):
                raise ValueError("Expected a positive finite index of refraction for each material code")
            self._ior = ti.field(ti.f32,shape=65536)
            self._ior.from_numpy(indices)

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
            self._ambient = ti.Vector.field(3, dtype=ti.f32, shape=())
            self._ambient[None] = ambient
        if self._lighting and self._directional:
            if point_lights_pos is not None or point_lights_intensity is not None:
                raise ValueError("Specify a directional light or point lights")
            direction, intensity = directional_light_parameters(directional_light)
            self._light_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
            self._light_direction[None] = direction
            self._light_color = ti.Vector.field(3, dtype=ti.f32, shape=())
            self._light_color[None] = intensity
            self._init_light_cache(data, np.vstack((ambient, intensity)))
        elif self._lighting:
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
            # Point-light positions follow the dense API's Z-up convention.
            self._lights_pos.from_numpy(np.ascontiguousarray(positions[:, [0, 2, 1]]))
            self._lights_intensity.from_numpy(intensities)
            self._init_light_cache(data, np.vstack((ambient, intensities)))
        self._saved_camera_pose = data.player_pose
        self.set_view(camera)
        self._long_edge_fov = True
        self.render = self._make_render()

    def set_view(self, camera):
        """Reset to the overview or to the saved player view, when provided."""
        if camera == "overview":
            self._target[None] = self._center
            self.camera_distance = 3
            self.set_camera_phi(0)
            self.set_camera_theta(0)
        elif camera == "saved" and self._saved_camera_pose is not None:
            position, rotation = self._saved_camera_pose
            forward = rotation[:, 2]
            self.camera_distance = 64.0 / self._scale
            self._target[None] = position + 64.0 * forward
            self._set_camera_direction(-forward)
        else:
            raise ValueError("Expected 'overview' or an available 'saved' camera pose")

    def _brick_field(self, channels):
        field = ti.Vector.field(channels, dtype=ti.f32)
        for channel in range(channels):
            # Keep each channel in its own Vulkan buffer.
            fields = ti.FieldsBuilder()
            fields.dense(ti.ij, (self.brick_count, 512)).place(field.get_scalar_field(channel))
            fields.finalize()
        return field

    def _load_voxels(self, blocks, materials, density, gamma, storage):
        blocks = np.asarray(blocks)
        shape = (self.brick_count, 512)
        indexed_input = blocks.dtype == np.uint16 and blocks.shape == shape
        if storage in (None, "auto"):
            storage = "indexed" if indexed_input else "direct"
        if storage not in ("indexed", "direct"):
            raise ValueError("storage must be 'auto', 'indexed', or 'direct'")
        self._direct = storage == "direct"

        if not indexed_input:
            if not np.issubdtype(blocks.dtype, np.floating) or blocks.shape not in (shape, (*shape, 4)):
                raise ValueError("Expected uint16 indices, float densities, or float RGB-density bricks")
            if materials is not None:
                raise ValueError("Float bricks already contain their own voxel values")
            if not self._direct:
                if not np.all(np.isfinite(blocks)) or np.any(blocks < 0):
                    raise ValueError("Voxel values must be finite and nonnegative")
                blocks, materials = index_brick_values(blocks)
                indexed_input = True

        if indexed_input:
            palette = packed_color_materials(density) if materials is None else np.asarray(materials, dtype=np.float32)
            if palette.shape != (65536, 4) or not np.all(np.isfinite(palette)) or np.any(palette < 0):
                raise ValueError("materials must be a finite, nonnegative (65536, 4) RGB-density array")
            if gamma != 1.0:
                palette = palette.copy()
                palette[:, :3] = apply_rgb_gamma(palette[:, :3], gamma)

            # Pair two codes in one u32 to avoid requiring 16-bit GPU storage features.
            packed = np.ascontiguousarray(blocks).view(np.uint32)
            self._blocks = ti.field(ti.u32)
            self._materials = ti.Vector.field(4, dtype=ti.f32)
            fields = ti.FieldsBuilder()
            fields.dense(ti.ij, packed.shape).place(self._blocks)
            fields.dense(ti.i, 65536).place(self._materials)
            indexed_tree = fields.finalize()
            self._blocks.from_numpy(packed)
            self._materials.from_numpy(palette)

            if self._direct:
                self._value_channels = 4
                self._values = self._brick_field(self._value_channels)
                self._expand_indexed_voxels()
                ti.sync()
                indexed_tree.destroy()
            return

        if blocks.shape == shape:
            blocks = blocks[..., None]
        self._value_channels = blocks.shape[2]
        self._values = self._brick_field(self._value_channels)
        for channel in range(self._value_channels):
            values = np.ascontiguousarray(blocks[..., channel], dtype=np.float32)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("Direct voxel values must be finite and nonnegative")
            if channel < self._value_channels - 1:
                values = apply_rgb_gamma(values, gamma)
            self._values.get_scalar_field(channel).from_numpy(values)

    @ti.kernel
    def _expand_indexed_voxels(self):
        for brick, voxel in self._blocks:
            for offset in ti.static(range(2)):
                index = 2 * voxel + offset
                code = self._block_code(brick, index)
                if code != 0:
                    self._values[brick, index] = self._materials[code]

    def _init_light_cache(self, data, intensities):
        # Store exactly the same light value once when all sources are achromatic.
        self._light_channels = light_cache_channels(intensities)
        self._light_cache = self._brick_field(self._light_channels)

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
            location = self._brick_locations[brick]
            if location.w >= 0 and self._sample_voxel(brick, voxel, False).w != 0:
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
    def _sample_voxel(self, brick, voxel, with_color: ti.template()):
        sample = ti.Vector.zero(ti.f32, 4)
        if ti.static(self._direct):
            if ti.static(with_color and self._value_channels == 4):
                sample = self._values[brick, voxel]
            else:
                sample.w = self._values[brick, voxel][self._value_channels - 1]
                if ti.static(with_color):
                    sample.xyz = ti.Vector([1.0, 1.0, 1.0])
        else:
            code = self._block_code(brick, voxel)
            if code != 0:
                sample = self._materials[code]
        return sample

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
        next_sample = t
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
        else:
            entry, exit = _intersect_box(local_origin, local_direction, voxel_lower - 1.0,
                                         upper * 128.0, t, end)
            next_sample = end
            if entry < exit:
                next_sample = entry
        return next_sample, brick, index

    @ti.func
    def _advance(self, t, next_sample, step):
        # Round down to stay on the sample lattice without stepping past the next occupied region.
        return ti.max(1.0, ti.floor((next_sample - t) / step))

    @ti.func
    def _volume_interval(self, origin, direction, padding):
        half_extent = self._world_extent[None] * 0.5 + padding
        center = self._world_center[None]
        return _intersect_box(origin, direction, center - half_extent, center + half_extent, 0.0, 1e30)

    @ti.func
    def _light_transmittance(self, origin, direction):
        transmittance, t = 1.0, 0.0
        _, end = self._volume_interval(origin, direction, 0.)
        while t <= end and transmittance != 0.0:
            density, next_sample = 0.0, 1e30
            for entity in range(self.entity_count):
                next_entity_sample, brick, voxel = self._lookup(origin, direction, t, 1e30, entity)
                next_sample = ti.min(next_sample, next_entity_sample)
                if brick >= 0:
                    density += self._sample_voxel(brick, voxel, False).w
            if next_sample > end:
                break
            transmittance *= 1.0 - density * self._light_step
            next_t = t + ti.min(self._advance(t, next_sample, self._light_step), 2**24) * self._light_step
            if next_t <= t:
                break
            t = next_t
        return transmittance

    @ti.func
    def _sample_refractive_index(self, origin, direction, t):
        ambient_ior = self._ior[0]
        index, next_sample = ambient_ior, 1e30
        for entity in range(self.entity_count):
            next_entity, brick, voxel = self._lookup(origin, direction, t, 1e30, entity)
            next_sample = ti.min(next_sample,next_entity)
            if index == ambient_ior and brick >= 0:
                index = self._ior[self._block_code(brick,voxel)]
        return index, next_sample

    @ti.func
    def _ior_gradient(self, origin, direction, t):
        gradient = ti.Vector.zero(ti.f32, 3)
        next_sample = 1e30
        for neighbor in range(6):
            axis = neighbor // 2
            sign = 1. if neighbor % 2 else -1.
            offset = ti.Vector.zero(ti.f32, 3)
            offset[axis] = sign
            index, next_neighbor = self._sample_refractive_index(origin + offset, direction, t)
            gradient[axis] += .5 * sign * index
            next_sample = ti.min(next_sample, next_neighbor)
        return gradient, next_sample

    @ti.func
    def _ior_at(self, point):
        # At t=0 only the material matters; discard the directional skip distance.
        index, _ = self._sample_refractive_index(point, ti.Vector([1., 1., 1.]), 0.)
        return index

    @ti.func
    def _interface_normal(self, point, direction):
        gradient = ti.Vector.zero(ti.f32, 3)
        for sample in range(6):
            axis = sample // 2
            wide = sample % 2 != 0
            radius = 2.576 if wide else 1.12
            weight = .58 if wide else 1.
            offset = ti.Vector.zero(ti.f32, 3)
            offset[axis] = radius
            gradient[axis] += weight * (self._ior_at(point + offset) - self._ior_at(point - offset))
        normal = -direction
        if gradient.norm_sqr() > 0.:
            normal = gradient.normalized()
        return normal

    @ti.func
    def _sample_light(self, voxel_center):
        light = ti.Vector([1.0, 1.0, 1.0])
        if ti.static(self._lighting):
            light = self._ambient[None]
            if ti.static(self._directional):
                visibility = self._light_transmittance(voxel_center, self._light_direction[None])
                light += self._light_color[None] * visibility
            else:
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
        origin = self._target[None] + pos * self._scale
        if ti.static(not self._interface):
            origin += direction * (ti.max(0.0, (origin - self._world_center[None]).norm() / self._scale - 0.866025) * self._scale)
        color = ti.Vector.zero(ti.f32, 3)
        transmittance = 1.0
        padding = ti.static(1. if self._refractive and not self._interface else 0.)
        entry, end = self._volume_interval(origin, direction, padding)
        index = ti.ceil(entry / self._step)
        previous_steps = 0.
        previous_medium, interfaces = 1., 0
        lower, upper = origin, origin
        if ti.static(self._interface):
            if entry >= end:
                end = -1.
            origin += direction * (entry + .004)
            index = 0.
            previous_medium = self._ior[0]
            half_extent = self._world_extent[None] * .5
            lower = self._world_center[None] - half_extent
            upper = self._world_center[None] + half_extent
        while entry <= end and transmittance >= self._stop_threshold:
            if ti.static(self._max_steps > 0):
                if index + previous_steps >= self._max_steps:
                    break
            t = index * self._step
            if ti.static(self._interface):
                if (origin < lower).any() or (origin >= upper).any():
                    break
                previous_steps += 1.
            else:
                if t > end:
                    break
            density, next_sample = 0.0, 1e30
            lit_color = ti.Vector.zero(ti.f32, 3)
            medium = 1.
            has_brick, occupied = False, False
            if ti.static(self._refractive):
                medium = self._ior[0]
            for entity in range(self.entity_count):
                next_entity_sample, brick, voxel = self._lookup(origin, direction, t, 1e30, entity)
                next_sample = ti.min(next_sample, next_entity_sample)
                if brick >= 0:
                    material = self._sample_voxel(brick, voxel, True)
                    density += material.w
                    if material.w != 0:
                        lit_color += material.xyz * self._cached_light(brick, voxel) * material.w
                    if ti.static(self._refractive):
                        code = self._block_code(brick, voxel)
                        if medium == self._ior[0]:
                            medium = self._ior[code]
                        if ti.static(self._interface):
                            has_brick = True
                            occupied = occupied or code != 0

            if ti.static(self._interface):
                # A non-refractive voxel receives the ray without an air interface.
                if medium != previous_medium and (medium != self._ior[0] or not occupied) and interfaces < 19:
                    normal = self._interface_normal(origin - direction * (.5 * self._step), direction)
                    bent, fresnel, reflected = refract_at_interface(
                        direction, normal, previous_medium, medium,
                    )
                    interfaces += 1
                    if reflected:
                        origin -= direction * self._step
                        origin += bent * (.35 * self._step)
                        direction = bent
                        continue
                    color += transmittance * fresnel * self._background[None]
                    transmittance *= 1. - fresnel
                    direction = bent
                previous_medium = medium

            if next_sample > end and ti.static(not self._refractive):
                break
            if density != 0:
                color, transmittance = composite_sample(
                    color, transmittance, lit_color / density, ti.Vector([1.0, 1.0, 1.0]),
                    density * self._step,
                )
            if ti.static(self._refractive and not self._interface):
                # Only skip space empty for the whole centered-difference stencil.
                gradient, next_optical_sample = self._ior_gradient(origin, direction, t)
                next_sample = ti.min(next_sample, next_optical_sample)
                bent, stop = refract_from_gradient(direction,medium,gradient,self._step)
                if stop:
                    break
                if (bent != direction).any():
                    origin += direction*t
                    direction = bent
                    entry, end = self._volume_interval(origin,direction,padding)
                    previous_steps += index
                    index = 1.
                    continue
            if ti.static(self._interface):
                advance = self._step
                if not has_brick:
                    advance = ti.max(advance, next_sample + .003)
                origin += direction * advance
            else:
                if next_sample > end:
                    break
                next_index = index + self._advance(t, next_sample, self._step)
                if next_index <= index:
                    break
                index = next_index
        return color + self._background[None] * transmittance
