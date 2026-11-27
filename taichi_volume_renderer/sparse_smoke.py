"""Taichi smoke ray marching over sparse 8³ voxel bricks."""

from pathlib import Path

import numpy as np
import taichi as ti

from . import _OrbitScene
from .cxw import load_cxw, packed_color_materials
from ._ground import _thicken_ground
from .sparse_data import SparseVolumeData, index_brick_values
from ._smoke import (INTERFACE_ENTRY_OFFSET, REFRACTIVE_MAX_STEPS, ambient_light_parameters, apply_rgb_gamma,
                     composite_sample, crosses_refractive_interface, directional_light_parameters,
                     intersect_box, light_cache_channels, refract_at_interface,
                     refract_from_gradient, validate_gamma)


@ti.func
def _distance_to_cell_exit(point, direction, cell, width):
    distance = 1e30
    for axis in ti.static(range(3)):
        if direction[axis] != 0:
            edge = (cell[axis] + int(direction[axis] > 0)) * width
            distance = ti.min(distance, (edge - point[axis]) / direction[axis])
    return ti.max(0.0, distance)


def _upload_lookup_field(values, components=None, builder=None):
    values = np.ascontiguousarray(values, dtype=np.float32 if values.dtype.kind == "f" else np.int32)
    shape = values.shape if components is None else (len(values),)
    kwargs = {"shape": shape} if builder is None else {}
    if isinstance(components, tuple):
        field = ti.Matrix.field(*components, ti.f32, **kwargs)
    elif components is None:
        field = ti.field(ti.f32 if values.dtype.kind == "f" else ti.i32, **kwargs)
    else:
        field = ti.Vector.field(components, ti.f32 if values.dtype.kind == "f" else ti.i32, **kwargs)
    if builder is not None:
        builder.dense(ti.axes(*range(len(shape))), shape).place(field)
        return field, values
    field.from_numpy(values)
    return field


@ti.data_oriented
class SparseSmokeScene(_OrbitScene):
    """Static sparse volume. Initialize Taichi before construction."""

    @classmethod
    def from_cxw(cls, path, *, gamma=1.0, thicken_ground=0, **options):
        data = _thicken_ground(load_cxw(path), thicken_ground)
        return cls(data, gamma=gamma, **options)

    @classmethod
    def from_file(cls, path, *, grid_name=None, thicken_ground=0,
                  caustics=True, caustic_photon_density=None, **options):
        suffix = Path(path).suffix.lower()
        if caustic_photon_density is not None and suffix != ".vxb":
            raise ValueError("Caustic baking requires a VXB file")
        if grid_name is not None and suffix != ".vdb":
            raise ValueError("grid_name applies only to VDB files")
        if suffix == ".cxw":
            return cls.from_cxw(path, thicken_ground=thicken_ground, **options)
        if suffix == ".vxb":
            from .vxb_scene import load_vxb_volume

            if any(name in options for name in ("materials", "index_of_refraction", "interface_normals", "emission")):
                raise ValueError("VXB supplies material values and optical properties from the file")
            data, palette, indices, normals, emission = load_vxb_volume(path, density=options.get("density", .22))
            data = _thicken_ground(data, thicken_ground)
            if caustics and indices is not None and options.get("directional_light") is not None and options.get("lighting", True):
                refractive = indices != 1.
                if any(refractive[data.blocks[start:start + 8192]].any()
                       for start in range(0, len(data.blocks), 8192)):
                    from .caustics import DEFAULT_PHOTON_DENSITY

                    options["caustic_photon_density"] = (DEFAULT_PHOTON_DENSITY if caustic_photon_density is None
                                                         else caustic_photon_density)
            return cls(data, materials=palette, index_of_refraction=indices, interface_normals=normals,
                       emission=emission, **options)
        if suffix == ".vdb":
            from .vdb import load_vdb

            data = _thicken_ground(load_vdb(path, grid_name=grid_name), thicken_ground)
            return cls(data, **options)
        raise ValueError("Expected a .cxw, .vdb, or .vxb file")

    def __init__(
        self, data: SparseVolumeData, background=(0.2, 0.2, 0.2), camera="overview",
        density=0.22, step_length=1.0, light_step_length=3.0,
        lighting=True, point_lights_pos=None, point_lights_intensity=None,
        materials=None, stop_threshold=0.01, max_steps=None, gamma=1.0,
        directional_light=None, ambient_light=(0.0, 0.0, 0.0), storage="auto",
        index_of_refraction=None, refraction_mode="interface", interface_normals=None, emission=None,
        reflection_environment=None, refractive_shadow_passthrough="auto", caustic_photon_density=None,
        caustic_multiplier=1.0, caustic_dispersion=0.0, caustic_spectral_samples=3,
    ):
        """Use voxel units for ray steps and the dense scene's units for point lights.

        A directional light is (direction towards the source in Z-up world axes, RGB intensity).
        Ambient light is a constant RGB intensity added at every occupied voxel.
        index_of_refraction is an optional 65,536-entry lookup table for indexed storage.
        Entry 0 is the ambient medium, including air outside allocated bricks.
        refraction_mode selects the dense gradient step or discrete interface optics.
        Nonzero Z-up interface_normals override the estimated normal in interface mode.
        emission is an optional 65,536-entry RGB table of linear outgoing radiance.
        reflection_environment optionally adds (sky RGB, sun RGB) to interface reflections.
        refractive_shadow_passthrough blends opaque refractive shadows (0) with ordinary shadows (1).
        Auto uses 0 when caustics are enabled and 1 otherwise.
        It applies to directional lighting when refraction is enabled.
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
        self._use_reflection_environment = reflection_environment is not None
        if self._use_reflection_environment:
            if not self._lighting or not self._directional:
                raise ValueError("Interface reflection lighting requires a directional light")
            sky, sun = reflection_environment
            self._reflection_environment = (
                tuple(ambient_light_parameters(sky)), tuple(ambient_light_parameters(sun)),
            )
        if max_steps is not None and (
            not isinstance(max_steps, (int, np.integer)) or not 0 < max_steps <= 2**31 - 1
        ):
            raise ValueError("max_steps must be None or a positive 32-bit integer")
        if refraction_mode not in ("gradient", "interface"):
            raise ValueError("refraction_mode must be 'gradient' or 'interface'")
        self._refractive = index_of_refraction is not None
        if not np.isfinite(caustic_multiplier) or caustic_multiplier < 0:
            raise ValueError("caustic_multiplier must be finite and nonnegative")
        self._caustic_multiplier = float(caustic_multiplier)
        if not np.isfinite(caustic_dispersion) or caustic_dispersion < 0:
            raise ValueError("caustic_dispersion must be finite and nonnegative")
        self._caustic_dispersion = float(caustic_dispersion)
        if (isinstance(caustic_spectral_samples, bool) or not isinstance(caustic_spectral_samples, (int, np.integer))
                or caustic_spectral_samples < 1):
            raise ValueError("caustic_spectral_samples must be a positive integer")
        self._caustic_spectral_samples = caustic_spectral_samples
        self._caustic_density = caustic_photon_density
        if self._caustic_density is not None:
            if not np.isfinite(self._caustic_density) or self._caustic_density <= 0:
                raise ValueError("caustic_photon_density must be finite and positive")
            if not (self._lighting and self._directional and self._refractive and len(data.entities) == 1):
                raise ValueError("Caustics require one indexed refractive entity and directional lighting")
            self._caustic_entity = data.entities[0]
            self._caustic_uniform_bricks = data.uniform_bricks
        mix = (float(self._caustic_density is None) if refractive_shadow_passthrough == "auto"
               else float(refractive_shadow_passthrough))
        if not 0. <= mix <= 1.:
            raise ValueError("refractive_shadow_passthrough must be auto or a number between 0 and 1")
        self._refractive_shadow_passthrough = mix if self._directional and self._refractive else 1.
        self._interface = self._refractive and refraction_mode == "interface"
        self._has_interface_normals = interface_normals is not None and (self._interface or self._caustic_density is not None)
        self._use_interface_normals = self._interface and self._has_interface_normals
        self._emissive = emission is not None
        default_max_steps = REFRACTIVE_MAX_STEPS if self._refractive else 0
        self._max_steps = default_max_steps if max_steps is None else int(max_steps)
        super().__init__(background)
        self._init_lookup(data, _upload_lookup_field)
        self._load_voxels(data.blocks, materials, density, gamma, storage)
        if self._emissive:
            if self._direct or data.blocks.dtype != np.uint16:
                raise ValueError("Emission requires indexed uint16 material codes")
            emitted = np.asarray(emission, dtype=np.float32)
            if (emitted.shape != (65536, 3) or not np.all(np.isfinite(emitted)) or np.any(emitted < 0.)):
                raise ValueError("Expected one finite, nonnegative RGB emission value per material code")
            self._emission = ti.Vector.field(3, dtype=ti.f32, shape=65536)
            self._emission.from_numpy(np.ascontiguousarray(emitted))
        if self._refractive:
            if self._direct or data.blocks.dtype != np.uint16:
                raise ValueError("Refraction requires indexed uint16 material codes")
            indices = np.asarray(index_of_refraction,dtype=np.float32)
            if indices.shape != (65536,) or not np.all(np.isfinite(indices) & (indices > 0)):
                raise ValueError("Expected a positive finite index of refraction for each material code")
            if self._caustic_density is not None and self._caustic_dispersion:
                spread = 1.7 * self._caustic_dispersion * np.abs(indices.astype(np.float64) - 1.)
                if np.any((indices - spread <= 0.) | (indices + spread > np.finfo(np.float32).max)):
                    raise ValueError("caustic_dispersion produces an out-of-range index of refraction")
            self._ior = ti.field(ti.f32,shape=65536)
            self._ior.from_numpy(indices)
        if self._has_interface_normals:
            normals = np.array(interface_normals, dtype=np.float32)
            if normals.shape != (65536, 3) or not np.all(np.isfinite(normals)):
                raise ValueError("Expected one finite interface normal per material code")
            lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
            present = lengths != 0.
            normals[present] /= lengths[present, None]
            self._interface_normals = ti.Vector.field(3, dtype=ti.f32, shape=65536)
            self._interface_normals.from_numpy(normals)

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
            self._lights_pos.from_numpy(np.ascontiguousarray(positions))
            self._lights_intensity.from_numpy(intensities)
            self._init_light_cache(data, np.vstack((ambient, intensities)))
        self._saved_camera_pose = data.player_pose
        self.set_view(camera)
        self.render = self._make_render()

    def _init_lookup(self, data, upload):
        self.entity_count = len(data.entities)
        self.sector_count = len(data.sectors)
        self.brick_count = len(data.blocks)
        self._regular_brick_count = self.brick_count - len(data.uniform_bricks)
        self._has_uniform_bricks = bool(data.uniform_bricks)
        self._native_hierarchy = data.hierarchy is not None
        self._root_shift = data.hierarchy.sizes[0].bit_length() - 1 if self._native_hierarchy else 7
        if self.sector_count:
            self._sectors = upload(np.asarray(data.sectors, dtype=np.int32), 3)
        if self._native_hierarchy:
            self._hierarchy_nodes = upload(data.hierarchy.nodes, 3)
            self._hierarchy_shifts = upload(np.asarray([size.bit_length() - 1 for size in data.hierarchy.sizes]))
        elif self.sector_count:
            self._brick_ids = upload(np.asarray(data.brick_ids, dtype=np.int32))
        if self._has_uniform_bricks and not self._native_hierarchy:
            if self.entity_count != 1:
                raise ValueError("Large uniform bricks require a single sparse entity")
            uniform = np.asarray(data.uniform_bricks, dtype=np.int32)
            levels, starts = np.unique(uniform[:, 0], return_index=True)
            ends = np.append(starts[1:], len(uniform))
            self._uniform_ranges = upload(np.column_stack((levels, starts, ends)), 3)
            self._uniform_coords = upload(uniform[:, 1:4], 3)
            self._uniform_rows = upload(uniform[:, 4])
        self._positions = upload(np.array([entity.position for entity in data.entities], dtype=np.float32), 3)
        self._rotations = upload(np.array([entity.rotation for entity in data.entities], dtype=np.float32), (3, 3))
        root_width = 1 << self._root_shift
        self._lower = upload(np.array([entity.lower // root_width for entity in data.entities]), 3)
        self._upper = upload(np.array([(entity.upper - 1) // root_width + 1 for entity in data.entities]), 3)
        self._ranges = upload(np.array([
            [entity.first_sector, entity.end_sector] for entity in data.entities
        ]), 2)

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

    def _brick_field(self, channels, brick_count):
        field = ti.Vector.field(channels, dtype=ti.f32)
        for channel in range(channels):
            # Keep each channel in its own Vulkan buffer.
            fields = ti.FieldsBuilder()
            fields.dense(ti.ij, (brick_count, 512)).place(field.get_scalar_field(channel))
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
            self._gray_materials = bool(np.all(palette[:, :3] == palette[:, :1]))

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
                self._values = self._brick_field(self._value_channels, self.brick_count)
                self._expand_indexed_voxels()
                ti.sync()
                indexed_tree.destroy()
            return

        if blocks.shape == shape:
            blocks = blocks[..., None]
        self._value_channels = blocks.shape[2]
        self._gray_materials = self._value_channels == 1
        self._values = self._brick_field(self._value_channels, self.brick_count)
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
        # A scalar still suffices when both the light and the outgoing color are gray.
        self._light_channels = (light_cache_channels(intensities)
                                if self._gray_materials and not self._emissive and self._caustic_density is None
                                else 3)
        locations = np.full((self._regular_brick_count, 4), -1, dtype=np.int32)
        uniform = np.zeros(self._regular_brick_count, dtype=bool)
        if self._native_hierarchy:
            locations[:, :3] = data.hierarchy.mixed_origins
            locations[:, 3] = 0
        else:
            for entity_id, entity in enumerate(data.entities):
                rows = slice(entity.first_sector, entity.end_sector)
                sector_ids, absolute_bricks = np.nonzero(data.brick_ids[rows] >= 0)
                brick_ids = data.brick_ids[rows][sector_ids, absolute_bricks]
                local = np.column_stack((absolute_bricks & 15, (absolute_bricks >> 4) & 15, absolute_bricks >> 8))
                locations[brick_ids, :3] = data.sectors[rows][sector_ids] * 128 + local * 8
                locations[brick_ids, 3] = entity_id
            regular = data.blocks[:self._regular_brick_count]
            axes = tuple(range(1, regular.ndim))
            for start in range(0, len(regular), 8192):
                bricks = regular[start:start + 8192]
                uniform[start:start + len(bricks)] = np.all(bricks == bricks[:, :1], axis=axes)
            uniform &= locations[:, 3] >= 0

        # A nonnegative row addresses the per-voxel cache. A negative row
        # addresses the per-brick cache.
        light_rows = np.zeros(self.brick_count, dtype=np.int32)
        shared = np.flatnonzero(uniform)
        light_rows[shared] = -(1 + np.arange(len(shared), dtype=np.int32))
        centers = locations[shared].astype(np.float64)
        centers[:, :3] += 4.
        if self._has_uniform_bricks:
            bricks = np.asarray(data.uniform_bricks, dtype=np.int32)
            width = np.exp2(bricks[:, :1].astype(np.float64))
            native_centers = (bricks[:, 1:4] + .5) * width
            light_rows[bricks[:, 4]] = -(len(shared) + 1 + np.arange(len(bricks), dtype=np.int32))
            centers = np.concatenate((centers, np.column_stack((native_centers, np.zeros(len(bricks))))))
            shared = np.concatenate((shared, bricks[:, 4]))
        self._uniform_light_count = len(centers)
        if self._uniform_light_count:
            entity_ids = centers[:, 3].astype(np.intp)
            rotations = np.asarray([entity.rotation for entity in data.entities], dtype=np.float32)
            positions = np.asarray([entity.position for entity in data.entities], dtype=np.float32)
            world_centers = np.einsum("nij,nj->ni", rotations[entity_ids], centers[:, :3]) + positions[entity_ids]
            self._uniform_light_centers = _upload_lookup_field(world_centers, 3)
            self._uniform_light_bricks = _upload_lookup_field(shared)
            self._uniform_light_cache = ti.Vector.field(self._light_channels, dtype=ti.f32,
                                                        shape=self._uniform_light_count)

        mixed = np.flatnonzero(~uniform).astype(np.int32)
        self._mixed_light_count = len(mixed)
        if self._mixed_light_count:
            light_rows[mixed] = np.arange(self._mixed_light_count, dtype=np.int32)
            self._light_cache = self._brick_field(self._light_channels, self._mixed_light_count)
            self._brick_locations = ti.Vector.field(4, dtype=ti.i32, shape=self._mixed_light_count)
            self._brick_locations.from_numpy(locations[mixed])
            self._mixed_light_bricks = _upload_lookup_field(mixed)
        if self._uniform_light_count:
            self._light_rows = _upload_lookup_field(light_rows)

    def update_light(self):
        """Overwrite the radiance cache with direct lighting and material emission."""
        if self._lighting:
            # Use short launches for desktop GPUs that enforce a kernel timeout.
            batch_size = 512
            for first in range(0, self._mixed_light_count, batch_size):
                self._cache_lighting(first, min(first + batch_size, self._mixed_light_count))
            uniform_batch_size = 8192
            for first in range(0, self._uniform_light_count, uniform_batch_size):
                self._cache_uniform_lighting(first, min(first + uniform_batch_size, self._uniform_light_count))
            ti.sync()

    def update_caustics(self):
        """Trace and add caustics for the current light. Call after update_light()."""
        if self._caustic_density is not None:
            from .caustics import _caustic_power

            for addresses, power in _caustic_power(
                self, self._caustic_entity, self._caustic_uniform_bricks, -self._light_direction[None].to_numpy(),
                self._caustic_density, self._caustic_dispersion, self._caustic_spectral_samples,
            ):
                if len(power):
                    self._add_caustic_radiance(addresses, power)
            ti.sync()

    @ti.func
    def _outgoing_radiance(self, brick, voxel, albedo, incident):
        radiance = albedo * incident
        if ti.static(self._emissive):
            radiance += self._emission[self._block_code(brick, voxel)]
        return radiance

    @ti.kernel
    def _cache_lighting(self, first: ti.i32, end: ti.i32):
        for row, voxel in ti.ndrange((first, end), 512):
            location = self._brick_locations[row]
            brick = self._mixed_light_bricks[row]
            material = self._sample_voxel(brick, voxel, True)
            if location.w >= 0 and material.w != 0:
                coordinates = location.xyz + ti.Vector([voxel & 7, (voxel >> 3) & 7, voxel >> 6])
                entity = location.w
                center = self._positions[entity] + self._rotations[entity] @ (ti.cast(coordinates, ti.f32) + 0.5)
                light = self._outgoing_radiance(brick, voxel, material.xyz, self._sample_light(center))
                for channel in ti.static(range(self._light_channels)):
                    self._light_cache[row, voxel][channel] = light[channel]

    @ti.kernel
    def _cache_uniform_lighting(self, first: ti.i32, end: ti.i32):
        for row in range(first, end):
            brick = self._uniform_light_bricks[row]
            material = self._sample_voxel(brick, 0, True)
            light = self._outgoing_radiance(brick, 0, material.xyz, self._sample_light(self._uniform_light_centers[row]))
            for channel in ti.static(range(self._light_channels)):
                self._uniform_light_cache[row][channel] = light[channel]

    @ti.kernel
    def _add_caustic_radiance(self, addresses: ti.types.ndarray(dtype=ti.i32, ndim=2),
                              power: ti.types.ndarray(dtype=ti.math.vec3, ndim=1)):
        for hit in range(power.shape[0]):
            brick, voxel = addresses[hit, 0], addresses[hit, 1]
            row = brick
            if ti.static(self._uniform_light_count):
                row = self._light_rows[brick]
            albedo = self._sample_voxel(brick, voxel, True).xyz
            radiance = (power[hit] * self._caustic_multiplier) * self._light_color[None] * albedo
            if ti.static(self._uniform_light_count):
                if row < 0:
                    self._uniform_light_cache[-row - 1] += radiance
            if ti.static(self._mixed_light_count):
                if row >= 0:
                    self._light_cache[row, voxel] += radiance

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
    def _voxel_radiance(self, brick, index):
        light = ti.Vector([1.0, 1.0, 1.0])
        if ti.static(self._lighting):
            row = brick
            value = ti.Vector.zero(ti.f32, self._light_channels)
            if ti.static(self._uniform_light_count):
                row = self._light_rows[brick]
                if row < 0:
                    value = self._uniform_light_cache[-row - 1]
            if ti.static(self._mixed_light_count):
                if row >= 0:
                    value = self._light_cache[row, index]
            if ti.static(self._light_channels == 1):
                light *= value[0]
            else:
                light = value
        return light

    @ti.func
    def _find_coord(self, coords: ti.template(), coord, bounds):
        lo, hi = bounds[0], bounds[1]
        found = -1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            other = coords[mid]
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
    def _native_brick(self, root, voxel):
        node = root
        brick, index, shift = -1, 0, ti.static(self._root_shift)
        for level in range(self._hierarchy_shifts.shape[0]):
            entry = self._hierarchy_nodes[node]
            if entry.y < 0:
                brick = ~entry.y
                if shift == 3:
                    local = voxel & 7
                    index = local.x + 8 * local.y + 64 * local.z
                break
            child_shift = self._hierarchy_shifts[level + 1]
            ratio = 1 << (shift - child_shift)
            shift = child_shift
            local = (voxel >> shift) & (ratio - 1)
            slot = local.x + ratio * (local.y + ratio * local.z)
            node = -1
            if entry.z == ratio * ratio * ratio:
                node = entry.y + slot
            else:
                lo, hi = entry.y, entry.y + entry.z
                while lo < hi:
                    mid = lo + (hi - lo) // 2
                    other = self._hierarchy_nodes[mid].x
                    if slot == other:
                        node = mid
                        break
                    if other < slot:
                        lo = mid + 1
                    else:
                        hi = mid
            if node < 0:
                break
        return brick, index, shift

    @ti.func
    def _lookup(self, origin, direction, t, end, entity, exact_voxel: ti.template() = False,
                forward_voxel: ti.template() = False):
        rotation = self._rotations[entity]
        position = self._positions[entity]
        local_direction = rotation.transpose() @ direction
        local_origin = rotation.transpose() @ (origin - position)
        p = local_origin + local_direction * t
        lower, upper = self._lower[entity], self._upper[entity]
        root_width = ti.static(1 << self._root_shift)
        voxel_lower = lower * root_width
        relative = p - voxel_lower
        if ti.static(forward_voxel):
            # At a boundary, sample the voxel the ray is about to enter.
            lower_edge = ti.floor(relative)
            relative = lower_edge - ti.cast((relative == lower_edge) & (local_direction < 0), ti.f32)
        elif ti.static(self._interface or exact_voxel):
            relative = ti.floor(relative)
        voxel = voxel_lower + ti.cast(relative, ti.i32)
        sector_cell = voxel >> ti.static(self._root_shift)
        next_sample = t
        brick, index = -1, 0
        if (sector_cell >= lower).all() and (sector_cell < upper).all():
            sector = -1
            if ti.static(self.sector_count):
                sector = self._find_coord(self._sectors, sector_cell, self._ranges[entity])
                if sector >= 0:
                    if ti.static(self._native_hierarchy):
                        brick, index, shift = self._native_brick(sector, voxel)
                        if brick < 0:
                            next_sample += _distance_to_cell_exit(p, local_direction, voxel >> shift, 1 << shift)
                    else:
                        brick_cell = voxel >> 3
                        local_brick = brick_cell - sector_cell * 16
                        brick = self._brick_ids[sector, local_brick.x + 16 * local_brick.y + 256 * local_brick.z]
                        if brick < 0:
                            next_sample += _distance_to_cell_exit(p, local_direction, brick_cell, 8)
                        else:
                            local_voxel = voxel - brick_cell * 8
                            index = local_voxel.x + 8 * local_voxel.y + 64 * local_voxel.z
            if sector < 0:
                if ti.static(self._has_uniform_bricks and not self._native_hierarchy):
                    for level in range(self._uniform_ranges.shape[0]):
                        level_range = self._uniform_ranges[level]
                        coord = voxel >> level_range[0]
                        found = self._find_coord(self._uniform_coords, coord, level_range.yz)
                        if found >= 0:
                            brick = self._uniform_rows[found]
                            break
                if brick < 0:
                    next_sample += _distance_to_cell_exit(p, local_direction, sector_cell, root_width)
        else:
            box_lower = voxel_lower - ti.static(0.0 if self._interface else 1.0)
            entry, exit = intersect_box(local_origin, local_direction, box_lower, upper * (1.0 * root_width), t, end)
            next_sample = end
            if entry < exit:
                next_sample = entry
        return next_sample, brick, index, voxel

    @ti.func
    def _advance(self, t, next_sample, step):
        # Round down to stay on the sample lattice without stepping past the next occupied region.
        return ti.max(1.0, ti.floor((next_sample - t) / step))

    @ti.func
    def _volume_interval(self, origin, direction):
        half_extent = self._world_extent[None] * 0.5
        center = self._world_center[None]
        return intersect_box(origin, direction, center - half_extent, center + half_extent, 0.0, 1e30)

    @ti.func
    def _refractive_shadow(self, origin, direction, end):
        blocked, t = False, 0.
        while t <= end and not blocked:
            next_sample = 1e30
            for entity in range(self.entity_count):
                next_entity, brick, index, voxel = self._lookup(origin, direction, t, 1e30, entity, True)
                if brick >= 0:
                    if self._ior[self._block_code(brick, index)] != self._ior[0]:
                        blocked = True
                        break
                    rotation = self._rotations[entity].transpose()
                    local_ray = rotation @ direction
                    local_pos = rotation @ (origin - self._positions[entity]) + local_ray * t
                    next_entity = t + _distance_to_cell_exit(local_pos, local_ray, voxel, 1)
                next_sample = ti.min(next_sample, next_entity)
            if next_sample > end:
                break
            next_t = next_sample + INTERFACE_ENTRY_OFFSET
            if next_t <= t:
                break
            t = next_t
        return blocked

    @ti.func
    def _light_transmittance(self, origin, direction):
        transmittance, t = 1.0, 0.0
        _, end = self._volume_interval(origin, direction)
        shadow_weight = 1.
        if ti.static(self._refractive_shadow_passthrough < 1.):
            if self._refractive_shadow(origin, direction, end):
                # The opaque endpoint contributes zero direct light at this voxel.
                shadow_weight *= self._refractive_shadow_passthrough
        if shadow_weight == 0.:
            transmittance = 0.
        while t <= end and transmittance != 0.0:
            sample_t = t
            if t != 0.:
                sample_t += .5 * self._light_step
            if sample_t > end:
                break
            density, next_sample = 0.0, 1e30
            for entity in range(self.entity_count):
                next_entity_sample, brick, voxel, _ = self._lookup(
                    origin, direction, sample_t, 1e30, entity, forward_voxel=True,
                )
                next_sample = ti.min(next_sample, next_entity_sample)
                if brick >= 0:
                    density += self._sample_voxel(brick, voxel, False).w
            if next_sample > end:
                break
            step = self._light_step
            next_t = t
            if t == 0.0:
                step = 0.5
                next_t = step
            else:
                next_t += ti.min(self._advance(sample_t, next_sample, step), 2**24) * step
            transmittance *= 1.0 - density * step
            if next_t <= t:
                break
            t = next_t
        return transmittance * shadow_weight

    @ti.func
    def _sample_refractive_index(self, origin, direction, t, exact_voxel: ti.template() = False):
        ambient_ior = self._ior[0]
        index, next_sample = ambient_ior, 1e30
        for entity in range(self.entity_count):
            next_entity, brick, voxel, _ = self._lookup(origin, direction, t, 1e30, entity, exact_voxel)
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
        index, _ = self._sample_refractive_index(point, ti.Vector([1., 1., 1.]), 0., True)
        return index

    @ti.func
    def _interface_face(self, point, direction, entity, voxel, entering):
        rotation = self._rotations[entity]
        local = rotation.transpose() @ (point - self._positions[entity])
        ray = rotation.transpose() @ direction
        distance = -1e30 if entering else 1e30
        axis = 0
        for i in ti.static(range(3)):
            if ray[i] != 0.:
                plane = voxel[i] + int((ray[i] < 0.) == entering)
                candidate = (plane - local[i]) / ray[i]
                if (entering and candidate > distance) or (not entering and candidate < distance):
                    distance, axis = candidate, i
        face = local + ray * distance
        face[axis] = voxel[axis] + int((ray[axis] < 0.) == entering)
        forward = ti.Vector.zero(ti.f32, 3)
        forward[axis] = 1. if ray[axis] > 0. else -1.
        return face, axis, rotation @ forward

    @ti.func
    def _interface_normal(self, face, axis, entity, forward):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        cell = ti.floor(face)
        delta = face - cell - .5
        gradient = ti.Vector.zero(ti.f32, 3)
        a, b = delta[u], delta[v]
        wu = ti.Vector([.5*(.5-a)**2, .75-a*a, .5*(.5+a)**2])
        wv = ti.Vector([.5*(.5-b)**2, .75-b*b, .5*(.5+b)**2])
        du, dv = ti.Vector([a-.5, -2.*a, a+.5]), ti.Vector([b-.5, -2.*b, b+.5])
        rotation = self._rotations[entity]
        position = self._positions[entity]
        for i, j in ti.ndrange(3, 3):
            sample = cell + .5
            sample[u] += i - 1
            sample[v] += j - 1
            high = self._ior_at(position + rotation @ sample)
            sample[axis] -= 1.
            low = self._ior_at(position + rotation @ sample)
            mean = .5 * (low + high)
            gradient[axis] += wu[i] * wv[j] * (high - low)
            gradient[u] += du[i] * wv[j] * mean
            gradient[v] += wu[i] * dv[j] * mean
        normal = forward
        if gradient.norm_sqr() > 1e-12:
            normal = rotation @ gradient.normalized()
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
        entry, end = self._volume_interval(origin, direction)
        index = ti.ceil(entry / self._step)
        previous_steps = 0.
        previous_medium, previous_code, interfaces = 1., 0, 0
        previous_entity = 0
        previous_voxel = ti.Vector.zero(ti.i32, 3)
        lower, upper = origin, origin
        if ti.static(self._refractive):
            half_extent = self._world_extent[None] * .5
            lower = self._world_center[None] - half_extent
            upper = self._world_center[None] + half_extent
        if ti.static(self._interface):
            if entry >= end:
                end = -1.
            origin += direction * (entry + INTERFACE_ENTRY_OFFSET)
            index = 0.
            previous_medium = self._ior[0]
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
            medium_code = 0
            medium_entity = 0
            medium_voxel = ti.Vector.zero(ti.i32, 3)
            boundary = 1e30
            has_brick = False
            if ti.static(self._refractive):
                medium = self._ior[0]
            for entity in range(self.entity_count):
                next_entity_sample, brick, voxel, coordinates = self._lookup(origin, direction, t, 1e30, entity)
                next_sample = ti.min(next_sample, next_entity_sample)
                if brick >= 0:
                    material = self._sample_voxel(brick, voxel, True)
                    density += material.w
                    code = 0
                    if ti.static(self._refractive):
                        code = self._block_code(brick, voxel)
                    if material.w != 0:
                        radiance = material.xyz
                        if ti.static(self._lighting):
                            radiance = self._voxel_radiance(brick, voxel)
                        else:
                            radiance = self._outgoing_radiance(brick, voxel, material.xyz, ti.Vector([1., 1., 1.]))
                        lit_color += radiance * material.w
                    if ti.static(self._refractive):
                        if medium == self._ior[0]:
                            medium = self._ior[code]
                            if ti.static(self._use_interface_normals):
                                medium_code = code
                            if ti.static(self._interface):
                                medium_entity, medium_voxel = entity, coordinates
                        if ti.static(self._interface):
                            has_brick = True
                            rotation = self._rotations[entity].transpose()
                            local = rotation @ (origin - self._positions[entity])
                            boundary = ti.min(boundary, _distance_to_cell_exit(local, rotation @ direction, coordinates, 1))

            if ti.static(self._interface):
                if crosses_refractive_interface(previous_medium, medium, self._ior[0], density != 0., interfaces):
                    entering = previous_medium == self._ior[0]
                    face_entity, face_voxel = previous_entity, previous_voxel
                    if entering:
                        face_entity, face_voxel = medium_entity, medium_voxel
                    face, axis, forward = self._interface_face(origin, direction, face_entity, face_voxel, entering)
                    normal = ti.Vector.zero(ti.f32, 3)
                    if ti.static(self._use_interface_normals):
                        if medium != self._ior[0]:
                            normal = self._interface_normals[medium_code]
                        if normal.norm_sqr() == 0. and previous_medium != self._ior[0]:
                            normal = self._interface_normals[previous_code]
                    if normal.norm_sqr() == 0.:
                        normal = self._interface_normal(face, axis, face_entity, forward)
                    bent, fresnel, reflected, mirror = refract_at_interface(direction, normal, previous_medium, medium)
                    side = -1. if reflected else 1.
                    if bent.dot(forward) * side <= 0.:
                        bent, fresnel, reflected, mirror = refract_at_interface(
                            direction, forward, previous_medium, medium,
                        )
                        side = -1. if reflected else 1.
                    direction = bent
                    origin = self._positions[face_entity] + self._rotations[face_entity] @ face
                    origin += forward * (side * INTERFACE_ENTRY_OFFSET)
                    interfaces += 1
                    if reflected:
                        continue
                    reflection = self._background[None]
                    if ti.static(self._use_reflection_environment):
                        reflection += ti.Vector(self._reflection_environment[0]) * ti.max(0., mirror.z)
                        alignment = ti.max(0., mirror.dot(self._light_direction[None]))
                        reflection += ti.Vector(self._reflection_environment[1]) * alignment**58
                    color += transmittance * fresnel * reflection
                    transmittance *= 1. - fresnel
                    previous_medium, previous_code = medium, medium_code
                    previous_entity, previous_voxel = medium_entity, medium_voxel
                    continue
                previous_medium = medium
                previous_entity, previous_voxel = medium_entity, medium_voxel
                if ti.static(self._use_interface_normals):
                    previous_code = medium_code

            if next_sample > end and ti.static(not self._refractive):
                break
            sample_length = self._step
            if ti.static(self._interface):
                sample_length = ti.min(sample_length, boundary)
            if density != 0:
                color, transmittance = composite_sample(
                    color, transmittance, lit_color / density, ti.Vector([1.0, 1.0, 1.0]),
                    density * sample_length,
                )
            if ti.static(self._refractive and not self._interface):
                # Only skip space empty for the whole centered-difference stencil.
                gradient, next_optical_sample = self._ior_gradient(origin, direction, t)
                next_sample = ti.min(next_sample, next_optical_sample)
                point = origin + direction * t
                if (point < lower + 1.).any() or (point >= upper - 1.).any():
                    gradient = ti.Vector.zero(ti.f32, 3)
                bent, stop = refract_from_gradient(direction,medium,gradient,self._step)
                if stop:
                    break
                if (bent != direction).any():
                    origin += direction*t
                    direction = bent
                    entry, end = self._volume_interval(origin, direction)
                    previous_steps += index
                    index = 1.
                    continue
            if ti.static(self._interface):
                advance = sample_length
                if boundary <= self._step:
                    advance += INTERFACE_ENTRY_OFFSET
                if not has_brick:
                    # Land just past the empty cell boundary before resuming regular steps.
                    advance = ti.max(advance, next_sample + .003)
                    previous_steps += ti.floor(advance / self._step) - 1.
                origin += direction * advance
            else:
                if next_sample > end:
                    break
                next_index = index + self._advance(t, next_sample, self._step)
                if next_index <= index:
                    break
                index = next_index
        return color + self._background[None] * transmittance
