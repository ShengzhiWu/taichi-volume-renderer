"""Write the fountain as a Z-up VXB scene with 200 pool surface normals."""

import numpy as np
from threadpoolctl import threadpool_limits

from taichi_volume_renderer.cxw import packed_color_materials
from taichi_volume_renderer.materials import cluster_materials
from taichi_volume_renderer.math import rotation_matrix_to_quaternion, rotation_quaternion_to_matrix
from taichi_volume_renderer.vxb import CameraPose, MaterialField, VxbScene, save_vxb

from .scene import BASIN_LEVEL, EYE, FOAM_CODE, LOWER, TARGET, UPPER, WATER_CODE, sectors
from .wave_field import water_wave


POOL_NORMAL_CLUSTERS = 200


def _camera():
    # Preserve the framing of the original Y-up scene when exporting to Z-up VXB.
    forward = TARGET - EYE
    forward /= np.linalg.norm(forward)
    yaw, pitch = np.arctan2(forward[0], forward[2]), -np.arcsin(forward[1])
    sy, cy = np.sin(yaw / 2), np.cos(yaw / 2)
    sx, cx = np.sin(pitch / 2), np.cos(pitch / 2)
    quaternion = np.array([cx * cy, sx * cy, cx * sy, -sx * sy], dtype=np.float32).astype(np.float64)
    rotation = rotation_quaternion_to_matrix(quaternion / np.linalg.norm(quaternion))[[0, 2, 1]]
    rotation[:, 2] *= -1
    xyzw = rotation_matrix_to_quaternion(rotation)[[1, 2, 3, 0]]
    return CameraPose(tuple(EYE[[0, 2, 1]]), tuple(xyzw))


def generate(path):
    world = dict(sectors())
    palette = packed_color_materials(.65)
    palette[WATER_CODE] = (.31, .42, .43, .0024)
    palette[FOAM_CODE] = (1., 1., 1., .70)
    used = np.zeros(65536, dtype=bool)
    used[0] = True
    for bricks in world.values():
        used[np.stack(tuple(bricks.values()))] = True
    codes = np.flatnonzero(used)
    index_for_code = np.zeros(65536, dtype=np.uint16)
    index_for_code[codes] = np.arange(len(codes))
    fields = tuple(MaterialField(name, "f32") for name in ("r", "g", "b", "density", "ior", "normal_x", "normal_y", "normal_z"))
    materials = [tuple(palette[code]) + (np.float32(1.33 if code == WATER_CODE else 1.), 0., 0., 0.) for code in codes]

    def uniform_or_children(children):
        value = next(iter(children.values()))
        if len(children) == 64 and isinstance(value, int) and all(child == value for child in children.values()):
            return value
        return children

    airborne = len(materials)
    materials.append((.50, .56, .59, .022, np.float32(1.33), 0., 0., 0.))
    local_z, local_y, local_x = np.indices((8, 8, 8), dtype=np.int32)

    def assign_airborne_water(values, grid, x, y, z):
        water = grid == WATER_CODE
        if not water.any():
            return
        ly = y * 8 + local_y + .5
        radius_sq = (x * 8 + local_x + .5)**2 + (z * 8 + local_z + .5)**2
        pool = ((ly >= 35.) & (ly < BASIN_LEVEL + 2.25) & (radius_sq < 397.**2))
        for tier in (LOWER, UPPER):
            top = tier[1]
            pool |= (ly >= top - 22.) & (ly < top + .1) & (radius_sq < (tier[3] + 10.)**2)
        values[(water & ~pool).ravel()] = airborne

    pools = ((BASIN_LEVEL, 397., 1.), (LOWER[1] - 1.8, LOWER[3] + 10., .74),
             (UPPER[1] - 1.8, UPPER[3] + 10., .74))
    empty = np.zeros((8, 8, 8), dtype=np.uint16)
    unit = 512
    water = materials[int(index_for_code[3])][:5]
    normal_ids = {}
    first_normal = len(materials)
    normal_weights = np.zeros(65536, dtype=np.int64)

    def adjacent_brick(x, y, z):
        sector = world.get((x // 16, y // 16, z // 16), {})
        slot = x % 16 + 16 * (y % 16 + 16 * (z % 16))
        return sector.get(slot, empty).reshape(8, 8, 8)

    def assign_surface_normals(values, grid, x, y, z):
        y0 = y * 8
        if not any(y0 < level + 4.5 and y0 + 8 > level - 4.5 for level, _, _ in pools):
            return
        exposed = grid == WATER_CODE
        if not exposed.any():
            return
        exposed[:, :-1, :] &= grid[:, 1:, :] == 0
        exposed[:, -1, :] &= adjacent_brick(x, y + 1, z)[:, 0, :] == 0
        exposed[:, 1:, :] &= grid[:, :-1, :] == WATER_CODE
        exposed[:, 0, :] &= adjacent_brick(x, y - 1, z)[:, -1, :] == WATER_CODE
        lz, ly, lx = np.nonzero(exposed)
        wx, wy, wz = x * 8 + lx + .5, y0 + ly + .5, z * 8 + lz + .5
        for level, radius, amplitude in pools:
            selected = (np.abs(wy - level) < 4.5) & (wx**2 + wz**2 < radius**2)
            if not selected.any():
                continue
            px, pz = wx[selected], wz[selected]
            h = .01
            dx = (water_wave(px + h, pz) - water_wave(px - h, pz)) * (amplitude / (2 * h))
            dz = (water_wave(px, pz + h) - water_wave(px, pz - h)) * (amplitude / (2 * h))
            slopes = np.rint(np.column_stack((-dx, -dz)) * unit).astype(int)
            ids = []
            for a, b in slopes.tolist():
                key = (a, b)
                index = normal_ids.get(key)
                if index is None:
                    index = len(materials)
                    if index >= 65536:
                        raise ValueError("Pool normal palette exceeds the sparse renderer's material capacity")
                    normal_ids[key] = index
                    normal = np.array((a / unit, b / unit, 1.))
                    materials.append(water + tuple(normal / np.linalg.norm(normal)))
                ids.append(index)
            flat = lx[selected] + 8 * (ly[selected] + 8 * lz[selected])
            values[flat] = ids

    roots = {}
    for (bx, by, bz), bricks in world.items():
        groups = {}
        for slot, block in bricks.items():
            x, y, z = slot % 16, slot >> 8, (slot >> 4) % 16
            outer = (x >> 2) + 4 * ((y >> 2) + 4 * (z >> 2))
            inner = (x & 3) + 4 * ((y & 3) + 4 * (z & 3))
            values = index_for_code[block]
            grid = block.reshape(8, 8, 8)
            assign_airborne_water(values, grid, int(bx) * 16 + x, int(by) * 16 + z, int(bz) * 16 + y)
            assign_surface_normals(values, grid,
                                   int(bx) * 16 + x, int(by) * 16 + z, int(bz) * 16 + y)
            np.add.at(normal_weights, values[values >= first_normal], 1)
            if np.all(values == values[0]):
                brick = int(values[0])
            else:
                brick = tuple(values.reshape(8, 8, 8).transpose(1, 0, 2).ravel().tolist())
            groups.setdefault(outer, {})[inner] = brick
        roots[(int(bx), int(bz), int(by))] = uniform_or_children(
            {slot: uniform_or_children(children) for slot, children in groups.items()})
    if len(normal_ids) > POOL_NORMAL_CLUSTERS:
        normals = np.asarray([row[5:] for row in materials[first_normal:]])
        weights = normal_weights[first_normal:len(materials)]
        with threadpool_limits(limits=4, user_api="openmp"):
            centers, labels = cluster_materials(normals, weights, POOL_NORMAL_CLUSTERS)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        remap = np.arange(len(materials), dtype=np.uint16)
        remap[first_normal:] = first_normal + labels

        def remap_brick(brick):
            if isinstance(brick, dict):
                return uniform_or_children({slot: remap_brick(child) for slot, child in brick.items()})
            if isinstance(brick, int):
                return int(remap[brick])
            values = remap[np.asarray(brick, dtype=np.uint16)]
            if np.all(values == values[0]):
                return int(values[0])
            return tuple(values.tolist())

        roots = {coord: remap_brick(brick) for coord, brick in roots.items()}
        materials[first_normal:] = [water + tuple(normal) for normal in centers]

    path.parent.mkdir(parents=True, exist_ok=True)
    save_vxb(path, VxbScene(fields, tuple(materials), (128, 32, 8), roots, _camera()))
    print(f"Saved {path} ({path.stat().st_size / 1e6:.2f} MB, {len(materials)} materials)")
