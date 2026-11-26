"""Generate the procedural limestone fountain VXB."""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from taichi_volume_renderer.cxw import packed_color_materials
from taichi_volume_renderer.materials import cluster_materials
from taichi_volume_renderer.math import rotation_matrix_to_quaternion, rotation_quaternion_to_matrix
from taichi_volume_renderer.vxb import CameraPose, MaterialField, VxbScene, merge_uniform_children, save_vxb
from taichi_volume_renderer.vxb_optimize import merge_hidden_materials

from .water_particles import DRAIN_OFFSET, simulate_water
from .wave_field import hash01, value_noise, water_wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Path for the generated VXB")
    parser.add_argument("--no-merge-hidden-materials", action="store_true", help="Preserve interior stone colors")
    args = parser.parse_args()

    generate(args.output, merge_hidden=not args.no_merge_hidden_materials)


# VXB export

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


def generate(path, *, merge_hidden=True):
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

    airborne = len(materials)
    materials.append((.50, .56, .59, .022, np.float32(1.33), 0., 0., 0.))
    local_z, local_y, local_x = np.indices((8, 8, 8), dtype=np.int32)

    def assign_airborne_water(values, grid, x, y, z):
        water = grid == WATER_CODE
        if not water.any():
            return
        ly = y * 8 + local_y + .5
        radius_sq = (x * 8 + local_x + .5)**2 + (z * 8 + local_z + .5)**2
        pool = ((ly >= 35.) & (ly < BASIN_LEVEL + POOL_WAVE_SCALE * 2.25) & (radius_sq < 397.**2))
        for tier in (LOWER, UPPER):
            top = tier[1]
            pool |= ((ly >= top - 22.) & (ly < top - 1.8 + POOL_WAVE_SCALE * 2.25)
                     & (radius_sq < (tier[3] + 10.)**2))
        values[(water & ~pool).ravel()] = airborne

    pools = ((BASIN_LEVEL, 397.), (LOWER[1] - 1.8, LOWER[3] + 10.),
             (UPPER[1] - 1.8, UPPER[3] + 10.))
    empty = np.zeros((8, 8, 8), dtype=np.uint16)
    unit = 512 / POOL_WAVE_SCALE
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
        surface_band = POOL_WAVE_SCALE * 3.
        if not any(y0 < level + surface_band and y0 + 8 > level - surface_band for level, _ in pools):
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
        for level, radius in pools:
            selected = (np.abs(wy - level) < surface_band) & (wx**2 + wz**2 < radius**2)
            if not selected.any():
                continue
            px, pz = wx[selected], wz[selected]
            h = .01
            dx = (water_wave(px + h, pz) - water_wave(px - h, pz)) * (POOL_WAVE_SCALE / (2 * h))
            dz = (water_wave(px, pz + h) - water_wave(px, pz - h)) * (POOL_WAVE_SCALE / (2 * h))
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
        roots[(int(bx), int(bz), int(by))] = merge_uniform_children(
            {slot: merge_uniform_children(children, 4) for slot, children in groups.items()}, 4,
        )
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
                return merge_uniform_children({slot: remap_brick(child) for slot, child in brick.items()}, 4)
            if isinstance(brick, int):
                return int(remap[brick])
            values = remap[np.asarray(brick, dtype=np.uint16)]
            if np.all(values == values[0]):
                return int(values[0])
            return tuple(values.tolist())

        roots = {coord: remap_brick(brick) for coord, brick in roots.items()}
        materials[first_normal:] = [water + tuple(normal) for normal in centers]

    scene = VxbScene(fields, tuple(materials), (128, 32, 8), roots, _camera())
    stone_materials = [index for index, code in enumerate(codes) if code & 0x8000 and code != FOAM_CODE]
    del world
    scene = merge_hidden_materials(scene, stone_materials, enabled=merge_hidden)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_vxb(path, scene)
    print(f"Saved {path} ({path.stat().st_size / 1e6:.2f} MB, {len(materials)} materials)")


# Fountain geometry

STONE = np.array([
    [13, 13, 12], [15, 15, 14], [17, 17, 16], [19, 19, 17],
    [21, 21, 19], [23, 22, 20], [24, 24, 22], [26, 25, 23],
    [27, 27, 25], [29, 28, 26], [30, 30, 28], [31, 31, 29],
], dtype=np.int32)
WATER_CODE = np.uint16(3)
FOAM_CODE = np.uint16(0xffff)
EYE = np.array([255., 395., -1400.])
TARGET = np.array([0., 114., 0.])

FRONT = np.arctan2(EYE[2], EYE[0])
GROUND_EDGE = 508
BASIN_LEVEL = 71
POOL_WAVE_SCALE = 2.
LOWER = (139, 211, 67, 232, 12)
UPPER = (270, 327, 43, 126, 10)
NOZZLE_HEIGHT = 336


def wrapped(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def solid_noise(x, y, z):
    return (
        np.sin(x * .063 + z * .035 + y * .041)
        * np.cos(z * .052 - x * .037 + y * .12)
        + .5 * np.sin(x * .18 + y * .65 + z * .114)
        + .32 * np.sin(x * .35 - y * .67 - z * .19)
    )


def petal(s, h, length, curve, spread):
    """Rounded leaf that curls away from the motif's central rib."""
    t = np.clip(h / length, 0, 1)
    center = curve * np.sin(t * 2.1)
    width = spread * np.maximum(.07, np.maximum(0., np.sin(t * np.pi)) ** .72)
    profile = np.maximum(0., 1 - ((s - center) / width) ** 2)
    return np.where((h > 0) & (h < length), np.sqrt(profile) * (1 - .30 * t), 0.)


def spiral(s, h, cx, cy, start, size, pitch, width):
    dx, dy = s-cx, h-cy
    direction = (np.arctan2(dy, dx)-start) % (2*np.pi)
    radius = np.hypot(dx, dy)
    expected = size + pitch*direction
    dist = np.minimum(abs(radius-expected), abs(radius-expected-2*np.pi*pitch))
    tube = np.sqrt(np.maximum(0, 1-(dist/width)**2))
    return tube * (radius < size+3.1*np.pi*pitch+width)


def spill_profile(theta, rim, count):
    stride = 2*np.pi/count
    angle = wrapped(theta-FRONT)
    band = np.floor(angle/stride+.5)
    phase = (angle+stride/2) % stride-stride/2
    tangential = rim*phase
    cosine = np.cos(count*phase)
    trough = (.5+.5*cosine)**6
    petal = (.5-.5*cosine)**.65
    relief = 14*petal - 10.5*trough
    half_width = np.where((rim > 200) & (band == 0), 28.,
                          9.7 + 4.6*(.5+.5*np.sin(2.83*band)))
    return tangential, band, trough, petal, relief, half_width


def make_sector(coord):
    sx, sy, sz = coord
    axis = np.arange(128, dtype=np.float32) + .5
    x = (axis + sx*128)[:, None, None]
    y = (axis + sy*128)[None, :, None]
    z = (axis + sz*128)[None, None, :]
    r = np.hypot(x, z)
    theta = np.arctan2(z, x)
    section = np.zeros((128, 128, 128), np.uint16)
    liquid = np.zeros(section.shape, dtype=np.bool_)
    rng = np.random.default_rng(np.random.SeedSequence([sx+5, sy+1, sz+5, 538]))
    random = rng.random(section.shape, dtype=np.float32)
    grain = solid_noise(x, y, z) + (random - .5) * 2.5
    sediment = np.sin(theta * 41 + r * .20 + y * .44) * np.sin(theta * 13 - y * .14)

    def fill_water(mask, collision_mask=None):
        # Keep the original water surface for particle collisions.
        if collision_mask is None:
            collision_mask = mask
        liquid[:] |= collision_mask & ((section == 0) | (section == WATER_CODE))
        mask = mask & (section == 0)
        section[mask] = WATER_CODE

    def stone(mask, level, texture=.95, lift=1.05):
        mask = np.broadcast_to(mask, section.shape)
        if not np.any(mask):
            return
        level = np.broadcast_to(level, section.shape)[mask] + grain[mask] * texture + lift
        red, green, blue = STONE[np.clip(np.rint(level).astype(np.intp), 0, len(STONE)-1)].T
        # Small warm and cool mineral flecks in the limestone.
        red = np.clip(red + rng.integers(-1, 2, size=red.size), 0, 31)
        blue = np.clip(blue + rng.integers(-1, 2, size=blue.size), 0, 31)
        section[mask] = 0x8000 | red << 10 | green << 5 | blue
        liquid[mask] = False

    if sy == -1:
        edge = np.maximum(abs(x), abs(z))
        variation = .75*(value_noise(x+183, z-711, 39)-.5) + .43*(value_noise(x, z, 15)-.5)
        row = np.floor(z/79.)
        staggered_x = x + 53.*(row % 2)
        col = np.floor(staggered_x/106.)
        # Leave one empty voxel between pavers through the full paving depth.
        paver = (staggered_x % 106. < 105.) & (z % 79. < 78.)
        ground = (edge < GROUND_EDGE) & (y >= -14) & (y < -2)
        damp = np.exp(-((r-433)/36.)**2) * (.46+.54*value_noise(x-71,z+36,69))
        natural = 4.28+.8*(hash01(col,row)-.5)+variation-.72*damp
        stone(ground & paver, natural, 0., lift=0.)
        stone((r < 432 + 1.3*y) & (y >= -3), 4.6)
        return section, liquid

    if sy == 0:
        # Solid basin foundation with a wider rounded foot and recessed marble panel.
        stone((r < 432) & (y < 36), 4.75 + .053*y)
        # The entire outer stone assembly uses its original profile at 1.5x height.
        b = y / 1.5
        stone((r > 411) & (r < 439 - .050*(b-7)**2) & (b < 15), 5.4 + b*.12)
        # Stretch from the foot of the relief so the crown rises above the rim.
        relief = relief_at(wrapped(theta-FRONT)*432, 18. + (y-18.)/1.32)
        wall_profile = 432 - 1.8*np.sin(np.clip((b-14)/36, 0, 1)*np.pi)
        wall = (r >= 397) & (r < wall_profile) & (b >= 14) & (b < 51)
        recess = np.clip(1.20-.088*relief, .08, 1) * (1+.32*np.exp(-((b-25)/11)**2))
        stone(wall, 5.65 + .62*sediment - recess - 1.2*np.exp(-((b-14)/4)**2))
        for height, radial, ry, rr, level in [
                (13, 429, 3.2, 8, 6.5), (18, 429, 1.75, 6.5, 5.6),
                (49, 430, 2.0, 7, 6.8), (55, 426, 7.2, 13, 8.4),
                (58, 410, 4, 15, 8.7), (49, 400, 6, 7, 5.8)]:
            radius = rr * np.sqrt(np.maximum(0, 1 - ((b-height)/ry)**2))
            stone((abs(r-radial) < radius) & (abs(b-height) < ry), level)
        stone((r >= 430) & (r < 435) & (abs(b-18) < 1.6) & (np.cos(theta*172) > .45), 8.5)

        # Apply the acanthus over the front of the mouldings, like the reference.
        carrier = wall_profile + 2.3*np.clip((b-49.)/5.,0,1) - 4.*np.clip((b-58.)/4.,0,1)
        carve = (r >= carrier-2.2) & (r < carrier+relief) & (relief > .64)
        stone(carve, 5.05+.254*np.minimum(relief,22)+.39*sediment, .52)

        tangential, _, _, _, _, _ = spill_profile(theta, 232, 12)
        impact = np.hypot(r-241, tangential)
        modulation = value_noise(x+105, z-81, 71)
        rings = .5 + .5*np.cos(impact*(.45+.13*modulation)+2.7*modulation)
        broken_rings = rings*np.exp(-impact/44)*(.13+.71*modulation)
        wave = water_wave(x, z)
        impact_wave = .43*broken_rings
        basin = (r < 397) & (y >= 36)
        joints = (abs((x+61.) % 113 - 56.5) < 1.) | (abs((z+48.) % 97 - 48.5) < 1.)
        stone((r < 397) & (y >= 33) & (y < 36), 4.9+.47*sediment-.85*joints)
        fill_water(basin & (y < BASIN_LEVEL + POOL_WAVE_SCALE * (wave + impact_wave)),
                   basin & (y < BASIN_LEVEL + wave + impact_wave))
        waterline = (r >= 397) & (r <= 402) & (y >= BASIN_LEVEL-2) & (y <= BASIN_LEVEL+4)
        stone(waterline & (section != 0), 4.1 + .59*sediment, .73)

    def pedestal(bottom, top, radius, lobes):
        if sy*128 >= top or (sy+1)*128 <= bottom:
            return
        h = y - bottom
        height = top-bottom
        t = np.clip(h/height, 0, 1)
        active = (h >= 0) & (h < height)
        base = radius * (.49 + .48*abs(2*t-1)**2.3)
        phi = wrapped(theta - FRONT + np.pi/lobes)
        period = 2*np.pi/lobes
        tangent = abs((phi + period/2) % period - period/2) * radius
        curve = radius*(.22 + .105*np.sin((t-.48)*2*np.pi))
        roll_width = radius*(.10+.025*np.cos(t*2*np.pi))

        band = np.maximum(0, 1 - ((tangent-curve)/roll_width)**2)
        groove = np.maximum(0, np.cos((tangent-curve)*.62))**8
        band = np.maximum(0, band*(1-.43*groove))
        scroll_depth = radius*(.24+.14*np.sin(t*np.pi))*np.sqrt(band)
        low_coil = abs(np.hypot(tangent-.26*radius, h-.22*height)-.142*radius)
        high_coil = abs(np.hypot(tangent-.23*radius, h-.80*height)-.121*radius)
        coil = np.maximum.reduce([
            np.sqrt(np.maximum(0, 1-(np.minimum(low_coil, high_coil)/(.063*radius))**2)),
            spiral(tangent, h, .27*radius, .25*height, -.8, .028*radius, .011*radius, 2.3),
            spiral(tangent, h, .21*radius, .78*height, 1.2, .026*radius, .009*radius, 2.2),
        ])
        leaf = np.maximum(
            petal(tangent, h+4, .45*height, .15*radius, .09*radius),
            petal(tangent-4, h+3, .34*height, .39*radius, .087*radius),
        )
        relief = np.maximum.reduce([scroll_depth, coil*.44*radius, leaf*.39*radius])
        wet = 1.6*np.exp(-((h-7)/7)**2)
        stone(active & (r < base), 4.4 + .65*sediment - wet - .83*groove)
        stone(active & (r >= base-2) & (r < base+relief),
              4.6 + 3.55*band + 2.2*coil + 2.55*leaf - wet - .67*groove)
        for elevation, thickness, flare in [(bottom+3, 4, .21), (bottom+9, 2, .11),
                                             (top-7, 4, .13), (top-2, 2, .17)]:
            stone((abs(y-elevation) < thickness) & (r < base + flare*radius), 7.25)

    def bowl(bottom, top, root, rim, lobes):
        if sy*128 >= top+13 or (sy+1)*128 <= bottom:
            return
        t = np.clip((y-bottom)/(top-bottom), 0, 1)
        tangent, _, trough, petals, rim_relief, width = spill_profile(theta, rim, lobes)
        silhouette = root + (rim-root)*np.sqrt(np.maximum(0, 1-(1-t)**1.8))
        fluting = (4 + 10*np.sin(t*np.pi/2))*petals - (4.5 + 6*t)*trough
        lip = top + 3.8*petals - 1.25*trough
        floor = top-16-3*np.cos(r/rim*np.pi*.75)
        hollow = (y > floor) & (r < silhouette+fluting-13)
        crest = rim + rim_relief
        slope = 2.8*np.clip((r-crest+7.)/12., 0, 1)
        carve_width = width + 2.4*np.clip((r-crest+6.)/11., 0, 1)
        channel = (abs(tangent) < carve_width) & (r >= crest-31) & (r < crest+11) & (y >= top-5.0-slope)
        outside = (y >= bottom) & (y <= lip) & (r < silhouette+fluting)
        wet = trough*np.clip((t-.2)*2.2, 0, 1) * (1.7 + .52*sediment)
        stone(outside & ~(hollow | channel),
              4.9 + 2.65*petals + .68*sediment + .025*(y-bottom) - wet - .73*trough)
        rolled = ((r-crest+1.5)/(8.8-1.8*trough))**2 + ((y-lip+2.4)/6.4)**2 < 1
        stone(rolled & ~channel, 7.85 + 1.15*petals - 1.85*trough)
        ripples = water_wave(x, z)
        level = top-1.8 + POOL_WAVE_SCALE * ripples
        drain = channel & (r < crest+DRAIN_OFFSET+1.85) & (y < top-2.0)
        fill_water((hollow & (y < level)) | drain,
                   (hollow & (y < top-1.8 + ripples)) | drain)

    pedestal(36, 149, 88, 4)
    # The lower pedestal carries its own traced upright acanthus instead of
    # reusing the apron motif. The existing structure remains behind it.
    if sy in (0,1):
        front_angle = wrapped(theta-FRONT)
        pedestal_leaf = relief_at(front_angle*81, y-60, pedestal=True)
        t = np.clip((y-36)/113, 0, 1)
        shoulder = 88*(.49+.48*abs(2*t-1)**2.3)+8.6
        carved = (abs(front_angle) < 1.08) & (pedestal_leaf > .68)
        stone(carved & (r >= shoulder-12) & (r < shoulder+pedestal_leaf),
              5.15+.239*pedestal_leaf+.34*sediment, .56)
    bowl(*LOWER)
    pedestal(197, 279, 54, 5)
    bowl(*UPPER)
    nozzle = (r < 13.3 + .8*np.cos(theta*8)) & (y >= 318) & (y < NOZZLE_HEIGHT)
    stone(nozzle, 7.8)
    # A thin ring of water wets the lip of the metal-free stone nozzle.
    fill_water((r < 12.8) & (y >= NOZZLE_HEIGHT) & (y < NOZZLE_HEIGHT+1.0))
    return section, liquid


def sectors():
    grids = {}
    # Collision cells are 0 for air, 1 for stone, 2 for standing water.
    # Airborne track samples occupy values >= 3 and do not act as colliders.
    collision = np.zeros((1024, 512, 1024), np.uint8)
    for sx in range(-4, 4):
        for sy in range(-1, 4):
            for sz in range(-4, 4):
                coord = sx, sy, sz
                near_x = max(sx*128, -(sx+1)*128, 0)
                near_z = max(sz*128, -(sz+1)*128, 0)
                if sy >= 0 and np.hypot(near_x, near_z) >= (463, 267, 145, 0)[sy]:
                    continue
                voxels, liquid = make_sector(coord)
                if np.any(voxels):
                    grids[coord] = voxels
                    if sy >= 0:
                        x, y, z = (sx+4)*128, sy*128, (sz+4)*128
                        collision[x:x+128, y:y+128, z:z+128] = np.where(liquid, 2, voxels >= 0x8000)
    simulate_water(collision, FRONT, LOWER, UPPER, NOZZLE_HEIGHT)

    for sx in range(-4, 4):
        for sy in range(-1, 4):
            for sz in range(-4, 4):
                coord = sx, sy, sz
                voxels = grids.pop(coord, None)
                if sy >= 0:
                    x, y, z = (sx+4)*128, sy*128, (sz+4)*128
                    cells = collision[x:x+128, y:y+128, z:z+128]
                    air_water = cells >= 3
                    if np.any(air_water):
                        if voxels is None:
                            voxels = np.zeros(cells.shape, np.uint16)
                        air_water &= voxels == 0
                        voxels[air_water] = np.where(cells[air_water] == 11, FOAM_CODE, WATER_CODE)
                if voxels is not None:
                    rows = (voxels.reshape(16, 8, 16, 8, 16, 8)
                            .transpose(4, 2, 0, 5, 3, 1).reshape(4096, 512))
                    nonempty = np.flatnonzero(np.any(rows != 0, axis=1))
                    if len(nonempty):
                        yield coord, {int(i): rows[i] for i in nonempty}


# Acanthus relief from hand-traced centerlines. Each control point stores
# reference-image x/y, brush half-width in pixels, and relief height in voxels.

# Central leaf, four overlapping lobes, and the rolled-over basal leaf tips.
CROWN = (
    ((772,969,3,6),(770,948,12,15),(775,903,12,19),(776,852,11,20),
     (776,803,14,20),(770,776,16,16),(779,762,7,13),(785,772,2,6)),
    ((782,956,7,11),(789,918,10,16),(798,868,16,21),(815,816,23,21),
     (836,788,20,20),(852,787,12,16),(861,792,3,9)),
    ((786,953,7,10),(805,918,12,18),(823,874,22,20),(848,836,23,22),
     (875,808,16,20),(887,805,10,13),(898,815,3,7)),
    ((799,942,5,8),(821,915,14,16),(842,886,20,20),(858,870,21,18),
     (877,853,14,19),(890,854,9,14),(883,866,3,9)),
    ((806,945,4,8),(833,942,11,13),(849,921,15,18),(852,902,11,21),
     (847,889,9,15),(836,896,4,11)),
    ((779,961,4,11),(790,954,11,13),(801,944,12,15),(813,950,9,15),
     (815,960,5,10),(804,966,2,7)),
)

# Smaller rolled lobe tips traced separately so they have their own edges.
LOBES = (
    ((809,849,4,11),(814,827,8,16),(827,816,10,18),(838,818,8,17),(844,827,3,10)),
    ((825,877,4,10),(836,858,9,14),(849,847,11,18),(861,848,8,15),(866,856,3,10)),
    ((846,869,4,9),(858,852,10,14),(872,839,10,17),(886,838,8,16),(891,846,3,10)),
    ((837,914,4,10),(831,900,9,15),(824,892,8,16),(817,897,4,10)),
    ((863,927,5,9),(875,911,8,14),(879,895,9,15),(876,885,7,14),(868,883,3,10)),
)

# Sweep rightward from the fan, then fold the living branch back into a tight curl.
VINES = (
    ((809,954,6,11),(843,949,11,15),(874,937,13,15),(908,914,12,16),
     (944,895,12,16),(981,878,14,19),(1017,874,15,19),(1044,882,12,18),
     (1060,898,10,17),(1055,916,9,15),(1039,917,8,12),(1030,906,6,12),
     (1038,897,4,10),(1047,904,2,6)),
    ((908,913,6,11),(930,889,13,14),(953,870,16,17),(981,851,15,17),
     (1015,837,12,14),(1033,837,3,8)),
    ((877,938,5,9),(898,921,10,14),(915,917,12,15),(943,921,12,16),
     (957,931,8,12),(954,939,3,7)),
    ((858,947,6,10),(876,947,10,13),(891,955,10,13),(894,968,4,8)),
    ((1000,872,7,11),(1028,861,11,14),(1056,862,12,16),(1084,874,11,16),
     (1101,891,8,14),(1109,908,4,8)),
    ((1063,871,5,9),(1091,876,8,12),(1118,889,10,13),(1144,902,11,14),
     (1177,906,8,12),(1205,899,5,9),(1237,883,3,6)),
    ((1112,887,5,9),(1100,870,9,12),(1082,853,10,13),(1064,849,6,10),
     (1051,855,2,6)),
    ((924,905,4,9),(932,885,8,12),(931,870,8,13),(919,856,6,10),
     (908,851,2,6)),
)

# Supporting leaf clusters farther around the stone apron.
SIDE = (
    ((1215,900,6,8),(1238,887,13,13),(1254,870,15,16),
     (1268,845,13,16),(1263,827,11,15),(1250,823,5,10)),
    ((1239,899,4,8),(1261,889,11,12),(1284,863,13,16),(1298,841,13,16),
     (1305,827,6,12),(1296,820,3,8)),
    ((1270,884,4,8),(1291,880,10,13),(1311,866,11,15),(1333,849,13,17),
     (1353,842,10,16),(1368,847,8,14),(1374,862,7,13),(1365,869,4,9)),
    ((1314,860,4,7),(1321,839,11,13),(1337,824,11,14),(1355,814,5,10)),
    ((1350,849,3,7),(1369,837,10,15),(1390,833,12,15),(1415,839,8,12)),
)

PEDESTAL = (
    ((772,650,4,6),(772,629,8,10),(773,600,9,14),(776,569,10,15),
     (781,547,7,13),(790,535,2,7)),
    ((780,650,4,7),(787,626,7,12),(799,597,12,15),(818,569,14,16),
     (833,555,10,13),(844,563,3,9)),
    ((781,653,4,7),(795,640,9,12),(812,619,12,14),(832,606,13,16),
     (854,605,9,14),(862,615,3,10)),
    ((788,654,3,7),(808,647,8,12),(830,636,11,15),(850,638,10,15),
     (863,649,3,9)),
    ((825,570,4,7),(851,546,9,11),(862,524,10,15),(871,499,8,14),
     (886,486,9,15),(902,489,8,12),(908,504,6,10),(900,511,3,8)),
    ((826,567,5,7),(848,571,11,12),(868,572,13,14),(887,583,12,15),
     (899,600,10,15),(895,615,9,13),(880,618,8,12),(871,609,4,10)),
)

H_STEP = .5


def smooth_segments(points):
    """Interpolate the centerline without the polygonal corners of a polyline."""
    p = np.vstack([points[0], points, points[-1]])
    for i in range(1, len(p)-2):
        a, b, c, d = p[i-1:i+3]
        samples = max(3, int(np.linalg.norm(c[:2]-b[:2])/.64))
        for t in np.arange(samples)/samples:
            q = .5*(2*b+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t)
            yield q
    yield points[-1]


def sweep(atlas, points, *, leaf):
    samples = tuple(smooth_segments(points))
    traveled = 0.
    for index, (cx, cy, width, depth) in enumerate(samples):
        width = max(.58, width)
        x0 = max(0, int((cx-width)/H_STEP))
        x1 = min(atlas.shape[0], int((cx+width)/H_STEP)+2)
        y0 = max(0, int((cy-width)/H_STEP))
        y1 = min(atlas.shape[1], int((cy+width)/H_STEP)+2)
        if x0 >= x1 or y0 >= y1:
            continue
        dx = np.arange(x0,x1)[:,None]*H_STEP-cx
        dy = np.arange(y0,y1)[None,:]*H_STEP-cy
        u = np.hypot(dx,dy)/width
        shape = np.sqrt(np.maximum(0.,1.-u*u))
        if leaf:
            prev = samples[max(0,index-1)][:2]
            ahead = samples[min(len(samples)-1,index+1)][:2]
            tangent = ahead-prev
            normal = np.array([-tangent[1],tangent[0]])/max(.001,np.linalg.norm(tangent))
            side = abs(normal[0]*dx+normal[1]*dy)/width
            traveled += np.linalg.norm(np.array([cx,cy])-prev)
            midrib = np.exp(-(side/.20)**2)
            folds = np.maximum(0,np.cos(1.15*traveled-4.2*side))**11
            grooves = np.exp(-((side-.38)/.12)**2)+.7*np.exp(-((side-.68)/.12)**2)
            shape *= .92+.16*midrib-.19*grooves-.14*folds*np.clip(side*1.45,0,1)
        tile = atlas[x0:x1,y0:y1]
        np.maximum(tile, shape*depth, out=tile)


@lru_cache(maxsize=2)
def relief_atlas(pedestal=False):
    if pedestal:
        atlas = np.zeros((205,221),dtype=np.float32)
        groups = [(PEDESTAL, None, .55, .41, 4., .52)]
    else:
        atlas = np.zeros((2721,193),dtype=np.float32)
        # Main frontal composition, followed by the smaller side cartouches.
        groups = [(CROWN, None, .78, .318, 14.5, .50),
                  (VINES, None, .80, .316, 14.6, .49),
                  (LOBES, None, .79, .318, 14.6, .46),
                  (SIDE, None, .85, .318, 14.6, .52),
                  (SIDE, 605., .54, .29, 16.3, .46),
                  (SIDE, 717., .72, .30, 15.8, .49),
                  (SIDE, 856., .76, .32, 14.8, .49),
                  (SIDE, 1011., .72, .30, 16.1, .46),
                  (SIDE, 1152., .75, .32, 15.3, .47),
                  (SIDE, 1298., .34, .33, 14.6, .40)]
    for paths, offset, sx, sy, base_y, width in groups:
        for i,path in enumerate(paths):
            points = np.array(path,dtype=np.float64)
            origin_x, origin_y = (772,656) if pedestal else (772,969)
            depth_scale = .87 if offset is not None else 1.
            if offset is not None:
                origin_x = 1215
            points[:,0] = (points[:,0]-origin_x)*sx+(offset or 0.)
            points[:,1] = base_y+(origin_y-points[:,1])*sy
            points[:,2] *= width
            points[:,3] *= depth_scale
            sweep(atlas, points, leaf=(i<5 if paths is CROWN else i not in (0,4,5)))
    return atlas


def relief_at(tangent, height, *, pedestal=False):
    """Map the traced design onto a cylindrical wall, with a subtly varied other side."""
    atlas = relief_atlas(pedestal)
    u = np.where(tangent < 0, -tangent*(1.+.021*np.sin(height*.13)), tangent)
    h = height+np.where(tangent < 0,.72*np.sin(u*.035),0.)
    u, h = np.broadcast_arrays(u/H_STEP,h/H_STEP)
    ix = np.clip(np.floor(u).astype(np.int32),0,atlas.shape[0]-2)
    iy = np.clip(np.floor(h).astype(np.int32),0,atlas.shape[1]-2)
    fx, fy = np.clip(u-ix,0,1), np.clip(h-iy,0,1)
    depth = ((1-fx)*((1-fy)*atlas[ix,iy]+fy*atlas[ix,iy+1])
             +fx*((1-fy)*atlas[ix+1,iy]+fy*atlas[ix+1,iy+1]))
    return np.where((u < atlas.shape[0]-1)&(h >= 0)&(h < atlas.shape[1]-1),depth,0.)


if __name__ == "__main__":
    main()
