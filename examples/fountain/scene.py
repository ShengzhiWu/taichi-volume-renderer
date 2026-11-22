"""Generate the v9 limestone fountain, its water streams, and its foam."""
import numpy as np

from .acanthus_relief import relief_at
from .water_particles import DRAIN_OFFSET, simulate_water
from .wave_field import hash01, value_noise, water_wave


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

    def fill_water(mask):
        mask = mask & (section == 0)
        section[mask] = WATER_CODE
        liquid[mask] = True

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
        # A single level of paving avoids the old one-voxel steps and their dark line shadows.
        ground = (edge < GROUND_EDGE) & (y >= -14) & (y < -2)
        stone(ground, 4.2+variation, 0., lift=0.)
        field_x = x+17.*(value_noise(x+401,z,121)-.5)
        field_z = z+13.*(value_noise(x,z-251,110)-.5)
        row = np.floor(field_z/79.)
        shift = 106.*hash01(row, 217)
        col = np.floor((field_x+shift)/106.)
        u, v = (field_x+shift) % 106., field_z % 79.
        seam = np.minimum(np.minimum(u,106.-u),np.minimum(v,79.-v))
        # Keep narrow mortar joints in the stone colors on the same plane.
        grout = np.clip((2.35-seam)/1.15, 0., 1.)
        damp = np.exp(-((r-433)/36.)**2) * (.46+.54*value_noise(x-71,z+36,69))
        natural = 4.28+.8*(hash01(col,row)-.5)+variation-1.14*grout-.72*damp
        stone((edge < GROUND_EDGE-2) & (y >= -5) & (y < -2), natural, 0., lift=0.)
        stone((r < 432 + 1.3*y) & (y >= -3), 4.6)
        return section, liquid

    if sy == 0:
        # Solid basin foundation with a wider rounded foot and recessed marble panel.
        stone((r < 432) & (y < 36), 4.75 + .053*y)
        # The entire outer stone assembly uses its original profile at 1.5x height.
        b = y / 1.5
        stone((r > 411) & (r < 439 - .050*(b-7)**2) & (b < 15), 5.4 + b*.12)
        relief = relief_at(wrapped(theta-FRONT)*432, y)
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
        carrier = wall_profile + 2.3*np.clip((b-49.)/5.,0,1)
        carve = (r >= carrier-2.2) & (r < carrier+relief) & (relief > .64)
        stone(carve, 5.05+.254*np.minimum(relief,22)+.39*sediment, .52)

        tangential, _, _, _, _, _ = spill_profile(theta, 232, 12)
        impact = np.hypot(r-241, tangential)
        modulation = value_noise(x+105, z-81, 71)
        rings = .5 + .5*np.cos(impact*(.45+.13*modulation)+2.7*modulation)
        broken_rings = rings*np.exp(-impact/44)*(.13+.71*modulation)
        surface = BASIN_LEVEL + water_wave(x, z) + .43*broken_rings
        joints = (abs((x+61.) % 113 - 56.5) < 1.) | (abs((z+48.) % 97 - 48.5) < 1.)
        stone((r < 397) & (y >= 33) & (y < 36), 4.9+.47*sediment-.85*joints)
        fill_water((r < 397) & (y >= 36) & (y < surface))
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
        level = top-1.8 + .74*ripples
        drain = channel & (r < crest+DRAIN_OFFSET+1.85) & (y < top-2.0)
        fill_water((hollow & (y < level)) | drain)

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
                        collision[x:x+128, y:y+128, z:z+128] = np.where(liquid, 2, voxels != 0)
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
                        if np.any(voxels[air_water]):
                            raise RuntimeError("Airborne water overlaps the static scene")
                        voxels[air_water] = np.where(cells[air_water] == 11, FOAM_CODE, WATER_CODE)
                if voxels is not None:
                    rows = (voxels.reshape(16, 8, 16, 8, 16, 8)
                            .transpose(4, 2, 0, 5, 3, 1).reshape(4096, 512))
                    nonempty = np.flatnonzero(np.any(rows != 0, axis=1))
                    if len(nonempty):
                        yield coord, {int(i): rows[i] for i in nonempty}
