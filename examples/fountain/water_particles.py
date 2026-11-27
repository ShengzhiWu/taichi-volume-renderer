"""Trace fountain particles through the stone and water voxels."""

import numpy as np
from numba import njit


GRAVITY = 155.
STONE_CELL = 1
STANDING_WATER = 2
DRAIN_OFFSET = 8.15


@njit
def mark(cells, ix, iy, iz, color):
    current = cells[ix, iy, iz]
    if current == 0 or current >= 3:
        cells[ix, iy, iz] = max(current, color)


@njit
def trace(cells, starts, velocities, colors, from_liquid, duration, scale, packet_phases=None):
    """Visit each voxel crossed in each integration step, in boundary order."""
    hits = np.empty((len(starts), 6), dtype=np.float32)
    kinds = np.zeros(len(starts), np.uint8)
    valid = 0
    offset = cells.shape[0] // 2
    for i in range(len(starts)):
        x, y, z = starts[i]
        x += offset
        z += offset
        vx, vy, vz = velocities[i]
        ix, iy, iz = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        if not (0 <= ix < cells.shape[0] and 0 <= iy < cells.shape[1] and 0 <= iz < cells.shape[2]):
            continue
        first = cells[ix, iy, iz]
        if first == STONE_CELL or (not from_liquid and first == STANDING_WATER):
            continue
        if scale == 1. and from_liquid and first != STANDING_WATER:
            continue
        valid += 1
        dry = first != STANDING_WATER
        if dry:
            mark(cells, ix, iy, iz, colors[i])
        elapsed = 0.
        packet_phase = -1.
        if packet_phases is not None:
            packet_phase = packet_phases[i]
        collided = False
        while elapsed < duration and not collided:
            dt = min(.008, .36 / max(1., abs(vx), abs(vy), abs(vz)))
            dx, dy, dz = vx*dt, (vy-.5*GRAVITY*scale*dt)*dt, vz*dt
            sx = 1 if dx > 0 else -1
            sy = 1 if dy > 0 else -1
            sz = 1 if dz > 0 else -1
            tx = (ix + (sx > 0) - x) / dx if dx != 0 else np.inf
            ty = (iy + (sy > 0) - y) / dy if dy != 0 else np.inf
            tz = (iz + (sz > 0) - z) / dz if dz != 0 else np.inf
            fraction = min(tx, ty, tz)
            while fraction <= 1.0:
                if tx == fraction:
                    ix += sx
                    tx += abs(1./dx)
                if ty == fraction:
                    iy += sy
                    ty += abs(1./dy)
                if tz == fraction:
                    iz += sz
                    tz += abs(1./dz)
                if (ix < 0 or ix >= cells.shape[0] or iy < 0 or iy >= cells.shape[1]
                        or iz < 0 or iz >= cells.shape[2]):
                    kinds[i] = 3
                    collided = True
                    break
                cell = cells[ix, iy, iz]
                if cell == STONE_CELL or (cell == STANDING_WATER and dry):
                    kinds[i] = cell
                    # Move a small distance backward onto the air side of the impacted face.
                    tx0, ty0, tz0 = x+fraction*dx, y+fraction*dy, z+fraction*dz
                    distance = .032 / max(1e-7, np.sqrt(dx*dx+dy*dy+dz*dz))
                    hits[i, 0] = tx0-dx*distance-offset
                    hits[i, 1] = ty0-dy*distance
                    hits[i, 2] = tz0-dz*distance-offset
                    hits[i, 3] = vx
                    hits[i, 4] = vy
                    hits[i, 5] = vz
                    collided = True
                    break
                if cell != STANDING_WATER:
                    dry = True
                    # A nozzle shares a release phase across its cross-section,
                    # so its gaps remain visible after all tracks are accumulated.
                    age = elapsed+fraction*dt
                    if packet_phase >= 0.:
                        phase = (23.+1.15*packet_phase)*age+packet_phase
                        visible = age < .086 or np.sin(phase+.47*np.sin(phase*.43)) > -.16
                    else:
                        visible = colors[i] != 10 or elapsed < .055 or np.sin(47.*elapsed+i*2.36) > -.26
                    if visible:
                        mark(cells, ix, iy, iz, colors[i])
                fraction = min(tx, ty, tz)
            x, y, z = x+dx, y+dy, z+dz
            vy -= GRAVITY*scale*dt
            elapsed += dt
    return kinds, hits, valid


@njit
def spread_on_surface(cells, hits, headings, lengths, phases):
    """Follow the waterline with thin, broken foam streaks after an impact."""
    count = 0
    offset = cells.shape[0] // 2
    for n in range(len(hits)):
        ix0 = int(np.floor(hits[n, 0])) + offset
        iz0 = int(np.floor(hits[n, 2])) + offset
        first_y = int(np.floor(hits[n, 1]))
        iy = first_y
        prior_x, prior_z = -1, -1
        for step in range(1, lengths[n]+1):
            heading = headings[n]+.19*np.sin(.25*step+phases[n])
            ix = ix0 + int(np.round(step*np.cos(heading)))
            iz = iz0 + int(np.round(step*np.sin(heading)))
            if ix == prior_x and iz == prior_z:
                continue
            prior_x, prior_z = ix, iz
            if ix <= 0 or iz <= 0 or ix >= cells.shape[0]-1 or iz >= cells.shape[2]-1:
                break
            surface = -1
            for level in range(min(iy+2, first_y+3), max(iy-3, first_y-4), -1):
                if cells[ix, level, iz] not in (STONE_CELL, STANDING_WATER):
                    if cells[ix, level-1, iz] == STANDING_WATER:
                        surface = level
                        break
            if surface < 0:
                break
            iy = surface
            if step < 4 or np.sin(phases[n]+step*1.82) > -.28+.73*step/lengths[n]:
                mark(cells, ix, iy, iz, 11)
                count += 1
    return count


def simulate_water(cells, front, lower, upper, nozzle_height, scale, grid_center):
    rng = np.random.default_rng(241037)
    starts, velocities, values = [], [], []
    groups = []
    jet_packets = []

    def add(x, y, z, vx, vy, vz, values_):
        starts.extend(zip(x, np.broadcast_to(y, x.shape), z))
        velocities.extend(zip(vx, vy, vz))
        values.extend(values_)

    # The plume has an energetic lobe and a quieter one. Each lobe also contains
    # a continuous spread of launch speeds so that its top stays irregular.
    n = 1080
    total = len(starts)
    theta = rng.uniform(0, 2*np.pi, n)
    theta += .34*np.sin(2*theta+.7)
    launch_radius = np.sqrt(rng.random(n))*(4.7+1.15*np.cos(theta-.62))
    rise = rng.beta(2.4, 2.85, n)
    radial_speed = (3.5+15.1*rng.random(n)**1.8)*(1.11-.58*rise)
    lift = 94+85*rise+11.2*np.cos(theta+.73)+5*np.cos(3*theta)
    add(launch_radius*np.cos(theta), nozzle_height+.50, launch_radius*np.sin(theta),
        radial_speed*np.cos(theta)+1.3, lift, radial_speed*np.sin(theta)-2.7,
        rng.integers(3, 9, n))
    groups.append(("vertical plume", total, len(starts)))

    # Submerged outlets sit halfway between the bowl center and its lip.
    total = len(starts)
    target_height = lower[1]-1.8
    outlet_radius = .5*upper[3]
    for k in range(7):
        flow = rng.uniform(.27, 1.57)
        n = max(42, int(127*flow))
        axis = front+(k+.54)*2*np.pi/7+rng.uniform(-.13, .13)
        angle_center = axis+rng.uniform(-.007, .007)
        thinner = 2./3.
        angle = angle_center+rng.normal(0, thinner*.5*(.0045+.003*flow), n)
        tangent = np.clip(rng.normal(0, thinner*.5*(.46+.23*flow), n), -.55, .55)
        offset = outlet_radius+rng.uniform(-.60, .60, n)
        reach = np.clip(rng.uniform(193.8, 204.5)+rng.normal(0, thinner*1.15*flow, n), 192., 207.)
        launch_y = upper[1]-4.
        clear_fraction = (upper[3]+24.-offset)/(reach-offset)
        clear_height = upper[1]+10.
        required = np.sqrt(2*(clear_height-launch_y-(target_height-launch_y)*clear_fraction)
                           /(GRAVITY*clear_fraction*(1.-clear_fraction)))
        center_duration = rng.uniform(2.235, 2.39)
        duration = np.maximum(required+.095, center_duration+rng.normal(0, thinner*.009, n))
        speed = (reach-offset)/duration
        lift = (target_height-launch_y)/duration+.5*GRAVITY*duration
        marks = np.where(rng.random(n) < .27-.12*flow, 10, 6)
        first_track = len(starts)
        add(offset*np.cos(axis)-tangent*np.sin(axis), launch_y,
            offset*np.sin(axis)+tangent*np.cos(axis), speed*np.cos(angle), lift,
            speed*np.sin(angle), marks)
        jet_packets.append((first_track, len(starts), (.91*k+.27*k*k)%(2*np.pi)))
    groups.append(("angled jets", total, len(starts)))

    for bowl, group in ((lower, "lower spillways"), (upper, "upper spillways")):
        total = len(starts)
        _, top, _, rim, count = bowl
        for k in range(count):
            band = k if k <= count/2 else k-count
            main_sheet = rim > 200 and k == 0
            width = 28. if main_sheet else 9.7+4.6*(.5+.5*np.sin(2.83*band))
            if k == count//2:
                width = min(width, 9.7+4.6*(.5+.5*np.sin(-2.83*band)))
            flow = 1. if main_sheet else rng.uniform(.11, 1.31)
            n = int(610 if main_sheet else 208*flow)
            n = max(18, n)
            mouth = width-.85
            span = mouth*(.22+.59*flow) if not main_sheet else .92*mouth
            center = rng.uniform(-.35, .35)*mouth
            cluster = rng.choice([-1., -.12, .83], size=n, p=[.23, .54, .23])
            sample = rng.beta(1.8, 2.2, n)*2.-1.
            tangent = np.clip(center+span*(.65*sample+.39*cluster), -mouth, mouth)
            phase = tangent/rim
            axis = front+k*2*np.pi/count+phase
            cosine = np.cos(count*phase)
            crest = rim+14*(.5-.5*cosine)**.65-10.5*(.5+.5*cosine)**6
            radius = crest+DRAIN_OFFSET+rng.uniform(-.44, .65, n)
            y = top-rng.uniform(2.20, 3.69, n)
            zero = np.zeros(n)
            marks = np.where(rng.random(n) < .39-.19*flow, 10, 6)
            add(radius*np.cos(axis), y, radius*np.sin(axis),
                zero, rng.uniform(-6.5, -.4, n), zero, marks)
        groups.append((group, total, len(starts)))

    starts = np.asarray(starts, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    colors = np.asarray(values, dtype=np.uint8)
    packet_phases = np.full(len(starts), -1., np.float32)
    for first, end, phase in jet_packets:
        packet_phases[first:end] = phase
    starts *= scale
    starts[:, (0, 2)] += grid_center
    # Start above the last stone voxel, whose center lies below the nozzle lip.
    lip = np.ceil(nozzle_height * scale - .5)
    plume = starts[:groups[0][2], 1]
    np.maximum(plume, lip, out=plume)
    velocities *= scale
    kinds, hits, valid = trace(cells, starts, velocities, colors, True, 4., scale, packet_phases)
    if scale == 1. and valid < .985 * len(starts):
        raise RuntimeError(f"{len(starts)-valid} primary particles did not start in standing water")
    for name, first, end in groups:
        local = kinds[first:end]
        wet = local == STANDING_WATER
        print(f"{name}: {end-first:,} launched, {np.count_nonzero(local == STONE_CELL):,} stone impacts, "
              f"{np.count_nonzero(wet):,} water impacts, "
              f"{np.count_nonzero(local == 3):,} exited scene", flush=True)
        if scale == 1. and not np.any(wet):
            raise RuntimeError(f"No water impacts from {name}")
        if "spillways" in name:
            assert not np.any(velocities[first:end, (0, 2)])
            fall_dist = np.linalg.norm(hits[first:end][wet][:, (0, 2)]-starts[first:end][wet][:, (0, 2)], axis=1)
            if np.max(fall_dist, initial=0.) > .002:
                raise RuntimeError("A vertical spillway moved horizontally")
            print(f"  Horizontal drift: {np.max(fall_dist, initial=0.):.5f} voxel", flush=True)
        if name == "angled jets":
            heights = hits[first:end][wet, 1]
            mid_bowl = np.count_nonzero(abs(heights-target_height * scale) <= max(2.6 * scale, .5))
            print(f"  {mid_bowl:,} of {len(local):,} trials reached the middle bowl's water", flush=True)
            radii = np.hypot(starts[first:end,0], starts[first:end,2])
            speeds = np.linalg.norm(velocities[first:end],axis=1)
            print(f"  Outlet radius {radii.min():.2f}–{radii.max():.2f}; "
                  f"mean launch speed {speeds.mean():.2f} voxels/s", flush=True)
            if scale == 1. and mid_bowl != len(local):
                raise RuntimeError("Top nozzles missed the middle bowl")

    impacted = hits[kinds == STANDING_WATER]
    impacted = impacted[impacted[:, 4] < -22 * scale]
    _, indices = np.unique(np.floor(impacted[:, :3]).astype(np.int32), axis=0, return_index=True)
    impacted = impacted[indices]
    m = len(impacted)

    # Most of the whitened water stays on the surface. Random thinning and
    # impact-biased headings leave patches and streaks rather than solid rings.
    runout = impacted[rng.random(m) < .360]
    n = len(runout)
    momentum = np.arctan2(runout[:, 5], runout[:, 3])
    headings = np.where(rng.random(n) < .58, momentum+rng.normal(0., .94, n),
                        rng.uniform(0., 2*np.pi, n))
    length = np.rint(np.clip(-runout[:, 4]*.044, 5 * scale, 16 * scale)+rng.uniform(2, 20, n)*scale).astype(np.int32)
    surface_samples = spread_on_surface(cells, runout, headings, length, rng.uniform(0, 2*np.pi, n))

    # A small subset rebounds, with a vertical speed capped below about one voxel of rise.
    rebound = impacted[rng.random(m) < .078]
    n = len(rebound)
    angle = rng.uniform(0, 2*np.pi, n)
    radial = rng.uniform(8., 25., n)
    lift = rng.uniform(5., 17., n)
    velocity = np.column_stack((radial*np.cos(angle), lift, radial*np.sin(angle))) * scale
    _, _, accepted = trace(cells, rebound[:, :3].copy(), velocity,
                            np.full(n, 11, dtype=np.uint8), False, .34, scale)
    print(f"{surface_samples:,} lateral foam samples, {accepted:,} low splash trajectories", flush=True)
