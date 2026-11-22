"""Shared wave coefficients for the CXW surface and the refractive normals."""
import numpy as np

# Two dominant wave vectors have half the spatial frequencies of v6.
MODES = (
    (.50, .055, .041, .7),
    (.39, .066, -.0655, 2.2),
    (.25, -.031, .072, 4.1),
    (.19, .035, -.027, 1.4),
    (.15, .074, .017, 5.7),
)
WARP = (
    (12., .016, .019, .4),
    (6.5, -.012, .026, 2.7),
    (10., -.014, .018, 2.0),
    (6., .025, .008, 5.4),
)


# v1's radial waves, expressed at the present basin scale.
RADIAL_AMPLITUDE = .77
RADIAL_FREQUENCY = .132
RADIAL_WARP = 2.20
RADIAL_CORE = 40.


def radial_wave(x, z):
    radius_sq = x*x+z*z
    radius = np.sqrt(radius_sq+9.)
    taper = radius_sq/(radius_sq+RADIAL_CORE**2)
    angle = np.arctan2(z, x)
    phase = RADIAL_FREQUENCY*radius+RADIAL_WARP*taper*np.sin(5*angle)
    return RADIAL_AMPLITUDE*taper*np.sin(phase)


def water_wave(x, z):
    shifts = [height*np.sin(kx*x+kz*z+phase) for height,kx,kz,phase in WARP]
    px, pz = x+shifts[0]+shifts[1], z+shifts[2]+shifts[3]
    base = sum(height*np.sin(kx*px+kz*pz+phase) for height,kx,kz,phase in MODES)
    return base+radial_wave(x,z)


def hash01(i, j):
    a = np.asarray(i, dtype=np.int32).astype(np.uint32)
    b = np.asarray(j, dtype=np.int32).astype(np.uint32)
    bits = (a*np.uint32(0x86d6d16b)) ^ (b*np.uint32(0xd16aa725)) ^ np.uint32(98431)
    bits ^= bits >> np.uint32(16)
    bits *= np.uint32(0x85ebca6b)
    bits ^= bits >> np.uint32(13)
    return (bits & np.uint32(65535)).astype(np.float32)/65535


def value_noise(x, z, size):
    u, v = x/size, z/size
    i, j = np.floor(u), np.floor(v)
    fx, fz = u-i, v-j
    fx, fz = fx*fx*(3-2*fx), fz*fz*(3-2*fz)
    top = (1-fx)*hash01(i,j) + fx*hash01(i+1,j)
    bottom = (1-fx)*hash01(i,j+1) + fx*hash01(i+1,j+1)
    return top*(1-fz) + bottom*fz
