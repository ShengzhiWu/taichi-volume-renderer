# VXB version 1

VXB stores one sparse, Z-up world grid. Voxels contain only material indices. A file supplies its own material schema, brick hierarchy, and optional initial camera pose.

All multi-byte integers and IEEE 754 floats are little-endian. Sizes and offsets are in bytes. World coordinates are in voxel units. The unit voxel at integer coordinate `(x, y, z)` has center `(x + 0.5, y + 0.5, z + 0.5)`.

## File order

1. Fixed header.
2. Brick size exponents, one byte per level.
3. Camera pose, if present.
4. Material field descriptors and rows.
5. Top-level brick directory.
6. Compressed top-level brick payloads, in directory order.

### Header

| Offset | Type | Value |
|---:|---|---|
| 0 | 4 bytes | `56 58 42 00` (`VXB\0`) |
| 4 | u16 | Version, `1` |
| 6 | u8 | Flags. Bit 0 indicates a camera pose. All other bits are zero |
| 7 | u8 | Number of brick levels, 1–28 |
| 8 | u32 | Number of material rows, including air, at least 1 |
| 12 | u16 | Number of fields per material, at least 1 |
| 14 | u32 | Number of top-level bricks |

Each exponent `e` encodes a brick side length of `2**e` voxels. Exponents strictly decrease, range from 3 to 30, and end in 3. Consecutive exponents differ by at most 10, so child slot numbers fit in a `u32`. For example, bytes `07 03` select sizes 128 and 8. Bytes `07 05 03` select sizes 128, 32, and 8. A decoder uses the file's list for all bricks.

The optional camera has seven finite `f64` values in the order `px, py, pz, qx, qy, qz, qw`. The position is world-space. The quaternion rotates camera-local +X (right), +Y (up), and −Z (forward) into world space. Its length must be nonzero. Normalize it when using it to construct a rotation.

### Materials

Each field descriptor is a `u16` UTF-8 byte count, a `u8` type tag, then that many name bytes. Names are nonempty and unique, with byte-for-byte, case-sensitive comparison. Descriptors are followed by all material rows. Each row stores one value per field, in descriptor order. There is no padding between values or rows.

| Tag | Type | Representation |
|---:|---|---|
| 0 | `f32` | Finite IEEE 754 binary32 |
| 1 | `i32` | Signed 32-bit integer |
| 2 | `bool` | One byte, exactly 0 or 1 |

Material indices are zero-based row numbers. Index 0 is always air. Its row follows the same layout as other rows, and its values are ignored when interpreting the scene. VXB codecs preserve unrecognized field names with recognized types.

The basic volume fields are `r`, `g`, `b`, and `density`, all `f32`. The first three give linear RGB and `density` is nonnegative extinction density per voxel unit. Applications can agree on more fields, such as an `ior` float, world-space `normal_x`, `normal_y`, and `normal_z` floats, or a `use_voxel_face_normal` boolean. The wire format assigns no additional layout to these fields.

For `M` material rows, the index width `w` is the smallest nonnegative integer with `2**w >= M`. For example, 300 rows use 9 bits. With one row, the scene is all air and the top-level brick count is zero.

### Top-level brick directory

Each 28-byte entry has this layout:

| Type | Field |
|---|---|
| i32 × 3 | Top-level brick coordinates `(bx, by, bz)` |
| u64 | Absolute offset of the compressed payload |
| u32 | Compressed byte count |
| u32 | Decompressed byte count |

Coordinates are signed brick coordinates, so their product with the top-level side length gives the minimum voxel coordinate of that brick. Entries are unique and sorted lexicographically by `(bx, by, bz)`. Payloads are consecutive, beginning immediately after the directory and ending at EOF. Both sizes are positive. Each payload is one independent raw Deflate stream, without a zlib or gzip wrapper. A decompressor must obtain exactly the stated decompressed size.

### Brick records

A decompressed payload contains exactly one top-level brick record. Every brick begins with a `u8` tag.

| Tag | Record | Allowed levels |
|---:|---|---|
| 0 | Uniform material index, unsigned little-endian in `ceil(w/8)` bytes | Any |
| 1 | Positive `u32` child count, then one `u32` child slot and one brick record per child | Any except the smallest |
| 2 | 512 material indices, packed at `w` bits per index | Smallest only |

Uniform bricks must use a nonzero material index. Missing top-level bricks and missing child slots represent air. A child brick may itself be uniform, so a decoder can reconstruct the tree without expanding it into voxels.

If the next level fits `n` bricks along one axis, child coordinates `(x, y, z)` in `[0, n)` use slot `x + n*(y + n*z)`. Child slots are unique and written in ascending order. The same ordering applies to the voxels in a mixed brick, with `n = 8`. A mixed brick contains at least two different valid indices.

Pack mixed indices least-significant bit first, with no gaps. Equivalently, index `i` starts at bit `i*w` of the little-endian bit stream. For 3-bit indices, the first four values `1, 2, 3, 4` contribute `D1 08` to the first two bytes if the next value is zero. A mixed brick always consumes exactly `64*w` bytes after the tag.

## Decoder reference

Read the header, sizes, camera, and material rows. For each directory entry, inflate its payload and parse one brick. On a child record, read a slot number and recursively parse at the next level. No external palette or scene metadata is needed.

The [Python reference codec](../taichi_volume_renderer/vxb.py) uses only the standard library and can be copied or imported as a standalone module. The renderer's existing CXW and OpenVDB loaders are separate.

The [sample file](../tests/fixtures/vxb-v1.vxb) has levels `[128, 32, 8]`, five materials including air, and 3-bit mixed indices. Its camera position is `(1.25, -2, 3)` with an identity quaternion. The top-level brick at `(-1, 0, 0)` is entirely material 1. The brick at `(0, 0, 0)` has child slot 0 uniformly filled with material 2, and child slot 1 has its own child slot 0 containing a mixed brick. That mixed brick starts with `1, 2, 3, 4` and is otherwise air. The child slot at `(0, 1, 0)` of the top-level brick is absent, so voxel `(0, 32, 0)` is air.
