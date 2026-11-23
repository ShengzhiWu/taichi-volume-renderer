# VXB sparse voxel scenes

VXB (`.vxb`) is the sparse voxel scene format developed for taichi-volume-renderer. A VXB file contains one Z-up world grid, its material schema, its brick hierarchy, and an optional initial camera pose. A decoder can read the entire scene from that one file. The format has a byte-level specification and a standalone Python codec that uses only the standard library. See the [VXB v1 specification](vxb-v1.md) and [reference codec](../taichi_volume_renderer/vxb.py).

## Material storage

Each file declares its material fields by name and type. Fields can be 32-bit floats, signed 32-bit integers, or booleans. A typical volume uses linear `r`, `g`, `b`, and `density`. The codec preserves custom fields with supported types, and each renderer can use the fields it understands.

| Mode | What a voxel stores | When to use it |
| --- | --- | --- |
| Indexed | A bit-packed index into the material table in the same file. Index 0 is air. | Many voxels share a limited set of materials. A table with 300 entries needs 9 bits per index. |
| Direct | One row of field values in the declared order. An all-zero row is air. | Values vary continuously and relatively few voxels share a material. |

The file header selects the mode. Both modes use the same brick layout.

## Brick hierarchy

The file declares its brick sizes from largest to smallest. For example, a file can use `[128, 8]` or `[128, 32, 8]`. Missing bricks represent air. A uniform brick stores one material for its entire volume, even for a large block of solid rock or water. A brick with children lists only the child slots it uses. Mixed bricks at the smallest level store values for all 512 voxels in an 8³ grid.

A directory gives the offset and sizes of each top-level brick. Each top-level brick and its descendants are compressed as an independent raw Deflate stream, so a reader can access one top-level brick without inflating the others. An [annotated sample file](vxb-v1.md#decoder-reference) illustrates the hierarchy, uniform bricks, and cross-byte indices.

## Open a scene

Run the sparse viewer from the repository root.

```bash
python -m examples.sparse_smoke "path/to/world.vxb"
```

Add `--camera saved` to start from the file's camera pose when there is one. The Python entry point is `DisplayWindow.from_file()`.

The current sparse renderer accepts the basic volume fields in either mode. Indexed materials can also supply the following properties:

| Fields | Meaning |
| --- | --- |
| `ior` | Index of refraction |
| `normal_x`, `normal_y`, `normal_z` | World-space interface normal. A zero vector uses the renderer's estimate. |
| `emission_r`, `emission_g`, `emission_b` | Linear outgoing radiance added after ordinary lighting. This can carry baked caustics. |

These rendering fields use `f32`. The current importer accepts up to 65536 indexed material rows and adapts the file hierarchy to the renderer's 128/8 storage. Uniform bricks with an edge length of at least 128 voxels stay compact. Smaller uniform bricks expand into the regular brick pool and share one cached light value per original brick. The renderer computes the value at the uniform brick's center.

## Convert an existing CXW scene

The repository includes a converter for packed-color CXW v7/v8 scenes with one untransformed occupied entity.

```bash
python -m scripts.cxw_to_vxb "path/to/world.cxw" "path/to/world.vxb"
```

The converter preserves the occupied voxels, original material codes, packed emission, and saved camera pose. It uses the CXW importer's RGB and density defaults and prints the file sizes for comparison. VXB has no CXW preview records or auxiliary voxel slots.
