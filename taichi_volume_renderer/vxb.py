"""Read and write VXB v1 sparse material scenes using the Python standard library."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from numbers import Integral, Real
import os
from pathlib import Path
import struct
import tempfile
from typing import Dict, Tuple, Union
import zlib


_HEADER = struct.Struct("<4sHBBIHI")
_ROOT = struct.Struct("<3iQII")
_POSE = struct.Struct("<7d")
_U32 = struct.Struct("<I")
_FIELD_TYPES = {"f32": (0, "f"), "i32": (1, "i"), "bool": (2, "B")}
_TYPE_NAMES = tuple(_FIELD_TYPES)
_MAGIC = b"VXB\0"
_VERSION = 1
_LEAF_VOXELS = 512

# A brick is a uniform index, a tuple of 512 indices, or a sparse map of child slots.
Brick = Union[int, Tuple[int, ...], Dict[int, "Brick"]]


@dataclass(frozen=True)
class MaterialField:
    name: str
    type: str


@dataclass(frozen=True)
class CameraPose:
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True)
class VxbScene:
    fields: tuple[MaterialField, ...]
    materials: tuple[tuple[float | int | bool, ...], ...]
    brick_sizes: tuple[int, ...]
    bricks: dict[tuple[int, int, int], Brick]
    camera: CameraPose | None = None


def _read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("Truncated VXB file or brick")
    return data


def _brick_ratios(sizes):
    if (not sizes or len(sizes) > 28 or sizes[-1] != 8
            or any(not isinstance(size, Integral) or not 8 <= size <= 2**30
                   or size & (size - 1) for size in sizes)):
        raise ValueError("VXB brick sizes must be powers of two ending in 8")
    ratios = tuple(int(larger // smaller) for larger, smaller in zip(sizes, sizes[1:]))
    if any(not 2 <= ratio <= 1024 for ratio in ratios):
        raise ValueError("VXB brick sizes must decrease by a factor of 2 to 1024")
    return ratios


def _check_camera(values):
    if not all(math.isfinite(value) for value in values) or not any(values[3:]):
        raise ValueError("VXB camera needs a finite position and nonzero finite quaternion")


def _material_layout(fields):
    names = set()
    codes, descriptors = [], []
    for field in fields:
        name = field.name.encode("utf-8")
        if not 1 <= len(name) <= 65535 or name in names:
            raise ValueError("VXB material field names must be nonempty and unique")
        if field.type not in _FIELD_TYPES:
            raise ValueError(f"Unknown VXB material field type: {field.type}")
        names.add(name)
        tag, code = _FIELD_TYPES[field.type]
        descriptors.append(struct.pack("<HB", len(name), tag) + name)
        codes.append(code)
    return descriptors, struct.Struct("<" + "".join(codes))


def _check_index(index, count):
    if not isinstance(index, Integral) or isinstance(index, bool) or not 0 <= index < count:
        raise ValueError("VXB material index is outside the table")
    return int(index)


def _pack_indices(indices, width, count):
    data = bytearray()
    pending = bits = 0
    for index in indices:
        pending |= _check_index(index, count) << bits
        bits += width
        while bits >= 8:
            data.append(pending & 255)
            pending >>= 8
            bits -= 8
    return data


def _unpack_indices(data, width):
    pending = bits = 0
    mask = (1 << width) - 1
    source = iter(data)
    for _ in range(_LEAF_VOXELS):
        while bits < width:
            pending |= next(source) << bits
            bits += 8
        yield pending & mask
        pending >>= width
        bits -= width


def _write_brick(stream, brick, ratios, level, width, count):
    if isinstance(brick, Integral):
        index = _check_index(brick, count)
        if index == 0:
            raise ValueError("Omit air bricks instead of writing a uniform index of zero")
        stream.write(b"\x00" + index.to_bytes((width + 7) // 8, "little"))
    elif isinstance(brick, dict):
        if level == len(ratios) or not brick or len(brick) >= 2**32:
            raise ValueError("Only non-leaf VXB bricks can have children")
        stream.write(b"\x01" + _U32.pack(len(brick)))
        for slot, child in sorted(brick.items()):
            if not isinstance(slot, Integral) or not 0 <= slot < ratios[level]**3:
                raise ValueError("VXB child slot is outside its brick")
            stream.write(_U32.pack(slot))
            _write_brick(stream, child, ratios, level + 1, width, count)
    else:
        if not isinstance(brick, tuple) or level != len(ratios) or len(brick) != _LEAF_VOXELS:
            raise ValueError("A mixed VXB brick needs 512 indices at the smallest level")
        if min(brick) == max(brick):
            raise ValueError("Store a constant VXB brick as one uniform index")
        stream.write(b"\x02" + _pack_indices(brick, width, count))


def _read_brick(stream, ratios, level, width, count):
    tag = _read_exact(stream, 1)[0]
    if tag == 0:
        index = int.from_bytes(_read_exact(stream, (width + 7) // 8), "little")
        if not 0 < index < count:
            raise ValueError("Invalid VXB uniform brick index")
        return index
    if tag == 1 and level < len(ratios):
        size, = _U32.unpack(_read_exact(stream, _U32.size))
        if not 0 < size <= ratios[level]**3:
            raise ValueError("Invalid VXB child count")
        children = {}
        previous = -1
        for _ in range(size):
            slot, = _U32.unpack(_read_exact(stream, _U32.size))
            if not previous < slot < ratios[level]**3:
                raise ValueError("VXB child slots must be sorted, unique, and in range")
            previous = slot
            children[slot] = _read_brick(stream, ratios, level + 1, width, count)
        return children
    if tag == 2 and level == len(ratios):
        indices = tuple(_unpack_indices(_read_exact(stream, _LEAF_VOXELS * width // 8), width))
        if max(indices) >= count or min(indices) == max(indices):
            raise ValueError("Invalid VXB mixed brick indices")
        return indices
    raise ValueError("Invalid VXB brick tag for this level")


def _max_brick_bytes(ratios, width):
    uniform = 1 + (width + 7) // 8
    size = max(uniform, 1 + _LEAF_VOXELS * width // 8)
    for ratio in reversed(ratios):
        size = max(uniform, 5 + ratio**3 * (4 + size))
    return size


def load_vxb(path) -> VxbScene:
    """Read material rows, all brick levels, and any initial camera pose."""
    with open(path, "rb") as stream:
        file_size = stream.seek(0, os.SEEK_END)
        stream.seek(0)
        magic, version, flags, levels, count, field_count, root_count = _HEADER.unpack(
            _read_exact(stream, _HEADER.size))
        if (magic != _MAGIC or version != _VERSION or flags & ~1 or not 1 <= levels <= 28
                or not count or not field_count or (count == 1 and root_count)):
            raise ValueError("Invalid or unsupported VXB header")
        sizes = tuple(1 << power for power in _read_exact(stream, levels))
        ratios = _brick_ratios(sizes)
        camera = None
        if flags & 1:
            values = _POSE.unpack(_read_exact(stream, _POSE.size))
            _check_camera(values)
            camera = CameraPose(values[:3], values[3:])

        fields = []
        for _ in range(field_count):
            length, tag = struct.unpack("<HB", _read_exact(stream, 3))
            if tag >= len(_TYPE_NAMES):
                raise ValueError("Unknown VXB material field type")
            fields.append(MaterialField(_read_exact(stream, length).decode("utf-8"), _TYPE_NAMES[tag]))
        _, row_layout = _material_layout(fields)
        rows_end = stream.tell() + row_layout.size * count
        payload_start = rows_end + root_count * _ROOT.size
        if payload_start > file_size:
            raise ValueError("Truncated VXB material table or brick directory")
        rows = []
        for _ in range(count):
            row = []
            for field, value in zip(fields, row_layout.unpack(_read_exact(stream, row_layout.size))):
                if field.type == "bool":
                    if value not in (0, 1):
                        raise ValueError("Invalid VXB boolean")
                    value = bool(value)
                elif field.type == "f32" and not math.isfinite(value):
                    raise ValueError("VXB float fields must be finite")
                row.append(value)
            rows.append(tuple(row))

        width = (count - 1).bit_length()
        max_raw = _max_brick_bytes(ratios, width)
        entries = []
        previous = None
        expected_offset = payload_start
        for _ in range(root_count):
            x, y, z, offset, packed, raw = _ROOT.unpack(_read_exact(stream, _ROOT.size))
            coord = (x, y, z)
            if ((previous is not None and coord <= previous) or offset != expected_offset
                    or not 0 < packed <= file_size - offset or not 0 < raw <= max_raw):
                raise ValueError("Invalid VXB brick directory")
            entries.append((coord, packed, raw))
            previous = coord
            expected_offset += packed
        if expected_offset != file_size:
            raise ValueError("Unexpected bytes after the VXB brick payloads")

        bricks = {}
        for coord, packed, raw in entries:
            inflater = zlib.decompressobj(-15)
            try:
                data = inflater.decompress(_read_exact(stream, packed), raw + 1)
            except zlib.error as error:
                raise ValueError("Invalid VXB Deflate payload") from error
            if len(data) != raw or not inflater.eof or inflater.unused_data:
                raise ValueError("Invalid VXB decompressed brick size")
            payload = BytesIO(data)
            bricks[coord] = _read_brick(payload, ratios, 0, width, count)
            if payload.read(1):
                raise ValueError("Unexpected bytes at the end of a VXB brick")
    return VxbScene(tuple(fields), tuple(rows), sizes, bricks, camera)


def save_vxb(path, scene):
    """Encode a VXB scene and replace the destination after a successful write."""
    ratios = _brick_ratios(scene.brick_sizes)
    count = len(scene.materials)
    if not 1 <= count < 2**32 or not 1 <= len(scene.fields) < 2**16 or len(scene.bricks) >= 2**32:
        raise ValueError("VXB scene counts exceed the format limits")
    if count == 1 and scene.bricks:
        raise ValueError("A VXB file with only air cannot contain bricks")
    descriptors, row_layout = _material_layout(scene.fields)
    width = (count - 1).bit_length()
    camera = scene.camera
    path = Path(path)
    with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
        temporary = Path(scratch) / "scene.vxb"
        with temporary.open("wb") as stream:
            stream.write(_HEADER.pack(_MAGIC, _VERSION, int(camera is not None), len(scene.brick_sizes),
                                      count, len(scene.fields), len(scene.bricks)))
            stream.write(bytes(int(size).bit_length() - 1 for size in scene.brick_sizes))
            if camera is not None:
                values = (*camera.position, *camera.rotation)
                if len(camera.position) != 3 or len(camera.rotation) != 4:
                    raise ValueError("VXB camera needs three position and four quaternion values")
                _check_camera(values)
                stream.write(_POSE.pack(*values))
            stream.writelines(descriptors)
            for row in scene.materials:
                if len(row) != len(scene.fields):
                    raise ValueError("VXB material row does not match the schema")
                for field, value in zip(scene.fields, row):
                    if field.type == "bool":
                        valid = isinstance(value, bool)
                    elif field.type == "i32":
                        valid = isinstance(value, Integral) and not isinstance(value, bool)
                    else:
                        valid = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
                    if not valid:
                        raise ValueError(f"Invalid value for VXB material field {field.name}")
                stream.write(row_layout.pack(*row))

            directory = stream.tell()
            stream.seek(directory + len(scene.bricks) * _ROOT.size)
            entries = []
            for coord, brick in sorted(scene.bricks.items()):
                payload = BytesIO()
                _write_brick(payload, brick, ratios, 0, width, count)
                raw = payload.getvalue()
                deflater = zlib.compressobj(wbits=-15)
                packed = deflater.compress(raw) + deflater.flush()
                entries.append(_ROOT.pack(*coord, stream.tell(), len(packed), len(raw)))
                stream.write(packed)
            stream.seek(directory)
            stream.writelines(entries)
        os.replace(temporary, path)
