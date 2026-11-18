"""Read packed-color CXW saves into compact sector tables and a brick pool."""

from dataclasses import dataclass
import os
import struct
import zlib

import numpy as np

from .math import rotation_quaternion_to_matrix


ENTITY = struct.Struct("<16s3f4fH B 6f B Q I")
REGION_HEADER_SIZE = 4096 * 4 + 4
MAX_INFLATED_SECTOR = 1 << 30


@dataclass
class Entity:
    position: np.ndarray
    rotation: np.ndarray
    first_sector: int
    end_sector: int
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class CXWData:
    entities: list
    sectors: np.ndarray
    brick_ids: np.ndarray
    blocks: np.ndarray
    player_pose: object


def _rotation(values):
    quaternion = np.asarray(values, dtype=np.float64)
    length = np.linalg.norm(quaternion)
    if not np.isfinite(length) or length == 0:
        raise ValueError("Invalid CXW quaternion")
    return rotation_quaternion_to_matrix(quaternion[[3, 0, 1, 2]] / length)


def _read_at(file, offset, size, file_size):
    if offset < 0 or size < 0 or offset + size > file_size:
        raise ValueError("CXW offset or record size is outside the file")
    file.seek(offset)
    return file.read(size)


def _decode_sector(compressed):
    inflate = zlib.decompressobj(-15)
    data = inflate.decompress(compressed, MAX_INFLATED_SECTOR)
    if not inflate.eof or inflate.unused_data or len(data) < 16406:
        raise ValueError("Invalid CXW deflate stream or sector size")

    magic, capacity, count, free_count = struct.unpack_from("<4s3I", data)
    if magic != b"VXS2" or capacity > 4096 or count + free_count != capacity:
        raise ValueError("Invalid CXW sector header")
    indices = np.frombuffer(data, dtype="<i2", count=4096 + free_count, offset=16)
    mapped = np.flatnonzero(indices[:4096] != -1)
    ids = indices[mapped]
    if len(mapped) != count or not np.array_equal(
        np.sort(np.concatenate((ids, indices[4096:]))), np.arange(capacity)
    ):
        raise ValueError("Invalid CXW brick indices or free list")

    offset = 16402 + 2 * free_count
    if offset + 4 > len(data):
        raise ValueError("Truncated CXW slot table")
    slot_count, = struct.unpack_from("<I", data, offset)
    offset += 4
    if slot_count > 16:
        raise ValueError("Too many CXW slots")
    seen = set()
    blocks = None
    for _ in range(slot_count):
        if offset + 3 > len(data):
            raise ValueError("Truncated CXW slot header")
        slot, stride = struct.unpack_from("<BH", data, offset)
        offset += 3
        end = offset + capacity * 512 * stride
        if slot in seen or slot >= 16 or stride == 0 or end > len(data):
            raise ValueError("Invalid CXW slot")
        seen.add(slot)
        if slot == 0:
            if stride != 2:
                raise ValueError("CXW Block slot must use 16-bit values")
            blocks = np.frombuffer(data, dtype="<u2", count=capacity * 512, offset=offset)
        offset = end

    if offset != len(data) or (count and blocks is None):
        raise ValueError("Invalid CXW slot data")
    if not count:
        return mapped, np.empty((0, 512), dtype=np.uint16)

    active = blocks.reshape(capacity, 512)[ids]
    if np.any((active >= 256) & (active < 0x4000)):
        raise ValueError("Invalid packed-color block code in CXW sector")
    occupied = np.any(active, axis=1)
    return mapped[occupied], active[occupied].astype(np.uint16, copy=False)


def load_cxw(path):
    """Load a packed-color CXW v7/v8 file. Taichi need not be initialized."""
    entities, sectors, tables, chunks = [], [], [], []
    brick_count = 0
    with open(path, "rb") as file:
        size = os.fstat(file.fileno()).st_size

        def read(offset, length):
            return _read_at(file, offset, length, size)

        magic, version, flags, table_offset = struct.unpack_from("<4sHHQ", read(0, 64))
        if magic not in (b"CXLS", b"VXLS") or version not in (7, 8) or flags != 1:
            raise ValueError("Expected a deflate-compressed CXW v7 or v8 world")
        pose_flag, *pose, entity_count = struct.unpack("<B7fI", read(table_offset, 33))
        player_pose = None
        if pose_flag & 1:
            position = np.array(pose[:3], dtype=np.float64)
            if not np.all(np.isfinite(position)):
                raise ValueError("Invalid CXW player position")
            player_pose = (position, _rotation(pose[3:]))

        records = read(table_offset + 33, ENTITY.size * entity_count)
        for record in ENTITY.iter_unpack(records):
            position = np.array(record[1:4], dtype=np.float64)
            if not np.all(np.isfinite(position)):
                raise ValueError("Invalid CXW entity position")
            rotation = _rotation(record[4:8])
            index_offset, sector_count = record[-2:]
            index = sorted(struct.iter_unpack("<3iQI", read(index_offset, sector_count * 24)))
            first = len(sectors)
            previous = None
            for x, y, z, region_offset, region_size in index:
                coord = (x, y, z)
                if coord == previous:
                    raise ValueError(f"Duplicate CXW sector {coord}")
                previous = coord
                if any(value < -(1 << 24) or value >= (1 << 24) for value in coord):
                    raise ValueError(f"CXW sector {coord} is outside signed voxel coordinates")
                compressed_size, = struct.unpack("<I", read(region_offset + REGION_HEADER_SIZE - 4, 4))
                if region_size != REGION_HEADER_SIZE + compressed_size:
                    raise ValueError(f"Invalid CXW region size for sector {coord}")
                try:
                    mapped, blocks = _decode_sector(read(region_offset + REGION_HEADER_SIZE, compressed_size))
                except (zlib.error, ValueError) as exc:
                    raise ValueError(f"Cannot decode CXW sector {coord}: {exc}") from exc
                if not len(blocks):
                    continue
                table = np.full(4096, -1, dtype=np.int32)
                table[mapped] = np.arange(brick_count, brick_count + len(blocks), dtype=np.int32)
                brick_count += len(blocks)
                sectors.append(coord)
                tables.append(table)
                chunks.append(blocks)

            end = len(sectors)
            if first < end:
                coords = np.asarray(sectors[first:end], dtype=np.int64)
                entities.append(Entity(position, rotation, first, end,
                                       128 * coords.min(axis=0), 128 * (coords.max(axis=0) + 1)))

    if not entities:
        raise ValueError("CXW world has no occupied bricks")
    return CXWData(entities, np.asarray(sectors, dtype=np.int32), np.stack(tables),
                   np.concatenate(chunks), player_pose)
