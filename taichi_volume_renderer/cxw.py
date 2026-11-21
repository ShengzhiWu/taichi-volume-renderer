"""Read packed-color CXW saves into compact sector tables and a brick pool."""

import os
import struct
import zlib

import numpy as np

from .math import rotation_quaternion_to_matrix
from .sparse_data import Entity, SparseVolumeData


ENTITY = struct.Struct("<16s3f4fH B 6f B Q I")
REGION_HEADER_SIZE = 4096 * 4 + 4
SECTOR_HEADER = struct.Struct("<4s3I")
MAX_INFLATED_SECTOR = 1 << 30


# Preserve imports from the format module for existing callers.
CXWData = SparseVolumeData


def packed_color_materials(density):
    """Decode packed CXW block colors into an RGB-density table."""
    codes = np.arange(65536, dtype=np.uint32)
    materials = np.zeros((len(codes), 4), dtype=np.float32)
    materials[1:, 3] = density
    opaque = codes >= 0x8000
    emissive = (codes & 0xc000) == 0x4000
    for channel, shift in enumerate((10, 5, 0)):
        materials[opaque, channel] = ((codes[opaque] >> shift) & 31) / 31
    for channel, shift in enumerate((10, 6, 2)):
        materials[emissive, channel] = ((codes[emissive] >> shift) & 15) / 15
    materials[1:256, :3] = (0.72, 0.83, 0.9)
    return materials


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


def _sector_header(data):
    magic, capacity, count, free_count = SECTOR_HEADER.unpack_from(data)
    if magic != b"VXS2" or capacity > 4096 or count + free_count != capacity:
        raise ValueError("Invalid CXW sector header")
    return capacity, count, free_count


def _count_sector_bricks(read, offset, length):
    inflate = zlib.decompressobj(-15)
    header = bytearray()
    while length and not inflate.eof:
        chunk_size = min(length, 4096)
        header.extend(inflate.decompress(read(offset, chunk_size), SECTOR_HEADER.size - len(header)))
        if len(header) == SECTOR_HEADER.size:
            return _sector_header(header)[1]
        offset += chunk_size
        length -= chunk_size
    raise ValueError("Truncated CXW sector header")


def _decode_sector(compressed, output=None):
    inflate = zlib.decompressobj(-15)
    data = inflate.decompress(compressed, MAX_INFLATED_SECTOR)
    if not inflate.eof or inflate.unused_data or len(data) < 16406:
        raise ValueError("Invalid CXW deflate stream or sector size")

    capacity, count, free_count = _sector_header(data)
    indices = np.frombuffer(data, dtype="<i2", count=4096 + free_count, offset=SECTOR_HEADER.size)
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
        return mapped

    active = blocks.reshape(capacity, 512)[ids]
    if np.any((active >= 256) & (active < 0x4000)):
        raise ValueError("Invalid packed-color block code in CXW sector")
    occupied = np.any(active, axis=1)
    mapped = mapped[occupied]
    if output is not None:
        if len(mapped) > len(output):
            raise ValueError("CXW sector grew during loading")
        output[:len(mapped)] = active[occupied]
    return mapped


def load_cxw(path):
    """Load a packed-color CXW v7/v8 file into Z-up world coordinates."""
    entities, sectors, tables = [], [], []
    entity_payloads = []
    allocated_bricks = 0
    with open(path, "rb") as file:
        size = os.fstat(file.fileno()).st_size

        def read(offset, length):
            return _read_at(file, offset, length, size)

        def decode(coord, offset, length, output=None):
            try:
                return _decode_sector(read(offset, length), output)
            except (zlib.error, ValueError) as exc:
                raise ValueError(f"Cannot decode CXW sector {coord}: {exc}") from exc

        magic, version, flags, table_offset = struct.unpack_from("<4sHHQ", read(0, 64))
        if magic not in (b"CXLS", b"VXLS") or version not in (7, 8) or flags != 1:
            raise ValueError("Expected a deflate-compressed CXW v7 or v8 world")
        pose_flag, *pose, entity_count = struct.unpack("<B7fI", read(table_offset, 33))
        player_pose = None
        if pose_flag & 1:
            position = np.array(pose[:3], dtype=np.float64)
            if not np.all(np.isfinite(position)):
                raise ValueError("Invalid CXW player position")
            player_pose = (position[[0, 2, 1]], _rotation(pose[3:])[[0, 2, 1]])

        records = read(table_offset + 33, ENTITY.size * entity_count)
        for record in ENTITY.iter_unpack(records):
            position = np.array(record[1:4], dtype=np.float64)
            if not np.all(np.isfinite(position)):
                raise ValueError("Invalid CXW entity position")
            # Change the world axes while preserving entity-local brick coordinates.
            position = position[[0, 2, 1]]
            rotation = _rotation(record[4:8])[[0, 2, 1]]
            index_offset, sector_count = record[-2:]
            index = sorted(struct.iter_unpack("<3iQI", read(index_offset, sector_count * 24)))
            payloads = []
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
                payload_offset = region_offset + REGION_HEADER_SIZE
                try:
                    count = _count_sector_bricks(read, payload_offset, compressed_size)
                except (zlib.error, ValueError) as exc:
                    raise ValueError(f"Cannot read CXW sector {coord}: {exc}") from exc
                allocated_bricks += count
                payloads.append((coord, payload_offset, compressed_size, count))
            entity_payloads.append((position, rotation, payloads))

        # Header counts include allocated air bricks. Compact the occupied bricks as we decode.
        try:
            blocks = np.empty((allocated_bricks, 512), dtype=np.uint16)
        except MemoryError:
            occupied_bricks = sum(
                len(decode(coord, offset, length))
                for _, _, payloads in entity_payloads
                for coord, offset, length, _ in payloads
            )
            blocks = np.empty((occupied_bricks, 512), dtype=np.uint16)
        brick_count = 0
        for position, rotation, payloads in entity_payloads:
            first = len(sectors)
            for coord, offset, length, count in payloads:
                mapped = decode(coord, offset, length, blocks[brick_count:brick_count + count])
                if not len(mapped):
                    continue
                table = np.full(4096, -1, dtype=np.int32)
                next_brick = brick_count + len(mapped)
                table[mapped] = np.arange(brick_count, next_brick, dtype=np.int32)
                brick_count = next_brick
                sectors.append(coord)
                tables.append(table)

            end = len(sectors)
            if first < end:
                coords = np.asarray(sectors[first:end], dtype=np.int64)
                entities.append(Entity(position, rotation, first, end,
                                       128 * coords.min(axis=0), 128 * (coords.max(axis=0) + 1)))

    if not entities:
        raise ValueError("CXW world has no occupied bricks")
    return SparseVolumeData(entities, np.asarray(sectors, dtype=np.int32), np.stack(tables),
                            blocks[:brick_count], player_pose)
