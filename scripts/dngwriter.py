"""Minimal uncompressed CFA DNG writer (no third-party dependencies).

Enough of the DNG 1.4 tag set for LibRaw/rawspeed to decode the file as a
Bayer raw with known black/white levels and a known camera identity — which
is what a synthetic denoising benchmark needs: the same clean image can be
written twice, once pristine and once with calibrated noise, and pushed
through the real Ansel pipeline.

The camera identity is a parameter: writing a real Make/Model makes Ansel
match that camera's noise profile, so a module that reads the profile
(denoiseprofile, rawdenoiseai) behaves exactly as it would on a real file
from that body.
"""

from __future__ import annotations

import struct
from pathlib import Path

# tag, type, count, value/offset — types: 1=BYTE 2=ASCII 3=SHORT 4=LONG 5=RATIONAL 10=SRATIONAL
_TYPESIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 10: 8}


def _pack_value(typ: int, values) -> bytes:
    if typ == 2:
        return values.encode() + b"\0"
    fmt = {1: "<B", 3: "<H", 4: "<I"}.get(typ)
    if fmt:
        return b"".join(struct.pack(fmt, int(v)) for v in values)
    if typ in (5, 10):  # (num, den) pairs
        f = "<II" if typ == 5 else "<ii"
        return b"".join(struct.pack(f, int(n), int(d)) for n, d in values)
    raise ValueError(f"unsupported tiff type {typ}")


def _ifd_bytes(entries, heap_start):
    """Serialize one IFD; returns (ifd, heap). Values > 4 bytes go to the heap
    at absolute offset heap_start."""
    heap, ifd = b"", b""
    for tag, typ, values in entries:
        payload = _pack_value(typ, values)
        count = len(values) if typ != 2 else len(payload)
        if len(payload) <= 4:
            field = payload.ljust(4, b"\0")
        else:
            field = struct.pack("<I", heap_start + len(heap))
            heap += payload + (b"\0" if len(payload) % 2 else b"")
        ifd += struct.pack("<HHI", tag, typ, count) + field
    return struct.pack("<H", len(entries)) + ifd + struct.pack("<I", 0), heap


def write_dng(path: str | Path, mosaic_u16, black: int, white: int,
              cfa_pattern=(0, 1, 1, 2), make: str = "Synthetic",
              model: str = "Synthetic", color_matrix=None, iso: int = 100) -> Path:
    """Write mosaic_u16 (H, W) uint16 as an uncompressed CFA DNG.

    cfa_pattern is the 2x2 colour index block in row-major order using DNG
    colour codes (0=R, 1=G, 2=B), i.e. (0,1,1,2) for RGGB. `iso` lands in a
    minimal Exif IFD: without it the ISO reads as 0 and any consumer that
    interpolates a noise profile per ISO (denoiseprofile, rawdenoiseai) would
    silently get the wrong one.
    """
    import numpy as np

    data = np.ascontiguousarray(mosaic_u16.astype("<u2"))
    h, w = data.shape
    # a plausible daylight matrix (sRGB-ish); only affects colour, not noise
    cm = color_matrix or [(6722, 10000), (-635, 10000), (-963, 10000),
                          (-4287, 10000), (12460, 10000), (2028, 10000),
                          (-908, 10000), (2162, 10000), (5668, 10000)]

    entries = [
        (254, 4, [0]),                       # NewSubfileType
        (256, 4, [w]),                       # ImageWidth
        (257, 4, [h]),                       # ImageLength
        (258, 3, [16]),                      # BitsPerSample
        (259, 3, [1]),                       # Compression = none
        (262, 3, [32803]),                   # PhotometricInterpretation = CFA
        (271, 2, make),                      # Make
        (272, 2, model),                     # Model
        (273, 4, [0]),                       # StripOffsets (patched below)
        (277, 3, [1]),                       # SamplesPerPixel
        (278, 4, [h]),                       # RowsPerStrip
        (279, 4, [data.nbytes]),             # StripByteCounts
        (284, 3, [1]),                       # PlanarConfiguration
        (33421, 3, [2, 2]),                  # CFARepeatPatternDim
        (33422, 1, list(cfa_pattern)),       # CFAPattern
        (50706, 1, [1, 4, 0, 0]),            # DNGVersion
        (50707, 1, [1, 1, 0, 0]),            # DNGBackwardVersion
        (50708, 2, model),                   # UniqueCameraModel
        (50714, 3, [black]),                 # BlackLevel
        (50717, 4, [white]),                 # WhiteLevel
        (50721, 10, cm),                     # ColorMatrix1
        (50728, 5, [(1, 1), (1, 1), (1, 1)]),  # AsShotNeutral
        (50778, 3, [21]),                    # CalibrationIlluminant1 = D65
        (34665, 4, [0]),                     # ExifIFD pointer (patched below)
    ]
    entries.sort(key=lambda e: e[0])
    exif = [(34855, 3, [int(iso)]),          # ISOSpeedRatings
            (36867, 2, "2026:01:01 00:00:00")]  # DateTimeOriginal
    exif.sort(key=lambda e: e[0])

    # layout: header | IFD0 | heap0 | Exif IFD | heap1 | strip data. Sizes are
    # known up front, so the two internal pointers can be filled in directly.
    header = b"II\x2a\x00" + struct.pack("<I", 8)
    ifd0_size = 2 + 12 * len(entries) + 4
    _, heap0_probe = _ifd_bytes(entries, 0)
    exif_off = len(header) + ifd0_size + len(heap0_probe)
    exif_ifd_size = 2 + 12 * len(exif) + 4
    exif_ifd, exif_heap = _ifd_bytes(exif, exif_off + exif_ifd_size)
    strip_off = exif_off + exif_ifd_size + len(exif_heap)

    entries = [(t, ty, [strip_off] if t == 273 else
                ([exif_off] if t == 34665 else v)) for t, ty, v in entries]
    ifd0, heap0 = _ifd_bytes(entries, len(header) + ifd0_size)

    path = Path(path)
    path.write_bytes(header + ifd0 + heap0 + exif_ifd + exif_heap + data.tobytes())
    return path
