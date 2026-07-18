"""Rawspeed black/white levels lookup — normalize training tiles in Ansel's domain.

The noise profiles are fit on data normalized by Rawspeed's per-camera sensor
levels (Ansel's rawprepare), while the harvest decodes with libraw, whose
levels occasionally disagree (typically the white level: Rawspeed curates the
measured sensor saturation, libraw often reports the format maximum — e.g.
Nikon D90: 3767 vs 4095, ~9%). Shards store raw ADUs untouched, so the fix is
a training-time normalization policy: when the shard's camera is found in the
table generated from Rawspeed's cameras.xml (scripts/update_rawspeed_levels.py
-> data/rawspeed_levels.json), normalize with Rawspeed's levels — the exact
runtime domain; otherwise fall back to the shard's libraw metadata.

Mode disambiguation: many cameras have several raw modes (12/14-bit,
compressed/sRAW) with different levels, and shards do not record the mode.
The libraw white level of the actual file arbitrates: the candidate whose
white is closest to it (within a sane ratio window) is the right bit depth.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_TABLE = Path(__file__).resolve().parents[2] / "data" / "rawspeed_levels.json"

# candidate white must be within this ratio window of the libraw white:
# rejects wrong-bit-depth modes (ratio ~4) while accepting saturation-margin
# disagreements (observed <= ~15%)
RATIO_MIN, RATIO_MAX = 0.6, 1.25


def norm_key(camera: str) -> str:
    """Canonical camera key: lowercase alnum tokens, corporate noise dropped,
    consecutive duplicate tokens collapsed ('NIKON CORPORATION NIKON D90' and
    Rawspeed's 'NIKON D90' both map to 'nikon d90')."""
    tokens = re.sub(r"[^a-z0-9]+", " ", camera.lower()).split()
    tokens = [t for t in tokens if t not in ("corporation", "corp", "co", "ltd", "gmbh")]
    out: list[str] = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return " ".join(out)


def _sensor_matches_iso(sensor: dict, iso: float | None) -> bool:
    if iso is None:
        return False
    if "iso_list" in sensor:
        return any(abs(iso - v) < 0.5 for v in sensor["iso_list"])
    lo, hi = sensor.get("iso_min"), sensor.get("iso_max")
    if lo is None and hi is None:
        return False
    return (lo is None or iso >= lo) and (hi is None or iso <= hi)


class RawspeedLevels:
    """Lookup of (black, white) by camera name, ISO and libraw white level."""

    def __init__(self, table: dict | Path | str | None = None):
        if isinstance(table, dict):
            self.cameras = table
        else:
            path = Path(table) if table else DEFAULT_TABLE
            self.cameras = (json.loads(path.read_text())["cameras"]
                            if path.is_file() else {})

    def lookup(self, camera: str, iso: float | None = None,
               libraw_white: float | None = None) -> tuple[float, float] | None:
        """Return (black, white) in Rawspeed's convention, or None if the
        camera is unknown or no candidate is consistent with libraw_white."""
        variants = self.cameras.get(norm_key(camera))
        if not variants:
            return None

        candidates: list[tuple[float, float]] = []
        for variant in variants:
            sensors = variant.get("sensors", [])
            picked = next((s for s in sensors if _sensor_matches_iso(s, iso)), None)
            if picked is None:  # default sensor: the one without ISO conditions
                picked = next((s for s in sensors
                               if "iso_list" not in s and "iso_min" not in s
                               and "iso_max" not in s), None)
            if picked and picked.get("white", 0) > picked.get("black", 0):
                candidates.append((float(picked["black"]), float(picked["white"])))

        if not candidates:
            return None
        if libraw_white is None or libraw_white <= 0:
            return candidates[0]
        best = min(candidates, key=lambda c: abs(c[1] / libraw_white - 1.0))
        ratio = best[1] / libraw_white
        return best if RATIO_MIN <= ratio <= RATIO_MAX else None
