#!/usr/bin/env python3
"""Generate data/rawspeed_levels.json from Rawspeed's cameras.xml.

Rawspeed's camera database carries the measured per-camera sensor black and
white levels that Ansel's rawprepare normalizes with — the domain the noise
profiles are fit in. This table lets the training normalize its tiles in the
same domain (src/ansel_denoise/levels.py).

Usage:
    python3 scripts/update_rawspeed_levels.py /path/to/ansel/src/external/rawspeed/data/cameras.xml

Keys are canonical camera names (levels.norm_key) built from both the
Camera make/model and the ID make/model aliases; each key maps to a list of
mode variants, each carrying its Sensor entries (optionally ISO-conditioned).
"""

from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ansel_denoise.levels import norm_key  # noqa: E402


def parse(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    cameras: dict[str, list[dict]] = {}
    n_cams = n_sensors = 0
    for cam in root.findall("Camera"):
        sensors = []
        for s in cam.findall("Sensor"):
            if "black" not in s.attrib or "white" not in s.attrib:
                continue
            entry: dict = {"black": float(s.get("black")), "white": float(s.get("white"))}
            if s.get("iso_list"):
                entry["iso_list"] = [float(v) for v in s.get("iso_list").split()]
            if s.get("iso_min"):
                entry["iso_min"] = float(s.get("iso_min"))
            if s.get("iso_max"):
                entry["iso_max"] = float(s.get("iso_max"))
            sensors.append(entry)
        if not sensors:
            continue
        n_cams += 1
        n_sensors += len(sensors)
        variant = {"mode": cam.get("mode", ""), "sensors": sensors}

        keys = {norm_key(f"{cam.get('make', '')} {cam.get('model', '')}")}
        cid = cam.find("ID")
        if cid is not None:
            keys.add(norm_key(f"{cid.get('make', '')} {cid.get('model', '')}"))
            if cid.text and cid.text.strip():
                keys.add(norm_key(cid.text))
        for k in keys:
            if k:
                cameras.setdefault(k, []).append(variant)
    print(f"{n_cams} cameras with levels, {n_sensors} sensor entries, {len(cameras)} keys")
    return cameras


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    xml_path = Path(sys.argv[1])
    cameras = parse(xml_path)
    out = REPO / "data" / "rawspeed_levels.json"
    out.write_text(json.dumps(
        {"source": "rawspeed cameras.xml", "sha1": hashlib.sha1(xml_path.read_bytes()).hexdigest(),
         "cameras": cameras}, sort_keys=True) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
