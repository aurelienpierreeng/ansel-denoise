"""Validate harvested shard files (.npz): structure, dtypes, ISO gate.

Shared by both ends of the community-contribution pipeline:
scripts/pack_contribution.py refuses to pack invalid shards on the
contributor side, and scripts/collect_contribution.sh refuses to merge them
on the maintainer side. Shards are loaded with allow_pickle=False, so a
hostile .npz cannot execute code on the maintainer's machine — the worst a
malformed file can do is get rejected.

Usage:
    python3.12 -m ansel_denoise.validate_shards shards/mine [--max-iso 200]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REQUIRED_KEYS = ("tiles", "offsets", "pattern", "pattern4", "black_per_channel",
                 "white", "wb", "iso", "camera", "source_path")


def validate_shard(path: Path, tile_size: int = 256, max_iso: int = 200) -> tuple[bool, dict]:
    """Check one shard; return (ok, info) — info carries 'reason' when not ok."""
    try:
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in REQUIRED_KEYS if k not in z]
            if missing:
                return False, {"reason": f"missing keys: {missing}"}
            tiles = z["tiles"]
            if tiles.dtype != np.uint16:
                return False, {"reason": f"tiles dtype {tiles.dtype}, expected uint16"}
            if tiles.ndim != 3 or tiles.shape[1:] != (tile_size, tile_size):
                return False, {"reason": f"tiles shape {tiles.shape}, "
                                         f"expected (N, {tile_size}, {tile_size})"}
            if len(tiles) == 0:
                return False, {"reason": "empty shard (0 tiles)"}
            if len(z["offsets"]) != len(tiles):
                return False, {"reason": "offsets/tiles length mismatch"}
            iso = float(z["iso"])
            if not 0 < iso <= max_iso:
                return False, {"reason": f"ISO {iso:.0f} outside (0, {max_iso}]"}
            return True, {"n_tiles": int(len(tiles)), "iso": iso, "camera": str(z["camera"])}
    except Exception as e:  # noqa: BLE001 — any unreadable file is simply invalid
        return False, {"reason": f"unreadable: {e!r}"}


def validate_dir(directory: Path, tile_size: int = 256, max_iso: int = 200,
                 verbose: bool = True) -> dict:
    """Validate every .npz in a directory (non-recursive). Returns a summary dict."""
    shards = sorted(directory.glob("*.npz"))
    bad, n_tiles, cameras = [], 0, Counter()
    for shard in shards:
        ok, info = validate_shard(shard, tile_size=tile_size, max_iso=max_iso)
        if ok:
            n_tiles += info["n_tiles"]
            cameras[info["camera"]] += info["n_tiles"]
        else:
            bad.append((shard.name, info["reason"]))
            if verbose:
                print(f"INVALID {shard.name}: {info['reason']}", file=sys.stderr)
    return {"n_shards": len(shards) - len(bad), "n_invalid": len(bad),
            "n_tiles": n_tiles, "cameras": dict(cameras), "invalid": bad}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: validate a shard directory, exit 1 if anything is invalid."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--max-iso", type=int, default=200)
    args = ap.parse_args(argv)

    summary = validate_dir(args.directory, tile_size=args.tile_size, max_iso=args.max_iso)
    print(f"{summary['n_shards']} valid shards, {summary['n_tiles']} tiles, "
          f"{len(summary['cameras'])} cameras, {summary['n_invalid']} invalid")
    for cam, n in sorted(summary["cameras"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:6d} tiles  {cam}")
    return 1 if summary["n_invalid"] or summary["n_shards"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
