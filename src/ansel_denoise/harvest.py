"""Streaming tile harvester: raw archive -> training shards, bounded disk use.

Designed around the raw.pixls.us git-annex mirror (https://raw.pixls.us/data.annex.git):
the clone holds only pointers; each file is materialized with `git annex get`,
mined for CFA-aligned tiles, then freed with `git annex drop`. Peak disk usage
is one raw file plus the growing shard directory. The same code runs on any
plain directory of raw files (community contributions, RAISE, ...) without
git-annex.

Every source file produces at most one .npz shard holding uint16 sensor-ADU
tiles plus the metadata needed to normalize them exactly like Ansel's
rawprepare does (per-channel black levels, white level, CFA pattern, WB, ISO).
Provenance (path + git-annex key) travels inside the shard, and the ledger
(ledger.jsonl in the output directory) makes the job resumable and the dataset
reproducible from the annex commit hash alone.

Usage:
    python -m ansel_denoise.harvest --source data.annex --annex --out shards/rpu
    python -m ansel_denoise.harvest --source ~/my-raws --out shards/mine
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .cfa import aligned_offset, normalize_pattern

RAW_EXTENSIONS = {
    ".dng", ".nef", ".nrw", ".cr2", ".cr3", ".crw", ".arw", ".srf", ".sr2",
    ".raf", ".orf", ".rw2", ".pef", ".srw", ".raw", ".rwl", ".iiq", ".3fr",
    ".fff", ".mef", ".mos", ".erf", ".kdc", ".dcr",
}

# Foveon and other non-CFA raw formats have no mosaic to learn.
EXCLUDE_EXTENSIONS = {".x3f"}


# ---------------------------------------------------------------------------
# pure functions (unit-tested without rawpy)
# ---------------------------------------------------------------------------

def normalize_mosaic(
    adu: np.ndarray, colors4: np.ndarray, black_per_channel: np.ndarray, white: float
) -> np.ndarray:
    """(ADU - black) / (white - black), like Ansel's rawprepare. colors4 is the
    libraw 4-color index map (G2 kept distinct, as black levels may differ)."""
    black = np.asarray(black_per_channel, dtype=np.float32)[colors4]
    scale = float(white) - float(np.mean(black_per_channel))
    return (adu.astype(np.float32) - black) / max(scale, 1.0)


def score_tile(norm: np.ndarray, clip_threshold: float = 0.98) -> tuple[float, float]:
    """Return (clipped_fraction, texture_energy) of a normalized tile.

    Texture energy is the mean absolute 2-pixel gradient — 2 pixels, not 1, so
    that CFA color alternation does not register as texture.
    """
    clipped = float(np.mean(norm >= clip_threshold))
    gy = np.abs(np.diff(norm[::2, :], axis=0)).mean() if norm.shape[0] > 2 else 0.0
    gx = np.abs(np.diff(norm[:, ::2], axis=1)).mean() if norm.shape[1] > 2 else 0.0
    return clipped, float(gy + gx)


def pick_tiles(
    adu: np.ndarray,
    norm: np.ndarray,
    period: tuple[int, int],
    rng: np.random.Generator,
    tile_size: int = 256,
    n_tiles: int = 16,
    candidates_per_tile: int = 4,
    max_clipped: float = 0.02,
    min_texture: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample CFA-aligned candidate crops, drop clipped/flat ones, keep the
    most textured. Returns (tiles uint16 (N, ts, ts), offsets int32 (N, 2))."""
    h, w = adu.shape
    ph, pw = period
    if h < tile_size + ph or w < tile_size + pw:
        return np.empty((0, tile_size, tile_size), np.uint16), np.empty((0, 2), np.int32)

    scored = []
    for _ in range(n_tiles * candidates_per_tile):
        oy = aligned_offset(rng, h, tile_size, ph)
        ox = aligned_offset(rng, w, tile_size, pw)
        tile_norm = norm[oy : oy + tile_size, ox : ox + tile_size]
        clipped, texture = score_tile(tile_norm)
        if clipped <= max_clipped and texture >= min_texture:
            scored.append((texture, oy, ox))

    # most textured first, deduplicated on position
    scored.sort(reverse=True)
    tiles, offsets, seen = [], [], set()
    for _, oy, ox in scored:
        if (oy, ox) in seen:
            continue
        seen.add((oy, ox))
        tiles.append(adu[oy : oy + tile_size, ox : ox + tile_size].astype(np.uint16))
        offsets.append((oy, ox))
        if len(tiles) >= n_tiles:
            break
    if not tiles:
        return np.empty((0, tile_size, tile_size), np.uint16), np.empty((0, 2), np.int32)
    return np.stack(tiles), np.asarray(offsets, dtype=np.int32)


# ---------------------------------------------------------------------------
# external tools
# ---------------------------------------------------------------------------

def read_exif(path: Path) -> dict:
    out = subprocess.run(
        ["exiftool", "-j", "-ISO", "-Make", "-Model", "-ExposureTime", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)[0]


def annex(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), "annex", *args], capture_output=True, text=True)


def decode_raw(path: Path) -> dict | None:
    """Decode with libraw; return the mosaic and normalization metadata, or
    None for non-mosaic files (already-demosaiced DNG, mono, Foveon...)."""
    import rawpy

    try:
        raw = rawpy.imread(str(path))
    except Exception:
        return None
    with raw:
        pattern = raw.raw_pattern
        if pattern is None or raw.raw_image_visible is None or raw.num_colors != 3:
            return None
        colors4 = np.ascontiguousarray(raw.raw_colors_visible.astype(np.uint8))
        ph, pw = np.asarray(pattern).shape
        # Re-read the pattern from the visible-area origin: libraw's raw_pattern
        # is phased on the full sensor incl. margins, while our tile offsets are
        # phased on the visible area. Since tile offsets are period-aligned,
        # this block reconstructs every tile's color map exactly.
        return {
            "adu": np.ascontiguousarray(raw.raw_image_visible),
            "colors4": colors4,
            "pattern4": colors4[:ph, :pw].copy(),
            "black_per_channel": np.asarray(raw.black_level_per_channel, dtype=np.float32),
            "white": float(raw.white_level),
            "wb": np.asarray(raw.camera_whitebalance, dtype=np.float32),
        }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def list_sources(source: Path, use_annex: bool) -> list[str]:
    if use_annex:
        out = subprocess.run(
            ["git", "-C", str(source), "ls-files"], capture_output=True, text=True, check=True
        ).stdout
        names = out.splitlines()
    else:
        names = [str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()]
    return sorted(
        n for n in names
        if Path(n).suffix.lower() in RAW_EXTENSIONS and Path(n).suffix.lower() not in EXCLUDE_EXTENSIONS
    )


def _pack_worker(path: str, rel: str, out_dir: str, tile_size: int, n_tiles: int, seed: int,
                 iso: int, camera: str, annex_key: str, q) -> None:
    """Decode + tile-extract + shard-write, run in a DISPOSABLE child process:
    raw.pixls.us deliberately hosts decoder-hostile files, and a libraw
    segfault must cost one ledger entry, not the harvest run."""
    try:
        dec = decode_raw(Path(path))
        if dec is None:
            q.put({"status": "rejected", "reason": "not a 3-color mosaic raw"})
            return
        # per-file rng: deterministic under resume regardless of processing order
        rng = np.random.default_rng((seed, int(hashlib.md5(rel.encode()).hexdigest()[:8], 16)))
        norm = normalize_mosaic(dec["adu"], dec["colors4"], dec["black_per_channel"], dec["white"])
        tiles, offsets = pick_tiles(
            dec["adu"], norm, dec["pattern4"].shape, rng, tile_size=tile_size, n_tiles=n_tiles
        )
        if len(tiles) == 0:
            q.put({"status": "rejected", "reason": "no usable tiles (clipped or flat)"})
            return

        shard = Path(out_dir) / (rel.replace("/", "_") + ".npz")
        tmp = shard.with_name(shard.name + ".tmp")
        with open(tmp, "wb") as f:  # write-then-rename: never leave a truncated shard
            np.savez_compressed(
                f,
                tiles=tiles,
                offsets=offsets,
                pattern=normalize_pattern(dec["pattern4"]),
                pattern4=dec["pattern4"],
                black_per_channel=dec["black_per_channel"],
                white=np.float32(dec["white"]),
                wb=dec["wb"],
                iso=np.int32(iso),
                camera=np.str_(camera),
                source_path=np.str_(rel),
                annex_key=np.str_(annex_key),
            )
        os.replace(tmp, shard)
        q.put({"status": "harvested", "n_tiles": int(len(tiles)), "shard": shard.name})
    except Exception as e:  # plain decode errors are data, not crashes
        q.put({"status": "error", "reason": repr(e)[:300]})


def run_isolated(target, args: tuple, timeout: int = 300) -> dict:
    """Run target(*args, queue) in a disposable child; survive segfaults/hangs.

    Start method is 'spawn', NOT 'fork': libraw is built with OpenMP, and a
    forked child inheriting the parent's OpenMP runtime state deadlocks in
    decode (observed: harvest stalled on the file after the first crash).
    Spawned children re-import the module, so target must be module-level.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(*args, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(10)
        return {"status": "error", "reason": f"decoder hang, killed after {timeout}s"}
    try:
        return q.get(timeout=5)
    except Exception:
        sig = f"signal {-p.exitcode}" if (p.exitcode or 0) < 0 else f"exit code {p.exitcode}"
        return {"status": "error", "reason": f"decoder crashed ({sig})"}


def harvest_one(source: Path, rel: str, out_dir: Path, args: argparse.Namespace) -> dict:
    """Process a single raw file; returns a ledger record."""
    path = source / rel
    record: dict = {"path": rel, "status": "rejected"}

    if args.annex:
        got = annex(source, "get", "--quiet", rel)
        if got.returncode != 0:
            return {**record, "reason": f"annex get failed: {got.stderr.strip()[:200]}"}
        key = annex(source, "lookupkey", rel)
        record["annex_key"] = key.stdout.strip() or None

    try:
        exif = read_exif(path)
        iso = exif.get("ISO")
        if iso is None or (isinstance(iso, str) and not iso.isdigit()):
            return {**record, "reason": "no ISO in EXIF"}
        iso = int(iso)
        record["iso"] = iso
        record["camera"] = f"{exif.get('Make', '?')} {exif.get('Model', '?')}"
        if iso > args.max_iso:
            return {**record, "reason": f"ISO {iso} > {args.max_iso}"}

        result = run_isolated(
            _pack_worker,
            (str(path), rel, str(out_dir), args.tile_size, args.tiles, args.seed,
             iso, record["camera"], record.get("annex_key") or ""),
        )
        return {**record, **result}
    finally:
        if args.annex:
            dropped = annex(source, "drop", "--quiet", rel)
            if dropped.returncode != 0:
                print(f"  warning: annex drop failed for {rel}: {dropped.stderr.strip()[:200]}",
                      file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True,
                    help="git-annex clone (with --annex) or plain directory of raw files")
    ap.add_argument("--out", type=Path, required=True, help="shard output directory")
    ap.add_argument("--annex", action="store_true", help="source is a git-annex repo: get/drop per file")
    ap.add_argument("--max-iso", type=int, default=200, help="reject files above this ISO (default 200)")
    ap.add_argument("--tiles", type=int, default=16, help="max tiles per source file")
    ap.add_argument("--tile-size", type=int, default=256, help="tile side in sensor pixels (CFA-aligned)")
    ap.add_argument("--include", default="", help="only paths containing this substring")
    ap.add_argument("--limit", type=int, default=0, help="stop after N processed files (0 = all)")
    ap.add_argument("--seed", type=int, default=0xA45E1)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    ledger_path = args.out / "ledger.jsonl"
    done = set()
    if ledger_path.exists():
        with open(ledger_path) as f:
            done = {json.loads(line)["path"] for line in f if line.strip()}

    sources = [s for s in list_sources(args.source, args.annex) if args.include in s and s not in done]
    if args.limit:
        sources = sources[: args.limit]
    print(f"{len(sources)} files to process ({len(done)} already in ledger)")

    n_ok = 0
    with open(ledger_path, "a") as ledger:
        for i, rel in enumerate(sources):
            try:
                record = harvest_one(args.source, rel, args.out, args)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # a bad file must never kill a week-long run
                record = {"path": rel, "status": "error", "reason": repr(e)[:300]}
            ledger.write(json.dumps(record) + "\n")
            ledger.flush()
            n_ok += record["status"] == "harvested"
            print(f"[{i + 1}/{len(sources)}] {record['status']:9s} {rel}"
                  + (f" ({record.get('reason', '')})" if record["status"] != "harvested" else
                     f" -> {record['n_tiles']} tiles"))
    print(f"done: {n_ok} files harvested into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
