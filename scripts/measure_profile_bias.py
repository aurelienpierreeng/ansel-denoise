#!/usr/bin/env python3
"""Measure how much a camera's shipped noise profile deviates from the real
mosaic-domain noise of raw files.

For each raw file, the true per-CFA-channel noise sigma is estimated from flat
16x16 blocks of the mosaic (MAD of a horizontal second difference, immune to
linear ramps; per intensity bin, a low percentile of block sigmas keeps only
the flat tail, so texture drops out). It is then compared to the profile
prediction sigma = sqrt(a*x + b) interpolated at the image ISO — exactly the
map Ansel's rawdenoiseai module feeds the network.

Validated on synthetic Poisson-Gaussian data: the per-bin median of block
sigmas tracks the true sigma within +/-3%; the default 10th percentile runs
~0.87x low, so reported ratios are a LOWER bound on the true deviation
(divide by 0.87 for a central estimate).

Context: the shipped profiles understate the physical mosaic-domain sigma by
~2.2-4x — an exact factor 2 from the lifting-Haar normalization in Ansel's
tools/noise/noiseprofile.c (std(HH) = sigma/2 for iid noise, squared into the
(a, b) fit uncorrected), times a channel-dependent 1.2-2x because profiles are
fitted on demosaiced pixels where interpolation has averaged away part of the
noise (most on the dense green lattice). This tool measured the calibration
now hard-coded in Ansel's rawdenoiseai module (x2 global, R 160% / G 200% /
B 140% per-channel defaults, pooled over 64 images, 3 cameras, ISO 64-12800).

Usage:
    python3 scripts/measure_profile_bias.py file.NEF [more.raw ...]
    python3 scripts/measure_profile_bias.py --csv < list_of_paths.txt

Requires exiftool, rawpy, numpy. Bayer CFA only (X-Trans subplane extraction
is not implemented).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import rawpy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ansel_denoise.levels import RawspeedLevels  # noqa: E402
from ansel_denoise.profiles import load_profiles  # noqa: E402

BLOCK = 16
PCTL = 10           # percentile of block sigmas per bin ("flat tail")
P10_BIAS = 0.87     # measured low bias of that percentile on synthetic flats
NBINS = 14


def exif(path: str) -> tuple[str, str, float | None]:
    out = subprocess.run(["exiftool", "-S", "-Make", "-Model", "-ISO", path],
                         capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        k, _, v = line.partition(": ")
        d[k] = v.strip()
    iso_str = "".join(ch for ch in d.get("ISO", "") if ch.isdigit())
    return d.get("Make", ""), d.get("Model", ""), float(iso_str) if iso_str else None


def find_profile(cameras, model: str):
    m = model.lower()
    best = None
    for cam in cameras:
        c = cam.model.lower()
        if c in m or m in c:
            if best is None or len(cam.model) > len(best.model):
                best = cam  # longest match wins: "Nikon D5300" over "Nikon D5"
    return best


def block_sigma(sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(block mean, robust block sigma) from a CFA subplane; both grids are
    built from the same cropped geometry so index i is the same block."""
    d = (sub[:, :-2] - 2.0 * sub[:, 1:-1] + sub[:, 2:]) / np.sqrt(6.0)
    m = sub[:, 1:-1]
    h = d.shape[0] - d.shape[0] % BLOCK
    w = d.shape[1] - d.shape[1] % BLOCK
    d, m = d[:h, :w], m[:h, :w]
    db = d.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).swapaxes(1, 2).reshape(-1, BLOCK * BLOCK)
    mb = m.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).swapaxes(1, 2).reshape(-1, BLOCK * BLOCK).mean(axis=1)
    med = np.median(db, axis=1, keepdims=True)
    sig = 1.4826 * np.median(np.abs(db - med), axis=1)
    return mb, sig


def measure(path: str, cameras, levels: RawspeedLevels, verbose: bool):
    """Return (model, iso, [ratio_R, ratio_G, ratio_B]) or None."""
    make, model, iso = exif(path)
    cam = find_profile(cameras, model)
    if cam is None or iso is None:
        if verbose:
            print(f"{path}: no noise profile for '{model}' or no ISO", file=sys.stderr)
        return None
    with rawpy.imread(path) as raw:
        if raw.raw_pattern.shape != (2, 2):
            if verbose:
                print(f"{path}: not a Bayer mosaic, skipped", file=sys.stderr)
            return None
        mosaic = raw.raw_image_visible.astype(np.float32)
        pattern = raw.raw_pattern.copy()
        libraw_white = float(raw.white_level)
    lv = levels.lookup(f"{make} {model}", iso=iso, libraw_white=libraw_white)
    if lv is None:
        if verbose:
            print(f"{path}: no rawspeed levels for '{make} {model}'", file=sys.stderr)
        return None
    black, white = lv
    x = (mosaic - black) / (white - black)
    prof = cam.interpolate(iso)

    chans: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    for dy in range(2):
        for dx in range(2):
            c = int(pattern[dy, dx])
            chans[{0: 0, 1: 1, 2: 2, 3: 1}[c]].append(x[dy::2, dx::2])

    if verbose:
        print(f"\n=== {Path(path).name} — {make} {model} ISO {iso:.0f} "
              f"[rawspeed black={black:.0f} white={white:.0f}] profile '{cam.name}'")
    names = "RGB"
    ratios = []
    for c in range(3):
        mb = np.concatenate([block_sigma(s)[0] for s in chans[c]])
        sg = np.concatenate([block_sigma(s)[1] for s in chans[c]])
        ok = (mb > 0.003) & (mb < 0.85) & (sg > 0)
        mb, sg = mb[ok], sg[ok]
        if len(mb) < 2000:
            ratios.append(float("nan"))
            continue
        edges = np.quantile(mb, np.linspace(0, 1, NBINS + 1))
        rs = []
        for i in range(NBINS):
            sel = (mb >= edges[i]) & (mb < edges[i + 1])
            if sel.sum() < 100:
                continue
            level = float(np.median(mb[sel]))
            if not 0.01 < level < 0.5:
                continue  # midtones only: away from clipping and the black floor
            pred = float(np.sqrt(max(prof.a[c] * level + prof.b[c], 1e-12)))
            if pred * (white - black) < 2.0:
                continue  # predicted sigma under ~2 DN: quantization floor
            meas = float(np.percentile(sg[sel], PCTL))
            rs.append(meas / pred)
            if verbose:
                print(f"  {names[c]} level {level:7.4f}  measured {meas:.3e}  "
                      f"profile {pred:.3e}  ratio {meas / pred:5.2f}")
        ratios.append(float(np.median(rs)) if rs else float("nan"))
    if verbose:
        r = ", ".join(f"{names[c]} {ratios[c]:.2f}" for c in range(3))
        print(f"  ==> sigma_measured / sigma_profile (lower bound): {r}")
    return model, iso, ratios


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("files", nargs="*", help="raw files; with --csv, read paths from stdin")
    ap.add_argument("--csv", action="store_true",
                    help="one CSV line per image on stdout: file,model,iso,ratio_R,ratio_G,ratio_B")
    ap.add_argument("--profiles", type=Path, default=REPO / "data" / "noiseprofiles.json",
                    help="path to Ansel's noiseprofiles.json")
    ap.add_argument("--levels", type=Path, default=REPO / "data" / "rawspeed_levels.json",
                    help="path to the rawspeed levels table")
    args = ap.parse_args()

    cameras = load_profiles(args.profiles)
    levels = RawspeedLevels(args.levels)
    paths = args.files if args.files else [line.strip() for line in sys.stdin if line.strip()]

    all_ratios = []
    for p in paths:
        try:
            r = measure(p, cameras, levels, verbose=not args.csv)
        except Exception as e:  # a broken raw must not kill the batch
            print(f"# {p}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if r is None:
            continue
        model, iso, ratios = r
        all_ratios.append(ratios)
        if args.csv:
            print(f"{Path(p).name},{model},{iso:.0f}," +
                  ",".join(f"{v:.3f}" for v in ratios), flush=True)

    a = np.array([r for r in all_ratios if not any(np.isnan(v) for v in r)])
    if len(a) > 1:
        med = np.median(a, axis=0)
        print(f"# pooled median over {len(a)} images: "
              f"R {med[0]:.2f} G {med[1]:.2f} B {med[2]:.2f}  "
              f"(central estimate: divide by {P10_BIAS})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
