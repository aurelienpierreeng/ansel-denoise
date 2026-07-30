#!/usr/bin/env python3
"""Reproducible benchmark for the high-ISO chroma-speckle defect.

Synthesizes calibration-corrected noise (the TRUE mosaic-domain magnitudes,
not the understated profile face values) on a fixed, seeded set of inputs:

- 32 held-out clean tiles from the local shard corpus (relative instrument:
  the same tiles score every model candidate);
- 4 synthetic Bayer charts — flat gray 0.18, flat 0.02, luma gradient,
  chroma gradient. On a flat chart ANY chroma structure in the output is
  error by construction.

Conditions: 3 reference cameras (Nikon D850, Sony ILCE-1, Canon EOS 5D
Mark III — three makers, all profiled to ISO 102400) at ISO 3200 / 12800 /
51200, (a, b) scaled by DEFAULT_SIGMA_CALIBRATION^2, sigma conditioning from
the same corrected (a, b) — exactly the inference-side convention.

Reported per condition: PSNR and LFCE at bins 4/16/64 (see metrics.py), for
the model and for the noisy input (baseline). Lower LFCE = fewer blotches.

Usage:
    python3 scripts/speckle_bench.py --model models/rawdenoiseai-v1-full.anselnn \
        --out bench/v1-full-baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ansel_denoise import noise as noise_mod  # noqa: E402
from ansel_denoise.cfa import BAYER_RGGB, colors_map, one_hot  # noqa: E402
from ansel_denoise.export import load_model_from_anselnn  # noqa: E402
from ansel_denoise.metrics import LFCE_BINS, lfce, psnr  # noqa: E402
from ansel_denoise.profiles import DEFAULT_SIGMA_CALIBRATION, load_profiles  # noqa: E402
from ansel_denoise.cfa import bin_for_pattern  # noqa: E402
from ansel_denoise.train import ms_forward  # noqa: E402

SEED = 20260729
N_TILES = 32
TILE = 192  # lcm-safe for both CFA bins x coarse stride pyramid (multiple of 96)
CAMERAS = ["Nikon D850", "Sony ILCE-1", "Canon EOS 5D Mark III"]
ISOS = [3200.0, 12800.0, 51200.0]


def collect_tiles(shards_dir: Path, rng: np.random.Generator):
    """Deterministic selection of N_TILES clean tiles (normalized) + colors."""
    paths = sorted(shards_dir.rglob("*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {shards_dir}")
    picks = rng.choice(len(paths), size=min(N_TILES, len(paths)), replace=False)
    tiles = []
    for i in sorted(picks):
        with np.load(paths[i], allow_pickle=False) as z:
            t = int(rng.integers(z["tiles"].shape[0]))
            raw = z["tiles"][t][:TILE, :TILE].astype(np.float32)
            black = float(np.mean(z["black_per_channel"]))
            white = float(z["white"])
            oy, ox = (int(v) for v in z["offsets"][t])
            colors = colors_map(z["pattern"], TILE, TILE, oy, ox)
        clean = np.clip((raw - black) / max(white - black, 1.0), 0.0, 1.0)
        tiles.append((clean, colors))
    return tiles


def charts():
    """4 synthetic Bayer (RGGB) charts. Returns [(name, clean, colors)]."""
    colors = colors_map(BAYER_RGGB, TILE, TILE)
    x = np.linspace(0.0, 1.0, TILE, dtype=np.float32)[None, :].repeat(TILE, axis=0)
    out = []
    for name, rgb in [
        ("flat-gray-0.18", (0.18, 0.18, 0.18)),
        ("flat-dark-0.02", (0.02, 0.02, 0.02)),
        ("luma-gradient", None),
        ("chroma-gradient", None),
        ("tungsten-gradient", None),
    ]:
        if name == "luma-gradient":
            planes = [0.02 + 0.78 * x] * 3
        elif name == "chroma-gradient":
            planes = [0.1 + 0.5 * x, np.full_like(x, 0.3), 0.6 - 0.5 * x]
        elif name == "tungsten-gradient":
            # raw-domain tungsten night regime (the DSC01047.ARW field bug):
            # strong out-of-corpus chroma at low signal. A chroma-faithful
            # model tracks it; a corpus-chroma prior AMPLIFIES it (~2x seen
            # on the pre-WB-augmentation multi-scale models).
            g = 0.004 + 0.056 * x
            planes = [0.75 * g, g, 0.35 * g]
        else:
            planes = [np.full_like(x, v) for v in rgb]
        mosaic = np.zeros((TILE, TILE), dtype=np.float32)
        for c in range(3):
            mosaic[colors == c] = planes[c][colors == c]
        out.append((name, mosaic, colors))
    return out


def corrected_ab(cam, iso):
    prof = cam.interpolate(iso)
    cal2 = np.asarray(DEFAULT_SIGMA_CALIBRATION, dtype=np.float64) ** 2
    return prof.a * cal2, prof.b * cal2


def run_model(model, clean, colors, a, b, rng, device, bin_factor=4):
    noisy = noise_mod.synthesize(clean, colors, a, b, rng, black_frac=0.03)
    sigma = noise_mod.sigma_map(noisy, colors, a, b)
    oh = one_hot(colors)
    x = np.concatenate([noisy[None], oh, sigma[None]]).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x[None]).to(device)
        if getattr(model, "cfg", {}).get("arch") == "unet-ms":
            bins = torch.tensor([bin_factor], device=device)
            pred = ms_forward(model, xt, bins).clamp(0.0, 1.0)
        else:
            pred = model(xt).clamp(0.0, 1.0)
    ct = torch.from_numpy(clean[None, None]).to(device)
    nt = torch.from_numpy(noisy[None, None]).to(device)
    oht = torch.from_numpy(oh[None]).to(device)
    return {
        "psnr": psnr(pred, ct),
        "psnr_noisy": psnr(nt.clamp(0, 1), ct),
        "lfce": lfce(pred - ct, oht),
        "lfce_noisy": lfce(nt - ct, oht),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model", required=True, help=".anselnn file to benchmark")
    ap.add_argument("--shards", type=Path, default=REPO / "shards")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg = load_model_from_anselnn(args.model)
    model = model.to(args.device)
    cams = {c.name: c for c in load_profiles()}
    refs = [cams[n] for n in CAMERAS]

    rng = np.random.default_rng(SEED)
    tiles = collect_tiles(args.shards, rng)
    chart_set = charts()

    results = []
    for cam in refs:
        for iso in ISOS:
            a, b = corrected_ab(cam, iso)
            nrng = np.random.default_rng((SEED, int(iso), hash(cam.name) & 0xFFFF))
            agg = {"psnr": [], "psnr_noisy": [],
                   **{f"lfce_{n}": [] for n in LFCE_BINS},
                   **{f"lfce_noisy_{n}": [] for n in LFCE_BINS}}
            for clean, colors in tiles:
                try:  # 2x2-periodic (Bayer-class) tile -> bin 4, else X-Trans -> 6
                    tile_bin = bin_for_pattern(colors[:2, :2])
                except ValueError:
                    tile_bin = 6
                r = run_model(model, clean, colors, a, b, nrng, args.device, tile_bin)
                agg["psnr"].append(r["psnr"])
                agg["psnr_noisy"].append(r["psnr_noisy"])
                for n in LFCE_BINS:
                    agg[f"lfce_{n}"].append(r["lfce"][n])
                    agg[f"lfce_noisy_{n}"].append(r["lfce_noisy"][n])
            row = {"camera": cam.name, "iso": iso, "set": "tiles",
                   **{k: round(float(np.mean(v)), 2) for k, v in agg.items()}}
            results.append(row)
            for name, clean, colors in chart_set:
                r = run_model(model, clean, colors, a, b, nrng, args.device)
                results.append({"camera": cam.name, "iso": iso, "set": name,
                                "psnr": round(r["psnr"], 2),
                                "psnr_noisy": round(r["psnr_noisy"], 2),
                                **{f"lfce_{n}": round(r["lfce"][n], 2) for n in LFCE_BINS},
                                **{f"lfce_noisy_{n}": round(r["lfce_noisy"][n], 2)
                                   for n in LFCE_BINS}})
            print(f"{cam.name} ISO {iso:.0f}: tiles psnr {row['psnr']} "
                  f"lfce16 {row['lfce_16']} (noisy {row['lfce_noisy_16']})", flush=True)

    payload = {"model": str(args.model), "cfg": cfg, "seed": SEED,
               "sigma_calibration": list(DEFAULT_SIGMA_CALIBRATION),
               "results": results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
