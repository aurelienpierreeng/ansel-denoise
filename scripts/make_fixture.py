#!/usr/bin/env python3.12
"""Generate the golden parity fixture for Ansel's C inference of an .anselnn model.

For arch "unet" (single-scale), writes next to --out:
    fixture-input.f32     5 x N x N float32 LE planes [mosaic, R, G, B, sigma]
    fixture-expected.f32  1 x N x N float32 LE, the torch model's output
    fixture-meta.json     shapes, model cfg, checksums, tolerances

For arch "unet-ms" (multi-scale pair), additionally:
    fixture-coarse-input.f32     6 x N/bin x N/bin  [R, G, B, sigmaRGB]
    fixture-coarse-expected.f32  3 x N/bin x N/bin  torch coarse output
    fixture-input.f32            8 x N x N  (5 base planes + torch guide)
    fixture-expected.f32         1 x N x N  fine output on that input
so the C selftest can gate each stage separately (binning contract, coarse
parity, fine parity) and then compose them end-to-end.

The input is fully deterministic and synthetic (seeded structured mosaic +
noise, RGGB one-hot, sigma from a real-profile-magnitude variance line), so
the fixture exercises every network path without shipping image data. The C
selftest must reproduce the expected outputs within the stated tolerances.

Usage: python3.12 scripts/make_fixture.py model.anselnn --out fixtures/
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ansel_denoise.cfa import (BAYER_RGGB, bin_mosaic_torch, bin_sigma_torch,  # noqa: E402
                               colors_map, one_hot)
from ansel_denoise.export import load_model_from_anselnn  # noqa: E402
from ansel_denoise.noise import sigma_map, synthesize  # noqa: E402


def base_planes(n: int):
    """Deterministic 5-plane input at n x n (RGGB): texture + edge + noise."""
    rng = np.random.default_rng(0xF17)
    yy, xx = np.mgrid[0:n, 0:n]
    clean = (0.25 + 0.2 * np.sin(xx / 7.0) * np.cos(yy / 11.0)
             + 0.1 * (xx > n // 2)).astype(np.float32)  # texture + an edge
    colors = colors_map(BAYER_RGGB, n, n)
    a = np.array([1.2e-4, 0.9e-4, 1.5e-4])  # high-ISO-magnitude variance line
    b = np.array([2.0e-6, 1.5e-6, 2.5e-6])
    noisy = synthesize(clean, colors, a, b, rng)
    sigma = sigma_map(noisy, colors, a, b)
    return np.concatenate([noisy[None], one_hot(colors), sigma[None]]).astype(np.float32)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model, cfg = load_model_from_anselnn(args.model)
    is_ms = cfg["arch"] == "unet-ms"
    n = 192 if is_ms else 96  # both multiples of 2**depth and the CFA period

    x5 = base_planes(n)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "n": n, "cfg": cfg,
        "model_sha256": sha(args.model.read_bytes()),
        "tolerance_abs": 2e-4,
    }

    if not is_ms:
        with torch.no_grad():
            y = model(torch.from_numpy(x5)[None])[0].numpy().astype(np.float32)
        (args.out / "fixture-input.f32").write_bytes(x5.tobytes())
        (args.out / "fixture-expected.f32").write_bytes(y.tobytes())
        meta.update({
            "input_sha256": sha(x5.tobytes()),
            "expected_sha256": sha(y.tobytes()),
            "input_planes": ["mosaic", "onehot_R", "onehot_G", "onehot_B", "sigma"],
        })
    else:
        bin_factor = cfg["bin"]["bayer"]
        xt = torch.from_numpy(x5)[None]
        noisy, onehot, sigma = xt[:, :1], xt[:, 1:4], xt[:, 4:5]
        binned, counts = bin_mosaic_torch(noisy, onehot, bin_factor)
        coarse_sigma = bin_sigma_torch(sigma, onehot, bin_factor, counts)
        c_in = torch.cat([binned, coarse_sigma], dim=1)
        with torch.no_grad():
            c_out = model.coarse(c_in)
            guide = torch.nn.functional.interpolate(c_out, scale_factor=bin_factor,
                                                    mode="nearest")
            f_in = torch.cat([xt, guide], dim=1)
            y = model.fine(f_in)[0].numpy().astype(np.float32)
        ci = c_in[0].numpy().astype(np.float32)
        co = c_out[0].numpy().astype(np.float32)
        fi = f_in[0].numpy().astype(np.float32)
        (args.out / "fixture-base-planes.f32").write_bytes(x5.tobytes())
        (args.out / "fixture-coarse-input.f32").write_bytes(ci.tobytes())
        (args.out / "fixture-coarse-expected.f32").write_bytes(co.tobytes())
        (args.out / "fixture-input.f32").write_bytes(fi.tobytes())
        (args.out / "fixture-expected.f32").write_bytes(y.tobytes())
        meta.update({
            "bin": bin_factor,
            "base_planes_sha256": sha(x5.tobytes()),
            "coarse_input_sha256": sha(ci.tobytes()),
            "coarse_expected_sha256": sha(co.tobytes()),
            "input_sha256": sha(fi.tobytes()),
            "expected_sha256": sha(y.tobytes()),
            "tolerance_bin_abs": 1e-6,       # pure arithmetic, no convs
            "tolerance_end_to_end_abs": 5e-4,  # allows coarse-error propagation
            "input_planes": ["mosaic", "onehot_R", "onehot_G", "onehot_B", "sigma",
                             "guide_R", "guide_G", "guide_B"],
            "coarse_planes": ["R", "G", "B", "sigma_R", "sigma_G", "sigma_B"],
        })

    (args.out / "fixture-meta.json").write_text(json.dumps(meta, indent=1))
    print(f"fixture in {args.out}: n={n} arch={cfg['arch']}, "
          f"expected range [{y.min():.4f}, {y.max():.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
