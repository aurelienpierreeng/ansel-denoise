#!/usr/bin/env python3.12
"""Generate the golden parity fixture for Ansel's C inference of an .anselnn model.

Writes, next to --out:
    fixture-input.f32     5 x N x N float32 LE planes [mosaic, R, G, B, sigma]
    fixture-expected.f32  1 x N x N float32 LE, the torch model's output
    fixture-meta.json     shapes, model cfg, checksums, tolerances

The input is fully deterministic and synthetic (seeded structured mosaic +
noise, RGGB one-hot, sigma from a real-profile-magnitude variance line), so
the fixture exercises every network path without shipping image data. The C
selftest must reproduce fixture-expected.f32 within the stated tolerance.

Usage: python3.12 scripts/make_fixture.py model.anselnn --out fixtures/
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ansel_denoise.cfa import BAYER_RGGB, colors_map, one_hot  # noqa: E402
from ansel_denoise.model import build_model  # noqa: E402
from ansel_denoise.noise import sigma_map, synthesize  # noqa: E402

N = 96  # multiple of 2**depth and of the CFA period


def load_anselnn(path: Path):
    raw = path.read_bytes()
    assert raw[:8] == b"ANSELDN1", "bad magic"
    (hlen,) = struct.unpack("<I", raw[8:12])
    hdr = json.loads(raw[12 : 12 + hlen])
    payload = raw[12 + hlen :]
    model = build_model(base=hdr["cfg"]["base"], depth=hdr["cfg"]["depth"])
    model.load_state_dict({
        t["name"]: torch.from_numpy(
            np.frombuffer(payload[t["offset"] : t["offset"] + t["size"]], dtype="<f4")
            .copy().reshape(t["shape"]))
        for t in hdr["tensors"]
    })
    model.eval()
    return model, hdr["cfg"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model, cfg = load_anselnn(args.model)
    rng = np.random.default_rng(0xF17)

    yy, xx = np.mgrid[0:N, 0:N]
    clean = (0.25 + 0.2 * np.sin(xx / 7.0) * np.cos(yy / 11.0)
             + 0.1 * (xx > N // 2)).astype(np.float32)  # texture + an edge
    colors = colors_map(BAYER_RGGB, N, N)
    a = np.array([1.2e-4, 0.9e-4, 1.5e-4])  # high-ISO-magnitude variance line
    b = np.array([2.0e-6, 1.5e-6, 2.5e-6])
    noisy = synthesize(clean, colors, a, b, rng)
    sigma = sigma_map(noisy, colors, a, b)
    x = np.concatenate([noisy[None], one_hot(colors), sigma[None]]).astype(np.float32)

    with torch.no_grad():
        y = model(torch.from_numpy(x)[None])[0].numpy().astype(np.float32)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fixture-input.f32").write_bytes(x.tobytes())
    (args.out / "fixture-expected.f32").write_bytes(y.tobytes())
    meta = {
        "n": N, "cfg": cfg,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "expected_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "tolerance_abs": 2e-4,
        "input_planes": ["mosaic", "onehot_R", "onehot_G", "onehot_B", "sigma"],
    }
    (args.out / "fixture-meta.json").write_text(json.dumps(meta, indent=1))
    print(f"fixture in {args.out}: input {x.shape}, expected {y.shape}, "
          f"expected range [{y.min():.4f}, {y.max():.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
