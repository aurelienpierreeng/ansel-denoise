"""Export a trained checkpoint to the flat weight format consumed by Ansel's
rawdenoise IOP (and optionally to ONNX for cross-checking).

.anselnn layout, all little-endian:
    8 bytes   magic "ANSELDN1"
    4 bytes   uint32 header length N
    N bytes   JSON header: {"cfg": {...model config...},
                            "tensors": [{"name", "shape", "offset", "size"}, ...]}
    payload   concatenated float32 tensor data, in header order

The C loader only needs the JSON header and one fread per tensor; the model
config tells it how to wire the (fixed) U-Net topology.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch

from .model import build_model


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    model = build_model(base=cfg["base"], depth=cfg["depth"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt.get("step")


def write_anselnn(model, cfg: dict, out: Path) -> int:
    tensors, blobs, offset = [], [], 0
    for name, t in model.state_dict().items():
        data = t.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
        tensors.append({"name": name, "shape": list(t.shape), "offset": offset, "size": len(data)})
        blobs.append(data)
        offset += len(data)
    header = json.dumps({"cfg": cfg, "tensors": tensors}).encode()
    with open(out, "wb") as f:
        f.write(b"ANSELDN1")
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        for blob in blobs:
            f.write(blob)
    return 12 + len(header) + offset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: checkpoint name + .anselnn")
    ap.add_argument("--onnx", type=Path, default=None, help="also export ONNX to this path")
    args = ap.parse_args(argv)

    model, cfg, step = load_model(args.checkpoint)
    out = args.out or args.checkpoint.with_suffix(".anselnn")
    size = write_anselnn(model, cfg, out)
    print(f"{out} ({size / 1e6:.1f} MB, step {step}, cfg {cfg})")

    if args.onnx:
        dummy = torch.zeros(1, cfg["in_channels"], 128, 128)
        torch.onnx.export(
            model, dummy, str(args.onnx), input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h", 3: "w"}},
        )
        print(f"{args.onnx} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
