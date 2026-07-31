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


def load_model(ckpt_path: Path, raw_weights: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = dict(ckpt["cfg"])
    # provenance signature: the training-code git hash stamped in the
    # checkpoint travels into the artifact cfg (the C loader ignores it)
    if ckpt.get("code_revision"):
        cfg["code_revision"] = ckpt["code_revision"]
    if cfg.get("arch") == "unet-ms":
        from .model import MSUNet

        model = MSUNet(coarse_base=cfg["coarse"]["base"], coarse_depth=cfg["coarse"]["depth"],
                       fine_base=cfg["fine"]["base"], fine_depth=cfg["fine"]["depth"])
        # the sigma convention the run synthesized with ships INSIDE the
        # artifact: the C side reads it so model and conditioning can never
        # drift apart (channel_sigma_scale = TOTAL per-channel multiplier)
        from .profiles import DEFAULT_SIGMA_CALIBRATION

        cfg.setdefault("anchor", 32)
        cfg.setdefault("sigma_calibration",
                       {"channel_sigma_scale": list(DEFAULT_SIGMA_CALIBRATION)})
    else:
        model = build_model(base=cfg["base"], depth=cfg["depth"])
    # the EMA weights are the shipping artifact when the run maintained them
    weights = ckpt["model"] if raw_weights or "ema" not in ckpt else ckpt["ema"]
    model.load_state_dict(weights)
    model.eval()
    return model, cfg, ckpt.get("step"), ("raw" if raw_weights or "ema" not in ckpt else "ema")


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
    ap.add_argument("--raw-weights", action="store_true",
                    help="export the live training weights instead of the EMA average")
    args = ap.parse_args(argv)

    model, cfg, step, which = load_model(args.checkpoint, raw_weights=args.raw_weights)
    out = args.out or args.checkpoint.with_suffix(".anselnn")
    size = write_anselnn(model, cfg, out)
    print(f"{out} ({size / 1e6:.1f} MB, step {step}, {which} weights, cfg {cfg})")

    if args.onnx:
        if cfg.get("arch") == "unet-ms":
            print("ONNX export is single-net only; skipping for unet-ms")
            return 0
        dummy = torch.zeros(1, cfg["in_channels"], 128, 128)
        torch.onnx.export(
            model, dummy, str(args.onnx), input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h", 3: "w"}},
        )
        print(f"{args.onnx} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def read_anselnn(path):
    """Parse an .anselnn file back into (cfg, {name: np.ndarray}).

    Inverse of write_anselnn — used by benchmarks and round-trip tests so a
    shipped artifact can be evaluated in torch exactly as the C executor
    sees it.
    """
    import numpy as np

    data = Path(path).read_bytes()
    if data[:8] != b"ANSELDN1":
        raise ValueError(f"{path}: not an .anselnn file")
    hlen = struct.unpack("<I", data[8:12])[0]
    header = json.loads(data[12:12 + hlen].decode())
    payload = data[12 + hlen:]
    tensors = {}
    for t in header["tensors"]:
        arr = np.frombuffer(payload, dtype=np.float32,
                            count=t["size"] // 4, offset=t["offset"])
        tensors[t["name"]] = arr.copy().reshape(t["shape"])
    return header["cfg"], tensors


def load_model_from_anselnn(path):
    """Instantiate the torch model matching an .anselnn file and load its
    weights bit-exactly ('unet' and 'unet-ms')."""
    from .model import MSUNet, UNet

    cfg, tensors = read_anselnn(path)
    arch = cfg.get("arch")
    if arch == "unet":
        model = UNet(base=cfg["base"], depth=cfg["depth"],
                     in_channels=cfg.get("in_channels", 5),
                     out_channels=cfg.get("out_channels", 1))
    elif arch == "unet-ms":
        model = MSUNet(coarse_base=cfg["coarse"]["base"],
                       coarse_depth=cfg["coarse"]["depth"],
                       fine_base=cfg["fine"]["base"],
                       fine_depth=cfg["fine"]["depth"])
    else:
        raise ValueError(f"unsupported arch {arch!r}")
    state = {k: torch.from_numpy(v) for k, v in tensors.items()}
    model.load_state_dict(state)
    model.eval()
    return model, cfg
