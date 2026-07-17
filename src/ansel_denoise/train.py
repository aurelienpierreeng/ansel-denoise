"""Training loop. Single device, CPU or CUDA — the same entry point runs on a
laptop (smoke test) and on a rented GPU box (real training); see docs/cloud.md.

    python -m ansel_denoise.train --shards shards/ --out runs/v1 --steps 300000

Checkpoints are self-contained (model config + weights + optimizer + step) so
a run can resume across machines. Validation reports PSNR on held-out cameras.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import RawTileDataset
from .model import build_model, count_params


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    return 99.0 if mse <= 1e-12 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


@torch.no_grad()
def validate(model, loader, device, max_batches: int = 50) -> tuple[float, float]:
    """Returns (denoised PSNR, noisy-input PSNR); their gap is the actual gain."""
    model.eval()
    scores, baselines = [], []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        scores.append(psnr(model(x), y))
        baselines.append(psnr(x[:, :1], y))
    model.train()
    n = max(len(scores), 1)
    return sum(scores) / n, sum(baselines) / n


def save_checkpoint(path: Path, model, opt, step: int) -> None:
    torch.save(
        {"cfg": model.cfg, "model": model.state_dict(), "opt": opt.state_dict(), "step": step},
        path,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=Path, required=True, help="harvested shard directory")
    ap.add_argument("--out", type=Path, required=True, help="run directory (checkpoints, log)")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=32, help="U-Net base width")
    ap.add_argument("--depth", type=int, default=4, help="U-Net depth")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=10_000)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "train.log", "a", encoding="utf-8")

    def say(msg: str) -> None:
        print(msg)
        log.write(msg + "\n")
        log.flush()

    if args.patch % 2**args.depth:
        raise SystemExit(f"--patch must be a multiple of {2**args.depth} (depth {args.depth})")

    train_set = RawTileDataset(args.shards, "train", patch=args.patch)
    try:
        val_set = RawTileDataset(args.shards, "val", patch=args.patch)
    except ValueError:
        say("warning: no held-out-camera tiles; validating on training cameras")
        val_set = train_set
    say(f"tiles: {len(train_set)} train / {len(val_set)} val | device: {device}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=0)

    model = build_model(base=args.base, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        for _ in range(step):
            sched.step()
        say(f"resumed from {args.resume} at step {step}")

    say(f"model: {json.dumps(model.cfg)} ({count_params(model) / 1e6:.2f}M params)")

    model.train()
    t0, loss_acc, n_acc = time.time(), 0.0, 0
    while step < args.steps:
        for x, y in train_loader:
            if step >= args.steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                loss = torch.nn.functional.l1_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            loss_acc += loss.item()
            n_acc += 1

            if step % 100 == 0:
                rate = n_acc * args.batch / (time.time() - t0)
                say(f"step {step:7d}  loss {loss_acc / n_acc:.5f}  {rate:.1f} patches/s"
                    f"  lr {sched.get_last_lr()[0]:.2e}")
                t0, loss_acc, n_acc = time.time(), 0.0, 0
            if step % args.val_every == 0:
                score, base = validate(model, val_loader, device)
                say(f"step {step:7d}  val PSNR {score:.2f} dB (noisy input: {base:.2f} dB)")
            if step % args.ckpt_every == 0:
                save_checkpoint(args.out / f"ckpt-{step:08d}.pt", model, opt, step)

    save_checkpoint(args.out / "ckpt-final.pt", model, opt, step)
    score, base = validate(model, val_loader, device)
    say(f"final val PSNR {score:.2f} dB (noisy input: {base:.2f} dB)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
