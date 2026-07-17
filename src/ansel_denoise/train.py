"""Training loop. Single device, CPU or CUDA — the same entry point runs on a
laptop (smoke test) and on a rented GPU box (real training); see docs/cloud.md.

    python -m ansel_denoise.train --shards shards/ --out runs/v1 --steps 300000

Checkpoints are self-contained (model config + weights + optimizer + step) so
a run can resume across machines. Validation reports PSNR on held-out cameras.
"""

from __future__ import annotations

import argparse
import json
import math
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


def save_checkpoint(path: Path, model, opt, step: int, best_psnr: float = float("-inf"),
                    stale_vals: int = 0) -> None:
    torch.save(
        {"cfg": model.cfg, "model": model.state_dict(), "opt": opt.state_dict(), "step": step,
         "best_psnr": best_psnr, "stale_vals": stale_vals},
        path,
    )


def rotate_checkpoints(out: Path, keep: int) -> None:
    """Delete the oldest numbered checkpoints beyond `keep`. A long run at a
    short --ckpt-every would otherwise fill the checkpoint volume (a free
    Google Drive dies after ~150 checkpoints of a 7.6M-param model)."""
    numbered = sorted(out.glob("ckpt-0*.pt"))
    for stale in numbered[:-keep] if keep > 0 else []:
        stale.unlink()


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
    ap.add_argument("--keep-ckpts", type=int, default=3,
                    help="numbered checkpoints to keep, oldest deleted first (0 = keep all)")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after N validations without a +0.05 dB val-PSNR improvement, "
                         "counted across resumed sessions (0 = never stop early)")
    ap.add_argument("--schedule", choices=["cosine", "constant"], default="cosine",
                    help="cosine: one-shot run annealing to 0 at --steps. constant: for "
                         "incremental sessions with a moving --steps target — a cosine pinned "
                         "to an ever-receding target keeps every later session in its dying "
                         "tail; anneal deliberately in a final cosine run instead")
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
        val_set = RawTileDataset(args.shards, "train", patch=args.patch, deterministic=True)
    say(f"tiles: {len(train_set)} train / {len(val_set)} val | device: {device}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=0)

    model = build_model(base=args.base, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8)
    if args.schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    step = 0
    best_psnr = float("-inf")
    stale_vals = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        best_psnr = ckpt.get("best_psnr", float("-inf"))
        stale_vals = ckpt.get("stale_vals", 0)
        # The checkpointed optimizer carries the PREVIOUS schedule's last lr —
        # 0.0 if that run completed its cosine. CosineAnnealingLR.step() is a
        # recursive multiplicative formula on the current lr, so fast-forwarding
        # from 0 stays 0 forever; reset to the base lr first, then the
        # fast-forward reproduces the closed-form value for the new T_max.
        for group in opt.param_groups:
            group["lr"] = args.lr
        import warnings
        with warnings.catch_warnings():
            # fast-forwarding the scheduler necessarily steps it before any
            # optimizer.step() of this process; the pytorch warning is moot
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(step):
                sched.step()
        say(f"resumed from {args.resume} at step {step}")
        if step >= args.steps:
            say(f"nothing to train: resumed step {step} >= --steps {args.steps} "
                f"(pass a higher --steps to continue this run)")

    say(f"model: {json.dumps(model.cfg)} ({count_params(model) / 1e6:.2f}M params)")

    model.train()
    t0, loss_acc, n_acc = time.time(), 0.0, 0
    stalled = False
    while step < args.steps and not stalled:
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
            li = loss.item()
            loss_acc += li
            n_acc += 1

            # broken-run invariants: abort loudly, do NOT checkpoint over good
            # state. Train-loss "stagnation" is deliberately not gated — a flat
            # L1 plateau is what healthy denoiser training looks like; only the
            # deterministic val metric below can diagnose real stagnation.
            if not math.isfinite(li):
                say(f"ABORT at step {step}: non-finite loss {li} — weights are corrupted, "
                    f"resume from the last checkpoint with a lower --lr")
                log.close()
                return 1
            lr_now = sched.get_last_lr()[0]
            if lr_now == 0.0 and step < args.steps:
                say(f"ABORT at step {step}: lr is 0 with {args.steps - step} steps remaining — "
                    f"schedule/resume bug, this run would burn compute without learning")
                log.close()
                return 1

            if step % 100 == 0:
                rate = n_acc * args.batch / (time.time() - t0)
                say(f"step {step:7d}  loss {loss_acc / n_acc:.5f}  {rate:.1f} patches/s"
                    f"  lr {lr_now:.2e}")
                t0, loss_acc, n_acc = time.time(), 0.0, 0
            if step % args.val_every == 0:
                score, base = validate(model, val_loader, device)
                if score > best_psnr + 0.05:
                    best_psnr, stale_vals = score, 0
                    save_checkpoint(args.out / "ckpt-best.pt", model, opt, step, best_psnr, stale_vals)
                else:
                    stale_vals += 1
                say(f"step {step:7d}  val PSNR {score:.2f} dB (noisy input: {base:.2f} dB, "
                    f"best {best_psnr:.2f}, stale {stale_vals})")
                if args.patience and stale_vals >= args.patience:
                    say(f"early stop at step {step}: {stale_vals} validations without improvement "
                        f"(best {best_psnr:.2f} dB, kept in ckpt-best.pt)")
                    stalled = True
                    break
            if step % args.ckpt_every == 0:
                save_checkpoint(args.out / f"ckpt-{step:08d}.pt", model, opt, step, best_psnr, stale_vals)
                rotate_checkpoints(args.out, args.keep_ckpts)

    # numbered checkpoint is the resume anchor (strictly increasing names);
    # ckpt-final.pt is a stable alias for the export step
    save_checkpoint(args.out / f"ckpt-{step:08d}.pt", model, opt, step, best_psnr, stale_vals)
    save_checkpoint(args.out / "ckpt-final.pt", model, opt, step, best_psnr, stale_vals)
    rotate_checkpoints(args.out, args.keep_ckpts)
    score, base = validate(model, val_loader, device)
    say(f"final val PSNR {score:.2f} dB (noisy input: {base:.2f} dB)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
