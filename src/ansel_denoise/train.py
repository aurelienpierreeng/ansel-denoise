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
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .cfa import bin_mosaic_torch, bin_sigma_torch
from .dataset import RawTileDataset
from .metrics import lfce
from .model import MSUNet, build_model, count_params


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    return 99.0 if mse <= 1e-12 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def coarse_planes(noisy, onehot, sigma, bin_factor: int):
    """Shared binning glue: 6-plane coarse input [RGB, sigmaRGB] from the
    fine planes. One function so training, validation and the benchmark all
    exercise the exact convention the C side mirrors."""
    binned, counts = bin_mosaic_torch(noisy, onehot, bin_factor)
    coarse_sigma = bin_sigma_torch(sigma, onehot, bin_factor, counts)
    return torch.cat([binned, coarse_sigma], dim=1), binned


def upsample_guide(guide, bin_factor: int):
    return torch.nn.functional.interpolate(guide, scale_factor=bin_factor, mode="nearest")


def chroma_l1(pred_rgb, clean_rgb) -> torch.Tensor:
    """L1 on WB-free pseudo-chroma U = R - G, V = B - G."""
    du = (pred_rgb[:, 0] - pred_rgb[:, 1]) - (clean_rgb[:, 0] - clean_rgb[:, 1])
    dv = (pred_rgb[:, 2] - pred_rgb[:, 1]) - (clean_rgb[:, 2] - clean_rgb[:, 1])
    return du.abs().mean() + dv.abs().mean()


def coarse_loss(pred_rgb, clean_rgb) -> torch.Tensor:
    pool = torch.nn.functional.avg_pool2d
    return (torch.nn.functional.l1_loss(pred_rgb, clean_rgb)
            + 0.5 * chroma_l1(pred_rgb, clean_rgb)
            + 0.5 * torch.nn.functional.l1_loss(pool(pred_rgb, 4), pool(clean_rgb, 4)))


def fine_loss(pred, clean, onehot, bin_factor: int) -> torch.Tensor:
    """L1 plus superpixel-binned L1 terms: binning shrinks white residual by
    1/sqrt(n) but leaves correlated low-frequency error intact — the terms
    penalize exactly the chroma-blotch failure mode."""
    loss = torch.nn.functional.l1_loss(pred, clean)
    residual = pred - clean
    for factor, weight in ((bin_factor, 0.5), (4 * bin_factor, 0.25)):
        rgb, _ = bin_mosaic_torch(residual, onehot, factor)
        loss = loss + weight * rgb.abs().mean()
    return loss


def ms_forward(model, x, bins, guide_net=None):
    """Full multi-scale inference glue for a mixed-CFA batch: bin, run the
    coarse net (guide_net if given, else model.coarse), upsample the guide,
    run the fine net. Returns the denoised batch (same order)."""
    noisy, onehot, sigma = x[:, :1], x[:, 1:4], x[:, 4:5]
    coarse = guide_net if guide_net is not None else model.coarse
    pred = torch.empty_like(noisy)
    for bf in (4, 6):
        idx = (bins == bf).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        c_in, _ = coarse_planes(noisy[idx], onehot[idx], sigma[idx], bf)
        guide = upsample_guide(coarse(c_in), bf)
        pred[idx] = model.fine(torch.cat([x[idx], guide], dim=1))
    return pred


@torch.no_grad()
def validate(model, loader, device, max_batches: int = 50) -> tuple[float, float]:
    """Returns (denoised PSNR, noisy-input PSNR); their gap is the actual
    gain. For MSUNet the batch carries bin factors and the full multi-scale
    glue runs; LFCE_16 is accumulated on model.last_val_lfce16."""
    model.eval()
    scores, baselines, lfces = [], [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        if len(batch) == 3:
            x, y, bins = batch
            x, y = x.to(device), y.to(device)
            pred = ms_forward(model, x, bins.to(device))
            lfces.append(lfce(pred - y, x[:, 1:4], bins=(16,))[16])
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)
        scores.append(psnr(pred, y))
        baselines.append(psnr(x[:, :1], y))
    model.train()
    n = max(len(scores), 1)
    if lfces:
        model.last_val_lfce16 = sum(lfces) / len(lfces)
    return sum(scores) / n, sum(baselines) / n


def save_checkpoint(path: Path, model, opt, step: int, best_psnr: float = float("-inf"),
                    stale_vals: int = 0, ema: dict | None = None,
                    opt_coarse=None) -> None:
    torch.save(
        {"cfg": model.cfg, "model": model.state_dict(), "opt": opt.state_dict(), "step": step,
         "best_psnr": best_psnr, "stale_vals": stale_vals,
         **({"ema": ema} if ema is not None else {}),
         **({"opt_coarse": opt_coarse.state_dict()} if opt_coarse is not None else {})},
        path,
    )


def ms_step(model, guide_net, x, y, bins, opt_coarse, opt_fine,
            scaler_coarse, scaler_fine, device, guide_noise_p: float) -> float:
    """One interleaved multi-scale step on one (mixed-CFA) batch.

    The wavelet-like collaboration: the coarse net trains on the binned batch
    (its own loss, its own optimizer), then the fine net trains guided by the
    slowly-moving coarse EMA (guide_net, detached) — both nets co-adapt for
    the whole run without the fine net chasing a fast-moving guide. With
    probability guide_noise_p the guide is the binned NOISY input instead:
    the fine net stays fully functional without a coarse pass, which is
    exactly what the runtime fast mode ("chroma pass off") feeds it.
    """
    noisy, onehot, sigma = x[:, :1], x[:, 1:4], x[:, 4:5]
    groups = []
    for bf in (4, 6):
        idx = (bins == bf).nonzero(as_tuple=True)[0]
        if idx.numel():
            groups.append((bf, idx, idx.numel() / bins.numel()))

    # --- coarse update
    opt_coarse.zero_grad(set_to_none=True)
    loss_coarse = torch.zeros((), device=device)
    fine_inputs = []
    for bf, idx, weight in groups:
        c_in, b_noisy = coarse_planes(noisy[idx], onehot[idx], sigma[idx], bf)
        b_clean, _ = bin_mosaic_torch(y[idx], onehot[idx], bf)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            loss_coarse = loss_coarse + weight * coarse_loss(model.coarse(c_in), b_clean)
        with torch.no_grad():
            if torch.rand(()) < guide_noise_p:
                guide = b_noisy
            else:
                guide = guide_net(c_in)
        fine_inputs.append((idx, weight, torch.cat([x[idx], upsample_guide(guide, bf)], dim=1)))
    scaler_coarse.scale(loss_coarse).backward()
    scaler_coarse.step(opt_coarse)
    scaler_coarse.update()

    # --- fine update
    opt_fine.zero_grad(set_to_none=True)
    loss_fine = torch.zeros((), device=device)
    for (bf, idx, _), (_, weight, f_in) in zip(groups, fine_inputs):
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            loss_fine = loss_fine + weight * fine_loss(model.fine(f_in), y[idx],
                                                       onehot[idx], bf)
    scaler_fine.scale(loss_fine).backward()
    scaler_fine.step(opt_fine)
    scaler_fine.update()
    return float(loss_coarse.detach()) + float(loss_fine.detach())


def ema_init(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def ema_update(ema: dict, model, decay: float, step: int) -> None:
    # warmup ramp: early steps track the fast-moving weights closely, so a
    # short run (or the start of a long one) is not stuck near initialization
    d = min(decay, (1.0 + step) / (10.0 + step))
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            ema[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
        else:
            ema[k].copy_(v)


@torch.no_grad()
def validate_ema(model, ema: dict, loader, device, max_batches: int = 50) -> tuple[float, float]:
    """Validate the EMA weights by swapping them into the model (VRAM-cheap:
    no second model instance), restoring the live weights afterwards."""
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema)
    try:
        return validate(model, loader, device, max_batches)
    finally:
        model.load_state_dict(backup)


def rotate_checkpoints(out: Path, keep: int) -> None:
    """Delete the oldest numbered checkpoints beyond `keep`. A long run at a
    short --ckpt-every would otherwise fill the checkpoint volume (a free
    Google Drive dies after ~150 checkpoints of a 7.6M-param model).

    On a Google Drive mount (Colab) a plain unlink only moves the file to
    Drive's trash, which STILL counts against the quota — a long session then
    bloats the account with tens of GB of "deleted" checkpoints. Truncate to
    zero bytes first so the copy that lands in trash occupies no space; on a
    normal filesystem this is just a harmless empty write before the delete."""
    numbered = sorted(out.glob("ckpt-0*.pt"))
    for stale in numbered[:-keep] if keep > 0 else []:
        try:
            stale.write_bytes(b"")  # 0-byte trash on Drive; no-op cost locally
        except OSError:
            pass
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
    ap.add_argument("--arch", choices=["unet", "unet-ms"], default="unet",
                    help="unet: the single-scale mosaic net. unet-ms: multi-scale pair "
                         "(coarse net on superpixel-binned RGB guiding the fine mosaic net), "
                         "trained interleaved — one coarse step and one EMA-guided fine step "
                         "per batch. --base/--depth configure the FINE net; --patch defaults "
                         "to 192 (must be a multiple of 96 for both CFA families).")
    ap.add_argument("--coarse-base", type=int, default=32, help="coarse U-Net base width")
    ap.add_argument("--coarse-depth", type=int, default=3, help="coarse U-Net depth")
    ap.add_argument("--guide-noise-p", type=float, default=0.15,
                    help="probability of feeding the fine net the binned NOISY input as guide "
                         "instead of the coarse output — trains the runtime fast mode "
                         "(chroma pass off) into the fine net")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-shards", type=int, default=None,
                    help="cap training to a deterministic random subset of this many shards. "
                         "The tile cache is one RAM-resident memmap (~2.1 MB/shard, ~5 GB per "
                         "2500 shards); on a RAM-limited box (Colab T4 ≈ 12.7 GB) a bigger corpus "
                         "thrashes the page cache and crawls. Cap it to ~2000 to fit. Omit for "
                         "unlimited on a machine with enough RAM.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=10_000)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--keep-ckpts", type=int, default=3,
                    help="numbered checkpoints to keep, oldest deleted first (0 = keep all)")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after N validations without a +0.05 dB val-PSNR improvement, "
                         "counted across resumed sessions (0 = never stop early)")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="exponential moving average of the weights; the EMA weights are "
                         "what gets validated, kept as ckpt-best and exported (0 = disable)")
    ap.add_argument("--schedule", choices=["cosine", "constant"], default="cosine",
                    help="cosine: one-shot run annealing to 0 at --steps. constant: for "
                         "incremental sessions with a moving --steps target — a cosine pinned "
                         "to an ever-receding target keeps every later session in its dying "
                         "tail; anneal deliberately in a final cosine run instead")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "train.log", "a", encoding="utf-8")

    # Under a notebook/pipe (Colab, Kaggle) stdout is block-buffered, so log
    # lines vanish into the buffer and the run looks hung for many minutes.
    # Line-buffer stdout so every line appears immediately; say() flushes too.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    is_ms = args.arch == "unet-ms"
    if is_ms and args.patch == 128:
        args.patch = 192  # the unet default is not divisible by both bins
        say("unet-ms: --patch defaulted to 192")
    if args.patch % 2**args.depth:
        raise SystemExit(f"--patch must be a multiple of {2**args.depth} (depth {args.depth})")
    if is_ms and args.patch % 96:
        # patch % 16 (fine), % 4 and % 6 (CFA bins), and the binned planes
        # must divide by 2^coarse_depth for both families -> lcm is 96
        raise SystemExit("--patch must be a multiple of 96 for unet-ms")

    train_set = RawTileDataset(args.shards, "train", patch=args.patch, max_shards=args.max_shards,
                               with_bin=is_ms)
    try:
        val_set = RawTileDataset(args.shards, "val", patch=args.patch, max_shards=args.max_shards,
                                 with_bin=is_ms)
    except ValueError:
        say("warning: no held-out-camera tiles; validating on training cameras")
        val_set = RawTileDataset(args.shards, "train", patch=args.patch, deterministic=True,
                                 max_shards=args.max_shards, with_bin=is_ms)
    say(f"tiles: {len(train_set)} train / {len(val_set)} val | device: {device}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=0)

    if is_ms:
        model = MSUNet(coarse_base=args.coarse_base, coarse_depth=args.coarse_depth,
                       fine_base=args.base, fine_depth=args.depth).to(device)
        opt = torch.optim.AdamW(model.fine.parameters(), lr=args.lr, weight_decay=1e-8)
        opt_coarse = torch.optim.AdamW(model.coarse.parameters(), lr=args.lr, weight_decay=1e-8)
        scaler_coarse = torch.amp.GradScaler(enabled=device.type == "cuda")
        # the guide is produced by a frozen copy of the coarse net refreshed
        # from the EMA — a slowly-moving target the fine net can rely on
        import copy
        guide_net = copy.deepcopy(model.coarse).eval()
        for p in guide_net.parameters():
            p.requires_grad_(False)
    else:
        model = build_model(base=args.base, depth=args.depth).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8)
        opt_coarse = scaler_coarse = guide_net = None
    if args.schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        sched_coarse = (torch.optim.lr_scheduler.CosineAnnealingLR(opt_coarse, T_max=args.steps)
                        if opt_coarse else None)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        sched_coarse = (torch.optim.lr_scheduler.LambdaLR(opt_coarse, lambda _: 1.0)
                        if opt_coarse else None)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    step = 0
    best_psnr = float("-inf")
    stale_vals = 0
    ema = ema_init(model) if args.ema_decay > 0 else None

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        best_psnr = ckpt.get("best_psnr", float("-inf"))
        stale_vals = ckpt.get("stale_vals", 0)
        if ema is not None:
            # older checkpoints carry no EMA state: seed it from the resumed
            # weights, which the warmup ramp then tracks closely
            ema = ckpt.get("ema") or ema_init(model)
            ema = {k: v.to(device) for k, v in ema.items()}
        # The checkpointed optimizer carries the PREVIOUS schedule's last lr —
        # 0.0 if that run completed its cosine. CosineAnnealingLR.step() is a
        # recursive multiplicative formula on the current lr, so fast-forwarding
        # from 0 stays 0 forever; reset to the base lr first, then the
        # fast-forward reproduces the closed-form value for the new T_max.
        for group in opt.param_groups:
            group["lr"] = args.lr
        if opt_coarse is not None and "opt_coarse" in ckpt:
            opt_coarse.load_state_dict(ckpt["opt_coarse"])
            for group in opt_coarse.param_groups:
                group["lr"] = args.lr
        import warnings
        with warnings.catch_warnings():
            # fast-forwarding the scheduler necessarily steps it before any
            # optimizer.step() of this process; the pytorch warning is moot
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(step):
                sched.step()
                if sched_coarse is not None:
                    sched_coarse.step()
        if guide_net is not None and ema is not None:
            guide_net.load_state_dict({k[len("coarse."):]: v for k, v in ema.items()
                                       if k.startswith("coarse.")})
        say(f"resumed from {args.resume} at step {step}")
        if step >= args.steps:
            say(f"nothing to train: resumed step {step} >= --steps {args.steps} "
                f"(pass a higher --steps to continue this run)")

    say(f"model: {json.dumps(model.cfg)} ({count_params(model) / 1e6:.2f}M params)")

    model.train()
    t0, loss_acc, n_acc = time.time(), 0.0, 0
    # data-vs-compute split: t_data is time blocked waiting for the loader to
    # deliver a batch (high => GPU starved by the input pipeline, the classic
    # "low GPU util, low vRAM, crawling" signature), t_compute is forward+
    # backward+step. Reported alongside patches/s every 100 steps.
    t_data, t_compute = 0.0, 0.0
    t_prev = time.perf_counter()
    stalled = False
    while step < args.steps and not stalled:
        for batch in train_loader:
            t_data += time.perf_counter() - t_prev
            if step >= args.steps:
                break
            t_c = time.perf_counter()
            if is_ms:
                x, y, bins = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                li = ms_step(model, guide_net, x, y, bins.to(device), opt_coarse, opt,
                             scaler_coarse, scaler, device, args.guide_noise_p)
                sched_coarse.step()
            else:
                x, y = batch
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                    loss = torch.nn.functional.l1_loss(model(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                li = loss.item()
            sched.step()
            step += 1
            if ema is not None:
                ema_update(ema, model, args.ema_decay, step)
                if is_ms and step % 100 == 0:
                    # refresh the frozen guide from the coarse EMA: the fine
                    # net's target moves slowly and deliberately
                    guide_net.load_state_dict(
                        {k[len("coarse."):]: v for k, v in ema.items()
                         if k.startswith("coarse.")})
            if device.type == "cuda":
                torch.cuda.synchronize()  # so t_compute isn't understated by async kernels
            t_compute += time.perf_counter() - t_c
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
                elapsed = time.time() - t0
                rate = n_acc * args.batch / elapsed
                data_pct = 100.0 * t_data / max(elapsed, 1e-9)
                say(f"step {step:7d}  loss {loss_acc / n_acc:.5f}  {rate:.1f} patches/s"
                    f"  lr {lr_now:.2e}  [data {data_pct:3.0f}% "
                    f"{1000 * t_data / n_acc:.0f}ms/b | compute {1000 * t_compute / n_acc:.0f}ms/b]")
                t0, loss_acc, n_acc = time.time(), 0.0, 0
                t_data, t_compute = 0.0, 0.0
            if step % args.val_every == 0:
                # the EMA weights are the shipping artifact: they drive the
                # metric, the best-checkpoint gate and the early stop; the raw
                # weights are reported alongside for visibility
                if ema is not None:
                    score, base = validate_ema(model, ema, val_loader, device)
                    raw_score, _ = validate(model, val_loader, device)
                    detail = f"(noisy input: {base:.2f} dB, raw {raw_score:.2f}, "
                    if is_ms and hasattr(model, "last_val_lfce16"):
                        detail += f"LFCE16 {model.last_val_lfce16:.1f} dB, "
                else:
                    score, base = validate(model, val_loader, device)
                    detail = f"(noisy input: {base:.2f} dB, "
                if score > best_psnr + 0.05:
                    best_psnr, stale_vals = score, 0
                    save_checkpoint(args.out / "ckpt-best.pt", model, opt, step, best_psnr,
                                    stale_vals, ema, opt_coarse)
                else:
                    stale_vals += 1
                say(f"step {step:7d}  val PSNR {score:.2f} dB {detail}"
                    f"best {best_psnr:.2f}, stale {stale_vals})")
                t_prev = time.perf_counter()  # don't bill validation time to t_data
                if args.patience and stale_vals >= args.patience:
                    say(f"early stop at step {step}: {stale_vals} validations without improvement "
                        f"(best {best_psnr:.2f} dB, kept in ckpt-best.pt)")
                    stalled = True
                    break
            if step % args.ckpt_every == 0:
                save_checkpoint(args.out / f"ckpt-{step:08d}.pt", model, opt, step, best_psnr,
                                stale_vals, ema, opt_coarse)
                rotate_checkpoints(args.out, args.keep_ckpts)
                t_prev = time.perf_counter()  # don't bill checkpoint I/O to t_data
            t_prev = time.perf_counter()

    # numbered checkpoint is the resume anchor (strictly increasing names);
    # ckpt-final.pt is a stable alias for the export step
    save_checkpoint(args.out / f"ckpt-{step:08d}.pt", model, opt, step, best_psnr,
                    stale_vals, ema, opt_coarse)
    save_checkpoint(args.out / "ckpt-final.pt", model, opt, step, best_psnr,
                    stale_vals, ema, opt_coarse)
    rotate_checkpoints(args.out, args.keep_ckpts)
    if ema is not None:
        score, base = validate_ema(model, ema, val_loader, device)
        raw_score, _ = validate(model, val_loader, device)
        say(f"final val PSNR {score:.2f} dB (noisy input: {base:.2f} dB, raw {raw_score:.2f})")
    else:
        score, base = validate(model, val_loader, device)
        say(f"final val PSNR {score:.2f} dB (noisy input: {base:.2f} dB)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
