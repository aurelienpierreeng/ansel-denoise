# Remote / cloud training

Nothing in this pipeline is laptop-bound: the laptop only needs the code. Both
heavy stages (harvest bandwidth, training compute) run on a rented box with
the *same commands* as locally — there is no separate cloud path to maintain.

## Sizing

- **Harvest**: bandwidth-bound (tens of GB downloaded once, ~5–15 GB of shards
  kept). Any cheap VPS or the GPU box itself. Hours, mostly network.
- **Training**: one consumer/datacenter GPU (RTX 4090 / A100 class), batch 32,
  ~300k steps ≈ 1–3 days. Roughly $50–200 of rented GPU time per full run at
  2025 spot prices. CPU-only works for smoke tests, not for real runs.

## Workflow on a fresh GPU box

```sh
# 1. environment
sudo apt install -y git-annex exiftool           # or dnf
git clone https://github.com/aurelienpierreeng/ansel-denoise.git
cd ansel-denoise
python3 -m pip install -e .[harvest,train]       # torch wheel matches the box's CUDA

# 2. harvest (resumable; tmux/screen recommended)
./scripts/harvest_rpu.sh shards/rpu

# 3. train
python3 -m ansel_denoise.train --shards shards/rpu --out runs/v1 \
    --steps 300000 --batch 32 --workers 8

# 4. bring home only what matters (a few MB)
python3 -m ansel_denoise.export runs/v1/ckpt-final.pt
scp box:ansel-denoise/runs/v1/ckpt-final.anseldn .
```

`--resume runs/v1/ckpt-XXXXXXXX.pt` continues an interrupted run, including on
a different machine — checkpoints are self-contained (config + weights +
optimizer + step). Shards are plain `.npz` files: `rsync` them between boxes
to avoid re-harvesting, or keep them in object storage.

## Reproducibility contract

A released weight file must be reproducible from:

1. this repository at a tagged commit,
2. the raw.pixls.us annex commit hash + the harvest ledger (`ledger.jsonl`),
3. the `noiseprofiles.json` shipped in the same tag,
4. the training command line (recorded in `runs/*/train.log`).

That chain — not the dataset itself — is what gets published, which keeps the
weights auditable and GPL-clean without redistributing anyone's images.
