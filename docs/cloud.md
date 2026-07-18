# Remote / cloud training

Nothing in this pipeline is laptop-bound: the laptop only needs the code. Both
heavy stages (harvest bandwidth, training compute) run on a rented box with
the *same commands* as locally — there is no separate cloud path to maintain.

For **zero-budget training**, two notebook twins run the same fixed-schedule,
resume-until-done contract on free GPU tiers:
[`notebooks/colab_train.ipynb`](../notebooks/colab_train.ipynb) (Colab T4,
checkpoints on Google Drive) and
[`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb) (Kaggle
T4 ×2/P100, ~30 h GPU/week, checkpoints pushed to a private Kaggle Dataset
via API secrets, committed version output as fallback, wall-clock budget so
the save beats the session cap). The rest of this page is the rented-box
path.

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

# 2. get the data: fetch the published cache (minutes), or re-harvest (days)
./scripts/fetch_shards.sh shards/rpu         # from the shards-v1 GitHub release
# ./scripts/harvest_rpu.sh shards/rpu        # full harvest, if cache is stale

# 3. train
python3 -m ansel_denoise.train --shards shards/rpu --out runs/v1 \
    --steps 300000 --batch 32 --workers 8

# 4. bring home only what matters (a few MB)
python3 -m ansel_denoise.export runs/v1/ckpt-final.pt
scp box:ansel-denoise/runs/v1/ckpt-final.anselnn .
```

`--resume runs/v1/ckpt-XXXXXXXX.pt` continues an interrupted run, including on
a different machine — checkpoints are self-contained (config + weights +
optimizer + step). Shards are plain `.npz` files: the GitHub release cache
(`scripts/publish_shards.sh` / `fetch_shards.sh`) is the canonical way to move
them between boxes; `rsync`/object storage work too.

## Reproducibility contract

A released weight file must be reproducible from:

1. this repository at a tagged commit,
2. the raw.pixls.us annex commit hash + the harvest ledger (`ledger.jsonl`),
3. the `noiseprofiles.json` shipped in the same tag,
4. the training command line (recorded in `runs/*/train.log`).

That chain — not the dataset itself — is what gets published, which keeps the
weights auditable and GPL-clean without redistributing anyone's images.
