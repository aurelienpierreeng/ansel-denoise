# ansel-denoise

Training pipeline for the neural raw denoiser of [Ansel](https://github.com/aurelienpierreeng/ansel)'s
`rawdenoise` module (CFA domain, before demosaicing).

## Principle

There is no paired noisy/clean photo dataset here — and none is needed. Sensor
noise is fully characterized by the Poisson-Gaussian model Ansel already ships
for hundreds of cameras ([`data/noiseprofiles.json`](data/noiseprofiles.json),
one `(a, b)` pair per channel per ISO, fitted so that `Var(x) = a·x + b` on
normalized raw values):

1. **Harvest** clean tiles from base-ISO raw files (CC0 archives, community
   contributions), stored as sensor ADUs with their normalization metadata.
2. **Synthesize** noise on those tiles at training time, drawing `(a, b)` from
   the whole profile database — Poisson shot noise, Gaussian read noise, ADU
   quantization, white clipping, signed shadow excursions.
3. **Train** a small CFA-agnostic U-Net conditioned on the per-pixel noise
   sigma map. One set of weights covers every profiled camera, Bayer and
   X-Trans alike; a newly profiled camera is supported with **no retraining**,
   because the sigma map is computed from its profile at runtime.
4. **Export** the weights to a flat `.anseldn` file loaded by Ansel's C/OpenCL
   inference in the `rawdenoise` IOP.

The network input is `[noisy mosaic, R/G/B one-hot CFA planes, sigma map]`
(5 planes, full resolution), output is the denoised 1-plane mosaic, predicted
as a residual. The user-facing "strength" control in Ansel scales the sigma
map — the training already covers mis-scaled sigmas via profile jitter.

## Requirements

- Python ≥ 3.10 with `numpy` (harvest additionally needs `rawpy` and the
  `exiftool` binary; training needs `torch`)
- `git-annex` for streaming the raw.pixls.us archive (not needed for plain
  directories of raw files)

```sh
python3.12 -m pip install --user -e .[harvest,train,dev]
```

## 1. Harvest clean tiles

Disk-bounded streaming from the [raw.pixls.us](https://raw.pixls.us) CC0
archive (metadata-only clone; each file is fetched, mined, dropped):

```sh
./scripts/harvest_rpu.sh shards/rpu          # clone + harvest, resumable
```

or from any local directory of raw files:

```sh
python3.12 -m ansel_denoise.harvest --source ~/photos --out shards/mine --max-iso 200
```

Each source file yields one compressed `.npz` shard (~1–4 MB): up to 16
CFA-aligned 256×256 uint16 tiles (clipped/flat crops rejected, most textured
kept) plus black/white levels, CFA pattern, WB, ISO, camera, source path and
git-annex key. `ledger.jsonl` records every decision, making runs resumable
and the dataset reproducible from the annex commit hash + this repo alone —
nothing needs redistribution.

## 2. Train

```sh
python3.12 -m ansel_denoise.train --shards shards/rpu --out runs/v1 \
    --steps 300000 --batch 32
```

Runs as-is on CPU (smoke test) or a single CUDA GPU (real training, 1–3 days
on one consumer GPU); see [docs/cloud.md](docs/cloud.md) for the remote
workflow. Validation PSNR is measured on **held-out cameras** (deterministic
hash split), so it reports cross-sensor generalization, not memorization.

## 3. Export for Ansel

```sh
python3.12 -m ansel_denoise.export runs/v1/ckpt-final.pt --onnx runs/v1/check.onnx
```

writes `ckpt-final.anseldn`: an 8-byte magic, a JSON header (model config +
tensor table), and raw float32 blobs — trivially parsed from C. The ONNX twin
is for numerical cross-checking of the C/OpenCL implementation.

## Layout

| path | role |
|---|---|
| `src/ansel_denoise/profiles.py` | noise profile DB loading, ISO interpolation, training-time sampling |
| `src/ansel_denoise/noise.py` | Poisson-Gaussian synthesis + sigma-map computation |
| `src/ansel_denoise/cfa.py` | CFA color maps, alignment, one-hot encoding |
| `src/ansel_denoise/harvest.py` | streaming tile harvester (git-annex or plain dir) |
| `src/ansel_denoise/dataset.py` | shards → noisy/clean training pairs (noise made on the fly) |
| `src/ansel_denoise/model.py` | CFA-agnostic U-Net (OpenCL-friendly ops only) |
| `src/ansel_denoise/train.py` | training loop, camera-holdout validation |
| `src/ansel_denoise/export.py` | `.anseldn` / ONNX export |
| `data/noiseprofiles.json` | profile DB from upstream darktable (superset of Ansel's); refresh with `scripts/update_noiseprofiles.sh`, source commit pinned in `data/noiseprofiles.upstream` |

## License

GPL-3.0-or-later, like Ansel. `data/noiseprofiles.json` comes from the
darktable project (GPL-3.0) and carries its profiling contributors' work.
