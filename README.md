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
4. **Export** the weights to a flat `.anselnn` file loaded by Ansel's C/OpenCL
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

or from the [PlayRaw category](https://discuss.pixls.us/c/playraw/30) of
discuss.pixls.us (~2000 topics of real photographs with declared CC licenses —
the content-diversity complement to raw.pixls.us's decoder-coverage archive):

```sh
python3.12 -m ansel_denoise.crawl_playraw --out shards/playraw
```

The crawler walks the category via Discourse's JSON API, verifies each
topic's declared license (all CC tags accepted by default — the category
rules require CC licensing to post; restrict with `--licenses cc0,by`),
applies the same ISO gate and tile pipeline, and records topic URL + author +
license in the ledger and shards for attribution.

Finally, your own Ansel library can contribute a **curated** set: select
images in the lighttable, note their IDs, and harvest them by ID — the ID
list is the curation, since tiles are viewable fragments of the photographs:

```sh
python3.12 -m ansel_denoise.harvest_library --ids 65345,65350-65360 --out shards/library
```

Paths and metadata resolve through `~/.config/ansel/library.db` (opened
read-only); the ISO gate uses the library's own EXIF data. By default the
output is publishable like any other source; pass `--private` to mark the
directory with a `.private` file instead, which `publish_shards.sh`
hard-refuses to upload — that mode is for personal training data that must
never reach the public release.

Each source file yields one compressed `.npz` shard (~1–4 MB): up to 16
CFA-aligned 256×256 uint16 tiles (clipped/flat crops rejected, most textured
kept) plus black/white levels, CFA pattern, WB, ISO, camera, source path and
git-annex key. `ledger.jsonl` records every decision, making runs resumable
and the dataset reproducible from the annex commit hash + this repo alone —
nothing needs redistribution.

Harvested shards are cached permanently as GitHub release assets so nobody
has to repeat the multi-day harvest (the git history itself stays lean):

```sh
./scripts/publish_shards.sh shards/rpu       # incremental, resumable, safe mid-harvest
./scripts/prune_shards.sh shards/rpu         # free local disk: delete published shards
./scripts/fetch_shards.sh shards/rpu         # restore / fast path on a fresh training box
```

`publish_shards.sh` packs only not-yet-published shards into ≤1.8 GB tarballs
(GitHub caps assets at 2 GiB) under the `shards-v1` release, and keeps a
`published.txt` index plus the latest `ledger.jsonl` alongside them. Re-run it
whenever the harvest has progressed.

`prune_shards.sh` is the local `git annex drop` equivalent for a disk-tight
machine: it deletes only shards whose names are on the release's
`published.txt` (updated strictly after each successful upload), never touches
`ledger.jsonl` (the harvester's resume state), and supports `DRY_RUN=1` to
preview. Chain them while harvesting:

```sh
./scripts/publish_shards.sh shards/rpu && ./scripts/prune_shards.sh shards/rpu
```

Frequent incremental publishes accumulate small tarballs on the release;
`./scripts/compact_shards.sh` occasionally merges them back into ~1.8 GB ones
(client-side — GitHub cannot merge assets server-side). It uploads the merged
tarball before deleting the originals, so an interruption can only leave
harmless duplicates, and `published.txt` tracks shard names rather than
tarball membership, so publish/prune/fetch are unaffected by the regrouping.

## 2. Train

```sh
python3.12 -m ansel_denoise.train --shards shards/rpu --out runs/v1 \
    --steps 300000 --batch 32
```

For a free test run, [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelienpierreeng/ansel-denoise/blob/master/notebooks/colab_train.ipynb)
trains on a free Colab T4 from the published shard cache, checkpoints to your
Google Drive, and auto-resumes after the session dies — re-run all cells.

Runs as-is on CPU (smoke test) or a single CUDA GPU (real training, 1–3 days
on one consumer GPU); see [docs/cloud.md](docs/cloud.md) for the remote
workflow. Validation PSNR is measured on **held-out cameras** (deterministic
hash split), so it reports cross-sensor generalization, not memorization.

## Model distribution (`models/`)

Trained models ship from this repo: Ansel's build fetches
[`models/manifest.json`](models/manifest.json) from the raw GitHub URL and
downloads each listed `.anselnn` verified by sha256 — so nightly builds bundle
the current models, and an override here reaches every fresh build (stale
local copies fail the hash check and re-download). Publish with:

```sh
python3.12 scripts/publish_model.py runs/v1/ckpt-final.anselnn --version v1 --variant full
```

**Versioning policy:** filenames (`rawdenoiseai-<version>-<variant>.anselnn`)
are permanent per (version, variant). During R&D a version may be overridden
(the manifest `revision` bumps; amend the model commit rather than stacking
30 MB blobs in history). **Once a version has shipped in a tagged stable
Ansel release it is frozen** — further training becomes a new version with a
new enum value in the Ansel module, so users' existing edits never change.

## 3. Export for Ansel

```sh
python3.12 -m ansel_denoise.export runs/v1/ckpt-final.pt --onnx runs/v1/check.onnx
```

writes `ckpt-final.anselnn`: an 8-byte magic, a JSON header (model config +
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
| `src/ansel_denoise/export.py` | `.anselnn` / ONNX export |
| `data/noiseprofiles.json` | profile DB from upstream darktable (superset of Ansel's); refresh with `scripts/update_noiseprofiles.sh`, source commit pinned in `data/noiseprofiles.upstream` |

## License

GPL-3.0-or-later, like Ansel. `data/noiseprofiles.json` comes from the
darktable project (GPL-3.0) and carries its profiling contributors' work.
