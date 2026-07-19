# ansel-denoise

Training pipeline for the neural raw denoiser of [Ansel](https://github.com/aurelienpierreeng/ansel)'s
`rawdenoiseai` module (CFA domain, before demosaicing).

## Why this is not just another AI denoiser

First, why a neural network at all — because this is not AI-FOMO, it is the
end of a road the field has walked for twenty years. Fine detail and noise
live in the **same high frequencies**: any filter that separates them by
frequency (gaussian, median, wavelet thresholding) must destroy one to
remove the other, by construction. The only way out is to be
**content- and context-aware** — decide from the surrounding image what is
signal and what is chance. That insight produced non-local means (denoise a
patch using every similar patch in the image), then BM3D (collaborative
filtering of matched 3D blocks), which has been exploited to the maximum of
its abilities and has plateaued for over a decade. A convolutional network
is the same idea carried further — context priors *learned* from data
instead of hand-coded self-similarity — and it is today the only known way
to push past that plateau. We use one because the classical context-aware
line is exhausted, not because AI is fashionable.

Second, why ours is different. Every general-purpose AI denoiser learns from developed images — demosaiced,
white-balanced, tone-mapped pixels, several destructive approximations away
from the measurement. This network trains on **non-demosaiced sensor data,
as close to the actual sensor reading as a raw file allows**, and runs at
the exact same point of Ansel's pipeline: before white balance, before
chromatic-aberration correction, before demosaicing. Training domain and
inference domain are the same domain.

That placement is where the magic compounds: noise is removed while it is
still the well-characterized Poisson-Gaussian process sensor physics
dictates — so it can be synthesized exactly from measured camera profiles —
and every downstream stage inherits clean data. Demosaicing interpolates
real detail instead of weaving noise into maze and zipper artifacts,
CA correction sees clean edges, and nothing later has to fight amplified,
correlated noise.

And because the same project controls the pipeline, the training and the
deployment point, there is no domain mismatch, no guessing about upstream
processing, no compromise or intermediate approximation: a purpose-built
network, excellent at exactly one thing, under conditions we define.

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

**[docs/architecture.md](docs/architecture.md)** is the deep dive: the exact
network topology, why every design choice was made (residual output, no
normalization, operator vocabulary matched to the C/OpenCL executor), how
the encoding lets the network exploit inter-channel correlations despite
their spatial offsets in the mosaic, and the full list of assumptions the
model makes (and what it cannot do, by design).

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
images in the lighttable and export the selection through
`File > Export image list...` — the list is the curation, since tiles are
viewable fragments of the photographs. All the dialog's outputs are accepted:
IDs or shell-quoted paths pasted from the clipboard as arguments, or the
saved one-item-per-line `.txt` files:

```sh
python3.12 -m ansel_denoise.harvest_library --ids 65345,65350-65360 --out shards/library
python3.12 -m ansel_denoise.harvest_library --out shards/library '/photos/IMG 1.NEF' '/photos/IMG 2.NEF'
python3.12 -m ansel_denoise.harvest_library --paths-file ansel-image-files.txt --out shards/library
python3.12 -m ansel_denoise.harvest_library --ids-file ansel-image-ids.txt --out shards/library
```

Paths work **with or without** `library.db`: when the library knows the
file, its metadata is used; otherwise camera and ISO are read from the file
itself (TIFF tags + libraw, exiftool filling gaps when installed) — so a
plain list of raw paths harvests on a machine that never ran Ansel.

**Community contributions:** anyone can feed the corpus from their own Ansel
library — see [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow
(one-script setup, curation = lighttable selection, `pack_contribution.py`
bundle, maintainer ingest via `collect_contribution.sh`, public bookkeeping
in [`contrib/registry.jsonl`](contrib/registry.jsonl)).

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

**Licensing of the shard release:** the corpus is an aggregate of three
provenance classes, each identifiable per shard — raw.pixls.us harvest
(**CC0-1.0**), PlayRaw photographs (**per-shard declared CC license**, with
author/topic attribution embedded in shard and ledger), and community
contributions (`<handle>_` prefix, **[ATDL-1.1](LICENSE-DATA.md)**: usable
by anyone with the ansel-denoise training stack — to audit, reproduce and
benchmark the training, or to train custom **denoising** models whose
weights are unrestricted, commercial use included; feeding the tiles to any
stack able to learn anything else — style, generative AI — is forbidden).
The reference use, training denoising networks with this stack, satisfies
all classes at once: attribution travels in the metadata and denoising is
the only ATDL-permitted learning task. Any other reuse requires per-shard
license filtering and excludes the contributed shards. The full terms live on the
[release page](https://github.com/aurelienpierreeng/ansel-denoise/releases/tag/shards-v1)
and in [LICENSE-DATA.md](LICENSE-DATA.md), which is attached to the release
alongside the data.

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

One notebook, [notebooks/train.ipynb](notebooks/train.ipynb), covers
**local, Colab and Kaggle** with automatic environment detection —
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelienpierreeng/ansel-denoise/blob/master/notebooks/train.ipynb)
or [import it on Kaggle](https://www.kaggle.com/kernels/welcome?src=https://github.com/aurelienpierreeng/ansel-denoise/blob/master/notebooks/train.ipynb).
Same contract everywhere: one fixed cosine schedule, run all cells, and
after any disconnect run them again — training resumes from the newest
checkpoint until done. Per-environment branching handles the rest:
checkpoints persist to `runs/` locally, to Google Drive on Colab, and on
Kaggle (~30 GPU-h/week, T4 ×2/P100) to a private Kaggle Dataset via API
secrets with the committed version output as automatic fallback, plus a
wall-clock budget so the save always beats the session kill. Between the
two free tiers, a full 100k-step training fits in one week of quota.

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
are permanent per (version, variant). During R&D a version may be overridden:
publish again and commit normally (the manifest `revision` bumps; history
simply grows, no rewriting). **Once a version has shipped in a tagged stable
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
| `data/rawspeed_levels.json` | per-camera sensor black/white levels from Rawspeed's `cameras.xml` — training normalizes in Ansel's exact runtime domain (the profiles' domain); refresh with `scripts/update_rawspeed_levels.py` |

## License

The **code** is GPL-3.0-or-later, like Ansel. `data/noiseprofiles.json` comes
from the darktable project (GPL-3.0) and carries its profiling contributors'
work.

The **data** (training shards on the release, and community-contributed
tiles) is licensed separately, per provenance class — see the shard-release
licensing paragraph above and [LICENSE-DATA.md](LICENSE-DATA.md) for the
Ansel Training Data License covering community contributions.
