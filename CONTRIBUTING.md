# Contributing training images to the Ansel raw denoiser

The neural denoiser in Ansel's `rawdenoiseai` module learns what clean sensor
data looks like from a corpus of **base-ISO raw tiles**, then noise is
synthesized on top at training time from Ansel's camera noise profiles. The
corpus is what limits quality: more scenes, more textures and more camera
models directly translate into a better denoiser for everyone — especially
for under-represented sensors (X-Trans, older CCDs, phones).

This page explains how to contribute images from your own Ansel library,
what exactly leaves your machine, and what happens to it.

## What is collected — read this before anything else

Your photographs are **not** uploaded. Your files never leave your machine
whole. Instead, a harvester decodes each raw file locally and extracts up to
16 tiles of 256×256 **raw sensor pixels** (mosaic data, before demosaicing),
packed into one compressed `.npz` "shard" per image (~1–4 MB).

Each shard contains exactly:

| field | content |
|---|---|
| `tiles` | up to 16 crops of 256×256 raw sensor values (16-bit) |
| `offsets` | where in the frame each tile was cropped |
| `pattern` / `pattern4` | the sensor's CFA layout (Bayer/X-Trans) |
| `black_per_channel`, `white`, `wb` | sensor calibration levels, white balance |
| `iso` | the ISO the image was shot at |
| `camera` | camera make and model, e.g. `NIKON CORPORATION D850` |
| `source_path` | `library/<image-id>/<filename>` — the bare filename only |

It contains **no** GPS coordinates, no timestamps, no serial numbers, no
author metadata, and no filesystem paths from your machine (the harvester's
local `ledger.jsonl` does contain absolute paths, and the pack script
therefore never includes it).

**However — and this is the important part — the tiles themselves are
viewable fragments of your photographs.** 256×256 crops are large enough to
recognize faces, license plates, documents, screens. Treat a contribution
exactly like publishing crops of the photos.

## Privacy and license

- Contributed tiles are published as **public assets** on this repository's
  `shards-v1` GitHub release, under **CC0-1.0** (public domain dedication).
  Anyone can download them. This is deliberate: it makes the training corpus
  reproducible and auditable by anyone, with no hidden data.
- You must own the rights to the photographs you contribute.
- **The curation step is your privacy control**: contribute only images whose
  content you are comfortable making public. No people, documents or places
  you would not post publicly. The Ansel lighttable selection *is* the
  consent boundary — nothing you did not explicitly select is ever touched.
- Removal: open an issue and your shards are deleted from the release (they
  are all prefixed with your handle, so this is mechanical). Be aware that
  model versions already trained on them cannot be untrained — removal
  applies to the corpus and to future trainings.
- Everything runs locally with open-source code you can read in this repo;
  there is no telemetry, no third-party service involved. You upload the
  final bundle yourself, wherever you choose.

## What makes a good candidate image

The harvester wants **clean** signal — the noise is added synthetically later.

Good:

- **Base ISO** (ISO ≤ 200 is enforced; ISO 64–100 is even better),
  correctly exposed or slightly bright — shadows contain noise even at base
  ISO, highlights are clean.
- **Sharp and textured**: foliage, fabric, hair, gravel, brick, grass —
  fine detail is exactly what the network must learn to preserve.
- **Diverse content**: landscapes, architecture, still life, macro...
  The tile picker prefers textured regions on its own, but variety in what
  you give it is what variety in the corpus is made of.
- **Rare cameras**: anything beyond the usual Canon/Nikon/Sony full-frame
  bodies is disproportionately valuable — Fuji X-Trans, Olympus/OM, Pentax,
  compacts, phones with DNG, old CCD bodies.

Not useful (the harvester rejects most of these on its own, but save
yourself the time):

- High-ISO, underexposed, or motion-blurred / out-of-focus images;
- Long-exposure night shots (hot pixels are not "clean");
- Bursts and near-duplicates of the same scene — pick one;
- Flat frames: clear skies, walls, defocused backgrounds (clipped and
  textureless crops are auto-rejected).

Fifty varied, sharp, base-ISO images are worth more than five hundred
near-duplicates.

## How to contribute, step by step

Requirements: Linux or macOS (Windows: use WSL), Python ≥ 3.10, an Ansel
library, and about 10 minutes.

**0. Install the tooling** (one script — clones this repo if needed and
installs the two Python dependencies, `numpy` and `rawpy`):

```sh
git clone https://github.com/aurelienpierreeng/ansel-denoise.git
cd ansel-denoise
sh scripts/setup_contributor.sh
```

**1. Curate in Ansel.** In the lighttable, select the images you are willing
to make public tiles of (base ISO — sort or filter by ISO to be quick), then
**File ▸ Export image list... ▸ Save as file...** and keep the proposed
`ansel-image-files.txt` name.

**2. Harvest.** This reads your Ansel library database read-only, gates on
ISO, decodes each file in a crash-isolated child process, and writes the
shards. Nothing is uploaded:

```sh
python3 -m ansel_denoise.harvest_library --paths-file ansel-image-files.txt --out shards/mine
```

**3. Pack.** This validates every shard, prefixes them with your handle
(so bundles from different people can never collide), writes a manifest with
per-file checksums and your CC0 grant, and produces a single tarball:

```sh
python3 scripts/pack_contribution.py shards/mine --handle your-github-name
```

**4. Share.** Upload the printed `.tar.gz` to any file host the maintainer
can download from — Google Drive, Dropbox, WeTransfer, Proton Drive, your
own server — then open a
[Shard contribution issue](https://github.com/aurelienpierreeng/ansel-denoise/issues/new/choose)
with the link and the sha256 the script printed.

That's it. You can delete `shards/mine` and the bundle afterwards.

## What happens next (maintainer side)

The maintainer runs:

```sh
./scripts/collect_contribution.sh <your-link> --sha256 <your-hash> --source <issue-url>
```

which downloads the bundle, verifies the tarball hash against the issue,
verifies every shard against the manifest checksums, re-validates shard
structure and the ISO gate (shards are loaded with `allow_pickle=False`, so
a bundle cannot execute code), merges the new shards into
`shards/contrib/<handle>/` skipping anything already known, and appends one
bookkeeping line to [`contrib/registry.jsonl`](contrib/registry.jsonl) —
who, when, from where, hash, tile statistics. The registry is committed, so
the provenance of the whole corpus is public git history.

Finally `./scripts/publish_shards.sh shards/contrib/<handle>` uploads the
shards to the `shards-v1` release, where every future training fetches them
with `scripts/fetch_shards.sh`, and your camera earns its place in the next
model version's changelog.
