# Contributing training images to the Ansel raw denoiser

> **Prefer the rendered version:** this guide also lives on the Ansel website
> at [ansel.photos/en/contribute/training-data](https://ansel.photos/en/contribute/training-data/) —
> same content, nicer to read. This file remains the reference the scripts
> and templates link to.

The neural denoiser in Ansel's `rawdenoiseai` module learns what clean sensor
data looks like from a corpus of **base-ISO raw tiles**, then noise is
synthesized on top at training time from Ansel's camera noise profiles.

This is what sets it apart from every general-purpose AI denoiser: it is
trained on **non-demosaiced sensor data, as close to the actual sensor
reading as a raw file allows**, and deployed at the exact pipeline stage it
was trained for — before white balance, chromatic-aberration correction and
demosaicing, so every downstream stage inherits clean data. We control both
the training and the usage, with no intermediate approximation in between.
The corpus is what limits quality: more scenes, more textures and more
camera models directly translate into a better denoiser for everyone —
especially for under-represented sensors (X-Trans, older CCDs, phones).

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
| `black_per_channel`, `white`, `wb` | sensor calibration levels; the as-shot WB coefficients are metadata only — **never applied**, tiles stay pre-white-balance |
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
  `shards-v1` GitHub release. This is deliberate: it makes the training
  corpus reproducible and auditable by anyone, with no hidden data.
- Public does **not** mean free-for-all: the tiles are licensed under the
  **[Ansel Training Data License 1.1](LICENSE-DATA.md)** (ATDL-1.1), which
  you keep the copyright under. In short: anyone may use the tiles **with
  the ansel-denoise training stack** (GPL-3.0) — to audit, review,
  reproduce and benchmark the Ansel denoiser (scientific and academic use
  welcome), or to **train their own denoising models**, whose weights are
  theirs without restriction, commercial applications included. The bright
  line: the tiles must never feed a training stack able to learn anything
  else than separating noise from signal — "style" learning, generative AI,
  dataset redistribution, identification of people or places are explicitly
  forbidden. Accepting the license is accepting that prohibition.
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

**Volume policy: up to ~1000 raw images per person.** The corpus needs
content and camera variability, not bulk from a single library — a thousand
images from one photographer already share one eye, one gear bag and one
processing habit. The pack script enforces the cap; if your library has
more to give, curate down to your most varied thousand.

## How to contribute, step by step

Requirements: Linux, macOS or Windows, Python ≥ 3.10, an Ansel library, and
about 10 minutes. Every step below shows the Linux/macOS command first and
the Windows (PowerShell) command after it.

**Windows first-timers — two one-time preparations:**

- **Install Python** if `py --version` in a PowerShell window says it is
  missing: either `winget install Python.Python.3.12` in PowerShell, or
  download it from [python.org](https://www.python.org/downloads/) and
  **check "Add python.exe to PATH"** in the installer's first screen.
- **Open PowerShell in the right folder**: press `Win+X`, choose *Terminal*
  (or *Windows PowerShell*), then `cd` to where you want the tools, e.g.
  `cd $HOME\Documents`. You can paste commands with a right-click.
  PowerShell refuses to run script files by default; that is why every
  script command below is written with `-ExecutionPolicy Bypass`, which
  allows just that one script for just that one run.

**0. Install the tooling** (one script — installs the two Python
dependencies, `numpy` and `rawpy`, and fetches this repository if needed;
on Windows it downloads it as a ZIP, no git required):

```sh
git clone https://github.com/aurelienpierreeng/ansel-denoise.git
cd ansel-denoise
sh scripts/setup_contributor.sh
```

On Windows (the script downloads the repository for you):

```powershell
Invoke-WebRequest https://raw.githubusercontent.com/aurelienpierreeng/ansel-denoise/master/scripts/setup_contributor.ps1 -OutFile setup_contributor.ps1
powershell -ExecutionPolicy Bypass -File setup_contributor.ps1
cd ansel-denoise
```

**1. Curate in Ansel.** In the lighttable, select the images you are willing
to make public tiles of (base ISO — sort or filter by ISO to be quick), then
**File ▸ Export image list... ▸ Save as file...** and keep the proposed
`ansel-image-files.txt` name. We accept up to 1000 images per contributors,
to maintain a proper diversity in the dataset.

**2. Harvest.** This gates on ISO and decodes each file in a crash-isolated
child process, then writes the shards. Your Ansel library database is used
read-only when present, but is not required — camera and ISO are read from
the files themselves otherwise. Nothing is uploaded:

```sh
python3 -m ansel_denoise.harvest_library --paths-file ansel-image-files.txt --out shards/mine
```

On Windows (the `py` launcher comes with Python; the library database is
found automatically under `%LOCALAPPDATA%\ansel`):

```powershell
py -m ansel_denoise.harvest_library --paths-file ansel-image-files.txt --out shards\mine
```

**3. Pack.** This validates every shard, prefixes them with your handle
(so bundles from different people can never collide), writes a manifest with
per-file checksums and your license grant, packs the license text with the
data, and produces a single tarball:

```sh
python3 scripts/pack_contribution.py shards/mine --handle your-github-name
```

On Windows:

```powershell
py scripts\pack_contribution.py shards\mine --handle your-github-name
```

**4. Upload and open an issue.** Put the printed `.tar.gz` on any file host that
gives a **direct download link** — Google Drive, Dropbox, WeTransfer, Proton
Drive, your own server — then open a
[Shard contribution issue](https://github.com/aurelienpierreeng/ansel-denoise/issues/new/choose)
and paste the link (the form also asks you to confirm the license grant).
No extra tools, no git, nothing to install beyond what step 0 set up — the
same command works on Linux, macOS and Windows.

That's the whole submission. The maintainer downloads the bundle from your
link, verifies it and merges it into the corpus. You can delete `shards/mine`
and the bundle afterwards.

## What happens next (maintainer side)

The maintainer opens your issue and runs:

```sh
./scripts/collect_contribution.sh <your-link> --source <issue-url>
```

which downloads the bundle from your link,
verifies every shard against the manifest checksums, re-validates shard
structure and the ISO gate (shards are loaded with `allow_pickle=False`, so
a bundle cannot execute code), merges the new shards into
`shards/contrib/<handle>/` skipping anything already known, and appends one
bookkeeping line to [`contrib/registry.jsonl`](contrib/registry.jsonl) —
who, when, from where, hash, tile statistics. The registry is committed, so
the provenance of the whole corpus is public git history.

Finally `./scripts/publish_shards.sh shards/contrib/<handle>` uploads the
shards to the `shards-v1` release (with the [data license](LICENSE-DATA.md)
alongside), the pending metadata file is removed in the same commit that
records the registry entry, your pull request is merged — and your camera
earns its place in the next model version's changelog.
