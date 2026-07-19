#!/usr/bin/env python3
"""Pack harvested shards into a contribution bundle for the Ansel denoiser corpus.

Contributor side of the community pipeline (see CONTRIBUTING.md):
  1. validates every shard (a broken shard aborts the pack — delete it and rerun);
  2. renames shards with your handle as a prefix, so bundles from different
     contributors can never collide in the shared corpus;
  3. writes a contribution-manifest.json (handle, date, license grant, per-file
     sha256, camera/tile statistics) — the bookkeeping the maintainer records;
  4. produces a single .tar.gz to upload anywhere the maintainer can download.

The local ledger.jsonl is deliberately NOT packed: it contains absolute paths
from your machine. Shards only embed the library-relative name, camera model,
ISO and sensor levels.

Usage:
    python3 scripts/pack_contribution.py shards/mine --handle your-name --yes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ansel_denoise.validate_shards import validate_dir  # noqa: E402

LICENSE_TAG = "ATDL-1.1"
MAX_IMAGES = 1000  # per person: the corpus needs variability, not single-library volume
GRANT = ("I own the rights to these photographs and I license the packed tiles "
         "under the Ansel Training Data License 1.1 (LICENSE-DATA.md): anyone "
         "may use them with the ansel-denoise training stack to audit, "
         "reproduce and benchmark the training, or to train their own "
         "denoising models — whose weights are theirs, commercial use "
         "included; feeding the tiles to any training stack able to learn "
         "anything else than denoising (style, generative AI) is explicitly "
         "forbidden.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path, help="shard directory (output of harvest_library)")
    ap.add_argument("--handle", required=True,
                    help="your public handle (GitHub username works well); "
                         "lowercase letters, digits, dashes")
    ap.add_argument("--out", type=Path, default=Path("."),
                    help="where to write the bundle (default: current directory)")
    ap.add_argument("--max-iso", type=int, default=200)
    ap.add_argument("--yes", action="store_true",
                    help="non-interactive: accept the ATDL-1.1 grant printed by the script")
    args = ap.parse_args(argv)

    handle = args.handle.lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,30}", handle):
        ap.error("handle must be 1-31 chars of lowercase letters, digits, dashes")
    if (args.directory / ".private").exists():
        sys.exit(f"REFUSING: {args.directory} is marked private (.private marker). "
                 f"Contributions are public by definition — harvest a curated, "
                 f"publishable directory instead.")

    print(f"validating {args.directory} ...")
    summary = validate_dir(args.directory, max_iso=args.max_iso)
    if summary["n_invalid"]:
        sys.exit(f"{summary['n_invalid']} invalid shards (listed above): "
                 f"delete them and rerun.")
    if summary["n_shards"] == 0:
        sys.exit(f"no shards in {args.directory} — run harvest_library first")
    if summary["n_shards"] > MAX_IMAGES:
        sys.exit(f"{summary['n_shards']} images exceeds the ~{MAX_IMAGES} images/person "
                 f"policy: the corpus needs content and camera variability more than "
                 f"volume from one library — curate down to your best {MAX_IMAGES}.")
    print(f"{summary['n_shards']} shards, {summary['n_tiles']} tiles, "
          f"{len(summary['cameras'])} cameras")

    print(f"\nBy packing you declare:\n  {GRANT}")
    print("  (full license text: LICENSE-DATA.md, packed with the bundle)")
    if not args.yes:
        if input("Type 'yes' to accept: ").strip().lower() != "yes":
            sys.exit("aborted: grant not accepted")

    date = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / f"contrib-{handle}-{date:%Y%m%d}"
        stage.mkdir()
        files = {}
        for shard in sorted(args.directory.glob("*.npz")):
            name = shard.name if shard.name.startswith(f"{handle}_") \
                else f"{handle}_{shard.name}"
            shutil.copyfile(shard, stage / name)
            files[name] = hashlib.sha256(shard.read_bytes()).hexdigest()

        # the license travels with the data
        shutil.copyfile(REPO / "LICENSE-DATA.md", stage / "LICENSE-DATA.md")

        manifest = {
            "format": 1,
            "handle": handle,
            "created": date.isoformat(timespec="seconds"),
            "license": LICENSE_TAG,
            "grant": GRANT,
            "n_shards": summary["n_shards"],
            "n_tiles": summary["n_tiles"],
            "cameras": summary["cameras"],
            "files": files,
        }
        (stage / "contribution-manifest.json").write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n")

        args.out.mkdir(parents=True, exist_ok=True)
        bundle = args.out / f"ansel-denoise-contrib-{handle}-{date:%Y%m%d}.tar.gz"
        # tarfile, not a tar subprocess: works identically on Windows
        with tarfile.open(bundle, "w:gz") as tar:
            tar.add(stage, arcname=stage.name)

    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    print(f"\nbundle: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)")
    print(f"sha256: {sha}")
    submit = (f"powershell -ExecutionPolicy Bypass -File scripts\\submit_contribution.ps1 "
              f"-Bundle {bundle} -Url <your-download-link>" if os.name == "nt"
              else f"sh scripts/submit_contribution.sh {bundle} --url <your-download-link>")
    print("\nNext:")
    print("  1. upload the bundle to any file host the maintainer can download from")
    print("     (Google Drive, Dropbox, WeTransfer, Proton Drive, your own server...)")
    print("  2. submit it — no git knowledge needed, the script does everything:")
    print(f"     {submit}")
    print("     (or, without the gh CLI, open an issue instead:")
    print("     https://github.com/aurelienpierreeng/ansel-denoise/issues/new/choose)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
