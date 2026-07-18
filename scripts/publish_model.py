#!/usr/bin/env python3.12
"""Publish a trained .anselnn model into models/ with manifest bookkeeping.

The models/ directory of this repo is the distribution channel for Ansel:
the app's build system fetches models/manifest.json from the raw GitHub URL
and downloads each listed file verified by its sha256, so an override here
propagates to every fresh build automatically (a stale local copy fails the
hash check and is re-fetched).

Publishing rules:
  - filenames are rawdenoiseai-<version>-<variant>.anselnn and never change
    for a given (version, variant);
  - during R&D a (version, variant) MAY be overridden: the manifest's
    revision counter bumps so consumers can see churn;
  - once a version has shipped in a tagged stable Ansel release it is FROZEN:
    further training must become a new version (new enum value in the module,
    new file here). Overriding a frozen version silently changes the render
    of users' existing edits, which the version parameter exists to prevent.

Usage:
    python3.12 scripts/publish_model.py path/to/model.anselnn --version v1 --variant full
Then commit models/ (amend the previous model commit during R&D churn) and push.
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path, help="trained .anselnn file to publish")
    ap.add_argument("--version", required=True, help="model version tag, e.g. v1")
    ap.add_argument("--variant", required=True, choices=["full", "distilled"])
    args = ap.parse_args()

    blob = args.model.read_bytes()
    if blob[:8] != b"ANSELDN1":
        sys.exit(f"{args.model} is not an ANSELDN1 model file")

    MODELS.mkdir(exist_ok=True)
    name = f"rawdenoiseai-{args.version}-{args.variant}.anselnn"
    manifest_path = MODELS / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"models": {}}

    entry = manifest["models"].get(name, {"revision": 0})
    new_sha = hashlib.sha256(blob).hexdigest()
    if entry.get("sha256") == new_sha:
        print(f"{name}: unchanged (revision {entry['revision']}), nothing to do")
        return 0

    shutil.copyfile(args.model, MODELS / name)
    manifest["models"][name] = {
        "sha256": new_sha,
        "size": len(blob),
        "revision": entry["revision"] + 1,
        "updated": date.today().isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"published {name}: {len(blob) / 1e6:.1f} MB, sha256 {new_sha[:16]}..., "
          f"revision {entry['revision'] + 1}")
    print("now: git add models/ && git commit (amend the model commit during R&D) && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
