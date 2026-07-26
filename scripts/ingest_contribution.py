#!/usr/bin/env python3
"""Maintainer helper: fix a contribution bundle's handle, then ingest it.

Contributors sometimes pack their bundle under the docs placeholder handle
(`your-github-name`, `your-github-<partial>`, ...) instead of their real
GitHub login. Ingesting as-is would poison the corpus namespace and break
removal-by-handle, so this wrapper renames every shard prefix and rewrites
the manifest (handle + file keys) to <desired_handle> — per-file sha256 are
carried over unchanged because the shard bytes are identical, and verified
before and after — then hands the corrected bundle to
`collect_contribution.sh` for the usual verify + merge + registry step.

When the bundle's handle already equals <desired_handle> it is ingested
untouched (no repack), so this is safe to run on every contribution.

Usage:
    python3 scripts/ingest_contribution.py <bundle.tar.gz> <github-login> <issue-url>

Big bundles (contributions can be >1 GB) extract to $TMPDIR — point it at a
real disk if /tmp is a small tmpfs:
    TMPDIR=/var/tmp python3 scripts/ingest_contribution.py ...
"""

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _verify(manifest_files: dict, d: Path) -> list[str]:
    return [n for n, s in manifest_files.items()
            if hashlib.sha256((d / n).read_bytes()).hexdigest() != s]


def main() -> int:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    bundle, desired, issue_url = Path(sys.argv[1]), sys.argv[2].lower(), sys.argv[3]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with tarfile.open(bundle) as t:
            t.extractall(tmp, filter="data")
        manifest_path = next(tmp.rglob("contribution-manifest.json"))
        d = manifest_path.parent
        m = json.loads(manifest_path.read_text())
        old = m["handle"]

        if _verify(m["files"], d):
            sys.exit(f"CONTENT MISMATCH in {bundle.name}: shard bytes do not match "
                     f"the manifest hashes — refusing (corrupted or tampered bundle)")

        cams = ", ".join(f"{k}:{v}" for k, v in m["cameras"].items())
        print(f"{bundle.name}: handle '{old}' -> '{desired}', "
              f"{m['n_shards']} shards / {m['n_tiles']} tiles [{cams}]")

        if old == desired:
            corrected = bundle
        else:
            oldpfx, newpfx = f"{old}_", f"{desired}_"
            for f in list(d.glob("*.npz")):
                if f.name.startswith(oldpfx):
                    f.rename(d / (newpfx + f.name[len(oldpfx):]))
            m["handle"] = desired
            m["files"] = {(newpfx + k[len(oldpfx):] if k.startswith(oldpfx) else k): v
                          for k, v in m["files"].items()}
            manifest_path.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
            if _verify(m["files"], d):
                sys.exit("post-rename hash mismatch — aborting")
            corrected = Path(tempfile.gettempdir()) / f"corrected-{desired}-{bundle.stem}.tar.gz"
            with tarfile.open(corrected, "w:gz") as t:
                t.add(d, arcname=d.name)

        return subprocess.run(
            ["./scripts/collect_contribution.sh", str(corrected),
             "--source", issue_url, "--dest", f"shards/contrib/{desired}"],
            cwd=REPO).returncode


if __name__ == "__main__":
    sys.exit(main())
