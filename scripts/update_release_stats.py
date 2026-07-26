#!/usr/bin/env python3
"""Count the shards on the shard release and write the total into its notes.

The release's `published.txt` lists one shard name per line, so it is the
authoritative global count. This script downloads it, tallies the total and
the community-contributed share (shards named `<handle>_...`, handles taken
from contrib/registry.jsonl), and rewrites a stats block — delimited by
`<!-- STATS -->` markers — at the top of the release notes, leaving the rest
(licensing terms) untouched. Idempotent: re-run it after every publish.

Usage:
    python3 scripts/update_release_stats.py [--tag shards-v1] [--dry-run]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import date, timezone, datetime
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
GH_REPO = "aurelienpierreeng/ansel-denoise"
BEGIN, END = "<!-- STATS -->", "<!-- /STATS -->"


def gh(*args: str, capture=True) -> str:
    return subprocess.run(["gh", *args], cwd=REPO_DIR, check=True,
                          capture_output=capture, text=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="shards-v1")
    ap.add_argument("--repo", default=GH_REPO)
    ap.add_argument("--dry-run", action="store_true", help="print the stats block, do not edit")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        gh("release", "download", args.tag, "--repo", args.repo,
           "--pattern", "published.txt", "--dir", tmp, "--clobber")
        names = [ln.strip() for ln in
                 (Path(tmp) / "published.txt").read_text().splitlines() if ln.strip()]
    total = len(names)

    # community share: shards prefixed with a registered contributor handle
    registry = REPO_DIR / "contrib" / "registry.jsonl"
    handles = sorted({json.loads(ln)["handle"]
                      for ln in registry.read_text().splitlines() if ln.strip()}) \
        if registry.exists() else []
    per_handle = {h: sum(1 for n in names if n.startswith(f"{h}_")) for h in handles}
    community = sum(per_handle.values())
    contributors = sum(1 for c in per_handle.values() if c)
    archive = total - community

    today = datetime.now(timezone.utc).date().isoformat()
    block = (
        f"{BEGIN}\n"
        f"### Corpus: {total:,} shards\n\n"
        f"Each shard holds up to 16 CFA tiles of 256×256 raw sensor pixels. "
        f"**{archive:,}** shards come from the raw.pixls.us and PlayRaw archives, "
        f"**{community:,}** from **{contributors}** community contributors. "
        f"_Updated {today}._\n"
        f"{END}"
    )
    if args.dry_run:
        print(block)
        return 0

    body = gh("release", "view", args.tag, "--repo", args.repo, "--json", "body")
    body = json.loads(body)["body"] or ""
    if BEGIN in body and END in body:
        pre, rest = body.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        body = pre + block + post
    else:
        body = block + "\n\n" + body

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
        notes = f.name
    gh("release", "edit", args.tag, "--repo", args.repo, "--notes-file", notes, capture=False)
    Path(notes).unlink()
    print(f"release '{args.tag}' updated: {total:,} shards "
          f"({archive:,} archive + {community:,} community from {contributors} contributors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
