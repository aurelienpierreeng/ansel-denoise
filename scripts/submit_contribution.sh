#!/bin/sh
# Submit a contribution bundle as a pull request — no git knowledge needed.
#
# Everything goes through the GitHub CLI (https://cli.github.com): it signs
# you in through your browser, forks the repository for you, and opens the
# pull request. What the PR contains is NOT the bundle (too big for git) but
# a small metadata file under contrib/pending/ — your handle, the download
# link, the sha256, the statistics and the license grant — which is the
# maintainer's review queue.
#
# Usage:
#   sh scripts/submit_contribution.sh <bundle.tar.gz> --url <download-link>
#
# Run it after pack_contribution.py and after uploading the bundle to any
# file host (the --url link must download the file directly).
set -eu

REPO="${ANSEL_DENOISE_REPO:-aurelienpierreeng/ansel-denoise}"
BUNDLE="${1:?usage: submit_contribution.sh <bundle.tar.gz> --url <download-link>}"
shift
URL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --url) URL="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done
[ -f "$BUNDLE" ] || { echo "no such file: $BUNDLE" >&2; exit 1; }
if [ -z "$URL" ]; then
    printf "Paste the download link of the uploaded bundle: "
    read -r URL
fi
[ -n "$URL" ] || { echo "a download link is required" >&2; exit 1; }

command -v gh >/dev/null || {
    echo "ERROR: the GitHub CLI 'gh' is required. Install it from https://cli.github.com" >&2
    echo "(Debian/Ubuntu: apt install gh | Fedora: dnf install gh | macOS: brew install gh)" >&2
    exit 1
}
# sign in through the browser if needed — this is the only 'account setup'
gh auth status >/dev/null 2>&1 || gh auth login --web

PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null || PY=python3
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# --- build the pending metadata file from the bundle's own manifest --------
SHA=$(sha256sum "$BUNDLE" | cut -d' ' -f1)
mkdir "$WORK/x"
tar xzf "$BUNDLE" -C "$WORK/x"
MANIFEST=$(find "$WORK/x" -name contribution-manifest.json | head -1)
[ -n "$MANIFEST" ] || { echo "not a contribution bundle (no manifest) — run pack_contribution.py first" >&2; exit 1; }

HANDLE=$("$PY" - "$MANIFEST" "$URL" "$SHA" "$WORK/pending.json" <<'EOF'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
pending = dict(manifest, url=sys.argv[2], bundle_sha256=sys.argv[3])
pending.pop("files", None)  # per-file hashes stay in the bundle itself
Path(sys.argv[4]).write_text(json.dumps(pending, indent=1, sort_keys=True) + "\n")
print(manifest["handle"])
EOF
)
STAMP=$(date -u +%Y%m%d-%H%M)
NAME="$HANDLE-$STAMP"

# --- fork, branch, commit, pull request — all through gh -------------------
echo "forking $REPO and opening the pull request..."
(cd "$WORK" && gh repo fork "$REPO" --clone -- --depth 1 --quiet)
CLONE="$WORK/$(basename "$REPO")"
mkdir -p "$CLONE/contrib/pending"
cp "$WORK/pending.json" "$CLONE/contrib/pending/$NAME.json"

cd "$CLONE"
git checkout -q -b "shards/$NAME"
git add "contrib/pending/$NAME.json"
# identity fallback so first-time users without a git config can commit
git -c user.name="$HANDLE" -c user.email="$HANDLE@users.noreply.github.com" \
    commit -q -m "Shard contribution from $HANDLE"
git push -q -u origin "shards/$NAME"

cat > "$WORK/pr-body.md" <<EOF
Shard contribution bundle (packed by \`pack_contribution.py\`, see CONTRIBUTING.md).

- download: $URL
- sha256: \`$SHA\`

By opening this pull request I confirm the grant recorded in the metadata
file: I own the rights to these photographs and license the tiles under the
[Ansel Training Data License 1.0](../blob/master/LICENSE-DATA.md) — denoiser
training only, every other use (generative AI included) forbidden.

The maintainer ingests with:
\`./scripts/collect_contribution.sh contrib/pending/$NAME.json --source <this PR URL>\`
EOF
gh pr create --repo "$REPO" --base master \
    --title "[shards] contribution from $HANDLE" \
    --body-file "$WORK/pr-body.md"

echo "done — the maintainer will fetch and validate the bundle from your link."
