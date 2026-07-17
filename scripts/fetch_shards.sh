#!/bin/sh
# Download the published shard cache from GitHub Releases into a local shard
# directory — the fast path for training boxes, instead of re-harvesting.
# Tarballs are fetched and extracted one at a time, so peak extra disk is one
# ~1.8 GB tarball.
#
# Usage: ./scripts/fetch_shards.sh [shard-dir] [release-tag]
set -eu

OUT="${1:-shards/rpu}"
TAG="${2:-shards-v1}"
REPO="${ANSEL_DENOISE_REPO:-aurelienpierreeng/ansel-denoise}"

command -v gh >/dev/null || { echo "gh CLI required" >&2; exit 1; }
mkdir -p "$OUT"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

gh release view "$TAG" --repo "$REPO" --json assets \
    --jq '.assets[].name | select(startswith("shards-"))' > "$WORK/assets.txt"
N=$(wc -l < "$WORK/assets.txt")
echo "$N tarballs on release '$TAG'"

I=0
while IFS= read -r asset; do
    I=$((I + 1))
    echo "[$I/$N] $asset"
    gh release download "$TAG" --repo "$REPO" --pattern "$asset" --dir "$WORK"
    tar xf "$WORK/$asset" -C "$OUT"
    rm -f "$WORK/$asset"
done < "$WORK/assets.txt"

gh release download "$TAG" --repo "$REPO" --pattern ledger.jsonl --dir "$OUT" --clobber 2>/dev/null || :
echo "done: $(find "$OUT" -name '*.npz' | wc -l) shards in $OUT"
