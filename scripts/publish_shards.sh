#!/bin/sh
# Publish harvested shards as GitHub release assets — a permanent public cache
# of the training set. Incremental and resumable: a published.txt index on the
# release records what is already up, so re-running after more harvesting (or
# after an interrupted upload) only packs and uploads the new shards. Safe to
# run while the harvest is still going.
#
# Usage: ./scripts/publish_shards.sh <shard-dir> [release-tag]
#
# The canonical provenance remains ledger.jsonl + the annex commit hash; the
# release spares contributors the multi-day re-harvest, nothing more.
set -eu

DIR="${1:?usage: publish_shards.sh <shard-dir> [release-tag]}"
TAG="${2:-shards-v1}"
REPO="${ANSEL_DENOISE_REPO:-aurelienpierreeng/ansel-denoise}"
MAX_BYTES=$((1800 * 1024 * 1024)) # stay under GitHub's 2 GiB per-asset cap

command -v gh >/dev/null || { echo "gh CLI required" >&2; exit 1; }
[ -d "$DIR" ] || { echo "no such directory: $DIR" >&2; exit 1; }
if [ -e "$DIR/.private" ]; then
    echo "REFUSING to publish: $DIR is marked private (.private marker)." >&2
    echo "These shards come from a personal library and must never reach the public release." >&2
    exit 1
fi

gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1 || gh release create "$TAG" \
    --repo "$REPO" --title "Harvested training shards ($TAG)" --latest=false \
    --notes "Clean-tile training shards for the Ansel neural raw denoiser. Cache only: the dataset is reproducible from ledger.jsonl + the harvest scripts. Fetch with scripts/fetch_shards.sh. LICENSING: mixed-provenance aggregate — raw.pixls.us shards are CC0-1.0, PlayRaw shards carry their per-shard declared CC license (attribution embedded), community-contributed shards (<handle>_ prefix) are under the Ansel Training Data License 1.1 (LICENSE-DATA.md, attached): usable by anyone with the ansel-denoise training stack to audit/reproduce/benchmark or to train custom denoising models (resulting weights unrestricted, commercial use included); feeding them to any stack able to learn anything else than denoising (style, generative AI) is forbidden. Intended use of the whole corpus: training denoising networks on your own infrastructure, which satisfies all classes at once. See the repository README and CONTRIBUTING.md for the full terms."

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
gh release download "$TAG" --repo "$REPO" --pattern published.txt --dir "$WORK" 2>/dev/null || :
touch "$WORK/published.txt"

# shards present locally but not yet on the release (newline-safe, sorted)
find "$DIR" -maxdepth 1 -name '*.npz' -printf '%f\n' | sort > "$WORK/local.txt"
grep -vxF -f "$WORK/published.txt" "$WORK/local.txt" > "$WORK/new.txt" || :
N_NEW=$(wc -l < "$WORK/new.txt")
[ "$N_NEW" -gt 0 ] || { echo "nothing new to publish ($(wc -l < "$WORK/published.txt") shards already up)"; exit 0; }
echo "$N_NEW new shards to publish"

STAMP=$(date -u +%Y%m%d-%H%M%S)
SEQ=0
: > "$WORK/batch.txt"
BATCH_BYTES=0

flush_batch() {
    [ -s "$WORK/batch.txt" ] || return 0
    SEQ=$((SEQ + 1))
    NAME="shards-$STAMP-$(printf '%03d' "$SEQ").tar"
    echo "packing $NAME ($(wc -l < "$WORK/batch.txt") shards, $((BATCH_BYTES / 1024 / 1024)) MB)"
    tar cf "$WORK/$NAME" -C "$DIR" --files-from="$WORK/batch.txt"
    gh release upload "$TAG" "$WORK/$NAME" --repo "$REPO"
    rm -f "$WORK/$NAME"
    # commit to the index only after a successful upload -> interruption-safe
    cat "$WORK/batch.txt" >> "$WORK/published.txt"
    sort -o "$WORK/published.txt" "$WORK/published.txt"
    gh release upload "$TAG" "$WORK/published.txt" --repo "$REPO" --clobber
    : > "$WORK/batch.txt"
    BATCH_BYTES=0
}

while IFS= read -r f; do
    SIZE=$(stat -c %s "$DIR/$f")
    if [ -s "$WORK/batch.txt" ] && [ $((BATCH_BYTES + SIZE)) -gt "$MAX_BYTES" ]; then
        flush_batch
    fi
    printf '%s\n' "$f" >> "$WORK/batch.txt"
    BATCH_BYTES=$((BATCH_BYTES + SIZE))
done < "$WORK/new.txt"
flush_batch

# keep the latest provenance ledger alongside the data
[ -f "$DIR/ledger.jsonl" ] && gh release upload "$TAG" "$DIR/ledger.jsonl" --repo "$REPO" --clobber

# the data license travels with the data (community shards are ATDL-1.0)
LICENSE_FILE="$(dirname "$0")/../LICENSE-DATA.md"
[ -f "$LICENSE_FILE" ] && gh release upload "$TAG" "$LICENSE_FILE" --repo "$REPO" --clobber

echo "done: $(wc -l < "$WORK/published.txt") shards published on release '$TAG'"
