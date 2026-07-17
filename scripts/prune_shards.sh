#!/bin/sh
# Free local disk by deleting shards that are safely stored on the GitHub
# release — the local equivalent of `git annex drop`. A shard is deleted only
# if its name appears in the release's published.txt index, which
# publish_shards.sh updates strictly AFTER each successful tarball upload, so
# nothing unpublished can ever be deleted. ledger.jsonl is always kept: it is
# the harvester's resume state, not data. Restore with scripts/fetch_shards.sh.
#
# Usage: ./scripts/prune_shards.sh <shard-dir> [release-tag]
#        DRY_RUN=1 ./scripts/prune_shards.sh <shard-dir>   # preview only
#
# Typical cycle while the harvest runs:
#   ./scripts/publish_shards.sh shards/rpu && ./scripts/prune_shards.sh shards/rpu
set -eu

DIR="${1:?usage: prune_shards.sh <shard-dir> [release-tag]}"
TAG="${2:-shards-v1}"
REPO="${ANSEL_DENOISE_REPO:-aurelienpierreeng/ansel-denoise}"

command -v gh >/dev/null || { echo "gh CLI required" >&2; exit 1; }
[ -d "$DIR" ] || { echo "no such directory: $DIR" >&2; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# the index is the source of truth for "safely stored"; no index -> refuse
gh release download "$TAG" --repo "$REPO" --pattern published.txt --dir "$WORK" 2>/dev/null \
    || { echo "no published.txt on release '$TAG' — nothing is confirmed stored, refusing to delete" >&2; exit 1; }

N_DEL=0
BYTES=0
N_KEPT=0
find "$DIR" -maxdepth 1 -name '*.npz' -printf '%f\n' | sort | while IFS= read -r f; do
    if grep -qxF "$f" "$WORK/published.txt"; then
        SIZE=$(stat -c %s "$DIR/$f")
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "would delete $f ($((SIZE / 1024)) KB)"
        else
            rm "$DIR/$f"
        fi
        echo "$SIZE" >> "$WORK/deleted.txt"
    else
        echo "keeping $f (not yet published)"
    fi
done

# the pipe above runs in a subshell; recount from the work files for the summary
if [ -f "$WORK/deleted.txt" ]; then
    N_DEL=$(wc -l < "$WORK/deleted.txt")
    BYTES=$(awk '{ s += $1 } END { print s }' "$WORK/deleted.txt")
fi
N_KEPT=$(find "$DIR" -maxdepth 1 -name '*.npz' | wc -l)

VERB=deleted
[ "${DRY_RUN:-0}" = "1" ] && VERB="would delete"
echo "$VERB $N_DEL shards ($((BYTES / 1024 / 1024)) MB); $N_KEPT remain locally (ledger.jsonl kept)"
