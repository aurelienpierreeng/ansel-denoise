#!/bin/sh
# Merge small shard tarballs on the GitHub release into ~1.8 GiB ones.
# GitHub cannot do this server-side (release assets are immutable blobs), so
# this downloads the small tarballs, repacks them, uploads the merged tarball
# and only THEN deletes the originals: an interruption can leave duplicate
# assets (which fetch_shards.sh handles harmlessly — same shards extracted
# twice), never lose data. published.txt lists shard names, not tarball
# membership, so publish/prune/fetch are all unaffected by regrouping.
#
# Run it occasionally after several incremental publishes, on any machine with
# decent bandwidth and ~4 GB of free temp space.
#
# Usage: ./scripts/compact_shards.sh [release-tag]
#        DRY_RUN=1 ./scripts/compact_shards.sh   # preview the plan only
set -eu

TAG="${1:-shards-v1}"
REPO="${ANSEL_DENOISE_REPO:-aurelienpierreeng/ansel-denoise}"
MAX_BYTES=$((1800 * 1024 * 1024)) # target tarball size (2 GiB cap upstream)
SMALL=$((1500 * 1024 * 1024))     # only tarballs below this get merged

command -v gh >/dev/null || { echo "gh CLI required" >&2; exit 1; }
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

gh release view "$TAG" --repo "$REPO" --json assets \
    --jq '.assets[] | select(.name | startswith("shards-")) | "\(.size)\t\(.name)"' \
    | sort -n > "$WORK/assets.tsv"

awk -F'\t' -v small="$SMALL" '$1 < small' "$WORK/assets.tsv" > "$WORK/candidates.tsv"
N=$(wc -l < "$WORK/candidates.tsv")
if [ "$N" -lt 2 ]; then
    echo "nothing to compact: $N tarball under $((SMALL / 1024 / 1024)) MB (of $(wc -l < "$WORK/assets.tsv") total)"
    exit 0
fi

# greedy bins over the size-sorted candidates
awk -F'\t' -v max="$MAX_BYTES" 'BEGIN { grp = 0 } {
    if (n > 0 && sum + $1 > max) { grp++; sum = 0; n = 0 }
    print grp "\t" $1 "\t" $2; sum += $1; n++
}' "$WORK/candidates.tsv" > "$WORK/groups.tsv"

STAMP=$(date -u +%Y%m%d-%H%M%S)
for g in $(cut -f1 "$WORK/groups.tsv" | sort -un); do
    awk -F'\t' -v g="$g" '$1 == g { print $3 }' "$WORK/groups.tsv" > "$WORK/members.txt"
    M=$(wc -l < "$WORK/members.txt")
    [ "$M" -ge 2 ] || continue # a lone tarball gains nothing from repacking
    NEW="shards-$STAMP-m$(printf '%03d' "$g").tar"
    SUM=$(awk -F'\t' -v g="$g" '$1 == g { s += $2 } END { print s }' "$WORK/groups.tsv")
    echo "merging $M tarballs ($((SUM / 1024 / 1024)) MB) -> $NEW"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        sed 's/^/    /' "$WORK/members.txt"
        continue
    fi

    mkdir "$WORK/merge"
    while IFS= read -r name; do
        gh release download "$TAG" --repo "$REPO" --pattern "$name" --dir "$WORK"
        tar xf "$WORK/$name" -C "$WORK/merge"
        rm -f "$WORK/$name"
    done < "$WORK/members.txt"
    tar cf "$WORK/$NEW" -C "$WORK/merge" .
    gh release upload "$TAG" "$WORK/$NEW" --repo "$REPO"
    rm -rf "$WORK/merge" "$WORK/$NEW"
    # originals go away only after the merged tarball is safely up
    while IFS= read -r name; do
        gh release delete-asset "$TAG" "$name" --yes --repo "$REPO"
    done < "$WORK/members.txt"
done
echo "done"
