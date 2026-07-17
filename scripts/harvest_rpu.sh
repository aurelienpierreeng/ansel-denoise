#!/bin/sh
# Stream-harvest the raw.pixls.us CC0 archive into training shards.
# Usage: ./scripts/harvest_rpu.sh [shard-output-dir] [extra harvest args...]
# Disk-bounded: the annex clone holds pointers only; each raw is fetched,
# mined for tiles, then dropped. Resumable — just re-run.
set -eu

OUT="${1:-shards/rpu}"
[ $# -gt 0 ] && shift
ANNEX_DIR="${RPU_ANNEX_DIR:-data.annex}"

command -v git-annex >/dev/null || { echo "git-annex is required (dnf/apt install git-annex)" >&2; exit 1; }
command -v exiftool >/dev/null || { echo "exiftool is required (dnf install perl-Image-ExifTool / apt install libimage-exiftool-perl)" >&2; exit 1; }

if [ ! -d "$ANNEX_DIR" ]; then
    echo "cloning raw.pixls.us annex (metadata only) into $ANNEX_DIR ..."
    git clone https://raw.pixls.us/data.annex.git "$ANNEX_DIR"
fi

# Pin the dataset state for the reproducibility record.
echo "annex commit: $(git -C "$ANNEX_DIR" rev-parse HEAD)"

exec python3.12 -m ansel_denoise.harvest --source "$ANNEX_DIR" --annex --out "$OUT" "$@"
