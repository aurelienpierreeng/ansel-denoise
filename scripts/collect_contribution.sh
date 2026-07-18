#!/bin/sh
# Maintainer side of the community shard pipeline (see CONTRIBUTING.md):
# ingest a contribution bundle produced by scripts/pack_contribution.py,
# verify and validate it, merge it into the local corpus, and record it in
# contrib/registry.jsonl (the committed bookkeeping of what was added, when,
# from where).
#
# Usage:
#   ./scripts/collect_contribution.sh <url-or-tarball> [options]
#     --sha256 HEX     verify the bundle hash (from the contribution issue)
#     --source NOTE    provenance note for the registry (e.g. the issue URL);
#                      defaults to the url/path argument
#     --publish        run publish_shards.sh on the merged directory afterwards
#     --dest DIR       merge target (default shards/contrib/<handle>)
#     --registry FILE  bookkeeping file (default contrib/registry.jsonl)
#
# Safety: only *.npz files are taken out of the bundle (a stray ledger or any
# other file is ignored), shards are loaded with allow_pickle=False during
# validation, every file must match the sha256 recorded in the bundle's own
# manifest, and already-known shard names are skipped, so re-collecting the
# same bundle is a no-op.
set -eu

cd "$(dirname "$0")/.."

SRC="${1:?usage: collect_contribution.sh <url-or-tarball> [--sha256 HEX] [--publish]}"
shift
SHA256="" SOURCE="$SRC" PUBLISH=0 DEST="" REGISTRY="contrib/registry.jsonl"
while [ $# -gt 0 ]; do
    case "$1" in
        --sha256) SHA256="$2"; shift 2 ;;
        --source) SOURCE="$2"; shift 2 ;;
        --publish) PUBLISH=1; shift ;;
        --dest) DEST="$2"; shift 2 ;;
        --registry) REGISTRY="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null || PY=python3
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# --- fetch -----------------------------------------------------------------
if [ -f "$SRC" ]; then
    cp "$SRC" "$WORK/bundle.tar.gz"
else
    echo "downloading $SRC"
    curl -fL --retry 3 -o "$WORK/bundle.tar.gz" "$SRC"
fi
GOT_SHA=$(sha256sum "$WORK/bundle.tar.gz" | cut -d' ' -f1)
if [ -n "$SHA256" ] && [ "$GOT_SHA" != "$SHA256" ]; then
    echo "SHA256 MISMATCH: expected $SHA256, got $GOT_SHA — refusing the bundle" >&2
    exit 1
fi

# --- extract: only the manifest and .npz files are taken -------------------
mkdir "$WORK/x" "$WORK/shards"
tar xzf "$WORK/bundle.tar.gz" -C "$WORK/x"
MANIFEST=$(find "$WORK/x" -name contribution-manifest.json | head -1)
[ -n "$MANIFEST" ] || { echo "no contribution-manifest.json in bundle (not packed by pack_contribution.py?)" >&2; exit 1; }
find "$WORK/x" -name '*.npz' | while IFS= read -r f; do mv "$f" "$WORK/shards/"; done

# --- verify manifest hashes + validate shard structure ---------------------
HANDLE=$("$PY" - "$MANIFEST" "$WORK/shards" <<'EOF'
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, "src")
from ansel_denoise.validate_shards import validate_dir

manifest = json.loads(Path(sys.argv[1]).read_text())
shards = Path(sys.argv[2])
listed = manifest["files"]
present = {p.name for p in shards.glob("*.npz")}
if set(listed) != present:
    sys.exit(f"manifest/file mismatch: {sorted(set(listed) ^ present)}")
for name, sha in listed.items():
    if hashlib.sha256((shards / name).read_bytes()).hexdigest() != sha:
        sys.exit(f"sha256 mismatch on {name} — corrupted or tampered bundle")
summary = validate_dir(shards)
if summary["n_invalid"] or summary["n_shards"] == 0:
    sys.exit(f"{summary['n_invalid']} invalid shards — refusing the bundle")
print(f"manifest OK: {summary['n_shards']} shards, {summary['n_tiles']} tiles, "
      f"{len(summary['cameras'])} cameras", file=sys.stderr)
Path(sys.argv[2], ".summary.json").write_text(json.dumps(summary))
print(manifest["handle"])
EOF
)

# --- merge, skipping shard names already known locally or on the release ---
DEST="${DEST:-shards/contrib/$HANDLE}"
mkdir -p "$DEST"
gh release download shards-v1 --repo "${ANSEL_DENOISE_REPO:-aurelienpierreeng/ansel-denoise}" \
    --pattern published.txt --dir "$WORK" 2>/dev/null \
    || { echo "note: no gh access — skipping release-side duplicate check"; touch "$WORK/published.txt"; }
[ -f "$WORK/published.txt" ] || touch "$WORK/published.txt"

N_NEW=0 N_DUP=0
for f in "$WORK/shards"/*.npz; do
    NAME=$(basename "$f")
    if [ -e "$DEST/$NAME" ] || grep -qxF "$NAME" "$WORK/published.txt"; then
        N_DUP=$((N_DUP + 1))
    else
        mv "$f" "$DEST/$NAME"
        N_NEW=$((N_NEW + 1))
    fi
done
echo "merged $N_NEW new shards into $DEST ($N_DUP already known, skipped)"

# --- bookkeeping -----------------------------------------------------------
mkdir -p "$(dirname "$REGISTRY")"
"$PY" - "$MANIFEST" "$WORK/shards/.summary.json" "$REGISTRY" "$SOURCE" "$GOT_SHA" "$N_NEW" <<'EOF'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
summary = json.loads(Path(sys.argv[2]).read_text())
entry = {
    "collected": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "handle": manifest["handle"],
    "created": manifest["created"],
    "source": sys.argv[4],
    "bundle_sha256": sys.argv[5],
    "n_shards_new": int(sys.argv[6]),
    "n_tiles": summary["n_tiles"],
    "cameras": summary["cameras"],
    "license": manifest["license"],
}
with open(sys.argv[3], "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, sort_keys=True) + "\n")
print(f"registry: appended to {sys.argv[3]}")
EOF

if [ "$PUBLISH" = 1 ] && [ "$N_NEW" -gt 0 ]; then
    ./scripts/publish_shards.sh "$DEST"
else
    echo "next: review, then ./scripts/publish_shards.sh $DEST"
fi
echo "and commit the bookkeeping: git add $REGISTRY && git commit"
