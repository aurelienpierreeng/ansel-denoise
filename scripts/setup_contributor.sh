#!/bin/sh
# One-shot setup of everything a contributor needs to harvest and pack
# training shards for the Ansel raw denoiser. Linux and macOS (on Windows,
# run it inside WSL or Git Bash).
#
# Run it from anywhere:
#   - inside a clone of ansel-denoise: installs the Python tooling;
#   - outside: clones the repo into ./ansel-denoise first.
#
#   sh scripts/setup_contributor.sh
set -eu

REPO_URL="https://github.com/aurelienpierreeng/ansel-denoise.git"

# --- find a Python >= 3.10 -------------------------------------------------
PY=""
for CAND in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$CAND" >/dev/null 2>&1 \
        && "$CAND" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
        PY="$CAND"
        break
    fi
done
[ -n "$PY" ] || { echo "ERROR: Python >= 3.10 not found. Install it first (https://python.org)." >&2; exit 1; }
echo "using $PY ($("$PY" --version 2>&1))"

# --- find or clone the repo ------------------------------------------------
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd || true)
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/../pyproject.toml" ]; then
    REPO_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
elif [ -f "./ansel-denoise/pyproject.toml" ]; then
    REPO_DIR="$(pwd)/ansel-denoise"
else
    command -v git >/dev/null || { echo "ERROR: git required to clone $REPO_URL" >&2; exit 1; }
    git clone --depth 1 "$REPO_URL" ansel-denoise
    REPO_DIR="$(pwd)/ansel-denoise"
fi
echo "repo: $REPO_DIR"

# --- install the Python tooling -------------------------------------------
# harvest extra = numpy + rawpy (libraw decode); exiftool is NOT needed for
# the library-based harvest (metadata comes from Ansel's library.db).
"$PY" -m pip install --user -e "$REPO_DIR[harvest]" \
    || { echo "ERROR: pip install failed. On Debian/Ubuntu try: apt install python3-pip" >&2; exit 1; }

# --- smoke check -----------------------------------------------------------
"$PY" - <<EOF
import numpy, rawpy
import ansel_denoise.harvest_library, ansel_denoise.validate_shards
print("tooling OK: numpy", numpy.__version__, "| rawpy", rawpy.__version__)
EOF

cat <<EOF

Setup complete. Next steps (details in $REPO_DIR/CONTRIBUTING.md):
  1. In Ansel, select images (base ISO, content you agree to publish),
     then File > Export image list... > Save as file...
  2. $PY -m ansel_denoise.harvest_library --paths-file ansel-image-files.txt --out shards/mine
  3. $PY $REPO_DIR/scripts/pack_contribution.py shards/mine --handle your-name
  4. Upload the bundle to any file host, then submit it (opens the pull
     request for you, no git knowledge needed; requires the gh CLI):
     sh $REPO_DIR/scripts/submit_contribution.sh <bundle.tar.gz> --url <link>
EOF
command -v gh >/dev/null \
    || echo "note: install the GitHub CLI (https://cli.github.com) before step 4."
