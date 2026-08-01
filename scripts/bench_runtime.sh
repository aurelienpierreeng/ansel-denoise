#!/bin/sh
# Runtime benchmark of every denoising option in Ansel, CPU and GPU.
#
# Wraps scripts/bench_runtime.py with the setup the measurement needs: an
# isolated config/library so the run cannot touch your own Ansel settings or
# collection, and the model files where the module looks for them.
#
# Usage:
#   ./scripts/bench_runtime.sh <raw-file> [options]
#     --ansel-cli PATH   ansel-cli binary   (default: /opt/ansel/bin/ansel-cli — see below)
#     --datadir DIR      Ansel data dir     (default: /opt/ansel/share/ansel)
#     --moduledir DIR    IOP plugins        (default: <prefix>/lib/ansel/plugins)
#     --repeat N         timed runs kept per configuration, min wins (default 3)
#     --crop N           time on an N x N sensor crop instead of the whole raw
#     --out FILE         results            (default: bench/runtime.json)
#
# Expect roughly 1.5-3 h on a full 24 MP raw with --repeat 3: 7 configurations
# x (1 warm-up + N runs) x 2 devices, and non-local means at high ISO is slow
# by itself. Pass --crop 2048 for a ~10 min sanity run; relative costs hold,
# absolute seconds obviously do not.
#
# The machine should be otherwise idle: the script keeps the MINIMUM of the
# timed runs, which absorbs brief interference but not a busy desktop.
set -eu

cd "$(dirname "$0")/.."

RAW="${1:?usage: bench_runtime.sh <raw-file> [--ansel-cli PATH] [--repeat N] [--crop N]}"
shift

# Default to the INSTALLED ansel-cli, not a build-tree one: ansel-cli resolves
# the OpenCL kernel directory from its own location, NOT from --datadir, and a
# build tree does not stage the kernel headers. A build-tree binary therefore
# fails to compile the kernels and silently runs the whole pipeline on the CPU,
# which would turn the GPU half of this benchmark into a second CPU run.
# `sudo ninja install` first, so the installed binary is the one you mean.
ANSEL_CLI="/opt/ansel/bin/ansel-cli"
DATADIR="/opt/ansel/share/ansel"   # must match the ansel-cli you use
MODULEDIR=""
REPEAT=3
CROP=0
OUT="bench/runtime.json"
while [ $# -gt 0 ]; do
    case "$1" in
        --ansel-cli) ANSEL_CLI="$2"; shift 2 ;;
        --datadir)   DATADIR="$2"; shift 2 ;;
        --moduledir) MODULEDIR="$2"; shift 2 ;;
        --repeat)    REPEAT="$2"; shift 2 ;;
        --crop)      CROP="$2"; shift 2 ;;
        --out)       OUT="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

[ -f "$RAW" ] || { echo "no such raw: $RAW" >&2; exit 1; }
[ -x "$ANSEL_CLI" ] || { echo "no ansel-cli at $ANSEL_CLI (pass --ansel-cli)" >&2; exit 1; }
[ -d "$DATADIR" ] || { echo "no datadir at $DATADIR (pass --datadir)" >&2; exit 1; }
if [ -z "$MODULEDIR" ]; then
    # <prefix>/bin/ansel-cli -> <prefix>/lib/ansel/plugins
    MODULEDIR="$(dirname "$(dirname "$ANSEL_CLI")")/lib/ansel/plugins"
fi
[ -d "$MODULEDIR" ] || { echo "no moduledir at $MODULEDIR (pass --moduledir)" >&2; exit 1; }

PY="${PYTHON:-python3.12}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
CONFIGDIR="$WORK/config"
mkdir -p "$CONFIGDIR"

# Seed the isolated config from the user's real anselrc when there is one.
# A *fresh* config dir cannot initialise OpenCL — it has no per-device
# configuration and resolves the kernel directory relative to the binary
# instead of --datadir — so the GPU half of the benchmark would silently run
# on the CPU. The library/collection stays isolated either way: only anselrc
# is copied, and --library points into the temp dir.
USER_RC="${XDG_CONFIG_HOME:-$HOME/.config}/ansel/anselrc"
if [ -f "$USER_RC" ]; then
    cp "$USER_RC" "$CONFIGDIR/anselrc"
    echo "seeded config from $USER_RC"
else
    echo "WARNING: no anselrc at $USER_RC — OpenCL may not initialise and the" >&2
    echo "         GPU results would then be CPU results. Check the 'on GPU/CPU'" >&2
    echo "         column in the output." >&2
fi

# The module loads weights from <configdir> before <datadir>, so a checkout's
# models/ can be benchmarked without installing them system-wide.
for f in models/*.anselnn; do
    [ -e "$f" ] && cp "$f" "$CONFIGDIR/"
done

echo "raw:       $RAW"
echo "ansel-cli: $ANSEL_CLI"
echo "datadir:   $DATADIR"
echo "moduledir: $MODULEDIR"
echo "models:    $(ls "$CONFIGDIR"/*.anselnn 2>/dev/null | wc -l) in the isolated config dir"
echo "repeat:    $REPEAT   crop: $CROP"
echo

# -u so progress streams while it runs instead of appearing at the end
exec "$PY" -u scripts/bench_runtime.py \
    --raw "$RAW" \
    --ansel-cli "$ANSEL_CLI" \
    --configdir "$CONFIGDIR" \
    --datadir "$DATADIR" \
    --moduledir "$MODULEDIR" \
    --repeat "$REPEAT" \
    --crop "$CROP" \
    --out "$OUT"
