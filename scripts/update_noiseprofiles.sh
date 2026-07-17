#!/bin/sh
# Refresh data/noiseprofiles.json from upstream darktable (superset of Ansel's
# copy: upstream keeps gaining newly profiled cameras). Records the source
# commit next to the file for the reproducibility chain.
set -eu

cd "$(dirname "$0")/.."
URL="https://raw.githubusercontent.com/darktable-org/darktable/master/data/noiseprofiles.json"
API="https://api.github.com/repos/darktable-org/darktable/commits?path=data/noiseprofiles.json&per_page=1"

curl -fsSL "$URL" -o data/noiseprofiles.json.new
python3.12 - <<'EOF'
from ansel_denoise.profiles import load_profiles
cams = load_profiles("data/noiseprofiles.json.new")
n = sum(len(c.isos) for c in cams)
print(f"validated: {len(cams)} cameras, {n} ISO profiles")
EOF
mv data/noiseprofiles.json.new data/noiseprofiles.json
curl -fsSL "$API" | python3.12 -c "
import json, sys
c = json.load(sys.stdin)[0]
print(f'{c[\"sha\"]} {c[\"commit\"][\"committer\"][\"date\"]}')" > data/noiseprofiles.upstream 2>/dev/null || true
cat data/noiseprofiles.upstream 2>/dev/null || true
