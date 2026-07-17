#!/bin/sh
# End-to-end smoke test on synthetic data, no raw files or GPU needed (~2 min
# on CPU): fake shard -> dataset -> 30 training steps -> checkpoint -> export.
set -eu
cd "$(dirname "$0")/.."
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

python3.12 - "$OUT" <<'EOF'
import sys
from pathlib import Path
import numpy as np
from ansel_denoise.cfa import XTRANS, BAYER_RGGB

out = Path(sys.argv[1]) / "shards"
out.mkdir()
rng = np.random.default_rng(0)
for name, pattern, camera in [("bayer", BAYER_RGGB, "Acme Alpha1"),
                              ("xtrans", XTRANS, "Acme Xray9")]:
    yy, xx = np.mgrid[0:600, 0:600]
    sig = 0.3 + 0.25 * np.sin(xx / 13.0) * np.cos(yy / 19.0)
    adu = (np.clip(sig, 0, 1) * 15000 + 512).astype(np.uint16)
    np.savez_compressed(
        out / f"{name}.npz",
        tiles=np.stack([adu[i * 60 : i * 60 + 256, i * 60 : i * 60 + 256] for i in range(4)]),
        offsets=np.array([[i * 60, i * 60] for i in range(4)], dtype=np.int32),
        pattern=pattern, pattern4=pattern,
        black_per_channel=np.full(4, 512, np.float32), white=np.float32(15512),
        wb=np.ones(4, np.float32), iso=np.int32(100),
        camera=np.str_(camera), source_path=np.str_(f"synthetic/{name}"), annex_key=np.str_(""),
    )
print(f"synthetic shards in {out}")
EOF

python3.12 -m ansel_denoise.train --shards "$OUT/shards" --out "$OUT/run" \
    --steps 30 --batch 2 --patch 96 --depth 3 --base 8 --workers 0 \
    --val-every 30 --ckpt-every 30
python3.12 -m ansel_denoise.export "$OUT/run/ckpt-final.pt" --out "$OUT/run/final.anseldn"
echo "smoke test OK"
