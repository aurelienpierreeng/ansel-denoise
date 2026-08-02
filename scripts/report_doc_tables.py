#!/usr/bin/env python3
"""Print every number the user documentation quotes, from the committed bench
files — so the published tables can be re-derived and checked at any time.

Chain of custody for each table in
`content/views/darkroom/modules/ai-raw-denoise.md`:

  model quality      speckle_bench.py  -> bench/m760-<size>-<variant>.json
  chroma (LFCE)      same files (lfce_16 column)
  vs denoiseprofile  compare_denoisers.py -> bench/vs-denoiseprofile.json
  processing cost    bench_runtime.py     -> bench/runtime.json

Usage: python3.12 scripts/report_doc_tables.py [--bench bench]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MATRIX = [("large", "single"), ("large", "multi"), ("half", "single"), ("half", "multi")]


def _tiles(path: Path, hi_only: bool = False):
    rows = [r for r in json.loads(path.read_text())["results"] if r["set"] == "tiles"]
    if hi_only:
        rows = [r for r in rows if r["iso"] > 12000]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--bench", type=Path, default=REPO / "bench")
    args = ap.parse_args()

    print("## Model quality (PSNR gain over the noisy input, held-out cameras)\n")
    for label, hi in (("all ISO", False), ("ISO > 12000", True)):
        print(f"| PSNR gain (dB), {label} | single-scale | multiscale |")
        print("| --- | --- | --- |")
        cells = {}
        for size, variant in MATRIX:
            f = args.bench / f"m760-{size}-{variant}.json"
            if not f.exists():
                cells[(size, variant)] = None
                continue
            rows = _tiles(f, hi)
            cells[(size, variant)] = np.mean([r["psnr"] - r["psnr_noisy"] for r in rows])
        for size in ("large", "half"):
            vals = [cells[(size, v)] for v in ("single", "multi")]
            print(f"| {size} | " + " | ".join(
                f"{v:+.1f}" if v is not None else "*(in training)*" for v in vals) + " |")
        print("| quarter | *(in training)* | *(in training)* |\n")

    print("## Chroma advantage of multiscale (LFCE at bin 16, lower is better)\n")
    for label, hi in (("all ISO", False), ("ISO > 12000", True)):
        for size in ("large", "half"):
            got = {}
            for variant in ("single", "multi"):
                f = args.bench / f"m760-{size}-{variant}.json"
                if f.exists():
                    got[variant] = np.mean([r["lfce_16"] for r in _tiles(f, hi)])
            if len(got) == 2:
                print(f"  {size:<6} {label:<12} single {got['single']:.1f} dB, "
                      f"multi {got['multi']:.1f} dB  -> multiscale better by "
                      f"{got['single'] - got['multi']:.1f} dB")
    print()

    vs = args.bench / "vs-denoiseprofile.json"
    if vs.exists():
        res = json.loads(vs.read_text())["results"]
        isos = sorted({r["iso"] for r in res})
        keys = [("denoise (profiled), default settings", "gain denoiseprofile defaults"),
                ("denoise (profiled), best per-image settings", "gain denoiseprofile"),
                ("AI, half single-scale (the default)", "gain ai half-single"),
                ("AI, large multiscale", "gain ai large-multi")]
        print("## Against denoise (profiled), full pipeline, "
              f"{len({r['picture'] for r in res})} pictures\n")
        print("| PSNR gain (dB) | " + " | ".join(f"ISO {i:.0f}" for i in isos) + " |")
        print("| --- | " + " | ".join("---" for _ in isos) + " |")
        for label, key in keys:
            cells = [np.mean([r[key] for r in res if r["iso"] == i]) for i in isos]
            print(f"| {label} | " + " | ".join(f"{c:+.1f}" for c in cells) + " |")
        print("\n  denoiseprofile optimum per condition (the settings the sweep found):")
        modes = {0: "non-local means", 1: "wavelets"}
        cmodes = {0: "RGB", 1: "Y0U0V0"}
        for r in res:
            b = r["denoiseprofile best params"]
            print(f"    {r['camera']:<22} ISO {r['iso']:>6.0f}  {modes[b['mode']]:<15} "
                  f"{cmodes[b['color_mode']]:<7} strength {b['strength'] * 100:.0f} %")
        n_grid = len(res[0]["denoiseprofile sweep"])
        print(f"\n  sweep size: {n_grid} settings per picture and ISO, "
              f"{n_grid * len(res)} renders total")

    rt = args.bench / "runtime.json"
    if rt.exists():
        data = json.loads(rt.read_text())
        res = data["results"]
        rows = list(res["cpu"].keys())
        print(f"\n## Processing cost ({Path(data['raw']).name}, "
              f"min of {data['repeat']} runs)\n")
        print("| relative cost | CPU | GPU |")
        print("| --- | --- | --- |")
        for k in rows:
            cells = []
            for dev in ("cpu", "gpu"):
                v = res[dev].get(k)
                ref = res[dev]["ai half-single"]["secs"]
                cells.append(f"x{v['secs'] / ref:.2g}" if v else "n/a")
            print(f"| {k} | " + " | ".join(cells) + " |")
        print("\n  absolute seconds and the device each module reported:")
        for dev in ("cpu", "gpu"):
            for k, v in res[dev].items():
                flag = "  <-- OpenCL did NOT engage" if dev == "gpu" and v["device"] == "CPU" else ""
                print(f"    {dev} {k:<46} {v['secs']:7.2f} s  [{v['device']}]{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
