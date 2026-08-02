#!/usr/bin/env python3
"""Wall-clock cost of each denoising option, CPU and GPU, on a real raw.

Timings come from Ansel's own per-module instrumentation (`-d perf`), which
reports each module's processing time and the device it ran on, so the
numbers are the module's real cost rather than a difference of whole-export
wall clocks. The device attribution is kept and checked: an OpenCL run that
silently fell back to CPU would otherwise be reported as a fast GPU run.

Each configuration is run `--repeat` times and the MINIMUM is kept: the
minimum is the run least disturbed by other activity on the machine, which
is what a "how expensive is this module" number should reflect. A warm-up
run precedes the measurements so OpenCL kernel compilation and the page
cache never land inside a timed run.

Usage:
    python3.12 scripts/bench_runtime.py --raw photo.ARW \
        --ansel-cli ../ansel/build/src/cli/ansel-cli \
        --configdir <scratch>/config --datadir <scratch>/datadir \
        --moduledir <scratch>/moduledir --out bench/runtime.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from compare_denoisers import ai_params, dp_params, write_xmp  # noqa: E402

# denoiseprofile is timed at the settings the quality sweep actually chose
# (scripts/compare_denoisers.py), not at its defaults — comparing a tuned
# module's quality against an untuned module's cost would be meaningless.
# The (a, b) must be the REAL profile of the timed camera and ISO: the patch
# size, search radius and scattering are inferred from a, and they dominate
# the non-local-means cost.
DP_CAMERA, DP_ISO = "Sony ILCE-6000", 12800.0
sys.path.insert(0, str(REPO / "src"))
from ansel_denoise.profiles import load_profiles  # noqa: E402

_prof = next(c for c in load_profiles() if c.name == DP_CAMERA).interpolate(DP_ISO)
DP_A, DP_B = _prof.a, _prof.b

CONFIGS = [
    ("off", [("rawdenoiseai", 1, ai_params(1, 0), 0)]),
    ("ai large-single", [("rawdenoiseai", 1, ai_params(0, 0), 1)]),
    ("ai large-multi", [("rawdenoiseai", 1, ai_params(0, 1), 1)]),
    ("ai half-single", [("rawdenoiseai", 1, ai_params(1, 0), 1)]),
    ("ai half-multi", [("rawdenoiseai", 1, ai_params(1, 1), 1)]),
    # the sweep's optima: non-local means/Y0U0V0 won 5 of 6 conditions,
    # wavelets/RGB won the remaining one; strength 200 % for this camera
    ("denoiseprofile non-local means (swept optimum)",
     [("denoiseprofile", 11, dp_params(DP_A, DP_B, 2.0, 0, 1), 1)]),
    ("denoiseprofile wavelets (swept optimum)",
     [("denoiseprofile", 11, dp_params(DP_A, DP_B, 2.0, 1, 0), 1)]),
    ("denoiseprofile wavelets (defaults)",
     [("denoiseprofile", 11, dp_params(DP_A, DP_B, 1.0, 1, 1), 1)]),
]


PERF_RE = re.compile(
    r"took ([0-9.]+) secs \([0-9.]+ CPU\) processed `([^']+)' on (GPU|CPU)([^\[]*)\[([^\]]+)\]")


def time_render(cli: str, raw: Path, xmp: Path, work: Path, opts: dict, gpu: bool):
    """Run one export with -d perf; return {module label: (secs, device, notes)}
    for the export pipe."""
    out = work / "out.tif"
    if out.exists():
        out.unlink()
    db = work / "timing.db"
    for suffix in ("", "-lock", "-wal", "-shm", "-pre-0.0.0"):
        q = Path(str(db) + suffix)
        if q.exists():
            q.unlink()
    cmd = [cli, str(raw), str(xmp), str(out), "--core", "-d", "perf",
           "--configdir", opts["configdir"], "--library", str(db),
           "--moduledir", opts["moduledir"], "--datadir", opts["datadir"]]
    if not gpu:
        # everything after --core is a core option; inserted before it,
        # ansel-cli rejects it as an unknown option of its own
        cmd.append("--disable-opencl")
    r = subprocess.run(cmd, capture_output=True, text=True, env=opts["env"], check=False)
    if not out.exists():
        raise SystemExit(f"render failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    per_module = {}
    for secs, label, device, notes, pipe in PERF_RE.findall(r.stdout + r.stderr):
        if "export" not in pipe:
            continue
        prev = per_module.get(label)
        # a tiled module is reported once per call; sum them
        per_module[label] = ((prev[0] if prev else 0.0) + float(secs), device, notes.strip())
    return per_module


def denoise_cost(per_module: dict):
    """Total seconds spent in denoising modules, plus their device."""
    hits = {k: v for k, v in per_module.items()
            if "denoise" in k.lower() or "noise" in k.lower()}
    if not hits:
        return 0.0, "-", ""
    secs = sum(v[0] for v in hits.values())
    dev = ",".join(sorted({v[1] for v in hits.values()}))
    notes = ";".join(sorted({v[2] for v in hits.values() if v[2]}))
    return secs, dev, notes


def _crop_dng(raw: Path, n: int, work: Path) -> Path:
    """Write a centred n x n sensor crop of `raw` as a DNG, keeping the CFA
    phase and the camera identity so profile lookups still match."""
    import numpy as np
    import rawpy

    from dngwriter import write_dng

    with rawpy.imread(str(raw)) as r:
        m = r.raw_image_visible
        oy, ox = ((m.shape[0] - n) // 2) & ~1, ((m.shape[1] - n) // 2) & ~1
        crop = np.ascontiguousarray(m[oy:oy + n, ox:ox + n])
        pat = r.raw_pattern
        black, white = int(np.mean(r.black_level_per_channel)), int(r.white_level)
    cfa = tuple(int(pat[(i // 2) % pat.shape[0], (i % 2) % pat.shape[1]]) for i in range(4))
    ident = subprocess.run(["exiftool", "-s", "-s", "-s", "-Make", "-Model", str(raw)],
                           capture_output=True, text=True, check=False).stdout.split("\n")
    make = ident[0].strip() if ident and ident[0].strip() else "Synthetic"
    model = ident[1].strip() if len(ident) > 1 and ident[1].strip() else "Synthetic"
    out = work / "bench-crop.dng"
    write_dng(out, crop, black, white, cfa_pattern=cfa, make=make, model=model,
              iso=int(DP_ISO))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--ansel-cli", required=True)
    ap.add_argument("--configdir", required=True)
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--moduledir", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--crop", type=int, default=0,
                    help="time on a centred NxN SENSOR crop instead of the whole raw "
                         "(0 = whole raw). Cropping the sensor image scales both module "
                         "families proportionally; shrinking the EXPORT would not, because "
                         "rawdenoiseai runs before demosaic and always at full sensor "
                         "resolution while denoiseprofile runs after it.")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    opts = {"configdir": args.configdir, "datadir": args.datadir,
            "moduledir": args.moduledir,
            "env": {"LD_LIBRARY_PATH": str(Path(args.ansel_cli).parents[1]), "LANG": "C",
                    "HOME": str(Path.home()), "PATH": "/usr/bin:/bin"}}
    work = Path(tempfile.mkdtemp(prefix="rtbench-"))
    raw = Path(args.raw)
    if args.crop:
        raw = _crop_dng(raw, args.crop, work)
        print(f"timing on a {args.crop}x{args.crop} sensor crop -> {raw}", flush=True)

    results = {}
    for device, gpu in (("cpu", False), ("gpu", True)):
        print(f"=== {device.upper()} ===", flush=True)
        rows = {}
        for label, entries in CONFIGS:
            if label == "off":
                continue
            xmp = write_xmp(work / f"{label.replace(' ', '_')}.xmp", entries)
            time_render(args.ansel_cli, raw, xmp, work, opts, gpu)  # warm-up
            runs, dev, notes = [], "-", ""
            for _ in range(args.repeat):
                secs, dev, notes = denoise_cost(
                    time_render(args.ansel_cli, raw, xmp, work, opts, gpu))
                runs.append(secs)
            rows[label] = {"secs": min(runs), "runs": [round(x, 3) for x in runs],
                           "device": dev, "notes": notes}
            print(f"  {label:<32} {min(runs):7.3f} s  on {dev:<3} {notes}"
                  + "  (" + ", ".join(f"{x:.3f}" for x in runs) + ")", flush=True)
        if gpu:
            on_cpu = [k for k, v in rows.items() if v["device"] == "CPU"]
            if on_cpu:
                print("  WARNING: OpenCL did not engage for: " + ", ".join(on_cpu)
                      + "\n           these are CPU timings, not GPU ones — check that the"
                      "\n           config dir has a working OpenCL setup.", flush=True)
        ref = rows["ai half-single"]["secs"]
        print(f"  -- relative, x1 = ai half-single ({ref:.3f} s) --")
        for k, v in rows.items():
            v["relative"] = round(v["secs"] / ref, 2)
            print(f"  {k:<32} x{v['relative']:.2f}", flush=True)
        results[device] = rows

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"raw": str(raw), "repeat": args.repeat, "results": results}, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
