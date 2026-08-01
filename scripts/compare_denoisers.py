#!/usr/bin/env python3
"""Compare Ansel's classical `denoiseprofile` against the neural `rawdenoiseai`
on the SAME pictures, through the SAME real pipeline.

Method — the only way to compare a mosaic-domain denoiser with a
demosaiced-domain one honestly is to measure what comes out of the pipeline:

1. a real base-ISO raw is the clean reference (centre crop, CFA phase kept);
2. calibrated Poisson-Gaussian noise for a target ISO is synthesized into the
   mosaic, i.e. the noise the sensor would really have produced;
3. both the clean and the noisy mosaic are written as uncompressed CFA DNGs
   carrying the SOURCE camera's identity and the TARGET ISO, so every module
   that looks up a noise profile gets the right one;
4. each DNG is rendered by ansel-cli with an otherwise identical history, and
   PSNR is measured on the output image against the clean render.

Reported per condition: PSNR gain over the un-denoised noisy render — the
same "how far from noisy toward clean" quantity the model bench reports.

Noise conventions (deliberate, not a bug): synthesis uses the CALIBRATED
(a, b) — the true mosaic-domain magnitudes — while denoiseprofile's params
carry the SHIPPED profile values, exactly what its reload_defaults() writes
when a user enables it. Those differ by the calibration factor because the
profiles were fitted on demosaiced data, which is the domain denoiseprofile
works in. Each module therefore runs on its own correct assumption.

Usage:
    python3.12 scripts/compare_denoisers.py --raws A.NEF B.CR2 \
        --ansel-cli ../ansel/build/src/cli/ansel-cli --out bench/vs-denoiseprofile.json
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rawpy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from ansel_denoise import noise as noise_mod  # noqa: E402
from ansel_denoise.cfa import colors_map  # noqa: E402
from ansel_denoise.profiles import DEFAULT_SIGMA_CALIBRATION, load_profiles  # noqa: E402
from dngwriter import write_dng  # noqa: E402

# a default (all-off) blend params blob, module-agnostic
BLENDOP = "gz12eJxjYGBgkGAAgRNODESDBnsIHll8ANNSGQM="
DP_BANDS, DP_CHANNELS = 7, 6


def dp_params(a: np.ndarray, b: np.ndarray, strength: float = 1.0) -> bytes:
    """denoiseprofile params blob, replicating init() + reload_defaults():
    the autoset fields are inferred from the profile's green a, the rest are
    the $DEFAULT annotations. Mirrors src/iop/denoiseprofile.c."""
    ag = float(a[1])
    radius = float(min(int(1.0 + ag * 15000.0 + ag * ag * 300000.0), 8))
    scattering = min(3000.0 * ag, 1.0)
    shadows = min(max(0.1 - 0.1 * math.log(ag), 0.7), 1.8)
    bias = -max(5 + 0.5 * math.log(ag), 0.0)
    vals = [radius, 7.0, strength, shadows, bias, scattering, 0.1, 1.0]
    out = b"".join(struct.pack("<f", v) for v in vals)
    out += b"".join(struct.pack("<f", float(v)) for v in a)
    out += b"".join(struct.pack("<f", float(v)) for v in b)
    out += struct.pack("<i", 1)  # mode = MODE_WAVELETS
    for _ in range(DP_CHANNELS):  # x[ch][k] = k / (BANDS - 1)
        out += b"".join(struct.pack("<f", k / (DP_BANDS - 1.0)) for k in range(DP_BANDS))
    out += b"".join(struct.pack("<f", 0.5) for _ in range(DP_CHANNELS * DP_BANDS))
    out += struct.pack("<iiii", 1, 1, 1, 1)  # wb_adaptive, fix_anscombe, new_vst, Y0U0V0
    return out


def ai_params(size: int, scale: int, strength: float = 1.0) -> bytes:
    """rawdenoiseai params: strength, version, size, global correction, the
    three per-channel corrections (the shipped calibration), variant."""
    return (struct.pack("<f", strength) + struct.pack("<ii", 0, size)
            + struct.pack("<ffff", 1.0, 2.82, 3.94, 2.96) + struct.pack("<i", scale))


def write_xmp(path: Path, entries) -> Path:
    """entries: (operation, modversion, params blob[, enabled]). Every render
    of a comparison MUST use the same history structure — an empty history is
    not equivalent to a disabled module, so the reference carries the same
    entry with enabled=0 rather than no entry at all."""
    entries = [(e + (1,))[:4] for e in entries]
    li = "".join(
        f'''
     <rdf:li
      darktable:operation="{op}"
      darktable:enabled="{en}"
      darktable:modversion="{ver}"
      darktable:params="{blob.hex()}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="7"
      darktable:blendop_params="{BLENDOP}"/>''' for op, ver, blob, en in entries)
    path.write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 4.4.0-Exiv2">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:darktable="http://darktable.sf.net/"
   darktable:xmp_version="2"
   darktable:auto_presets_applied="1"
   darktable:history_end="{len(entries)}">
   <darktable:history>
    <rdf:Seq>{li}
    </rdf:Seq>
   </darktable:history>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
''')
    return path


def render(cli: str, raw: Path, xmp: Path, out: Path, work: Path, opts: dict) -> np.ndarray:
    from PIL import Image
    if out.exists():
        out.unlink()
    db = work / f"lib-{out.stem}.db"
    cmd = [cli, str(raw), str(xmp), str(out), "--core", "--disable-opencl",
           "--configdir", opts["configdir"], "--library", str(db),
           "--moduledir", opts["moduledir"], "--datadir", opts["datadir"]]
    env = dict(opts["env"])
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if not out.exists():
        raise SystemExit(f"render failed for {out.name}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    img = np.asarray(Image.open(out))
    # normalize by the actual container depth: ansel-cli writes 8- or 16-bit
    # TIFF depending on plugins/imageio/format/tiff/bpp
    return img.astype(np.float64) / (65535.0 if img.dtype == np.uint16 else 255.0)


def psnr(x: np.ndarray, ref: np.ndarray) -> float:
    mse = float(np.mean((x - ref) ** 2))
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--raws", nargs="+", required=True, help="base-ISO source raws")
    ap.add_argument("--isos", nargs="+", type=float, default=[3200.0, 12800.0, 51200.0])
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--ansel-cli", required=True)
    ap.add_argument("--configdir", required=True)
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--moduledir", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--keep", type=Path, help="keep the generated DNGs/renders here")
    args = ap.parse_args()

    cams = {c.name: c for c in load_profiles()}
    cal2 = np.asarray(DEFAULT_SIGMA_CALIBRATION, dtype=np.float64) ** 2
    opts = {"configdir": args.configdir, "datadir": args.datadir,
            "moduledir": args.moduledir,
            "env": {"LD_LIBRARY_PATH": str(Path(args.ansel_cli).parents[1]), "LANG": "C",
                    "HOME": str(Path.home()), "PATH": "/usr/bin:/bin"}}
    # denoiseprofile is measured twice: at its own defaults (what a user gets
    # by enabling it) and at 200 % strength, which compensates the factor-2
    # understatement the shipped profiles carry — its fairest showing.
    variants = [("denoiseprofile", [("denoiseprofile", 11, None)]),
                ("denoiseprofile 200%", [("denoiseprofile", 11, "strength2")]),
                ("ai large-multi", [("rawdenoiseai", 1, ai_params(0, 1))]),
                ("ai large-single", [("rawdenoiseai", 1, ai_params(0, 0))]),
                ("ai half-multi", [("rawdenoiseai", 1, ai_params(1, 1))]),
                ("ai half-single", [("rawdenoiseai", 1, ai_params(1, 0))])]

    results = []
    tmp = Path(tempfile.mkdtemp(prefix="cmpdn-")) if not args.keep else args.keep
    tmp.mkdir(parents=True, exist_ok=True)
    for raw_path in args.raws:
        raw_path = Path(raw_path)
        with rawpy.imread(str(raw_path)) as r:
            mosaic = r.raw_image_visible.astype(np.float32)
            black = float(np.mean(r.black_level_per_channel))
            white = float(r.white_level)
            pattern = r.raw_pattern.copy()
        cam = _match_profile(cams, raw_path)
        if cam is None:
            print(f"{raw_path.name}: no noise profile, skipped", flush=True)
            continue
        # Pick the best-exposed crop, on an even offset so the CFA phase is
        # preserved. The reference is a REAL base-ISO capture and therefore
        # carries its own noise; in a well-exposed region that noise is far
        # below the injected high-ISO noise, but in deep shadows it is not —
        # a dark crop makes every denoiser look bad, because it also removes
        # the reference's own grain and PSNR counts that as error.
        h, w = mosaic.shape
        n = min(args.crop, h - h % 2, w - w % 2)
        best, crop = -1.0, None
        for fy in np.linspace(0, h - n, 5):
            for fx in np.linspace(0, w - n, 5):
                oy, ox = int(fy) & ~1, int(fx) & ~1
                cand = mosaic[oy:oy + n, ox:ox + n]
                norm = (cand - black) / (white - black)
                if float(np.mean(norm > 0.98)) > 0.01:  # skip blown regions
                    continue
                m = float(np.mean(norm))
                if m > best:
                    best, crop = m, cand
        if crop is None:
            crop = mosaic[:n, :n]
        print(f"{raw_path.name}: crop mean signal {best:.3f} of white", flush=True)
        clean01 = np.clip((crop - black) / (white - black), 0.0, 1.0)
        colors = colors_map(pattern, n, n, 0, 0)
        # CFA codes for the DNG tag, read off the (possibly rolled) crop
        cfa = tuple(int(colors[i // 2, i % 2]) for i in range(4))

        clean_dng = write_dng(tmp / f"{raw_path.stem}-clean.dng",
                              np.rint(clean01 * 65535).astype(np.uint16), 0, 65535,
                              cfa_pattern=cfa, make=cam.maker, model=cam.model, iso=100)
        # same structure as every other render, module switched off
        ref_xmp = write_xmp(tmp / "off.xmp",
                            [("rawdenoiseai", 1, ai_params(1, 0), 0)])
        ref = render(args.ansel_cli, clean_dng, ref_xmp, tmp / f"{raw_path.stem}-ref.tif",
                     tmp, opts)

        for iso in args.isos:
            # a profile clamps above its highest measured ISO, which would
            # silently duplicate the ceiling row instead of testing that ISO
            if iso > cam.isos[-1].iso * 1.01:
                print(f"  ISO {iso:.0f} above {cam.name}'s profiled max "
                      f"({cam.isos[-1].iso:.0f}) — skipped", flush=True)
                continue
            prof = cam.interpolate(iso)
            rng = np.random.default_rng((20260801, int(iso), abs(hash(raw_path.name)) % 9973))
            noisy01 = noise_mod.synthesize(clean01, colors, prof.a * cal2, prof.b * cal2,
                                           rng, black_frac=0.0)
            tag = f"{raw_path.stem}-{int(iso)}"
            noisy_dng = write_dng(tmp / f"{tag}-noisy.dng",
                                  np.rint(np.clip(noisy01, 0, 1) * 65535).astype(np.uint16),
                                  0, 65535, cfa_pattern=cfa, make=cam.maker,
                                  model=cam.model, iso=int(iso))
            base = render(args.ansel_cli, noisy_dng, ref_xmp, tmp / f"{tag}-noisy.tif", tmp, opts)
            row = {"picture": raw_path.name, "camera": cam.name, "iso": iso,
                   "psnr_noisy": round(psnr(base, ref), 2)}
            for label, entries in variants:
                built = [(op, ver,
                          dp_params(prof.a, prof.b) if blob is None
                          else (dp_params(prof.a, prof.b, 2.0) if blob == "strength2" else blob),
                          1) for op, ver, blob in entries]
                xmp = write_xmp(tmp / f"{tag}-{label.replace(' ', '_')}.xmp", built)
                img = render(args.ansel_cli, noisy_dng, xmp,
                             tmp / f"{tag}-{label.replace(' ', '_')}.tif", tmp, opts)
                row[label] = round(psnr(img, ref), 2)
                row[f"gain {label}"] = round(row[label] - row["psnr_noisy"], 2)
            results.append(row)
            gains = "  ".join(f"{k.replace('gain ', '')} {v:+.2f}"
                              for k, v in row.items() if k.startswith("gain "))
            print(f"{cam.name} ISO {iso:.0f} [{raw_path.stem}]: noisy "
                  f"{row['psnr_noisy']:.2f} dB | {gains}", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"protocol": "full-pipeline, synthetic calibrated noise on real base-ISO raws",
             "sigma_calibration": list(DEFAULT_SIGMA_CALIBRATION),
             "results": results}, indent=1))
        print(f"wrote {args.out}")
    return 0


def _match_profile(cams: dict, raw_path: Path):
    """Find the noise profile of the camera that took raw_path (exiftool
    identity, matched on the model part of the profile name)."""
    r = subprocess.run(["exiftool", "-s", "-s", "-s", "-Model", str(raw_path)],
                       capture_output=True, text=True)
    model = r.stdout.strip().upper()
    if not model:
        return None
    best = None
    for c in cams.values():
        m = c.model.upper()
        if m == model or m in model or model in m:
            if best is None or len(c.model) > len(best.model):
                best = c
    return best


if __name__ == "__main__":
    sys.exit(main())
