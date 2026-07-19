"""Harvest shards from a hand-picked list of images (Ansel library optional).

The Ansel lighttable is the curation UI: select the images you are willing to
feed the training (safe content, base ISO) and pass them here — either as
image IDs (--ids / --ids-file, which need the library database) or as file
paths (positional arguments or --paths-file, as produced by Ansel's
"File > Export image list..." dialog). **Paths do not need the library at
all**: when library.db is absent or does not know a path, camera and ISO are
read from the file itself (TIFF tags + libraw; exiftool fills gaps when
installed). Everything funnels through the same crash-isolated decode ->
tile pipeline as every other source.

PRIVACY CONTRACT: the list is the curation. Tiles are viewable fragments
of your photographs and the default assumption is that they will be PUBLISHED
to the public shard release — list only images whose content you are willing
to make public. For a local/private-only harvest instead, pass --private:
the output directory then gets a `.private` marker that publish_shards.sh
hard-refuses to upload.

Usage:
    python -m ansel_denoise.harvest_library --ids 65345,65350-65360 --out shards/library
    python -m ansel_denoise.harvest_library --ids-file keepers.txt --out shards/library
    python -m ansel_denoise.harvest_library --ids-file all.txt --out shards/personal --private
    python -m ansel_denoise.harvest_library --out shards/library '/photos/2024/IMG 1.NEF' ...
    python -m ansel_denoise.harvest_library --paths-file ansel-image-files.txt --out shards/library
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import struct
import sys
from pathlib import Path

from .harvest import EXCLUDE_EXTENSIONS, RAW_EXTENSIONS, _pack_worker, read_exif, run_isolated


def default_db() -> Path:
    """Ansel's library.db location: g_get_user_config_dir()/ansel/library.db,
    which GLib resolves to %LOCALAPPDATA% on Windows and ~/.config elsewhere."""
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA",
                                   Path.home() / "AppData" / "Local")) / "ansel" / "library.db"
    return Path.home() / ".config" / "ansel" / "library.db"


DEFAULT_DB = default_db()

_QUERY = """SELECT i.id, f.folder || '/' || i.filename, i.maker, i.model, i.iso, i.filename
            FROM images i JOIN film_rolls f ON i.film_id = f.id"""


def parse_ids(spec: str) -> list[int]:
    """'12,15,100-103' -> [12, 15, 100, 101, 102, 103]"""
    ids: list[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def _entries(rows: list) -> list[dict]:
    images, seen_paths = [], set()
    for iid, path, maker, model, iso, filename in sorted(rows):
        if path in seen_paths:  # darktable-style duplicates share the file
            continue
        seen_paths.add(path)
        images.append({"id": iid, "path": path, "filename": filename,
                       "rel": f"library/{iid}/{filename}",
                       "camera": f"{(maker or '?').strip()} {(model or '?').strip()}",
                       "iso": iso})
    return images


def resolve_images(db_path: Path, ids: list[int]) -> list[dict]:
    """Resolve image IDs to paths + metadata via library.db, read-only."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        marks = ",".join("?" * len(ids))
        rows = db.execute(f"{_QUERY} WHERE i.id IN ({marks}) ORDER BY i.id", ids).fetchall()
    finally:
        db.close()
    images = _entries(rows)
    for missing in sorted(set(ids) - {r[0] for r in rows}):
        images.append({"id": missing, "path": None, "filename": None,
                       "rel": f"library/{missing}/None", "camera": None, "iso": None})
    return images


def resolve_paths_db(db_path: Path, paths: list[str]) -> tuple[list[dict], list[str]]:
    """Match file paths against library.db; returns (entries, unmatched paths).
    Unmatched paths are NOT an error — they fall back to file metadata."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows, matched = [], set()
    try:
        for arg in paths:
            # as given first, then resolved (symlinks, relative paths)
            for candidate in dict.fromkeys([arg, str(Path(arg).resolve())]):
                row = db.execute(f"{_QUERY} WHERE f.folder || '/' || i.filename = ?",
                                 (candidate,)).fetchone()
                if row:
                    rows.append(row)
                    matched.add(arg)
                    break
    finally:
        db.close()
    return _entries(rows), [p for p in paths if p not in matched]


def _tiff_camera(path: Path) -> str | None:
    """Make + Model from a raw file's TIFF IFD0, stdlib only. Covers the
    TIFF-based containers (NEF/CR2/ARW/DNG/PEF/ORF/RW2...); Fuji RAF carries
    the model in its own header instead."""
    try:
        with open(path, "rb") as f:
            data = f.read(65536)
    except OSError:
        return None
    if data[:8] == b"FUJIFILM":
        model = data[0x1C:0x3C].split(b"\0")[0].decode("ascii", "replace").strip()
        return f"FUJIFILM {model}" if model else None
    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        return None

    def u16(o: int) -> int:
        return struct.unpack_from(endian + "H", data, o)[0]

    def u32(o: int) -> int:
        return struct.unpack_from(endian + "I", data, o)[0]

    try:
        ifd = u32(4)
        if ifd + 2 > len(data):
            return None
        make = model = None
        for i in range(min(u16(ifd), 512)):
            e = ifd + 2 + 12 * i
            if e + 12 > len(data):
                break
            tag, typ, cnt = u16(e), u16(e + 2), u32(e + 4)
            if typ != 2 or tag not in (0x010F, 0x0110) or cnt == 0:
                continue
            src = e + 8 if cnt <= 4 else u32(e + 8)
            if src + cnt > len(data):
                continue
            val = data[src:src + cnt].split(b"\0")[0].decode("ascii", "replace").strip()
            if tag == 0x010F:
                make = val
            else:
                model = val
        if make or model:
            return " ".join(v for v in (make, model) if v)
    except struct.error:
        return None
    return None


def _file_iso(path: Path) -> float | None:
    """ISO from libraw's metadata block (cheap: opens the file, no decode)."""
    try:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            iso = float(getattr(raw.other, "iso_speed", 0.0) or 0.0)
        return iso if iso > 0 else None
    except Exception:  # noqa: BLE001 — any unreadable file simply has no ISO
        return None


def file_entry(path_str: str) -> dict:
    """Metadata for a path harvested WITHOUT the library: camera from the
    file's TIFF tags, ISO from libraw, exiftool filling gaps when installed."""
    p = Path(path_str)
    digest = hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:8]
    entry = {"id": None, "path": str(p), "filename": p.name,
             "rel": f"file/{digest}/{p.name}", "camera": None, "iso": None}
    if not p.is_file():
        entry["reason"] = "file missing on disk"
        return entry
    entry["camera"] = _tiff_camera(p)
    entry["iso"] = _file_iso(p)
    if (entry["camera"] is None or entry["iso"] is None) and shutil.which("exiftool"):
        try:
            exif = read_exif(p)
            if entry["camera"] is None and (exif.get("Make") or exif.get("Model")):
                entry["camera"] = f"{exif.get('Make', '')} {exif.get('Model', '')}".strip()
            if entry["iso"] is None and exif.get("ISO"):
                entry["iso"] = float(exif["ISO"])
        except Exception:  # noqa: BLE001 — exiftool is best-effort
            pass
    if entry["iso"] is None:
        entry["reason"] = "cannot read ISO from the file (unsupported raw?)"
    elif entry["camera"] is None:
        entry["camera"] = "unknown"
    return entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB,
                    help="Ansel library.db (read-only; only needed for --ids/--ids-file — "
                         "paths work without it)")
    ap.add_argument("paths", nargs="*", metavar="FILE",
                    help="image file paths, as pasted from Ansel's 'File > Export image list...' "
                         "(the shell removes the quoting); no library needed")
    ap.add_argument("--ids", default="", help="image IDs, comma list with ranges: 12,15,100-120")
    ap.add_argument("--ids-file", type=Path, default=None,
                    help="file with one ID (or range) per line, '#' comments allowed "
                         "(as saved by Ansel's 'Export image list' in ID mode)")
    ap.add_argument("--paths-file", type=Path, default=None,
                    help="file with one image path per line, unquoted, lines starting with '#' "
                         "ignored (as saved by Ansel's 'Export image list' in filename mode)")
    ap.add_argument("--out", type=Path, required=True, help="shard output directory")
    ap.add_argument("--private", action="store_true",
                    help="mark the output directory private: publish_shards.sh will refuse it")
    ap.add_argument("--license", default="ATDL-1.1",
                    help="license tag recorded for these shards (you are the rights holder; "
                         "default ATDL-1.1, the Ansel Training Data License — denoiser "
                         "training only, see LICENSE-DATA.md)")
    ap.add_argument("--max-iso", type=int, default=200)
    ap.add_argument("--tiles", type=int, default=16)
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0xA45E1)
    args = ap.parse_args(argv)

    spec = args.ids
    if args.ids_file:
        lines = args.ids_file.read_text(encoding="utf-8").splitlines()
        spec += "," + ",".join(line.split("#")[0].strip() for line in lines)
    ids = parse_ids(spec)

    paths = list(args.paths)
    if args.paths_file:
        # whole-line comments only: '#' is a legal character inside a path
        paths += [line.strip() for line in
                  args.paths_file.read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    if not ids and not paths:
        ap.error("no images given (use --ids, --ids-file, --paths-file "
                 "and/or file path arguments)")

    db_available = Path(args.db).is_file()
    if ids and not db_available:
        ap.error(f"--ids/--ids-file need the Ansel library at {args.db}, which does not "
                 f"exist — pass file paths instead, or point --db at your library")

    images: list[dict] = []
    if ids:
        images += resolve_images(args.db, ids)
    if paths:
        if db_available:
            found, leftover = resolve_paths_db(args.db, paths)
            images += found
        else:
            print(f"note: no Ansel library at {args.db} — reading camera/ISO metadata "
                  f"directly from the files")
            leftover = paths
        images += [file_entry(p) for p in leftover]

    # one file can arrive both by id and by path
    seen: set[str] = set()
    images = [img for img in images
              if (key := img.get("path") or img["rel"]) not in seen and not seen.add(key)]

    args.out.mkdir(parents=True, exist_ok=True)
    if args.private:
        (args.out / ".private").write_text(
            "Shards from a personal image library. publish_shards.sh refuses this directory.\n")
    elif (args.out / ".private").exists():
        print(f"note: {args.out} carries a .private marker from an earlier run; "
              f"it stays private until you delete the marker")
    else:
        print("NOTE: this output is publishable — every listed image's tiles may become "
              "public on the shard release. Curate the list accordingly (--private otherwise).")
    ledger_path = args.out / "ledger.jsonl"
    done = set()
    if ledger_path.exists():
        with open(ledger_path, encoding="utf-8") as f:
            done = {json.loads(line)["path"] for line in f if line.strip()}

    images = [img for img in images if img["rel"] not in done]
    print(f"{len(images)} images to process ({len(done)} entries already in ledger)")

    n_ok = 0
    with open(ledger_path, "a", encoding="utf-8") as ledger:
        for i, img in enumerate(images):
            rel = img["rel"]
            record = {"path": rel, "image_id": img["id"], "source": img["path"],
                      "camera": img["camera"], "iso": img["iso"],
                      "license": args.license, "status": "rejected"}
            if img.get("reason"):
                record["reason"] = img["reason"]
            elif img["path"] is None:
                record["reason"] = "id not found in library.db"
            elif not Path(img["path"]).is_file():
                record["reason"] = "file missing on disk (stale film roll?)"
            elif (ext := Path(img["path"]).suffix.lower()) not in RAW_EXTENSIONS \
                    or ext in EXCLUDE_EXTENSIONS:
                record["reason"] = f"not a supported raw ({ext})"
            elif img["iso"] is None or img["iso"] <= 0:
                record["reason"] = "no ISO available"
            elif img["iso"] > args.max_iso:
                record["reason"] = f"ISO {img['iso']:.0f} > {args.max_iso}"
            else:
                origin = f"library:{img['id']}" if img["id"] is not None \
                    else f"file:{img['filename']}"
                result = run_isolated(
                    _pack_worker,
                    (img["path"], rel, str(args.out), args.tile_size, args.tiles, args.seed,
                     int(img["iso"]), img["camera"], origin),
                )
                record.update(result)
            ledger.write(json.dumps(record) + "\n")
            ledger.flush()
            n_ok += record["status"] == "harvested"
            print(f"[{i + 1}/{len(images)}] {record['status']:9s} {rel}"
                  + (f" ({record.get('reason', '')})" if record["status"] != "harvested"
                     else f" -> {record['n_tiles']} tiles"))
    mode = "PRIVATE — publish_shards.sh will refuse this directory" if args.private \
        else "publishable — publish_shards.sh will upload these"
    print(f"done: {n_ok} images harvested into {args.out} ({mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
