"""Harvest shards from an Ansel library, for a hand-picked list of images.

The Ansel lighttable is the curation UI: select the images you are willing to
feed the training (safe content, base ISO) and pass them here — either as
image IDs (--ids / --ids-file) or as file paths (positional arguments, as
produced by Ansel's "File > Export image list..." dialog, which shell-quotes
them for direct pasting). Both forms resolve through library.db (opened
READ-ONLY), gate on the DB's own ISO and metadata, and funnel the files
through the same crash-isolated decode -> tile pipeline as every other
source.

PRIVACY CONTRACT: the ID list is the curation. Tiles are viewable fragments
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
import json
import os
import sqlite3
import sys
from pathlib import Path

from .harvest import EXCLUDE_EXTENSIONS, RAW_EXTENSIONS, _pack_worker, run_isolated

def default_db() -> Path:
    """Ansel's library.db location: g_get_user_config_dir()/ansel/library.db,
    which GLib resolves to %LOCALAPPDATA% on Windows and ~/.config elsewhere."""
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA",
                                   Path.home() / "AppData" / "Local")) / "ansel" / "library.db"
    return Path.home() / ".config" / "ansel" / "library.db"


DEFAULT_DB = default_db()


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


def resolve_images(db_path: Path, ids: list[int], paths: list[str] | None = None) -> list[dict]:
    """Resolve image IDs and/or file paths to paths + metadata via library.db, read-only."""
    paths = paths or []
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = []
        if ids:
            marks = ",".join("?" * len(ids))
            rows += db.execute(
                f"""SELECT i.id, f.folder || '/' || i.filename, i.maker, i.model, i.iso, i.filename
                    FROM images i JOIN film_rolls f ON i.film_id = f.id
                    WHERE i.id IN ({marks}) ORDER BY i.id""",
                ids,
            ).fetchall()
        # paths match on the DB's own folder/filename join; try the argument as
        # given first, then resolved (symlinks, relative paths)
        matched_paths = set()
        for arg in paths:
            for candidate in dict.fromkeys([arg, str(Path(arg).resolve())]):
                row = db.execute(
                    """SELECT i.id, f.folder || '/' || i.filename,
                              i.maker, i.model, i.iso, i.filename
                       FROM images i JOIN film_rolls f ON i.film_id = f.id
                       WHERE f.folder || '/' || i.filename = ?""",
                    (candidate,),
                ).fetchone()
                if row:
                    rows.append(row)
                    matched_paths.add(arg)
                    break
    finally:
        db.close()
    found = {r[0] for r in rows}
    images, seen_paths = [], set()
    for iid, path, maker, model, iso, filename in sorted(rows):
        if path in seen_paths:  # darktable-style duplicates share the file
            continue
        seen_paths.add(path)
        images.append({"id": iid, "path": path, "filename": filename,
                       "camera": f"{(maker or '?').strip()} {(model or '?').strip()}",
                       "iso": iso})
    for missing in sorted(set(ids) - found):
        images.append({"id": missing, "path": None, "filename": None,
                       "camera": None, "iso": None})
    for missing_path in sorted(set(paths) - matched_paths):
        images.append({"id": "?", "path": None, "filename": Path(missing_path).name,
                       "camera": None, "iso": None})
    return images


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="Ansel library.db (read-only)")
    ap.add_argument("paths", nargs="*", metavar="FILE",
                    help="image file paths, as pasted from Ansel's 'File > Export image list...' "
                         "(the shell removes the quoting; paths are matched against library.db)")
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

    args.out.mkdir(parents=True, exist_ok=True)
    if args.private:
        (args.out / ".private").write_text(
            "Shards from a personal image library. publish_shards.sh refuses this directory.\n")
    elif (args.out / ".private").exists():
        print(f"note: {args.out} carries a .private marker from an earlier run; "
              f"it stays private until you delete the marker")
    else:
        print("NOTE: this output is publishable — every listed image's tiles may become "
              "public on the shard release. Curate the ID list accordingly (--private otherwise).")
    ledger_path = args.out / "ledger.jsonl"
    done = set()
    if ledger_path.exists():
        with open(ledger_path, encoding="utf-8") as f:
            done = {json.loads(line)["path"] for line in f if line.strip()}

    images = [img for img in resolve_images(args.db, ids, paths)
              if f"library/{img['id']}/{img['filename']}" not in done]
    print(f"{len(images)} images to process ({len(done)} entries already in ledger)")

    n_ok = 0
    with open(ledger_path, "a", encoding="utf-8") as ledger:
        for i, img in enumerate(images):
            rel = f"library/{img['id']}/{img['filename']}"
            record = {"path": rel, "image_id": img["id"], "source": img["path"],
                      "camera": img["camera"], "iso": img["iso"],
                      "license": args.license, "status": "rejected"}
            if img["path"] is None:
                record["reason"] = ("file path not in library.db (pass paths exactly as "
                                    "Ansel's 'Export image list' produces them)"
                                    if img["id"] == "?" else "id not found in library.db")
            elif not Path(img["path"]).is_file():
                record["reason"] = "file missing on disk (stale film roll?)"
            elif (ext := Path(img["path"]).suffix.lower()) not in RAW_EXTENSIONS \
                    or ext in EXCLUDE_EXTENSIONS:
                record["reason"] = f"not a supported raw ({ext})"
            elif img["iso"] is None or img["iso"] <= 0:
                record["reason"] = "no ISO in library.db"
            elif img["iso"] > args.max_iso:
                record["reason"] = f"ISO {img['iso']:.0f} > {args.max_iso}"
            else:
                result = run_isolated(
                    _pack_worker,
                    (img["path"], rel, str(args.out), args.tile_size, args.tiles, args.seed,
                     int(img["iso"]), img["camera"], f"library:{img['id']}"),
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
