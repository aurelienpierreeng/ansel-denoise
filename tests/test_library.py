"""Tests for the Ansel-library harvester (offline, temp sqlite db)."""

import sqlite3
import subprocess
from pathlib import Path

from ansel_denoise.harvest_library import (_tiff_camera, file_entry, parse_ids,
                                           resolve_images, resolve_paths_db)


def make_db(path: Path) -> None:
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, access_timestamp INTEGER,
                                 folder VARCHAR(1024) NOT NULL);
        CREATE TABLE images (id INTEGER PRIMARY KEY, film_id INTEGER, filename VARCHAR,
                             maker VARCHAR, model VARCHAR, iso REAL);
        INSERT INTO film_rolls VALUES (1, 0, '/photos/roll1'), (2, 0, '/photos/roll2');
        INSERT INTO images VALUES (10, 1, 'a.nef', 'NIKON', 'D850', 100.0);
        INSERT INTO images VALUES (11, 1, 'b.nef', 'NIKON', 'D850', 3200.0);
        INSERT INTO images VALUES (12, 2, 'c.raf', 'FUJIFILM', 'X-T4', 160.0);
        INSERT INTO images VALUES (13, 2, 'c.raf', 'FUJIFILM', 'X-T4', 160.0); -- duplicate
    """)
    db.commit()
    db.close()


def test_parse_ids():
    assert parse_ids("12,15,100-103") == [12, 15, 100, 101, 102, 103]
    assert parse_ids(" 5 , 5 ,3-4") == [3, 4, 5]
    assert parse_ids("") == []


def test_resolve_images(tmp_path):
    db = tmp_path / "library.db"
    make_db(db)
    imgs = resolve_images(db, [10, 11, 12, 13, 99])
    by_id = {i["id"]: i for i in imgs}
    assert by_id[10]["path"] == "/photos/roll1/a.nef"
    assert by_id[10]["camera"] == "NIKON D850" and by_id[10]["iso"] == 100.0
    assert by_id[99]["path"] is None  # unknown id reported, not dropped
    assert 13 not in by_id  # duplicate of 12 (same file) resolved once
    assert by_id[12]["path"] == "/photos/roll2/c.raf"


def test_publish_refuses_private_dir(tmp_path):
    (tmp_path / ".private").write_text("private\n")
    (tmp_path / "x.npz").write_bytes(b"fake")
    script = Path(__file__).resolve().parents[1] / "scripts" / "publish_shards.sh"
    r = subprocess.run(["sh", str(script), str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 1
    assert "REFUSING" in r.stderr


def make_tiff(path: Path, make=b"NIKON CORPORATION", model=b"NIKON D90") -> None:
    """Minimal little-endian TIFF: IFD0 with Make (0x010F) and Model (0x0110)."""
    import struct
    strings = make + b"\x00" + model + b"\x00"
    str_off = 8 + 2 + 2 * 12 + 4
    header = b"II" + struct.pack("<HI", 42, 8)
    entries = struct.pack("<HHII", 0x010F, 2, len(make) + 1, str_off)
    entries += struct.pack("<HHII", 0x0110, 2, len(model) + 1, str_off + len(make) + 1)
    path.write_bytes(header + struct.pack("<H", 2) + entries + struct.pack("<I", 0) + strings)


def test_tiff_camera(tmp_path):
    f = tmp_path / "x.nef"
    make_tiff(f)
    assert _tiff_camera(f) == "NIKON CORPORATION NIKON D90"
    f2 = tmp_path / "junk.nef"
    f2.write_bytes(b"not a tiff at all")
    assert _tiff_camera(f2) is None


def test_file_entry_missing_and_no_iso(tmp_path):
    e = file_entry(str(tmp_path / "nope.nef"))
    assert e["reason"] == "file missing on disk" and e["rel"].startswith("file/")

    f = tmp_path / "fake.nef"
    make_tiff(f)
    e = file_entry(str(f))  # camera parses, but libraw can't read ISO from a stub
    assert e["camera"] == "NIKON CORPORATION NIKON D90"
    assert "reason" in e  # no ISO -> rejected later with a clear message


def test_resolve_paths_db(tmp_path):
    db = tmp_path / "library.db"
    make_db(db)
    found, leftover = resolve_paths_db(db, ["/photos/roll1/a.nef", "/photos/elsewhere/z.nef"])
    assert len(found) == 1 and found[0]["id"] == 10 and found[0]["iso"] == 100.0
    assert leftover == ["/photos/elsewhere/z.nef"]


def test_main_paths_without_library(tmp_path, capsys):
    """The reported bug: --paths-file must work with no library.db at all."""
    from ansel_denoise.harvest_library import main
    raw = tmp_path / "fake.nef"
    make_tiff(raw)
    listing = tmp_path / "list.txt"
    listing.write_text(f"{raw}\n{tmp_path / 'gone.nef'}\n")
    rc = main(["--db", str(tmp_path / "no-such-library.db"),
               "--paths-file", str(listing), "--out", str(tmp_path / "out")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no Ansel library" in out and "reading camera/ISO metadata" in out
    ledger = (tmp_path / "out" / "ledger.jsonl").read_text().splitlines()
    assert len(ledger) == 2  # both files processed (rejected for ISO / missing), not a crash


def test_main_ids_need_library(tmp_path):
    from ansel_denoise.harvest_library import main
    import pytest
    with pytest.raises(SystemExit):
        main(["--db", str(tmp_path / "absent.db"), "--ids", "1", "--out", str(tmp_path / "o")])
