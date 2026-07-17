"""Tests for the Ansel-library harvester (offline, temp sqlite db)."""

import sqlite3
import subprocess
from pathlib import Path

from ansel_denoise.harvest_library import parse_ids, resolve_images


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
