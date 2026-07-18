"""Shard validator: the gatekeeper of community contributions."""

import numpy as np
import pytest

from ansel_denoise.validate_shards import validate_dir, validate_shard


def write_shard(path, n_tiles=2, tile_size=256, iso=100, dtype=np.uint16, drop=None):
    fields = {
        "tiles": np.zeros((n_tiles, tile_size, tile_size), dtype=dtype) + 1000,
        "offsets": np.zeros((n_tiles, 2), dtype=np.int32),
        "pattern": np.str_("RGGB"),
        "pattern4": np.zeros((2, 2), dtype=np.uint8),
        "black_per_channel": np.zeros(4, dtype=np.float32),
        "white": np.float32(16383.0),
        "wb": np.ones(4, dtype=np.float32),
        "iso": np.int32(iso),
        "camera": np.str_("Testmaker Model X"),
        "source_path": np.str_("library/1/test.raw"),
    }
    if drop:
        fields.pop(drop)
    np.savez_compressed(path, **fields)


def test_valid_shard(tmp_path):
    write_shard(tmp_path / "good.npz")
    ok, info = validate_shard(tmp_path / "good.npz")
    assert ok and info["n_tiles"] == 2 and info["camera"] == "Testmaker Model X"


@pytest.mark.parametrize("kwargs,reason", [
    ({"drop": "wb"}, "missing keys"),
    ({"dtype": np.float32}, "dtype"),
    ({"tile_size": 128}, "shape"),
    ({"n_tiles": 0}, "empty"),
    ({"iso": 1600}, "ISO"),
])
def test_invalid_shards(tmp_path, kwargs, reason):
    write_shard(tmp_path / "bad.npz", **kwargs)
    ok, info = validate_shard(tmp_path / "bad.npz")
    assert not ok and reason in info["reason"]


def test_unreadable(tmp_path):
    (tmp_path / "junk.npz").write_bytes(b"not a zip at all")
    ok, info = validate_shard(tmp_path / "junk.npz")
    assert not ok and "unreadable" in info["reason"]


def test_validate_dir_summary(tmp_path):
    write_shard(tmp_path / "a.npz", n_tiles=3)
    write_shard(tmp_path / "b.npz", n_tiles=5)
    write_shard(tmp_path / "c.npz", iso=1600)
    summary = validate_dir(tmp_path, verbose=False)
    assert summary["n_shards"] == 2 and summary["n_invalid"] == 1
    assert summary["n_tiles"] == 8
    assert summary["cameras"] == {"Testmaker Model X": 8}
