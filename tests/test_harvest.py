"""Tests of the pure harvesting logic on synthetic mosaics (no rawpy needed)."""

import os
import time

import numpy as np
import pytest

from ansel_denoise.cfa import XTRANS, BAYER_RGGB, colors_map
from ansel_denoise.harvest import normalize_mosaic, pick_tiles, score_tile


def synthetic_raw(h=1200, w=1600, black=512, white=15000, seed=0):
    """A textured synthetic sensor readout with a clipped bright band."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    signal = 0.25 + 0.2 * np.sin(xx / 17.0) * np.cos(yy / 23.0) + 0.05 * rng.standard_normal((h, w))
    signal[:, : w // 8] = 1.2  # overexposed strip
    signal[h // 2 : h // 2 + 100, :] = 0.4  # flat strip
    adu = np.clip(signal * (white - black) + black, 0, white).astype(np.uint16)
    return adu, black, white


def test_normalize_mosaic_roundtrip():
    adu, black, white = synthetic_raw()
    colors4 = colors_map(BAYER_RGGB, *adu.shape)
    norm = normalize_mosaic(adu, colors4, np.full(4, black, np.float32), white)
    assert norm.max() <= 1.0 + 1e-5
    assert norm.min() >= -black / (white - black) - 1e-5


def test_score_tile_flags_clipping_and_flatness():
    flat = np.full((64, 64), 0.4, np.float32)
    clipped, texture = score_tile(flat)
    assert clipped == 0.0 and texture == 0.0

    hot = np.ones((64, 64), np.float32)
    assert score_tile(hot)[0] == 1.0

    rng = np.random.default_rng(0)
    textured = rng.uniform(0, 0.9, (64, 64)).astype(np.float32)
    assert score_tile(textured)[1] > 0.01


def test_pick_tiles_alignment_and_rejection():
    adu, black, white = synthetic_raw()
    for pattern in (BAYER_RGGB, XTRANS):
        colors4 = colors_map(pattern, *adu.shape)
        norm = normalize_mosaic(adu, colors4, np.full(4, black, np.float32), white)
        rng = np.random.default_rng(1)
        tiles, offsets = pick_tiles(adu, colors4, np.full(4, black, np.float32), white,
                                    pattern.shape, rng, tile_size=256, n_tiles=8)
        assert 0 < len(tiles) <= 8
        assert tiles.dtype == np.uint16 and tiles.shape[1:] == (256, 256)
        ph, pw = pattern.shape
        for oy, ox in offsets:
            assert oy % ph == 0 and ox % pw == 0
            # no tile from the overexposed strip
            tile_norm = norm[oy : oy + 256, ox : ox + 256]
            assert np.mean(tile_norm >= 0.98) <= 0.02


def test_pick_tiles_too_small_image():
    adu = np.zeros((100, 100), np.uint16)
    colors4 = colors_map(BAYER_RGGB, 100, 100)
    tiles, offsets = pick_tiles(adu, colors4, np.zeros(4, np.float32), 1.0, (2, 2),
                                np.random.default_rng(0))
    assert len(tiles) == 0 and len(offsets) == 0


# module-level targets: run_isolated uses the 'spawn' start method, whose
# children import this module and unpickle targets by qualified name
def _crasher(q):
    os.abort()


def _hang(q):
    time.sleep(60)


def _healthy(q):
    q.put({"status": "harvested", "n_tiles": 3})


def test_run_isolated_survives_native_crash():
    # raw.pixls.us hosts decoder-hostile files that segfault libraw; a native
    # crash must become a ledger record, not kill the harvest
    from ansel_denoise.harvest import run_isolated

    rec = run_isolated(_crasher, (), timeout=30)
    assert rec["status"] == "error" and "crashed" in rec["reason"]

    rec = run_isolated(_hang, (), timeout=2)
    assert rec["status"] == "error" and "hang" in rec["reason"]

    assert run_isolated(_healthy, ())["n_tiles"] == 3


def test_pack_worker_rejects_garbage_file(tmp_path):
    pytest.importorskip("rawpy")
    from ansel_denoise.harvest import _pack_worker, run_isolated

    bad = tmp_path / "garbage.nef"
    bad.write_bytes(b"\x00\x01this is not a raw file" * 100)
    rec = run_isolated(_pack_worker, (str(bad), "x/garbage.nef", str(tmp_path), 256, 4, 0,
                                      100, "Fake Cam", ""))
    assert rec["status"] == "rejected"
    assert not list(tmp_path.glob("*.npz"))  # nothing half-written
