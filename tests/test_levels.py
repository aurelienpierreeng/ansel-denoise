"""Rawspeed levels lookup: the training-side normalization domain switch."""

import numpy as np
import pytest

from ansel_denoise.levels import RawspeedLevels, norm_key


def test_norm_key_collapses_vendor_noise():
    assert norm_key("NIKON CORPORATION NIKON D90") == "nikon d90"
    assert norm_key("Canon Canon EOS 5D") == "canon eos 5d"
    assert norm_key("FUJIFILM X-T4") == "fujifilm x t4"


def test_lookup_real_table():
    L = RawspeedLevels()  # committed data/rawspeed_levels.json
    assert L.lookup("NIKON CORPORATION NIKON D90", iso=200, libraw_white=4095) == (0.0, 3767.0)
    assert L.lookup("Canon EOS 5D", iso=100, libraw_white=3692) == (127.0, 3692.0)
    assert L.lookup("Frobnicator X1000", iso=100, libraw_white=4095) is None
    # wrong-bit-depth candidates must be rejected, not misapplied
    assert L.lookup("NIKON D90", iso=200, libraw_white=16383) is None


SYNTH = {
    "acme mk1": [
        {"mode": "12bit", "sensors": [{"black": 10.0, "white": 4000.0}]},
        {"mode": "14bit", "sensors": [{"black": 40.0, "white": 16000.0}]},
    ],
    "acme mk2": [
        {"mode": "", "sensors": [
            {"black": 5.0, "white": 4090.0},
            {"black": 7.0, "white": 3900.0, "iso_min": 3200.0},
            {"black": 6.0, "white": 4000.0, "iso_list": [50.0, 64.0]},
        ]},
    ],
}


def test_mode_disambiguation_by_libraw_white():
    L = RawspeedLevels(SYNTH)
    assert L.lookup("Acme Mk1", libraw_white=4095) == (10.0, 4000.0)
    assert L.lookup("Acme Mk1", libraw_white=16383) == (40.0, 16000.0)


def test_iso_conditioned_sensors():
    L = RawspeedLevels(SYNTH)
    assert L.lookup("Acme Mk2", iso=100, libraw_white=4095) == (5.0, 4090.0)   # default
    assert L.lookup("Acme Mk2", iso=6400, libraw_white=4095) == (7.0, 3900.0)  # iso_min
    assert L.lookup("Acme Mk2", iso=64, libraw_white=4095) == (6.0, 4000.0)    # iso_list


def test_dataset_normalizes_in_rawspeed_domain(tmp_path):
    """A D90-style shard (libraw white 4095, rawspeed white 3767) must come out
    brighter under rawspeed normalization; unknown cameras keep libraw levels."""
    torch = pytest.importorskip("torch")
    from ansel_denoise.cfa import BAYER_RGGB
    from ansel_denoise.dataset import RawTileDataset

    def write(path, camera):
        adu = np.full((256, 256), 3000, np.uint16)
        adu[::7, ::5] = 2000  # some texture so the tile isn't degenerate
        np.savez_compressed(
            path, tiles=adu[None], offsets=np.zeros((1, 2), np.int32),
            pattern=BAYER_RGGB, pattern4=BAYER_RGGB,
            black_per_channel=np.zeros(4, np.float32), white=np.float32(4095),
            wb=np.ones(4, np.float32), iso=np.int32(200),
            camera=np.str_(camera), source_path=np.str_("s"), annex_key=np.str_(""))

    d1, d2 = tmp_path / "rs", tmp_path / "lr"
    d1.mkdir(), d2.mkdir()
    write(d1 / "a.npz", "NIKON CORPORATION NIKON D90")
    write(d2 / "a.npz", "Frobnicator X1000")

    def clean_of(shards):
        ds = RawTileDataset(shards, "train", patch=64, deterministic=True,
                            level_jitter=False, exposure_push_ev=0.0)
        # target y is the normalized clean tile (deterministic: no jitter)
        _, y = ds[0]
        return y.max().item()

    rs_max = clean_of(d1)   # normalized by 3767
    lr_max = clean_of(d2)   # normalized by 4095
    assert rs_max == pytest.approx(3000 / 3767, abs=1e-4)
    assert lr_max == pytest.approx(3000 / 4095, abs=1e-4)


def test_level_jitter_perturbs_but_stays_sane(tmp_path):
    torch = pytest.importorskip("torch")
    from ansel_denoise.cfa import BAYER_RGGB
    from ansel_denoise.dataset import RawTileDataset

    adu = (np.random.default_rng(0).uniform(500, 3500, (256, 256))).astype(np.uint16)
    np.savez_compressed(
        tmp_path / "a.npz", tiles=adu[None], offsets=np.zeros((1, 2), np.int32),
        pattern=BAYER_RGGB, pattern4=BAYER_RGGB,
        black_per_channel=np.full(4, 128, np.float32), white=np.float32(4095),
        wb=np.ones(4, np.float32), iso=np.int32(100),
        camera=np.str_("Frobnicator X1000"), source_path=np.str_("s"), annex_key=np.str_(""))

    ds = RawTileDataset(tmp_path, "train", patch=64, deterministic=False,
                        level_jitter=True, exposure_push_ev=0.0)
    ys = [ds[0][1] for _ in range(4)]
    assert all(0.0 <= y.min() and y.max() <= 1.0 for y in ys)
    # jitter draws differ across samples -> normalized values differ
    assert not torch.allclose(ys[0], ys[1])
