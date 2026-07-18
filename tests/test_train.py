"""Training loop smoke test: EMA lifecycle across checkpoint, resume, export."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ansel_denoise.cfa import BAYER_RGGB
from ansel_denoise.dataset import camera_split
from ansel_denoise.export import load_model
from ansel_denoise.train import main as train_main


def _write_shard(path, camera):
    yy, xx = np.mgrid[0:600, 0:600]
    sig = 0.3 + 0.25 * np.sin(xx / 13.0) * np.cos(yy / 19.0)
    adu = (np.clip(sig, 0, 1) * 15000 + 512).astype(np.uint16)
    np.savez_compressed(
        path,
        tiles=np.stack([adu[i * 60 : i * 60 + 256, i * 60 : i * 60 + 256] for i in range(4)]),
        offsets=np.array([[i * 60, i * 60] for i in range(4)], dtype=np.int32),
        pattern=BAYER_RGGB, pattern4=BAYER_RGGB,
        black_per_channel=np.full(4, 512, np.float32), white=np.float32(15512),
        wb=np.ones(4, np.float32), iso=np.int32(100),
        camera=np.str_(camera), source_path=np.str_("synthetic"), annex_key=np.str_(""),
    )


@pytest.fixture
def shard_dir(tmp_path):
    train_cam = next(f"Cam {i}" for i in range(100) if camera_split(f"Cam {i}") == "train")
    val_cam = next(f"Cam {i}" for i in range(100) if camera_split(f"Cam {i}") == "val")
    shards = tmp_path / "shards"
    shards.mkdir()
    _write_shard(shards / "a.npz", train_cam)
    _write_shard(shards / "b.npz", val_cam)
    return shards


def _run(shards, out, steps, extra=()):
    rc = train_main([
        "--shards", str(shards), "--out", str(out), "--steps", str(steps),
        "--batch", "2", "--patch", "64", "--base", "8", "--workers", "0",
        "--device", "cpu", "--val-every", "4", "--ckpt-every", "4", *extra,
    ])
    assert rc == 0


def test_ema_checkpoint_resume_export(shard_dir, tmp_path):
    out = tmp_path / "run"
    _run(shard_dir, out, steps=6)

    ckpt = torch.load(out / "ckpt-final.pt", map_location="cpu")
    assert "ema" in ckpt and set(ckpt["ema"]) == set(ckpt["model"])
    # after only 6 steps the warmup ramp keeps EMA near the live weights but
    # they must not be identical (the average lags by construction)
    k = next(iter(ckpt["ema"]))
    assert not torch.equal(ckpt["ema"][k], ckpt["model"][k])

    # export prefers the EMA weights; --raw-weights opts out
    model_ema, _, _, which = load_model(out / "ckpt-final.pt")
    assert which == "ema"
    assert torch.equal(model_ema.state_dict()[k], ckpt["ema"][k])
    model_raw, _, _, which = load_model(out / "ckpt-final.pt", raw_weights=True)
    assert which == "raw"
    assert torch.equal(model_raw.state_dict()[k], ckpt["model"][k])

    # resume from an EMA-carrying checkpoint continues the average
    _run(shard_dir, out, steps=10, extra=("--resume", str(out / "ckpt-final.pt")))
    ckpt2 = torch.load(out / "ckpt-final.pt", map_location="cpu")
    assert ckpt2["step"] == 10 and "ema" in ckpt2


def test_resume_from_pre_ema_checkpoint(shard_dir, tmp_path):
    out = tmp_path / "run"
    _run(shard_dir, out, steps=4, extra=("--ema-decay", "0"))
    ckpt = torch.load(out / "ckpt-final.pt", map_location="cpu")
    assert "ema" not in ckpt  # disabled -> old-style checkpoint

    # resuming with EMA enabled seeds the average from the resumed weights
    _run(shard_dir, out, steps=8, extra=("--resume", str(out / "ckpt-final.pt")))
    ckpt2 = torch.load(out / "ckpt-final.pt", map_location="cpu")
    assert "ema" in ckpt2
    _, _, _, which = load_model(out / "ckpt-final.pt")
    assert which == "ema"
