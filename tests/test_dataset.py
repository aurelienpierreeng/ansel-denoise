"""Dataset behaviour that the training metrics depend on."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ansel_denoise.cfa import XTRANS, BAYER_RGGB
from ansel_denoise.dataset import RawTileDataset, camera_split


def _write_shard(path, pattern, camera):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:600, 0:600]
    sig = 0.3 + 0.25 * np.sin(xx / 13.0) * np.cos(yy / 19.0)
    adu = (np.clip(sig, 0, 1) * 15000 + 512).astype(np.uint16)
    np.savez_compressed(
        path,
        tiles=np.stack([adu[i * 60 : i * 60 + 256, i * 60 : i * 60 + 256] for i in range(4)]),
        offsets=np.array([[i * 60, i * 60] for i in range(4)], dtype=np.int32),
        pattern=pattern, pattern4=pattern,
        black_per_channel=np.full(4, 512, np.float32), white=np.float32(15512),
        wb=np.ones(4, np.float32), iso=np.int32(100),
        camera=np.str_(camera), source_path=np.str_("synthetic"), annex_key=np.str_(""),
    )


@pytest.fixture
def shard_dir(tmp_path):
    # camera names hand-picked so one lands in 'train' and one in 'val'
    train_cam = next(f"Cam {i}" for i in range(100) if camera_split(f"Cam {i}") == "train")
    val_cam = next(f"Cam {i}" for i in range(100) if camera_split(f"Cam {i}") == "val")
    _write_shard(tmp_path / "a.npz", BAYER_RGGB, train_cam)
    _write_shard(tmp_path / "b.npz", XTRANS, val_cam)
    return tmp_path


def test_shapes_and_channels(shard_dir):
    ds = RawTileDataset(shard_dir, "train", patch=96)
    x, y = ds[0]
    assert x.shape == (5, 96, 96) and y.shape == (1, 96, 96)
    assert (x[1:4].sum(dim=0) == 1).all()  # one-hot CFA planes
    assert (x[4] > 0).all()  # sigma map strictly positive
    assert torch.isfinite(x).all() and torch.isfinite(y).all()


def test_val_is_deterministic_train_is_not(shard_dir):
    val = RawTileDataset(shard_dir, "val", patch=96)
    x1, y1 = val[0]
    x2, y2 = RawTileDataset(shard_dir, "val", patch=96)[0]
    assert torch.equal(x1, x2) and torch.equal(y1, y2)

    train = RawTileDataset(shard_dir, "train", patch=96)
    a, _ = train[0]
    b, _ = train[0]
    assert not torch.equal(a, b)  # fresh noise/crop per access

    # validation fallback: a train-split dataset forced deterministic
    fb1, _ = RawTileDataset(shard_dir, "train", patch=96, deterministic=True)[0]
    fb2, _ = RawTileDataset(shard_dir, "train", patch=96, deterministic=True)[0]
    assert torch.equal(fb1, fb2)


def test_tile_cache_reuse_and_invalidation(shard_dir):
    from ansel_denoise.dataset import consolidate_tiles

    bin_path, ts, records = consolidate_tiles(shard_dir)
    assert bin_path.exists() and ts == 256 and len(records) == 8
    assert bin_path.stat().st_size == len(records) * ts * ts * 2
    mtime = bin_path.stat().st_mtime_ns

    _, _, again = consolidate_tiles(shard_dir)  # unchanged shards -> reuse
    assert bin_path.stat().st_mtime_ns == mtime and len(again) == 8

    _write_shard(shard_dir / "c.npz", BAYER_RGGB, "Cam extra")  # new shard -> rebuild
    _, _, rebuilt = consolidate_tiles(shard_dir)
    assert len(rebuilt) == 12 and bin_path.stat().st_mtime_ns != mtime


def test_split_partitions_cameras(shard_dir):
    train = RawTileDataset(shard_dir, "train", patch=96)
    val = RawTileDataset(shard_dir, "val", patch=96)
    assert len(train) == 4 and len(val) == 4
    train_cams = {train.records[i]["camera"] for i in train.index}
    val_cams = {val.records[i]["camera"] for i in val.index}
    assert train_cams.isdisjoint(val_cams)
