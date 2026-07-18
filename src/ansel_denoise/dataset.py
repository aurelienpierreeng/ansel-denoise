"""Torch dataset: harvested shards -> (noisy, sigma, colors) / clean pairs.

Noise is synthesized on the fly, so every epoch sees fresh realizations and a
fresh (camera, ISO) draw per patch — the shards only store clean tiles. The
train/val split is by *camera* (hash of the camera string), never by image, so
validation measures cross-sensor generalization.

Tile storage: on first use the compressed shards are consolidated into one
flat memory-mapped file next to them (.tiles-cache.bin/.json). Random tile
sampling then costs a page-cache read instead of an npz decompression — an
LRU of decompressed shards stops scaling around a few hundred shards (observed
on Colab: 178 -> 78 patches/s), while the OS page cache handles the full
~2000-shard harvest (~4 GB) shared across dataloader workers. The cache is
fingerprinted against the shard listing and rebuilt when shards change.

Sample tensors (float32):
    input  (5, H, W): [noisy mosaic, R one-hot, G one-hot, B one-hot, sigma map]
    target (1, H, W): clean mosaic
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cfa import aligned_offset, colors_map, one_hot
from .harvest import normalize_mosaic
from .levels import RawspeedLevels
from .noise import sigma_map, synthesize
from .profiles import ProfileSampler

# Fraction of patches trained with (near-)zero noise so the network stays an
# identity on clean input instead of eating base-ISO texture.
CLEAN_FRACTION = 0.05

# Consolidated-cache schema version: bump when the per-tile record fields
# change, so stale caches rebuild even though the shard fingerprint matches.
CACHE_VERSION = 2

# Training-time level jitter: white scaled, black shifted (as a fraction of
# white), modeling decoder disagreement on sensor levels (Rawspeed curates
# measured saturation, libraw often reports the format maximum — observed up
# to ~9% on white) so the network is explicitly invariant to it.
WHITE_JITTER = (0.92, 1.05)
BLACK_JITTER_FRAC = 0.002


def camera_split(camera: str, val_buckets: int = 10) -> str:
    """Deterministic 'train'/'val' split on the camera identity."""
    h = int(hashlib.md5(camera.encode()).hexdigest(), 16)
    return "val" if h % val_buckets == 0 else "train"


def _fingerprint(shards: list[Path]) -> str:
    h = hashlib.sha1()
    for s in shards:
        st = s.stat()
        h.update(f"{s.name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def consolidate_tiles(shard_dir: Path) -> tuple[Path, int, list[dict]]:
    """Merge all shards into a flat uint16 tile file + per-tile records.
    Returns (bin path, tile size, records); reuses the existing cache when the
    shard listing is unchanged."""
    shard_dir = Path(shard_dir)
    shards = sorted(shard_dir.glob("*.npz"))
    if not shards:
        raise ValueError(f"no shards under {shard_dir}")
    bin_path = shard_dir / ".tiles-cache.bin"
    meta_path = shard_dir / ".tiles-cache.json"
    fp = _fingerprint(shards)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fp and meta.get("version") == CACHE_VERSION \
                and bin_path.exists():
            return bin_path, meta["tile_size"], meta["records"]

    records: list[dict] = []
    tile_size = None
    with open(bin_path, "wb") as out:
        for shard in shards:
            with np.load(shard) as z:
                tiles = z["tiles"]
                if tiles.shape[0] == 0:
                    continue
                if tile_size is None:
                    tile_size = int(tiles.shape[1])
                if tiles.shape[1] != tile_size or tiles.shape[2] != tile_size:
                    raise ValueError(f"{shard.name}: tile size {tiles.shape[1:]} != {tile_size}")
                base = {
                    "camera": str(z["camera"]),
                    "pattern": z["pattern"].tolist(),
                    "black": np.asarray(z["black_per_channel"][:3], dtype=float).tolist(),
                    "white": float(z["white"]),
                    "iso": float(z["iso"]),
                }
                for t in range(tiles.shape[0]):
                    oy, ox = (int(v) for v in z["offsets"][t])
                    records.append({**base, "oy": oy, "ox": ox})
                out.write(np.ascontiguousarray(tiles, dtype=np.uint16).tobytes())
    meta_path.write_text(json.dumps(
        {"fingerprint": fp, "version": CACHE_VERSION, "tile_size": tile_size,
         "count": len(records), "records": records}
    ))
    return bin_path, tile_size, records


class RawTileDataset(Dataset):
    """Iterates over all tiles of all shards in `shard_dir` for one split."""

    def __init__(
        self,
        shard_dir: Path | str,
        split: str = "train",
        patch: int = 128,
        sampler: ProfileSampler | None = None,
        exposure_push_ev: float = 5.0,
        seed: int = 0,
        deterministic: bool | None = None,
        levels: RawspeedLevels | None = None,
        level_jitter: bool = True,
    ):
        self.patch = patch
        self.split = split
        # val items are frozen (crop, flips, profile, noise) so metrics are
        # comparable across steps and runs; override for a train-split dataset
        # used as validation fallback
        self.deterministic = (split == "val") if deterministic is None else deterministic
        self.sampler = sampler or ProfileSampler()
        self.exposure_push_ev = exposure_push_ev
        self.seed = seed

        self.tiles_path, self.tile_size, self.records = consolidate_tiles(Path(shard_dir))
        self.index = [i for i, r in enumerate(self.records) if camera_split(r["camera"]) == split]
        if not self.index:
            raise ValueError(f"no '{split}' tiles under {shard_dir}")
        self._tiles: np.memmap | None = None  # opened lazily, once per worker process
        self.level_jitter = level_jitter and not self.deterministic

        # Normalization levels: Rawspeed's per-camera sensor levels when the
        # camera is in the table — the exact domain rawprepare (and the noise
        # profiles) use at runtime — else the shard's libraw metadata.
        # Rawspeed's rawprepare subtracts one scalar black (the mean of the
        # four CFA blacks), so the resolved black is scalar in both cases.
        levels = levels if levels is not None else RawspeedLevels()
        n_rs = 0
        rs_cams, lr_cams = set(), set()
        for rec in self.records:
            rs = levels.lookup(rec["camera"], iso=rec.get("iso"), libraw_white=rec["white"])
            if rs is not None:
                rec["norm_black"], rec["norm_white"] = rs
                n_rs += 1
                rs_cams.add(rec["camera"])
            else:
                rec["norm_black"] = float(np.mean(rec["black"]))
                rec["norm_white"] = rec["white"]
                lr_cams.add(rec["camera"])
        if split == "train":
            print(f"levels: rawspeed for {n_rs}/{len(self.records)} tiles "
                  f"({len(rs_cams)} cameras), libraw metadata for {len(lr_cams)} cameras")

    def _tile(self, record_idx: int) -> np.ndarray:
        if self._tiles is None:
            ts = self.tile_size
            self._tiles = np.memmap(self.tiles_path, dtype=np.uint16, mode="r",
                                    shape=(len(self.records), ts, ts))
        return self._tiles[record_idx]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[self.index[i]]
        tile = self._tile(self.index[i])
        if self.deterministic:
            rng = np.random.default_rng((self.seed, i))
        else:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info else 0
            rng = np.random.default_rng(
                (self.seed, wid, i, np.random.SeedSequence().entropy % (2**32))
            )

        pattern = np.asarray(rec["pattern"], dtype=np.uint8)  # phase-correct
        ph, pw = pattern.shape

        # resolved normalization levels (rawspeed domain when known), plus
        # training-time jitter modeling decoder disagreement on the levels
        white = float(rec["norm_white"])
        black_scalar = float(rec["norm_black"])
        if self.level_jitter:
            white *= float(rng.uniform(*WHITE_JITTER))
            black_scalar += float(rng.uniform(-BLACK_JITTER_FRAC, BLACK_JITTER_FRAC)) * white
        black = np.full(3, black_scalar, dtype=np.float32)

        # colors for the tile, then crop tile and colors together, CFA-aligned
        colors = colors_map(pattern, tile.shape[0], tile.shape[1], rec["oy"], rec["ox"])
        cy = aligned_offset(rng, tile.shape[0], self.patch, ph)
        cx = aligned_offset(rng, tile.shape[1], self.patch, pw)
        tile = np.asarray(tile[cy : cy + self.patch, cx : cx + self.patch])
        colors = colors[cy : cy + self.patch, cx : cx + self.patch]

        # normalize like rawprepare: scalar black (rawprepare averages the four
        # CFA blacks into one), white - black scale
        clean = normalize_mosaic(tile, colors, black, white)
        clean = np.clip(clean, 0.0, 1.0)

        # flips are valid augmentation as long as mosaic and colors flip together
        for axis in (0, 1):
            if rng.random() < 0.5:
                clean = np.flip(clean, axis=axis)
                colors = np.flip(colors, axis=axis)
        clean = np.ascontiguousarray(clean)
        colors = np.ascontiguousarray(colors)

        # simulate underexposure that will be pushed later in the pipeline
        clean = clean * float(2.0 ** -rng.uniform(0.0, self.exposure_push_ev))

        a, b, _ = self.sampler.sample(rng)
        if rng.random() < CLEAN_FRACTION:
            a = a * 1e-3
            b = b * 1e-3
        black_frac = black_scalar / max(white, 1.0)
        noisy = synthesize(clean, colors, a, b, rng, black_frac=black_frac)
        sigma = sigma_map(noisy, colors, a, b)

        x = np.concatenate([noisy[None], one_hot(colors), sigma[None]], axis=0)
        y = clean.astype(np.float32)[None]
        return torch.from_numpy(x), torch.from_numpy(y)
