"""Torch dataset: harvested shards -> (noisy, sigma, colors) / clean pairs.

Noise is synthesized on the fly, so every epoch sees fresh realizations and a
fresh (camera, ISO) draw per patch — the shards only store clean tiles. The
train/val split is by *camera* (hash of the camera string), never by image, so
validation measures cross-sensor generalization.

Sample tensors (float32):
    input  (5, H, W): [noisy mosaic, R one-hot, G one-hot, B one-hot, sigma map]
    target (1, H, W): clean mosaic
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cfa import aligned_offset, colors_map, one_hot
from .harvest import normalize_mosaic
from .noise import sigma_map, synthesize
from .profiles import ProfileSampler

# Fraction of patches trained with (near-)zero noise so the network stays an
# identity on clean input instead of eating base-ISO texture.
CLEAN_FRACTION = 0.05


def camera_split(camera: str, val_buckets: int = 10) -> str:
    """Deterministic 'train'/'val' split on the camera identity."""
    h = int(hashlib.md5(camera.encode()).hexdigest(), 16)
    return "val" if h % val_buckets == 0 else "train"


@lru_cache(maxsize=64)
def _open_shard(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


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
        self.index: list[tuple[str, int]] = []  # (shard path, tile index)
        for shard in sorted(Path(shard_dir).glob("*.npz")):
            with np.load(shard) as z:
                if camera_split(str(z["camera"])) != split:
                    continue
                self.index.extend((str(shard), t) for t in range(z["tiles"].shape[0]))
        if not self.index:
            raise ValueError(f"no '{split}' tiles under {shard_dir}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_path, t = self.index[i]
        shard = _open_shard(shard_path)
        if self.deterministic:
            rng = np.random.default_rng((self.seed, i))
        else:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info else 0
            rng = np.random.default_rng(
                (self.seed, wid, i, np.random.SeedSequence().entropy % (2**32))
            )

        tile = shard["tiles"][t]
        oy, ox = (int(v) for v in shard["offsets"][t])
        pattern = shard["pattern"]  # phase-correct for the visible area
        ph, pw = pattern.shape

        # colors for the tile, then crop tile and colors together, CFA-aligned
        colors = colors_map(pattern, tile.shape[0], tile.shape[1], oy, ox)
        cy = aligned_offset(rng, tile.shape[0], self.patch, ph)
        cx = aligned_offset(rng, tile.shape[1], self.patch, pw)
        tile = tile[cy : cy + self.patch, cx : cx + self.patch]
        colors = colors[cy : cy + self.patch, cx : cx + self.patch]

        # normalize like rawprepare; colors4 == colors is fine for black
        # subtraction because per-channel blacks rarely differ between greens,
        # and the mean-black scale matches normalize_mosaic()
        clean = normalize_mosaic(tile, colors, shard["black_per_channel"][:3], float(shard["white"]))
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
        black_frac = float(np.mean(shard["black_per_channel"])) / max(float(shard["white"]), 1.0)
        noisy = synthesize(clean, colors, a, b, rng, black_frac=black_frac)
        sigma = sigma_map(noisy, colors, a, b)

        x = np.concatenate([noisy[None], one_hot(colors), sigma[None]], axis=0)
        y = clean.astype(np.float32)[None]
        return torch.from_numpy(x), torch.from_numpy(y)
