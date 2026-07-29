"""CFA (color filter array) geometry helpers.

Everything downstream works on the 1-channel mosaic exactly as Ansel's raw
pipeline sees it after rawprepare (black-subtracted, normalized to [0, 1]),
plus a per-pixel color index map in {0=R, 1=G, 2=B}. One network architecture
serves both Bayer and X-Trans: the CFA layout is an *input* (one-hot color
planes), not an architectural assumption. Crops must stay aligned to the CFA
period so the color map of a tile is derivable from the pattern alone.
"""

from __future__ import annotations

import numpy as np

# Canonical X-Trans 6x6 pattern (Fuji), for tests and synthetic data.
XTRANS = np.array(
    [
        [1, 1, 0, 1, 1, 2],
        [1, 1, 2, 1, 1, 0],
        [2, 0, 1, 0, 2, 1],
        [1, 1, 2, 1, 1, 0],
        [1, 1, 0, 1, 1, 2],
        [0, 2, 1, 2, 0, 1],
    ],
    dtype=np.uint8,
)

BAYER_RGGB = np.array([[0, 1], [1, 2]], dtype=np.uint8)


def normalize_pattern(pattern: np.ndarray) -> np.ndarray:
    """Map libraw color indices {0=R,1=G,2=B,3=G2} to {0=R,1=G,2=B}."""
    p = np.asarray(pattern, dtype=np.uint8).copy()
    p[p == 3] = 1
    return p


def colors_map(pattern: np.ndarray, height: int, width: int, oy: int = 0, ox: int = 0) -> np.ndarray:
    """Per-pixel color index map for a (height, width) window whose top-left
    corner sits at (oy, ox) in sensor coordinates."""
    p = normalize_pattern(pattern)
    ph, pw = p.shape
    rows = (np.arange(height) + oy) % ph
    cols = (np.arange(width) + ox) % pw
    return p[np.ix_(rows, cols)]


def one_hot(colors: np.ndarray) -> np.ndarray:
    """(H, W) color index map -> (3, H, W) float32 one-hot planes."""
    return (colors[None, :, :] == np.arange(3, dtype=colors.dtype)[:, None, None]).astype(np.float32)


def aligned_offset(rng: np.random.Generator, extent: int, crop: int, period: int) -> int:
    """Random crop offset in [0, extent - crop], aligned to the CFA period."""
    span = (extent - crop) // period
    if span < 0:
        raise ValueError(f"crop {crop} larger than extent {extent}")
    return int(rng.integers(span + 1)) * period


def bin_for_pattern(pattern: np.ndarray) -> int:
    """Superpixel bin factor for the coarse (low-frequency) stage: 4 for the
    2x2 Bayer period (a 4x4 sensor block holds 4 R, 8 G, 4 B sites), 6 for
    the 6x6 X-Trans period (8 R, 20 G, 8 B). Any bin must keep every channel
    represented in every block, so the factor is tied to the CFA period."""
    shape = normalize_pattern(pattern).shape
    if shape == (2, 2):
        return 4
    if shape == (6, 6):
        return 6
    raise ValueError(f"unsupported CFA pattern shape {shape}")


def bin_mosaic_torch(mosaic, onehot, bin_factor: int):
    """Superpixel-bin a mosaic per CFA channel (differentiable).

    mosaic (B, 1, H, W), onehot (B, 3, H, W) -> (rgb (B, 3, H/b, W/b),
    counts (B, 3, H/b, W/b)). Each coarse pixel is the count-weighted mean of
    the same-channel sensels inside its b x b block — the exact contract the
    C inference side mirrors. With bin 4 on Bayer / 6 on X-Trans every count
    is > 0 by construction; the clamp is a numerical safety, not a fallback.
    """
    import torch.nn.functional as F

    area = float(bin_factor * bin_factor)
    sums = F.avg_pool2d(mosaic * onehot, bin_factor) * area
    counts = F.avg_pool2d(onehot, bin_factor) * area
    return sums / counts.clamp(min=1.0), counts
