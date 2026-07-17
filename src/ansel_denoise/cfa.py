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
