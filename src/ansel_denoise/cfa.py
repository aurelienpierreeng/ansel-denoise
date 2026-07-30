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
    """Superpixel bin factor for the coarse (low-frequency) stage: 4 for
    periods dividing 4 — plain 2x2 Bayer (a 4x4 sensor block holds 4 R, 8 G,
    4 B sites) and 4x4 quad-Bayer-class sensors (one full period per block) —
    and 6 for the 6x6 X-Trans period (8 R, 20 G, 8 B). Any bin must keep
    every channel represented in every block, so the factor is tied to the
    CFA period."""
    p = normalize_pattern(pattern)
    shape = p.shape
    if shape in ((2, 2), (4, 4)) and {0, 1, 2} <= set(p.flatten().tolist()):
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


def bin_sigma_torch(sigma, onehot, bin_factor: int, counts):
    """Coarse sigma from the fine sigma plane: the variance of the mean of n
    sensels is mean(sigma_i^2) / n, i.e. sigma_c = sqrt(sum(sigma^2)) / n.
    sigma (B, 1, H, W), onehot (B, 3, H, W), counts from bin_mosaic_torch."""
    import torch.nn.functional as F

    area = float(bin_factor * bin_factor)
    s2 = F.avg_pool2d(sigma * sigma * onehot, bin_factor) * area
    return s2.sqrt() / counts.clamp(min=1.0)


def gaussian_bin_torch(mosaic, onehot, scale: int):
    """Gaussian-windowed superpixel means (overlapping 2s x 2s windows,
    sigma = s/2, stride s), per CFA channel with weight normalization.

    Used by the low-band fusion INSTEAD of box binning: a box window samples
    R, G and B at different effective positions relative to a luminance edge
    (their sensel sites differ inside the block), so edge-straddling blocks
    acquire a false chroma skew that the fusion's measurement bands inject
    into the output as a saturated outline along high-contrast edges. A
    Gaussian aperture is (nearly) identical for all three channels, which
    collapses the skew; the overlap also removes residual grid signature.
    The coarse-net input keeps box binning (bin_mosaic_torch): that is the
    training contract for the guide."""
    import torch
    import torch.nn.functional as F

    r = torch.arange(2 * scale, dtype=mosaic.dtype, device=mosaic.device) - (scale - 0.5)
    g1 = torch.exp(-(r * r) / (2.0 * (scale / 2.0) ** 2))
    k = (g1[:, None] * g1[None, :])[None, None]
    b, _, h, w = mosaic.shape
    weighted = (mosaic * onehot).reshape(b * 3, 1, h, w)
    weights = onehot.reshape(b * 3, 1, h, w)
    num = F.conv2d(weighted, k, stride=scale, padding=scale // 2)
    den = F.conv2d(weights, k, stride=scale, padding=scale // 2)
    return (num / den.clamp(min=1e-6)).reshape(b, 3, h // scale, w // scale)


def fuse_low_bands(pred, noisy, onehot, sigma, scales=(16, 32, 64)):
    """Hybrid low-band fusion: per-band self-calibrated Wiener weights at the
    fine/mid scales, pure measurement at the coarsest (the anchor's dilution
    floor). Measured to dominate both the hard anchor and full Wiener on
    PSNR and LFCE at every scale. Bayer densities; the anchored/coarsest
    band is exact measurement so the hallucination-free guarantee holds."""
    import torch
    import torch.nn.functional as F

    s0, S = scales[0], scales[-1]
    M = {s: bin_mosaic_torch(noisy, onehot, s)[0] for s in scales}
    D = {s: bin_mosaic_torch(pred, onehot, s)[0] for s in scales}
    dens = torch.tensor([0.25, 0.5, 0.25], device=pred.device).view(1, 3, 1, 1)
    s2 = torch.stack([((sigma ** 2) * onehot[:, c:c + 1]).sum()
                      / onehot[:, c:c + 1].sum().clamp(min=1.0)
                      for c in range(3)]).view(1, 3, 1, 1)
    # BILINEAR upsampling throughout: the fused bands carry measurement
    # noise (sigma/n per block, non-negligible on very noisy images) and
    # nearest upsampling turns it into visible checkers of colored squares;
    # tent interpolation makes every injected correction piecewise-linear.
    # The finest fusion band starts at 16 for the same reason: at 8 the
    # per-block measurement noise is strongest and the model needs no help.
    up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
    blur = lambda t: F.avg_pool2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), 3, stride=1)

    # Weights are LOCAL (per cell, 3x3-smoothed): a global per-tile weight
    # injects the measurement's band-pass response of hard edges — a
    # difference-of-smoothings halo that shows as a saturated outline along
    # high-contrast borders. Locally, structure makes the band discrepancy
    # huge, which must drive the blend toward the model exactly there.
    # Floor band: structure-gated anchor — pure measurement where the local
    # discrepancy is noise-sized (the dilution guarantee on smooth content),
    # the model where structure dominates (edges).
    # T is a chi^2-quantile guard: the local mean of d^2 over a 3x3 cell
    # neighbourhood has ~9 effective samples, so pure-noise cells fluctuate
    # up to ~2x their expectation; subtracting T*vn (not vn) keeps them
    # clamped at zero (anchor intact) while structure exceeds it by orders
    # of magnitude and is unaffected.
    T = 2.5
    # Floor gate: the discrepancy D-M cannot discriminate a real edge from
    # the model drifting on flat content (both are large) — the MEASUREMENT
    # must attest the structure. Local variance of the binned measurement,
    # noise-corrected: flat cells anchor to the measurement no matter what
    # the model says (the dilution guarantee), structured cells keep the
    # model (a block average across an edge mixes both sides and would
    # bleed chroma as a saturated outline).
    vn_S = s2 / (dens * S * S)
    mloc = M[S] - blur(M[S])
    struct_S = (blur(mloc * mloc) - T * vn_S).clamp(min=0.0)
    w_S = struct_S / (struct_S + vn_S + 1e-20)  # ->0 flat (anchor), ->1 edges
    fused = w_S * D[S] + (1 - w_S) * M[S]
    for s in reversed(scales[:-1]):
        band_d = D[s] - up(D[2 * s])
        band_m = M[s] - up(M[2 * s])
        vn = s2 * (1.0 / (dens * s * s) - 1.0 / (dens * 4 * s * s))
        vm = (blur((band_d - band_m) ** 2) - T * vn).clamp(min=0.0)
        w = vn / (vn + vm + 1e-20)  # ->1 flat (model band), ->1 nowhere harmful
        fused = up(fused) + w * band_d + (1 - w) * band_m
    corr = F.interpolate(fused - D[s0], scale_factor=s0, mode="bilinear",
                         align_corners=False)
    return pred + (corr * onehot).sum(dim=1, keepdim=True)


def anchor_low_band(pred, noisy, onehot, scale: int):
    """Replace the prediction's per-channel low band with the NOISY input's.

    Below the last scale a denoiser can learn, the n-averaged measurement is
    the true diluted estimate (sigma/n is sub-visible) while the network's
    low band carries accumulated model error — measured 14+ dB WORSE than the
    raw input on flat charts. Anchoring restores the dilution floor exactly
    and, as a side effect, zeroes the low-band gradient so training
    specializes the networks on their passband. Mirrored bit-exactly by the
    C inference side."""
    import torch.nn.functional as F

    rgb_in, _ = bin_mosaic_torch(noisy, onehot, scale)
    rgb_out, _ = bin_mosaic_torch(pred, onehot, scale)
    corr = F.interpolate(rgb_in - rgb_out, scale_factor=scale, mode="nearest")
    return pred + (corr * onehot).sum(dim=1, keepdim=True)
