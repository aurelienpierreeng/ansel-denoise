"""Quality metrics beyond PSNR.

LFCE (low-frequency chroma error) is the headline metric for the multi-scale
work: the defect it measures — colored low-frequency speckles surviving
denoising at high ISO — is nearly invisible to PSNR (small per-pixel
amplitude) but dominates perceived quality. Binning the residual per CFA
channel shrinks white noise by 1/sqrt(n) while leaving spatially-correlated
error intact, so the binned chroma planes isolate exactly the blotch failure
mode.
"""

from __future__ import annotations

import math

import torch

from .cfa import bin_mosaic_torch

LFCE_BINS = (4, 16, 64)


def psnr(pred: torch.Tensor, clean: torch.Tensor) -> float:
    """PSNR in dB on the [0, 1] normalized domain."""
    mse = torch.mean((pred - clean) ** 2).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def lfce(residual: torch.Tensor, onehot: torch.Tensor,
         bins: tuple[int, ...] = LFCE_BINS) -> dict[int, float]:
    """Low-frequency chroma error of a mosaic residual (pred - clean).

    residual (B, 1, H, W), onehot (B, 3, H, W). For each bin size N the
    residual is superpixel-binned per channel; U = R - G and V = B - G are
    WB-free pseudo-chroma planes; the reported value is
    10*log10(mean(U^2 + V^2)) in dB (lower = better). A perfect denoiser
    scores -inf; correlated chroma blotches raise it sharply.
    """
    out = {}
    for n in bins:
        rgb, _ = bin_mosaic_torch(residual, onehot, n)
        u = rgb[:, 0] - rgb[:, 1]
        v = rgb[:, 2] - rgb[:, 1]
        energy = torch.mean(u * u + v * v).item()
        out[n] = 10.0 * math.log10(max(energy, 1e-20))
    return out
