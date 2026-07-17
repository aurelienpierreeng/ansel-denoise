"""Poisson-Gaussian raw noise synthesis from Ansel noise profiles.

Model (per pixel, on the normalized black-subtracted mosaic x in [0, 1]):

    x_noisy = a * Poisson(x / a) + Normal(0, sqrt(b))

which yields exactly Var = a * x + b, the model fitted by Ansel's profiling.
`a` and `b` are per-channel and routed through the CFA color map.

The shipped database is an unconstrained least-squares fit: individual
channels can have a <= 0 or b < 0 (e.g. Canon EOS 550D blue channel). The
parameters are therefore treated as a fitted *variance line*, not as physical
gains: the exact Poisson+Gaussian decomposition is used only where a > 0 and
b >= 0 with photon counts low enough to be skewed; everywhere else the noise
is heteroscedastic Gaussian with Var = max(a*x + b, 0), which reproduces the
profiled statistics in every case.

Sensor realism knobs (all on by default):
  - quantization to the sensor's ADU grid (given a bit depth and black level),
  - clipping at the white level,
  - negative excursions below black are preserved down to -black_level
    (rawprepare subtracts black without clamping, so real shadow noise is
    signed; clamping at 0 during synthesis would bias the mean and teach the
    network a wrong shadow prior).

The sigma map used to condition the network is computed from the *noisy*
values because that is all the inference side has:  sigma = sqrt(a*max(x,0)+b).
"""

from __future__ import annotations

import numpy as np

# Above this photon count the Poisson is numerically Gaussian; sampling the
# normal approximation is much faster and avoids float32 saturation.
_POISSON_GAUSSIAN_CROSSOVER = 1000.0


def synthesize(
    clean: np.ndarray,
    colors: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    white: float = 1.0,
    black_frac: float = 0.0,
    quant_step: float | None = None,
) -> np.ndarray:
    """Return a noisy realization of `clean` (float32, (H, W) mosaic).

    clean       normalized mosaic, values in [0, white]
    colors      (H, W) color index map in {0, 1, 2}
    a, b        per-channel Poisson gain / Gaussian variance, shape (3,)
    black_frac  black level as a fraction of (white - black); sets the lower
                clip bound to -black_frac
    quant_step  ADU quantization step in normalized units, e.g.
                1 / (2**14 - black_adu) for a 14-bit sensor; None disables
    """
    clean = np.asarray(clean, dtype=np.float64)
    a_map = np.asarray(a, dtype=np.float64)[colors]
    b_map = np.asarray(b, dtype=np.float64)[colors]

    noisy = np.empty_like(clean)

    # Exact Poisson + Gaussian decomposition where it is physical (a > 0,
    # b >= 0) and photon counts are small enough to be genuinely skewed.
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(a_map > 0, np.maximum(clean, 0.0) / np.where(a_map > 0, a_map, 1.0), 0.0)
    pz = (a_map > 0) & (b_map >= 0) & (lam < _POISSON_GAUSSIAN_CROSSOVER)
    noisy[pz] = (
        rng.poisson(lam[pz]) * a_map[pz]
        + rng.normal(0.0, 1.0, size=int(pz.sum())) * np.sqrt(b_map[pz])
    )

    # Heteroscedastic Gaussian on the fitted variance line everywhere else
    # (high counts, and the non-physical a <= 0 / b < 0 fit artifacts).
    gz = ~pz
    var = np.maximum(a_map[gz] * np.maximum(clean[gz], 0.0) + b_map[gz], 0.0)
    noisy[gz] = clean[gz] + rng.normal(0.0, 1.0, size=int(gz.sum())) * np.sqrt(var)

    if quant_step is not None and quant_step > 0:
        noisy = np.round(noisy / quant_step) * quant_step

    np.clip(noisy, -abs(black_frac), white, out=noisy)
    return noisy.astype(np.float32)


def sigma_map(noisy: np.ndarray, colors: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel noise standard deviation estimated from noisy values — the
    conditioning input of the network, identical at training and inference."""
    a_map = np.asarray(a, dtype=np.float64)[colors]
    b_map = np.asarray(b, dtype=np.float64)[colors]
    var = a_map * np.maximum(np.asarray(noisy, dtype=np.float64), 0.0) + b_map
    return np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
