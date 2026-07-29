"""The noise synthesizer must reproduce the profiled statistics exactly:
Var(x) = a*x + b is the contract between this repo and Ansel's profiling."""

import numpy as np
import pytest

from ansel_denoise.cfa import BAYER_RGGB, colors_map
from ansel_denoise.noise import sigma_map, synthesize


@pytest.mark.parametrize("level", [0.02, 0.2, 0.8])
def test_variance_matches_model(level):
    rng = np.random.default_rng(42)
    h = w = 512
    colors = colors_map(BAYER_RGGB, h, w)
    clean = np.full((h, w), level, dtype=np.float32)
    # realistic magnitudes straight out of noiseprofiles.json
    a = np.array([4.5e-6, 2.2e-6, 3.2e-6]) * 100  # ~ISO 3200 territory
    b = np.array([2.6e-8, 2.1e-8, 2.7e-8]) * 100

    noisy = synthesize(clean, colors, a, b, rng)
    for c in range(3):
        sel = colors == c
        expected_var = a[c] * level + b[c]
        measured_var = np.var(noisy[sel] - level)
        assert measured_var == pytest.approx(expected_var, rel=0.10)
        # unbiased mean (no clipping at this level)
        assert np.mean(noisy[sel]) == pytest.approx(level, abs=5 * np.sqrt(expected_var / sel.sum()))


def test_variance_line_with_fit_artifacts():
    # Real database entries can have a < 0 or b < 0 on a channel; the
    # synthesizer must still realize Var = max(a*x + b, 0).
    rng = np.random.default_rng(3)
    h = w = 512
    colors = colors_map(BAYER_RGGB, h, w)
    level = 0.3
    clean = np.full((h, w), level, dtype=np.float32)
    a = np.array([-4.6e-7, 1.7e-5, 5e-6])  # negative shot-noise slope on R
    b = np.array([5.4e-6, -1.4e-6, 1e-7])  # negative intercept on G

    noisy = synthesize(clean, colors, a, b, rng)
    for c in range(3):
        expected = max(a[c] * level + b[c], 0.0)
        assert np.var(noisy[colors == c] - level) == pytest.approx(expected, rel=0.10)


def test_zero_noise_is_identity():
    rng = np.random.default_rng(0)
    colors = colors_map(BAYER_RGGB, 64, 64)
    clean = rng.uniform(0, 1, (64, 64)).astype(np.float32)
    noisy = synthesize(clean, colors, np.zeros(3), np.zeros(3), rng)
    assert np.allclose(noisy, clean, atol=1e-7)


def test_clipping_and_shadow_floor():
    rng = np.random.default_rng(1)
    colors = colors_map(BAYER_RGGB, 128, 128)
    a = np.full(3, 1e-3)
    b = np.full(3, 1e-4)

    bright = synthesize(np.ones((128, 128), np.float32), colors, a, b, rng)
    assert bright.max() <= 1.0

    dark = synthesize(np.zeros((128, 128), np.float32), colors, a, b, rng, black_frac=0.05)
    assert dark.min() >= -0.05 - 1e-6
    assert dark.min() < 0.0  # signed shadow noise is preserved


def test_quantization():
    rng = np.random.default_rng(2)
    colors = colors_map(BAYER_RGGB, 32, 32)
    step = 1.0 / 4096
    noisy = synthesize(
        np.full((32, 32), 0.3, np.float32), colors, np.full(3, 1e-5), np.full(3, 1e-7), rng,
        quant_step=step,
    )
    assert np.allclose(noisy / step, np.round(noisy / step), atol=1e-4)


def test_sigma_map_matches_model():
    colors = colors_map(BAYER_RGGB, 16, 16)
    a, b = np.array([1e-4, 2e-4, 3e-4]), np.array([1e-6, 2e-6, 3e-6])
    noisy = np.full((16, 16), 0.5, dtype=np.float32)
    s = sigma_map(noisy, colors, a, b)
    for c in range(3):
        assert np.allclose(s[colors == c], np.sqrt(a[c] * 0.5 + b[c]), rtol=1e-5)
    # negative values must not produce NaNs
    assert np.isfinite(sigma_map(np.full((4, 4), -0.02, np.float32), colors[:4, :4], a, b)).all()


def test_synthesize_binned_variance():
    """Var of the binned synthesis matches (a*mean + b)/n within 2%."""
    import numpy as np

    from ansel_denoise.noise import synthesize_binned

    rng = np.random.default_rng(5)
    a = np.array([2e-4, 1e-4, 3e-4])
    b = np.array([1e-6, 2e-6, 1.5e-6])
    n = np.array([4, 8, 4])
    clean = np.full((3, 200, 200), 0.3)
    noisy = synthesize_binned(clean, a, b, n, rng)
    for c in range(3):
        want = (a[c] * 0.3 + b[c]) / n[c]
        got = float(np.var(noisy[c] - clean[c]))
        assert abs(got / want - 1.0) < 0.02


def test_synthesize_quant_step_quantizes():
    """With quant_step set, mosaic synthesis lands on exact ADU levels."""
    import numpy as np

    from ansel_denoise.noise import synthesize

    clean = np.full((32, 32), 0.05, np.float32)
    colors = np.zeros((32, 32), np.uint8)
    noisy = synthesize(clean, colors, np.array([1e-5] * 3), np.array([1e-8] * 3),
                       np.random.default_rng(2), quant_step=1.0 / 4000)
    adu = noisy * 4000.0
    assert np.allclose(adu, np.round(adu), atol=1e-3)
