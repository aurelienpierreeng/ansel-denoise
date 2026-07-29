import numpy as np
import pytest

from ansel_denoise.cfa import XTRANS, BAYER_RGGB, aligned_offset, colors_map, normalize_pattern, one_hot


def test_normalize_pattern_maps_g2():
    p = np.array([[0, 1], [3, 2]], dtype=np.uint8)  # libraw RGGB with G2=3
    assert (normalize_pattern(p) == BAYER_RGGB.T).all() or (normalize_pattern(p) == [[0, 1], [1, 2]]).all()


def test_colors_map_periodicity_and_offset():
    m = colors_map(XTRANS, 24, 24)
    assert (m[:6, :6] == XTRANS).all()
    assert (m[6:12, 12:18] == XTRANS).all()
    # offset by one period == no offset
    assert (colors_map(XTRANS, 12, 12, 6, 6) == colors_map(XTRANS, 12, 12)).all()
    # sub-period offset shifts the pattern
    assert (colors_map(BAYER_RGGB, 2, 2, 1, 1) == [[2, 1], [1, 0]]).all()


def test_xtrans_color_ratios():
    m = colors_map(XTRANS, 6, 6)
    counts = np.bincount(m.ravel(), minlength=3)
    assert counts.tolist() == [8, 20, 8]  # X-Trans: 8R, 20G, 8B per 6x6


def test_one_hot():
    m = colors_map(BAYER_RGGB, 4, 4)
    oh = one_hot(m)
    assert oh.shape == (3, 4, 4)
    assert (oh.sum(axis=0) == 1).all()
    assert oh[1].sum() == 8  # half the Bayer sensels are green


def test_aligned_offset():
    rng = np.random.default_rng(0)
    for _ in range(100):
        off = aligned_offset(rng, 100, 30, 6)
        assert off % 6 == 0 and 0 <= off <= 70
    with pytest.raises(ValueError):
        aligned_offset(rng, 10, 20, 2)


def test_bin_for_pattern():
    from ansel_denoise.cfa import BAYER_RGGB, XTRANS, bin_for_pattern

    assert bin_for_pattern(BAYER_RGGB) == 4
    assert bin_for_pattern(XTRANS) == 6


def test_bin_mosaic_counts_and_means():
    """Superpixel counts are exact per CFA family and the count-weighted mean
    reproduces a flat field bit-exactly — the cross-repo binning contract."""
    import torch

    from ansel_denoise.cfa import (BAYER_RGGB, XTRANS, bin_mosaic_torch,
                                   colors_map, one_hot)

    for pattern, bin_factor, expected in [
        (BAYER_RGGB, 4, (4, 8, 4)),
        (XTRANS, 6, (8, 20, 8)),
    ]:
        size = 2 * bin_factor
        oh = torch.from_numpy(one_hot(colors_map(pattern, size, size)))[None]
        mosaic = torch.full((1, 1, size, size), 0.25)
        rgb, counts = bin_mosaic_torch(mosaic, oh, bin_factor)
        assert tuple(int(v) for v in counts[0, :, 0, 0]) == expected
        assert torch.allclose(rgb, torch.full_like(rgb, 0.25))


def test_bin_sigma_matches_analytic():
    """Coarse sigma derived from the fine sigma plane equals the analytic
    sigma of the mean of n sensels."""
    import numpy as np
    import torch

    from ansel_denoise.cfa import (BAYER_RGGB, bin_mosaic_torch, bin_sigma_torch,
                                   colors_map, one_hot)
    from ansel_denoise.noise import sigma_map, sigma_map_binned

    a = np.array([2e-4, 1e-4, 3e-4])
    b = np.array([1e-6, 2e-6, 1.5e-6])
    colors = colors_map(BAYER_RGGB, 64, 64)
    oh = torch.from_numpy(one_hot(colors))[None]
    fine = sigma_map(np.full((64, 64), 0.3, np.float32), colors, a, b)
    _, counts = bin_mosaic_torch(torch.zeros(1, 1, 64, 64), oh, 4)
    from_fine = bin_sigma_torch(torch.from_numpy(fine)[None, None], oh, 4, counts)
    direct = sigma_map_binned(np.full((3, 16, 16), 0.3, np.float32), a, b,
                              np.array([4, 8, 4]))
    assert float((from_fine[0] - torch.from_numpy(direct)).abs().max()) < 1e-6
