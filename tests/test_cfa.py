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
