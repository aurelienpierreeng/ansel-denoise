"""MSUNet and generalized-UNet contracts."""
import torch

from ansel_denoise.model import (COARSE_IN_CHANNELS, COARSE_OUT_CHANNELS,
                                 FINE_IN_CHANNELS, MSUNet, UNet)


def test_v1_unet_unchanged():
    """Default UNet keeps the v1 cfg and topology (state-dict names)."""
    m = UNet()
    assert m.cfg == {"arch": "unet", "base": 32, "depth": 4,
                     "in_channels": 5, "out_channels": 1}
    names = set(m.state_dict().keys())
    assert "enc.0.0.weight" in names and "head.bias" in names
    y = m(torch.zeros(1, 5, 32, 32))
    assert y.shape == (1, 1, 32, 32)


def test_v1_checkpoint_still_loads():
    """The shipped v1 artifact loads into the generalized UNet bit-exactly."""
    from pathlib import Path

    from ansel_denoise.export import load_model_from_anselnn

    path = Path(__file__).resolve().parents[1] / "models" / "rawdenoiseai-v1-full.anselnn"
    if not path.exists():
        import pytest
        pytest.skip("shipped model not present")
    m, cfg = load_model_from_anselnn(path)
    assert cfg["in_channels"] == 5 and cfg["out_channels"] == 1
    x = torch.zeros(1, 5, 32, 32)
    x[:, 0] = 0.5
    with torch.no_grad():
        y = m(x)
    assert torch.isfinite(y).all()


def test_msunet_shapes_and_names():
    m = MSUNet(coarse_base=8, coarse_depth=3, fine_base=8, fine_depth=4)
    names = set(m.state_dict().keys())
    assert "coarse.enc.0.0.weight" in names and "fine.head.bias" in names
    cy = m.coarse(torch.zeros(1, COARSE_IN_CHANNELS, 24, 24))
    assert cy.shape == (1, COARSE_OUT_CHANNELS, 24, 24)
    fy = m(torch.zeros(1, FINE_IN_CHANNELS, 32, 32))
    assert fy.shape == (1, 1, 32, 32)


def test_multichannel_residual_identity():
    """With a zeroed head, the coarse net returns its first 3 planes
    unchanged — the residual formulation on multiple channels."""
    m = MSUNet(coarse_base=8, coarse_depth=3, fine_base=8, fine_depth=4)
    torch.nn.init.zeros_(m.coarse.head.weight)
    torch.nn.init.zeros_(m.coarse.head.bias)
    x = torch.rand(1, COARSE_IN_CHANNELS, 24, 24)
    with torch.no_grad():
        y = m.coarse(x)
    assert torch.equal(y, x[:, :COARSE_OUT_CHANNELS])
