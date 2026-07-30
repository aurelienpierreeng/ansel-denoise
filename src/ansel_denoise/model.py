"""CFA-agnostic denoising U-Nets.

Deliberately boring: plain convolutions, GELU, nearest-neighbor upsampling,
skip concatenations, residual output (a network predicts noise, which is
subtracted from its input planes). No attention, no normalization layers —
every op has a direct OpenCL translation in Ansel's rawdenoise module, and
the receptive field (which sets the tiling overlap on the C side) is a simple
function of depth.

Two architectures share the same building blocks:

- `UNet` (arch "unet") — the single-scale mosaic denoiser.
  Input (in_channels, H, W); the first `out_channels` planes are the noisy
  signal the residual is subtracted from. The v1 layout is
  [noisy mosaic, R/G/B one-hot CFA planes, sigma map] -> denoised mosaic.
- `MSUNet` (arch "unet-ms") — the multi-scale pair: a `coarse` UNet on
  superpixel-binned RGB planes [R, G, B, sigmaR, sigmaG, sigmaB] -> denoised
  RGB (3-plane residual), whose nearest-upsampled output is injected as 3
  guide planes into a `fine` UNet
  [mosaic, one-hots, sigma, guide RGB] -> denoised mosaic.
  The stages are separate submodules on purpose: they run at different
  resolutions with the binning/upsampling glue living OUTSIDE the network
  (training loop and C inference do it identically), and the state-dict
  prefixes `coarse.` / `fine.` are the .anselnn tensor-naming contract.

H and W must be multiples of 2**depth (the trainer's patch size guarantees
it; the C side pads tiles).
"""

from __future__ import annotations

import torch
import torch.nn as nn

IN_CHANNELS = 5
OUT_CHANNELS = 1

# unet-ms plane layouts (cross-repo contract with the C inference side)
COARSE_IN_CHANNELS = 6   # [R, G, B, sigmaR, sigmaG, sigmaB]
COARSE_OUT_CHANNELS = 3  # denoised RGB
FINE_IN_CHANNELS = IN_CHANNELS + COARSE_OUT_CHANNELS


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GELU(),
    )


class UNet(nn.Module):
    def __init__(self, base: int = 32, depth: int = 4,
                 in_channels: int = IN_CHANNELS, out_channels: int = OUT_CHANNELS):
        super().__init__()
        self.cfg = {"arch": "unet", "base": base, "depth": depth,
                    "in_channels": in_channels, "out_channels": out_channels}
        widths = [base * 2**i for i in range(depth + 1)]

        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        cin = in_channels
        for w in widths[:-1]:
            self.enc.append(_block(cin, w))
            self.down.append(nn.Conv2d(w, w, 2, stride=2))
            cin = w
        self.bottleneck = _block(widths[-2], widths[-1])

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for w_skip, w in zip(reversed(widths[:-1]), reversed(widths[1:])):
            self.up.append(nn.Conv2d(w, w_skip, 1))
            self.dec.append(_block(2 * w_skip, w_skip))
        self.head = nn.Conv2d(widths[0], out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = x[:, : self.cfg["out_channels"]]
        skips = []
        h = x
        for enc, down in zip(self.enc, self.down):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            h = up(nn.functional.interpolate(h, scale_factor=2, mode="nearest"))
            h = dec(torch.cat([skip, h], dim=1))
        return signal - self.head(h)


class MSUNet(nn.Module):
    """Multi-scale pair. `forward` runs only the FINE stage on a pre-built
    8-plane input — the binning / coarse forward / upsample glue is the
    caller's job (training loop, benchmark, C inference), so that the exact
    same glue code is exercised everywhere. `coarse` is called explicitly."""

    def __init__(self, coarse_base: int = 32, coarse_depth: int = 3,
                 fine_base: int = 32, fine_depth: int = 4):
        super().__init__()
        self.coarse = UNet(base=coarse_base, depth=coarse_depth,
                           in_channels=COARSE_IN_CHANNELS,
                           out_channels=COARSE_OUT_CHANNELS)
        self.fine = UNet(base=fine_base, depth=fine_depth,
                         in_channels=FINE_IN_CHANNELS,
                         out_channels=OUT_CHANNELS)
        self.cfg = {
            "arch": "unet-ms",
            "coarse": {"base": coarse_base, "depth": coarse_depth,
                       "in_channels": COARSE_IN_CHANNELS,
                       "out_channels": COARSE_OUT_CHANNELS},
            "fine": {"base": fine_base, "depth": fine_depth,
                     "in_channels": FINE_IN_CHANNELS,
                     "out_channels": OUT_CHANNELS},
            "bin": {"bayer": 4, "xtrans": 6},
            "guide": "coarse-rgb-nearest",
            "anchor": 32,
            # floor mode of the low-band fusion: "gated" is only safe for
            # models trained with the DC-ownership loss (unfused output) —
            # they own their local means, so the structure gate can hand
            # edges to the model and kill the box-mixing outline. Models
            # trained with the fused loss MUST keep "anchored" (their DC
            # drifts and the gate would let it through).
            "floor": "gated",
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fine(x)


def build_model(base: int = 32, depth: int = 4) -> UNet:
    return UNet(base=base, depth=depth)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
