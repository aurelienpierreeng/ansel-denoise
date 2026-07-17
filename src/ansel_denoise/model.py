"""CFA-agnostic denoising U-Net.

Deliberately boring: plain convolutions, GELU, nearest-neighbor upsampling,
skip concatenations, residual output (the network predicts noise, which is
subtracted from the noisy channel). No attention, no normalization layers —
every op has a direct OpenCL translation in Ansel's rawdenoise module, and
the receptive field (which sets the tiling overlap on the C side) is a simple
function of depth.

Input  (5, H, W): [noisy mosaic, R/G/B one-hot CFA planes, sigma map]
Output (1, H, W): denoised mosaic
H and W must be multiples of 2**depth (the trainer's patch size guarantees it;
the C side pads tiles).
"""

from __future__ import annotations

import torch
import torch.nn as nn

IN_CHANNELS = 5
OUT_CHANNELS = 1


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GELU(),
    )


class UNet(nn.Module):
    def __init__(self, base: int = 32, depth: int = 4):
        super().__init__()
        self.cfg = {"arch": "unet", "base": base, "depth": depth,
                    "in_channels": IN_CHANNELS, "out_channels": OUT_CHANNELS}
        widths = [base * 2**i for i in range(depth + 1)]

        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        cin = IN_CHANNELS
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
        self.head = nn.Conv2d(widths[0], OUT_CHANNELS, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noisy = x[:, :1]
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
        return noisy - self.head(h)


def build_model(base: int = 32, depth: int = 4) -> UNet:
    return UNet(base=base, depth=depth)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
