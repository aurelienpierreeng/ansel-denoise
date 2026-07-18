# Network architecture: how the denoiser works and what it assumes

This documents the neural network trained by this repository and executed by
Ansel's `rawdenoiseai` module — the exact architecture, the input encoding,
the assumptions baked into the design, and how the network exploits
correlations between color channels despite their spatial offsets in the
mosaic. The runtime integration is documented on the
[Ansel side](https://github.com/aurelienpierreeng/ansel/blob/master/doc/rawdenoiseai.md);
this page is the training-side deep dive. The reference implementation is
[`src/ansel_denoise/model.py`](../src/ansel_denoise/model.py) — deliberately
~70 lines, and this page explains every decision in them.

## The problem, stated precisely

Given a **non-demosaiced sensor mosaic** — one intensity per photosite,
black-subtracted and normalized by the white level (`rawprepare`'s output),
**before white balance** — estimate the noise-free mosaic. The noise is
heteroscedastic Poisson-Gaussian: at a photosite of channel $c$ with true
normalized intensity $x$, the observed value is a random variable with

$$\mathrm{Var}(x) = a_c \cdot x + b_c$$

where $(a_c, b_c)$ come from Ansel's per-camera, per-ISO
[noise profiles](../data/noiseprofiles.json). The term $a_c x$ is shot noise
(photon counting is a Poisson process), $b_c$ is the signal-independent
electronic floor (read noise, quantization).

## Input encoding: 5 planes, full resolution

| plane | content |
|---|---|
| 0 | the noisy mosaic, one value per photosite |
| 1–3 | one-hot CFA maps: plane 1 is 1.0 where the photosite is red, etc. |
| 4 | per-pixel noise sigma $\sigma = \sqrt{a_c \max(x, 0) + b_c}$ |

Three properties of this encoding do the heavy lifting:

**Full resolution, single mosaic plane.** The common alternative — packing a
Bayer mosaic into four aligned half-resolution R/G1/G2/B planes — buys
explicit channel alignment but halves the spatial resolution of every
feature map, discards the sampling phase, and is structurally Bayer-only.
Keeping the mosaic in one full-resolution plane preserves the exact spatial
relationship between every sample, and works identically for any CFA period
(Bayer 2×2, X-Trans 6×6, or anything a future sensor ships).

**The CFA layout is data, not architecture.** The one-hot planes tell the
network which color each photosite measured. Nothing about the CFA is
hard-coded in the weights, which is why *one* set of weights serves Bayer
and X-Trans simultaneously, and why a newly profiled camera needs no
retraining.

**The noise level is data, not architecture, either.** The sigma plane
conditions the network on how much noise to expect *at each photosite* —
strong in shadows of a high-ISO file, nearly zero in base-ISO highlights.
The same weights therefore serve every ISO of every profiled camera, and
Ansel's user-facing *strength* slider is nothing but a multiplier on this
plane. Sigma is computed **from the noisy values themselves** (`sigma_map`
in [`noise.py`](../src/ansel_denoise/noise.py)) because true intensities are
unavailable at inference — see the assumptions below.

## The network: a deliberately boring U-Net

Plain convolutions, GELU activations, nearest-neighbor upsampling, skip
concatenations, residual output. No normalization layers, no attention.
For width $w$ = 32 (full model, 7.59M parameters) or 16 (distilled, 1.90M)
and depth 4:

```
input (5, H, W)
encoder level i = 0..3:   [conv3x3 → GELU → conv3x3 → GELU]  width w·2^i
                          then learned 2×2 stride-2 downsampling conv
bottleneck:               [conv3x3 → GELU → conv3x3 → GELU]  width w·16
decoder level i = 3..0:   2× nearest upsample → 1×1 conv (halve width)
                          → concat encoder skip → [conv3x3 → GELU → ×2]
head:                     conv3x3 → 1 plane = predicted noise
output:                   noisy mosaic − predicted noise
```

Each design choice has one owner:

- **Residual output** (predict the noise, subtract it): the identity
  transform — return the input untouched — is the zero function, the
  easiest thing a network can learn. Clean regions therefore cost nothing,
  and the network spends its capacity modeling noise, not reconstructing
  images. A fraction of training patches carries (near-)zero synthesized
  noise specifically to anchor this identity.
- **No normalization layers**: batch/instance statistics make the output of
  a pixel depend on the rest of the batch or frame — unacceptable for a
  deterministic, tile-based pipeline where the same pixels must produce the
  same result regardless of tiling. Conditioning comes from the sigma plane
  instead.
- **No attention, nearest-neighbor upsampling, exact GELU**: every operator
  has a direct, bit-parity C and OpenCL translation in Ansel
  (`src/common/nn_model.c`, `data/kernels/rawdenoiseai.cl`, verified to
  1.79e-07 against this implementation). The operator vocabulary is chosen
  for the executor we control, not for benchmark fashion.
- **Depth 4** puts the bottleneck at 1/16 resolution: at that scale a 3×3
  convolution spans 48 mosaic pixels, several X-Trans periods — enough
  context to tell texture from chance. The **measured** effective receptive
  field (impulse response of the trained network) decays to ~1e-6 of peak
  at a radius of 32 pixels and to zero by 96, which is what sets the
  48-pixel tile overlap on the C side — measured, not assumed.

## How inter-channel correlations are exploited

Natural images have strongly correlated color channels — edges, textures and
gradients appear in R, G and B together — and exploiting a neighbor's sample
of *another* channel is the whole secret of demosaicing. In a mosaic these
correlated samples are spatially offset, which is exactly why the encoding
above matters:

Because the mosaic stays in **one full-resolution plane**, every 3×3
convolution — from the very first layer — sees a neighborhood containing
samples of **all** channels at their native offsets, and the one-hot planes
tell it which channel each sample is. The network can therefore learn
channel-conditional context: "a red photosite whose green neighbors show a
vertical edge probably sits on that edge too". This is available at every
layer and every scale of the U-Net, so cross-channel evidence flows into
the estimate of every photosite, spatial offset included. Green — sampled
2× (Bayer) to 2.5× (X-Trans) denser than red/blue — effectively serves as a
high-SNR luminance scaffold for the sparser channels, the same asymmetry
demosaicing algorithms exploit deliberately, here learned from data.

Two empirical confirmations that this is used, not merely available: the
same weights generalize across Bayer *and* X-Trans (impossible with a
learned fixed channel geometry — the network must be reading the CFA planes),
and chroma noise is suppressed (an inherently cross-channel inference).

## Assumptions — what the model relies on

1. **The noise is Poisson-Gaussian per photosite, spatially independent.**
   Synthesis (and therefore the learned prior) models shot noise, Gaussian
   read noise, ADU quantization, white clipping and signed shadow
   excursions. It does **not** model row/banding noise, fixed-pattern noise,
   PRNU, or hot pixels — structured artifacts are out of distribution
   (Ansel's `hotpixels` module still owns that job).
2. **The noise profile is approximately right.** The sigma map comes from
   the camera's $(a, b)$ profile interpolated at the image ISO. Training
   jitters profiles deliberately, so a mis-estimated sigma degrades
   gracefully rather than catastrophically — this same tolerance is what
   makes the strength slider meaningful.
3. **Sigma from noisy values.** $\sigma$ is computed from the *observed*
   intensity, not the (unknown) true one — an approximation that
   overestimates noise on positive excursions and underestimates it on
   negative ones. Training uses the identical approximation, so the network
   learns with the exact conditioning statistics it will see at inference.
   Training/inference symmetry beats theoretical exactness here.
4. **The input domain is Ansel's, exactly.** Black-subtracted, white-
   normalized, **pre-white-balance** mosaic — the module runs at that exact
   pipeline point (before `temperature`, CA correction and demosaic).
   Feeding developed or white-balanced data is out of domain by design.
5. **Exactly one color per photosite.** The one-hot encoding assumes a CFA;
   monochrome sensors (Leica M Monochrom, Pentax Monochrome) are excluded
   at harvest (`num_colors != 3`) and at runtime (no mosaic → module
   unavailable). Supporting them is a training-time extension, not an
   inference switch.
6. **Base-ISO tiles are "clean".** Ground truth carries the base-ISO noise
   floor of the source cameras; the achievable PSNR is capped by it. The
   harvest gates at ISO ≤ 200 to keep that floor low.
7. **Held-out-camera validation measures what we care about.** The
   train/validation split is by *camera model* (deterministic hash), so
   reported gains are cross-sensor generalization, never memorization of a
   sensor's quirks.

## What it cannot do, by design

The network is a noise-splitting filter, not a generative model: the
residual formulation and the L1 objective tie every output pixel to its
noisy measurement — there is no mechanism to invent content, and the
training corpus contains no semantic supervision to invent it from. Its
failure mode under excessive strength is over-smoothing toward the local
mean, not hallucination.
