# Ansel Training Data License, version 1.0 (ATDL-1.0)

This license applies to the raw-sensor image tiles ("the Tiles") distributed
as community-contributed training shards of the ansel-denoise corpus.
Contributors keep the copyright on their photographs. By contributing, they
grant the rights below; by downloading or using the Tiles, you accept all of
the conditions below.

## 1. Who may use the Tiles

The Tiles may be used solely by:

- the [Ansel project](https://ansel.photos), and
- any person or organization reproducing the Ansel denoiser training
  workflow, as published in this repository, on their own infrastructure.

## 2. What the Tiles may be used for

The Tiles may be used **only to train, validate and test image-denoising
neural networks**, in the manner documented in the
[Ansel denoiser design page](https://github.com/aurelienpierreeng/ansel/blob/master/doc/rawdenoiseai.md):
the Tiles serve as **clean ground truth** and are **synthetically corrupted
with sensor noise** so that the network learns to separate noise from image
detail. Incidental technical operations required by that purpose —
downloading, caching, format conversion, computing statistics over the
corpus — are permitted.

## 3. What the Tiles may not be used for

**Every use not explicitly permitted by section 2 is forbidden, and by
accepting this license you explicitly accept that prohibition.** In
particular, and without limiting the previous sentence, the Tiles may not
be:

- used to train, fine-tune, evaluate or prompt **generative models** of any
  kind — image synthesis, inpainting, upscaling-as-generation, multimodal
  or language models;
- incorporated into or redistributed as part of any other dataset;
- used to identify, profile or locate persons, property or places depicted;
- sold, sublicensed, or used for advertising;
- displayed or published as photographs — the Tiles are training data, not
  an image collection.

## 4. Redistribution

Copies of the Tiles, whole or partial, must carry this license file, and
recipients are bound by it. The canonical distribution point is the shard
release of this repository, where this license is published alongside the
data.

## 5. Removal

A contributor may request the removal of their Tiles at any time by opening
an issue on this repository. Removed Tiles must not be used in any
subsequent training. Neural-network weights trained before the removal are
unaffected, as the Tiles cannot be extracted from them.

## 6. Termination

Any breach of these conditions immediately terminates your rights under
this license. Continued use after termination is copyright infringement.

## 7. No warranty

The Tiles are provided as-is, with no warranty of any kind, including
fitness for a particular purpose. Neither the contributors nor the Ansel
project are liable for any use of the Tiles.

## 8. Contributor grant

By contributing Tiles, the contributor declares owning the necessary rights
to the source photographs and grants the Ansel project and the users defined
in section 1 a non-exclusive, worldwide, royalty-free license to use the
Tiles under the conditions of sections 2–5.
