# Ansel Training Data License, version 1.1 (ATDL-1.1)

This license applies to the raw-sensor image tiles ("the Tiles") distributed
as community-contributed training shards of the ansel-denoise corpus.
Contributors keep the copyright on their photographs. By contributing, they
grant the rights below; by downloading or using the Tiles, you accept all of
the conditions below.

The intent, in one sentence: the Tiles exist to make image **denoising**
better and auditable for everyone — the models you train with them are
yours, but the Tiles must never feed a training stack capable of learning
anything else than separating noise from signal.

## 1. Who may use the Tiles

Anyone — the Ansel project, individuals, academic teams, companies —
provided every use complies with sections 2 and 3. There is no fee and no
registration.

## 2. What the Tiles may be used for

The Tiles may be used only through the
[ansel-denoise training stack](https://github.com/aurelienpierreeng/ansel-denoise)
— the GPL-3.0 code published in that repository, or a derivative of it that
can train image-denoising networks **and nothing else** — for the following
purposes:

1. **Audit, review, reproduction and benchmarking.** Re-run the published
   training to verify the Ansel project's results, study the method,
   measure alternatives against it — scientific and academic use is
   explicitly welcome, as is publishing your findings. The
   [design documentation](https://github.com/aurelienpierreeng/ansel/blob/master/doc/rawdenoiseai.md)
   describes the reference workflow.
2. **Training your own noise models.** Train custom image-denoising
   networks with the stack on these Tiles. **The resulting weights are
   yours**: this license places no restriction on them — use, ship, sell or
   license them however you see fit, in open-source or commercial
   applications, related to Ansel or not.
3. **Incidental technical operations** required by the above: downloading,
   caching, format conversion, computing statistics over the corpus.

## 3. What the Tiles may not be used for

Every use not explicitly permitted by section 2 is forbidden, and by
accepting this license you explicitly accept that prohibition. The bright
line is the **capability of the training stack**: the Tiles must never be
fed to a stack able to learn anything else than separating noise from image
signal. In particular, and without limiting the previous sentences, the
Tiles may not be:

- used to train, fine-tune, evaluate or prompt models that synthesize,
  restyle or complete images — "style" learning, generative models of any
  kind, inpainting, upscaling-as-generation, multimodal or language models;
- incorporated into or redistributed as part of any other dataset;
- used to identify, profile or locate persons, property or places depicted;
- sold or sublicensed **as data**, or used for advertising;
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
Weights produced by a use that breaches section 3 are not covered by the
freedom of section 2.2.

## 7. No warranty

The Tiles are provided as-is, with no warranty of any kind, including
fitness for a particular purpose. Neither the contributors nor the Ansel
project are liable for any use of the Tiles.

## 8. Contributor grant

By contributing Tiles, the contributor declares owning the necessary rights
to the source photographs and grants the Ansel project and every user
compliant with sections 2–5 a non-exclusive, worldwide, royalty-free
license to use the Tiles under the conditions of this document.

---

*Version history — 1.1 (2026-07-19): clarified permitted uses: anyone may
audit, reproduce and benchmark the training, and train custom denoising
models whose weights are unrestricted (including commercial use); the
prohibition is restated as a bright line on the training stack's
capability. Replaces 1.0 (2026-07-18), under which no contribution had been
accepted.*
