# Reference XMPs for the denoiser comparison

The exact histories used by `scripts/compare_denoisers.py` for one condition
(Sony ILCE-6000, ISO 12800), kept so the benchmark can be inspected and
replayed by hand:

| file | history |
| ---- | ------- |
| `off.xmp` | `rawdenoiseai` present but **disabled** — the reference and the noisy renders |
| `DSC00578-12800-denoiseprofile.xmp` | `denoiseprofile` at its own defaults |
| `DSC00578-12800-denoiseprofile_200%.xmp` | same, `strength` 200 % |
| `DSC00578-12800-ai_<size>-<variant>.xmp` | `rawdenoiseai`, one per model |

Two things these encode, both of which cost a debugging round when they were
wrong:

- **Every render of a comparison must have the same history structure.** An
  empty history is *not* equivalent to a disabled module: Ansel renders them
  differently, which shows up as a large constant PSNR offset on every
  denoised variant at once. Hence `off.xmp` carries the module, disabled.
- **`denoiseprofile`'s `a[3]`/`b[3]` are per camera and per ISO**, copied
  from the noise profile the way its `reload_defaults()` does. The blobs here
  are therefore only valid for the ILCE-6000 at ISO 12800; for any other
  condition regenerate them with the script rather than editing by hand.

Replay one by hand:

```sh
ansel-cli noisy.dng bench/xmp/DSC00578-12800-denoiseprofile.xmp out.tif --core \
    --configdir <scratch> --library <scratch>/lib.db
```

The DNGs themselves are not committed (they are large and regenerable): pass
`--keep <dir>` to `compare_denoisers.py` to retain them.
