"""Load and sample Ansel/darktable noise profiles.

The profile database (data/noiseprofiles.json, copied from the Ansel source tree)
stores, per camera and per ISO, the Poisson-Gaussian parameters (a, b) of the
noise variance model fitted on normalized raw values x in [0, 1]:

    Var(x) = a * x + b        (one (a, b) pair per RGB channel)

`a` is the photon (shot) noise gain, `b` the signal-independent (read) noise
variance. This module is the single source of noise statistics for training:
the synthesizer draws (a, b) from here, so the trained network is by
construction matched to the cameras Ansel supports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "noiseprofiles.json"


@dataclass(frozen=True)
class IsoProfile:
    iso: float
    a: np.ndarray  # shape (3,), per RGB channel
    b: np.ndarray  # shape (3,)


@dataclass(frozen=True)
class CameraProfile:
    maker: str
    model: str
    isos: tuple[IsoProfile, ...]  # sorted by ISO ascending

    @property
    def name(self) -> str:
        return f"{self.maker} {self.model}"

    def interpolate(self, iso: float) -> IsoProfile:
        """Linear interpolation of (a, b) against ISO, clamped to the profiled
        range — same behaviour as dt_noiseprofile_interpolate() in Ansel."""
        isos = self.isos
        if iso <= isos[0].iso:
            return isos[0]
        if iso >= isos[-1].iso:
            return isos[-1]
        for lo, hi in zip(isos, isos[1:]):
            if lo.iso <= iso <= hi.iso:
                t = (iso - lo.iso) / (hi.iso - lo.iso)
                return IsoProfile(iso=iso, a=(1 - t) * lo.a + t * hi.a, b=(1 - t) * lo.b + t * hi.b)
        raise AssertionError("unreachable")


def load_profiles(path: Path | str = DEFAULT_DB) -> list[CameraProfile]:
    with open(path) as f:
        db = json.load(f)
    cameras = []
    for maker in db["noiseprofiles"]:
        for model in maker["models"]:
            isos = sorted(
                (
                    IsoProfile(
                        iso=float(p["iso"]),
                        a=np.asarray(p["a"], dtype=np.float64),
                        b=np.asarray(p["b"], dtype=np.float64),
                    )
                    for p in model["profiles"]
                ),
                key=lambda p: p.iso,
            )
            if isos:
                cameras.append(CameraProfile(maker=maker["maker"], model=model["model"], isos=tuple(isos)))
    return cameras


class ProfileSampler:
    """Draw random (a, b) pairs for training-noise synthesis.

    Sampling strategy: uniform over cameras, then log-uniform in each camera's
    profiled ISO range (so rare high-ISO regimes are not underrepresented),
    then a mild log-uniform jitter on both parameters to cover in-between
    sensors and profile fitting error.
    """

    def __init__(
        self,
        cameras: list[CameraProfile] | None = None,
        jitter: float = 2.0,
        holdout: set[str] | None = None,
        sigma_calibration: tuple[float, float, float] | None = "default",
    ):
        cameras = cameras if cameras is not None else load_profiles()
        holdout = holdout or set()
        self.cameras = [c for c in cameras if c.name not in holdout]
        if not self.cameras:
            raise ValueError("no cameras left after holdout filter")
        self.jitter = float(jitter)
        # The DB (a, b) understate the physical mosaic-domain variance (see
        # DEFAULT_SIGMA_CALIBRATION below): synthesis must be centered on the
        # TRUTH so that calibration-corrected inference sigma is
        # in-distribution by construction. None turns the correction off
        # (pre-calibration behavior, for regression comparisons only).
        if sigma_calibration == "default":
            sigma_calibration = DEFAULT_SIGMA_CALIBRATION
        self.variance_calibration = (
            np.asarray(sigma_calibration, dtype=np.float64) ** 2
            if sigma_calibration is not None else np.ones(3)
        )

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
        cam = self.cameras[rng.integers(len(self.cameras))]
        lo, hi = cam.isos[0].iso, cam.isos[-1].iso
        iso = float(np.exp(rng.uniform(np.log(lo), np.log(max(hi, lo + 1e-3)))))
        prof = cam.interpolate(iso)
        j = self.jitter
        cal = self.variance_calibration
        a = prof.a * cal * np.exp(rng.uniform(-np.log(j), np.log(j), size=3))
        b = prof.b * cal * np.exp(rng.uniform(-np.log(j), np.log(j), size=3))
        return a, b, {"camera": cam.name, "iso": iso}


# Measured deviation between the shipped noise profiles and the true
# mosaic-domain sigma, in sigma units per RGB channel: an exact factor 2 from
# the profiling tool's lifting-Haar normalization (std(HH) = sigma/2 for iid
# noise, squared into the (a, b) fit uncorrected), times the demosaic
# high-frequency attenuation measured on flat regions of real raws vs the
# profile prediction (cross-camera medians of the 253-camera
# raw.pixls.us sweep, bench/rpu-sweep-calibration.csv, estimator-bias
# corrected; ISO-stable per the 64-image local study). Training synthesis and the
# sigma conditioning must share this convention with the C inference side —
# the constant also ships inside the exported model cfg.
DEFAULT_SIGMA_CALIBRATION = (2.0 * 1.41, 2.0 * 1.97, 2.0 * 1.48)
