import numpy as np

from ansel_denoise.profiles import ProfileSampler, load_profiles


def test_load_ansel_database():
    cams = load_profiles()
    assert len(cams) > 200  # the DB covers hundreds of camera models
    for cam in cams:
        isos = [p.iso for p in cam.isos]
        assert isos == sorted(isos)
        for p in cam.isos:
            assert p.a.shape == (3,) and p.b.shape == (3,)


def test_interpolation_is_local_and_clamped():
    cam = next(c for c in load_profiles() if len(c.isos) >= 3)
    lo, hi = cam.isos[0], cam.isos[1]
    mid = cam.interpolate((lo.iso + hi.iso) / 2)
    for c in range(3):
        assert min(lo.a[c], hi.a[c]) <= mid.a[c] <= max(lo.a[c], hi.a[c])
    assert (cam.interpolate(1.0).a == lo.a).all()
    assert (cam.interpolate(1e9).a == cam.isos[-1].a).all()


def test_sampler_draws_usable_params():
    # The shipped database is an unconstrained least-squares fit: individual
    # channels can have a <= 0 or b < 0 (e.g. EOS 550D blue channel). The
    # synthesizer treats (a, b) as a variance line and clamps Var at zero,
    # so the sampler's only contract is finiteness.
    sampler = ProfileSampler()
    rng = np.random.default_rng(7)
    for _ in range(200):
        a, b, meta = sampler.sample(rng)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert meta["iso"] > 0 and " " in meta["camera"]


def test_sampler_holdout():
    cams = load_profiles()
    excluded = cams[0].name
    sampler = ProfileSampler(cams, holdout={excluded})
    assert all(c.name != excluded for c in sampler.cameras)
