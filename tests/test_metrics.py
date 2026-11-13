import numpy as np

from src.common.metrics import (relative_l2, radial_enstrophy_spectrum,
                                spectrum_tail_ratio, vorticity_statistics)


def test_relative_l2_shapes():
    rng = np.random.default_rng(0)
    t = rng.standard_normal((64, 64))
    p = t[None].repeat(3, axis=0)
    assert np.allclose(relative_l2(p, t), 0.0)
    assert np.allclose(relative_l2(p, t[None].repeat(3, axis=0)), 0.0)


def test_spectrum_peak():
    n = 64
    x = np.arange(n) * 2 * np.pi / n
    f = np.cos(2 * x)[None, :].repeat(n, axis=0)
    k, z = radial_enstrophy_spectrum(f)
    peak = k[np.argmax(z[1:]) + 1]
    print("peak k", peak)
    assert peak == 2


def test_tail_ratio_identity():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((64, 64))
    f -= f.mean()
    assert abs(spectrum_tail_ratio(f, f) - 1.0) < 1e-10
    st = vorticity_statistics(f)
    assert abs(st["mean"]) < 1e-10
