"""Skill metrics used in the paper (relative L2, spectra, vorticity statistics, UQ)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def relative_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-sample relative L2 error.

    pred/truth: (N, H, W) or (H, W).  Returns shape (N,) or scalar.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.ndim == 2:
        return np.linalg.norm(pred - truth) / np.linalg.norm(truth)
    if truth.ndim == 2:
        truth = np.broadcast_to(truth, pred.shape)
    num = np.linalg.norm((pred - truth).reshape(pred.shape[0], -1), axis=1)
    den = np.linalg.norm(truth.reshape(truth.shape[0], -1), axis=1)
    return num / den


def ensemble_mean(samples: np.ndarray) -> np.ndarray:
    return samples.mean(axis=0)


def ensemble_std(samples: np.ndarray) -> np.ndarray:
    return samples.std(axis=0, ddof=1)


def radial_enstrophy_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Angle-averaged enstrophy spectrum Z(k) = 1/2 <|omega_hat|^2> over the ring.

    Uses integer wavenumbers on the 64x64 periodic grid.  The absolute
    normalization is not important for log-spectrum comparisons, but we keep
    Parseval-consistent scaling.
    """
    field = np.asarray(field, dtype=np.float64)
    n0, n1 = field.shape
    fh = np.fft.fft2(field)
    power = (np.abs(fh) ** 2) / field.size
    kx = np.fft.fftfreq(n0, d=1.0 / n0)
    ky = np.fft.fftfreq(n1, d=1.0 / n1)
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    kr = np.sqrt(kx ** 2 + ky ** 2)
    kmax = int(np.floor(kr.max()))
    k = np.arange(kmax + 1)
    z = np.zeros(kmax + 1)
    for i in range(kmax + 1):
        m = (kr >= i - 0.5) & (kr < i + 0.5)
        z[i] = 0.5 * power[m].sum()
    return k, z


def radial_kinetic_energy_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Angle-averaged kinetic energy spectrum E(k) from vorticity (psi = -lap^-1 omega)."""
    field = np.asarray(field, dtype=np.float64)
    n0, n1 = field.shape
    fh = np.fft.fft2(field)
    kx = np.fft.fftfreq(n0, d=1.0 / n0) * 2 * np.pi
    ky = np.fft.fftfreq(n1, d=1.0 / n1) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.0
    ph = -fh / k2
    uh = 1j * ky * ph
    vh = -1j * kx * ph
    power = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2) / field.size
    kr = np.sqrt(kx ** 2 + ky ** 2)
    kmax = int(np.floor(kr.max()))
    k = np.arange(kmax + 1)
    e = np.zeros(kmax + 1)
    for i in range(kmax + 1):
        m = (kr >= i - 0.5) & (kr < i + 0.5)
        e[i] = power[m].sum()
    return k, e


def vorticity_statistics(field: np.ndarray) -> dict:
    """Single-point vorticity statistics (mean, variance, skewness, kurtosis)."""
    x = np.asarray(field, dtype=np.float64).ravel()
    return {
        "mean": float(x.mean()),
        "var": float(x.var()),
        "std": float(x.std()),
        "skew": float(stats.skew(x, bias=True)),
        "kurtosis": float(stats.kurtosis(x, fisher=True, bias=True)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def spread_error_correlation(samples: np.ndarray, truth: np.ndarray) -> dict:
    """Correlation between ensemble spread and ensemble-mean error over pixels."""
    sd = ensemble_std(samples).ravel()
    err = np.abs(ensemble_mean(samples) - truth).ravel()
    pear = float(np.corrcoef(sd, err)[0, 1])
    spear = float(stats.spearmanr(sd, err).statistic)
    ratio = float(sd.mean() / (np.sqrt(np.mean(err ** 2)) + 1e-12))
    return {"pearson": pear, "spearman": spear, "spread_skill_ratio": ratio,
            "spread_mean": float(sd.mean()), "rmse": float(np.sqrt(np.mean(err ** 2)))}


def spectrum_tail_ratio(pred: np.ndarray, truth: np.ndarray, kmin: int = 16) -> float:
    """Mean ratio of enstrophy spectra for k >= kmin (1.0 = matched tail)."""
    k, zp = radial_enstrophy_spectrum(pred)
    _, zt = radial_enstrophy_spectrum(truth)
    m = k >= kmin
    return float(np.mean(zp[m] / (zt[m] + 1e-12)))


def log_spectrum_relative_error(field: np.ndarray, truth: np.ndarray,
                                floor: float = 1e-12) -> float:
    """Relative L2 error between log enstrophy spectra (paper skill metric)."""
    k, zf = radial_enstrophy_spectrum(field)
    _, zt = radial_enstrophy_spectrum(truth)
    lf = np.log10(np.maximum(zf, floor))
    lt = np.log10(np.maximum(zt, floor))
    return float(np.linalg.norm(lf - lt) / np.linalg.norm(lt))


def summarize_ensemble(samples: np.ndarray, truth: np.ndarray) -> dict:
    """All headline metrics for one case/method/seed."""
    rel = relative_l2(samples, truth)
    mean = ensemble_mean(samples)
    out = {
        "n": int(samples.shape[0]),
        "rel_l2_mean": float(rel.mean()),
        "rel_l2_std": float(rel.std(ddof=1)),
        "rel_l2_min": float(rel.min()),
        "rel_l2_max": float(rel.max()),
        "rel_l2_ensmean": float(relative_l2(mean, truth)),
        "log_spectrum_rel_error": log_spectrum_relative_error(mean, truth),
        "spectrum_tail_ratio": spectrum_tail_ratio(mean, truth),
        "vorticity_ensmean": vorticity_statistics(mean),
        "vorticity_truth": vorticity_statistics(truth),
        "spread_error": spread_error_correlation(samples, truth),
    }
    out["skew_error"] = out["vorticity_ensmean"]["skew"] - out["vorticity_truth"]["skew"]
    out["kurtosis_error"] = out["vorticity_ensmean"]["kurtosis"] - out["vorticity_truth"]["kurtosis"]
    return out
