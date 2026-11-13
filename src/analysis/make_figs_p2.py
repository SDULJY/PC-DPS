"""P2 failure-mode figures: DPS spectral smoothing and error vs distance."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import FIGURES, RESULTS, FIELD_STD  # noqa: E402
from common.metrics import radial_enstrophy_spectrum, ensemble_mean  # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7, "lines.linewidth": 1.4, "figure.dpi": 150,
})


def load_run(case, method, seed=0):
    p = ROOT.parent / "results" / "p1" / case / method / f"seed{seed}" / "samples.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["samples"], d["truth"], d["observed"]


def distance_curve(samples, truth, observed, n_bins=25):
    mask = observed != 0
    dist = distance_transform_edt(~mask)
    rmse = np.sqrt(((samples - truth[None]) ** 2).mean(axis=0))
    scale = truth.std()
    edges = np.arange(0, n_bins + 1)
    centers, means, sems = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() < 5:
            continue
        v = rmse[m] / (scale + 1e-12)
        centers.append(0.5 * (lo + hi))
        means.append(v.mean())
        sems.append(v.std(ddof=1) / np.sqrt(m.sum()))
    return np.array(centers), np.array(means), np.array(sems), dist.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="jet_re10000_partial,eddy_re10000_partial")
    ap.add_argument("--methods", default="dps,cond")
    args = ap.parse_args()
    cases = args.cases.split(",")
    methods = args.methods.split(",")

    out = {}
    fig, axes = plt.subplots(1, len(cases), figsize=(7.2, 2.8), squeeze=False)
    fig2, axes2 = plt.subplots(1, len(cases), figsize=(7.2, 2.8), squeeze=False)
    for j, case in enumerate(cases):
        out[case] = {}
        ref = load_run(case, methods[0])
        if ref is None:
            print(f"skip {case}: no runs")
            continue
        truth = ref[1]
        k, zt = radial_enstrophy_spectrum(truth)
        axes[0, j].loglog(k[1:], zt[1:], "k-", label="truth")
        for method in methods:
            d = load_run(case, method)
            if d is None:
                continue
            samples, _, observed = d
            zmean = radial_enstrophy_spectrum(ensemble_mean(samples))[1]
            zper = np.mean([radial_enstrophy_spectrum(s)[1] for s in samples], axis=0)
            axes[0, j].loglog(k[1:], zmean[1:], label=f"{method} ens-mean")
            axes[0, j].loglog(k[1:], zper[1:], "--", label=f"{method} sample-mean")
            tail = float(np.mean(zmean[k >= 16] / (zt[k >= 16] + 1e-12)))
            centers, means, sems, dmax = distance_curve(samples, truth, observed)
            axes2[0, j].plot(centers, means, label=method)
            axes2[0, j].fill_between(centers, means - sems, means + sems, alpha=0.2)
            out[case][method] = {
                "spectrum_tail_ratio_k_ge_16": tail,
                "distance_curve": {"distance": centers.tolist(),
                                   "rmse_over_std": means.tolist()},
                "far_field_rmse_over_std": float(means[-1]) if len(means) else None,
                "max_distance": float(dmax),
            }
        axes[0, j].axvline(16, color="gray", ls=":", lw=0.8)
        axes[0, j].set_xlabel("k")
        axes[0, j].set_ylabel(r"$Z(k)$")
        axes[0, j].set_title(case)
        axes[0, j].legend()
        axes2[0, j].set_xlabel("distance to nearest observation (pixels)")
        axes2[0, j].set_ylabel("RMSE / std(truth)")
        axes2[0, j].set_title(case)
        axes2[0, j].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05_dps_spectrum_smoothing.png", dpi=300)
    fig.savefig(FIGURES / "fig05_dps_spectrum_smoothing.pdf")
    fig2.tight_layout()
    fig2.savefig(FIGURES / "fig06_error_vs_distance.png", dpi=300)
    fig2.savefig(FIGURES / "fig06_error_vs_distance.pdf")
    (RESULTS / "p2").mkdir(parents=True, exist_ok=True)
    (RESULTS / "p2" / "failure_modes.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
