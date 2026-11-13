"""P3 figures: PC-DPS panels, error maps, spectra, and lambda sweep."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import FIGURES, RESULTS  # noqa: E402
from common.metrics import radial_enstrophy_spectrum, ensemble_mean  # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7, "lines.linewidth": 1.3, "figure.dpi": 150,
})


def load(case, lam, seed=0):
    p = ROOT.parent / "results" / "p3" / case / f"lambda{lam:g}" / f"seed{seed}" / "samples.npz"
    return np.load(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="eddy_re1000_partial")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d0 = load(args.case, 0.0, args.seed)
    d1 = load(args.case, args.lam, args.seed)
    if d0 is None or d1 is None:
        raise SystemExit("missing lambda0 or best-lambda run")
    truth = d0["truth"][0]
    obs = d0["observed"][0]
    m0 = ensemble_mean(d0["samples"][: d0["samples"].shape[0] // 2])
    m1 = ensemble_mean(d1["samples"][: d1["samples"].shape[0] // 2])
    vmax = float(np.percentile(np.abs(truth), 99)) or 1.0

    fig, axes = plt.subplots(1, 5, figsize=(12, 2.6))
    panels = [(truth, "truth"), (obs, "observation"), (m1, f"PC-DPS λ={args.lam:g}"),
              (m0, "DPS (λ=0)"), (np.abs(m1 - truth), "|PC-DPS error|")]
    for ax, (arr, title) in zip(axes, panels):
        cmap = "magma" if title.startswith("|") else "RdBu_r"
        if title.startswith("|"):
            im = ax.imshow(arr, origin="lower", cmap=cmap)
        else:
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig08_pcdps_panels.png", dpi=300)
    fig.savefig(FIGURES / "fig08_pcdps_panels.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    k, zt = radial_enstrophy_spectrum(truth)
    ax.loglog(k[1:], zt[1:], "k-", label="truth")
    for d, label in [(d0, "DPS (λ=0)"), (d1, f"PC-DPS λ={args.lam:g}")]:
        z = radial_enstrophy_spectrum(ensemble_mean(d["samples"][: d["samples"].shape[0] // 2]))[1]
        ax.loglog(k[1:], z[1:], label=label)
    ax.axvline(16, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("k"); ax.set_ylabel(r"$Z(k)$"); ax.set_title(args.case)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig09_pcdps_spectra.png", dpi=300)
    fig.savefig(FIGURES / "fig09_pcdps_spectra.pdf")
    print("wrote fig08/fig09")


if __name__ == "__main__":
    main()
