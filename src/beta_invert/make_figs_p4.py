"""P4 figures: beta estimator distribution and convergence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import FIGURES, RESULTS  # noqa: E402

plt.rcParams.update({"font.size": 8, "figure.dpi": 150})


def load(case, source="pcdps", seed=0):
    p = ROOT.parent / "results" / "p4" / case / source / f"seed{seed}" / "beta_estimate.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jet-case", default="jet_re1000_full")
    ap.add_argument("--eddy-case", default="eddy_re1000_partial")
    ap.add_argument("--source", default="pcdps")
    args = ap.parse_args()


    jet = load(args.jet_case, args.source)
    eddy = load(args.eddy_case, args.source)
    if jet is None:
        raise SystemExit("missing jet beta_estimate.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    vals = np.array(jet["beta_hat_values"])
    axes[0].hist(vals, bins=20, alpha=0.7, label=r"$\hat\beta$ per pair")
    axes[0].axvline(jet["beta_true"], color="k", ls="--", label=r"true $\beta=2.5$")
    axes[0].axvline(jet["beta_hat_mean"], color="C1", label="mean")
    axes[0].set_xlabel(r"$\hat\beta$"); axes[0].set_ylabel("count")
    axes[0].set_title("jet: per-pair estimates"); axes[0].legend(fontsize=6)

    cases = ["jet", "eddy"]
    means, errs = [], []
    for c in cases:
        d = jet if c == "jet" else (eddy if eddy is not None else None)
        if d is None:
            means.append(np.nan); errs.append(np.nan)
        else:
            means.append(d["beta_hat_mean"])
            errs.append(d["beta_hat_mean"] - d["beta_hat_ci95"][0])
    axes[1].bar(cases, means, yerr=errs, capsize=4)
    axes[1].axhline(2.5, color="k", ls="--", label="true jet beta")
    axes[1].axhline(0.0, color="gray", ls=":", label="true eddy beta")
    axes[1].set_ylabel(r"$\hat\beta$"); axes[1].set_title("ensemble estimate")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig11_beta_estimate.png", dpi=300)
    fig.savefig(FIGURES / "fig11_beta_estimate.pdf")

    # convergence of the running mean
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    run = np.cumsum(vals) / np.arange(1, len(vals) + 1)
    ax.plot(np.arange(1, len(vals) + 1), run, marker="o", ms=2)
    ax.axhline(jet["beta_true"], color="k", ls="--")
    ax.set_xlabel("number of pairs"); ax.set_ylabel(r"running mean $\hat\beta$")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig12_beta_convergence.png", dpi=300)
    fig.savefig(FIGURES / "fig12_beta_convergence.pdf")
    print("wrote fig11/fig12")


if __name__ == "__main__":
    main()
