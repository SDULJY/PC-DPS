"""P1 figures: reconstruction panels, paper comparison, spectra, statistics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import FIGURES, RESULTS  # noqa: E402
from common.metrics import radial_enstrophy_spectrum, ensemble_mean  # noqa: E402
from reproduce.case_matrix import METHOD_LABEL  # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7, "lines.linewidth": 1.3, "figure.dpi": 150,
})
METHODS = ["sdedit", "dps", "cond", "cfg"]


def load_run(case, method, seed=0):
    p = ROOT.parent / "results" / "p1" / case / method / f"seed{seed}" / "samples.npz"
    return np.load(p) if p.exists() else None


def panel_figure(cases):
    n = len(cases)
    # 第 7 列专门放 colorbar；边距按两行 1.69 in 正方形子图调好，
    # 避免 imshow 在上下留白边。不要再加 tight_layout()。
    fig = plt.figure(figsize=(11.6, 3.1 * n))
    gs = GridSpec(n, 7, figure=fig,
                  width_ratios=[1, 1, 1, 1, 1, 1, 0.075],
                  left=0.015, right=0.965, top=0.85, bottom=0.15,
                  wspace=0.07, hspace=0.56)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(6)]
                     for i in range(n)])
    caxes = [fig.add_subplot(gs[i, 6]) for i in range(n)]
    for i, case in enumerate(cases):
        ref = load_run(case, "dps")
        if ref is None:
            raise FileNotFoundError(f"missing samples.npz: {case}/dps/seed0")
        truth, obs = ref["truth"], ref["observed"]
        vmax = float(np.percentile(np.abs(truth), 99)) or 1.0
        im = axes[i, 0].imshow(truth, origin="lower", cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax)
        axes[i, 0].set_title(f"{case}: truth")
        axes[i, 1].imshow(obs, origin="lower", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax)
        axes[i, 1].set_title("observation")
        for j, method in enumerate(METHODS, start=2):
            d = load_run(case, method)
            if d is None:
                raise FileNotFoundError(f"missing samples.npz: {case}/{method}/seed0")
            arr = ensemble_mean(d["samples"])
            axes[i, j].imshow(arr, origin="lower", cmap="RdBu_r",
                              vmin=-vmax, vmax=vmax)
            axes[i, j].set_title(METHOD_LABEL[method])
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, cax=caxes[i])

    # 回归断言：colorbar 不得与任何场子图有像素级 bbox 交集
    fig.canvas.draw()
    for i in range(n):
        cb_box = caxes[i].get_window_extent()
        for j in range(6):
            assert not cb_box.overlaps(axes[i, j].get_window_extent()), \
                f"colorbar overlaps panel row={i} col={j}"

    fig.savefig(FIGURES / "fig01_recon_panels.png", dpi=300, facecolor="white")
    fig.savefig(FIGURES / "fig01_recon_panels.pdf")
    plt.close(fig)
    print("fig01 OK:", cases)


def comparison_figure():
    csv = RESULTS / "tab_reproduce_16.csv"
    if not csv.exists():
        csv = RESULTS / "tab_reproduce.csv"
    df = pd.read_csv(csv)
    cases = list(dict.fromkeys(df["case"]))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    width = 0.1
    x = np.arange(len(cases))
    for k, method in enumerate(METHODS):
        sub = df[df["method"] == method].set_index("case").reindex(cases)
        ax.bar(x + (k - 1.5) * width, sub["rel_l2_mean"], width,
               label=f"{METHOD_LABEL[method]} (ours)")
        ax.errorbar(x + (k - 1.5) * width, sub["rel_l2_mean"],
                    yerr=sub["rel_l2_seedstd"].fillna(0),
                    fmt="none", ecolor="k", lw=0.6)
        if "paper_rel_l2" in sub:
            ax.scatter(x + (k - 1.5) * width, sub["paper_rel_l2"],
                       marker="D", s=16, facecolors="none", edgecolors="k",
                       linewidths=0.8, zorder=6)
    ax.set_xticks(x); ax.set_xticklabels(cases, rotation=20, ha="right")
    ax.set_xlabel("test case")
    ax.set_yscale("log")
    # leave headroom above the tallest bar/diamond so the legend and the
    # explanatory caption never cover data
    upper = np.nanmax(np.maximum(
        df["rel_l2_mean"].to_numpy() + df["rel_l2_seedstd"].fillna(0).to_numpy(),
        df["paper_rel_l2"].to_numpy(),
    ))
    ax.set_ylim(top=float(upper) * 2.0)
    ax.set_ylabel("relative L2")
    ax.legend(ncol=2, fontsize=6, loc="upper right", framealpha=0.9)
    ax.set_title("Reproduction vs paper (bars: ours; diamonds: paper)")
    ax.text(
        0.01, 0.985,
        "bar: ours (mean over 3 seeds)\n"
        "vertical line: ±1 seed std\n"
        "black diamond: paper (N_ens=16)",
        transform=ax.transAxes, va="top", ha="left", fontsize=7,
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.92,
                  boxstyle="round,pad=0.25"),
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02_rel_l2_vs_paper.png", dpi=300)
    fig.savefig(FIGURES / "fig02_rel_l2_vs_paper.pdf")
    plt.close(fig)


def spectra_figure(cases):
    fig, axes = plt.subplots(1, len(cases), figsize=(7.2, 2.8), squeeze=False)
    for j, case in enumerate(cases):
        ref = load_run(case, "dps")
        if ref is None:
            continue
        k, zt = radial_enstrophy_spectrum(ref["truth"])
        axes[0, j].loglog(k[1:], zt[1:], "k-", label="truth")
        for method in ["dps", "cond"]:
            d = load_run(case, method)
            if d is None:
                continue
            z = radial_enstrophy_spectrum(ensemble_mean(d["samples"]))[1]
            axes[0, j].loglog(k[1:], z[1:], label=method)
        axes[0, j].axvline(16, color="gray", ls=":", lw=0.8)
        axes[0, j].set_xlabel("k"); axes[0, j].set_ylabel(r"$Z(k)$")
        axes[0, j].set_title(case); axes[0, j].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03_spectra.png", dpi=300)
    fig.savefig(FIGURES / "fig03_spectra.pdf")
    plt.close(fig)


def stats_figure():
    csv = RESULTS / "tab_reproduce_16.csv"
    if not csv.exists():
        csv = RESULTS / "tab_reproduce.csv"
    df = pd.read_csv(csv)
    cases = list(dict.fromkeys(df["case"]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, col, title in [(axes[0], "spread_error_pearson", "spread-error correlation"),
                           (axes[1], "spread_skill_ratio", "spread / RMSE")]:
        for k, method in enumerate(METHODS):
            sub = df[df["method"] == method].set_index("case").reindex(cases)
            ax.bar(np.arange(len(cases)) + (k - 1.5) * 0.1, sub[col], 0.1,
                   label=METHOD_LABEL[method])
        ax.set_xticks(np.arange(len(cases)))
        ax.set_xticklabels(cases, rotation=20, ha="right", fontsize=6)
        ax.set_title(title); ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04_stats_spread.png", dpi=300)
    fig.savefig(FIGURES / "fig04_stats_spread.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="jet_re10000_full,eddy_re10000_full")
    args = ap.parse_args()
    cases = args.cases.split(",")
    panel_figure(cases)
    comparison_figure()
    spectra_figure(cases)
    stats_figure()
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
