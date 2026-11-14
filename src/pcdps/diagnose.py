"""Decompose PC-DPS vs DPS error by wavenumber band and region (P2.1)."""
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


def load(case, lam):
    p = ROOT.parent / "results" / "p3" / case / f"lambda{lam:g}" / "seed0" / "samples.npz"
    return np.load(p)


def band_rel(err, truth, lo, hi):
    n = err.shape[-1]
    k = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    kr = np.sqrt(kx ** 2 + ky ** 2)
    m = (kr >= lo) & (kr < hi)
    ef = np.fft.fft2(err) * m
    tf = np.fft.fft2(truth) * m
    return float(np.sqrt((np.abs(ef) ** 2).sum() / ((np.abs(tf) ** 2).sum() + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="eddy_re10000_partial")
    ap.add_argument("--lambda-baseline", type=float, default=0.0)
    ap.add_argument("--lambda-pcdps", type=float, default=2.0)
    ap.add_argument("--out", default=str(RESULTS / "p3" / "pcdps_diagnosis.json"))
    args = ap.parse_args()

    out = {"case": args.case, "bands": {}, "regions": {}}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    methods = [("DPS", args.lambda_baseline), ("PC-DPS", args.lambda_pcdps)]
    for name, lam in methods:
        d = load(args.case, lam)
        s, t, o = d["samples"], d["truth"], d["observed"]
        n = s.shape[0] // 2
        m0 = s[:n].mean(0)
        m1 = s[n:].mean(0)
        err = np.stack([m0 - t[0], m1 - t[1]])
        truth = np.stack([t[0], t[1]])
        bands = {}
        for lo, hi in [(0, 9), (9, 17), (17, 33)]:
            vals = [band_rel(err[i], truth[i], lo, hi) for i in range(2)]
            bands[f"k{lo}-{hi-1}"] = float(np.mean(vals))
        out["bands"][name] = bands
        obs0, obs1 = o[0], o[1]
        mask = (obs0 != 0) | (obs1 != 0)
        gap = ~mask
        rmse = np.sqrt(((err ** 2).mean(axis=0)))
        out["regions"][name] = {
            "observed_rmse_over_std": float(rmse[mask].mean() / (truth.std() + 1e-12)),
            "gap_rmse_over_std": float(rmse[gap].mean() / (truth.std() + 1e-12)),
            "seen_fraction": float(mask.mean()),
        }
        axes[0].bar(np.arange(3) + (-0.2 if name == "DPS" else 0.2),
                    list(bands.values()), width=0.35, label=name)
        axes[1].bar([name + "\nobs", name + "\ngap"],
                    [out["regions"][name]["observed_rmse_over_std"],
                     out["regions"][name]["gap_rmse_over_std"]])
    axes[0].set_xticks(np.arange(3)); axes[0].set_xticklabels(["k0-8", "k9-16", "k17-32"])
    axes[0].set_xlabel("wavenumber band"); axes[0].set_ylabel("relative error")
    axes[0].legend(); axes[0].set_title(args.case)
    axes[1].set_ylabel("RMSE / std(truth)"); axes[1].set_title("region")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig15_pcdps_diagnosis.png", dpi=300)
    fig.savefig(FIGURES / "fig15_pcdps_diagnosis.pdf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
