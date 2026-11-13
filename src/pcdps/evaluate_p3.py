"""Aggregate the PC-DPS sweep and check the P3 gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def main():
    rows = []
    for p in sorted(RESULTS.glob("p3/*/lambda*/seed*/metrics.json")):
        d = json.loads(p.read_text())
        parts = p.parts
        case = parts[parts.index("p3") + 1]
        lam = float([x for x in parts if x.startswith("lambda")][0].replace("lambda", ""))
        seed = int([x for x in parts if x.startswith("seed")][0].replace("seed", ""))
        rows.append({"case": case, "lambda": lam, "seed": seed,
                     "rel_l2_mean": d["rel_l2_mean"],
                     "rel_l2_ensmean": d["rel_l2_ensmean"],
                     "log_spectrum_rel_error": d["log_spectrum_rel_error"],
                     "spectrum_tail_ratio": d.get("spectrum_tail_ratio", np.nan)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no P3 runs found")
    agg = (df.groupby(["case", "lambda"], as_index=False)
             .agg(n=("seed", "nunique"),
                  rel_l2_mean=("rel_l2_mean", "mean"),
                  rel_l2_std=("rel_l2_mean", "std"),
                  rel_l2_ensmean=("rel_l2_ensmean", "mean"),
                  log_spectrum_rel_error=("log_spectrum_rel_error", "mean"),
                  spectrum_tail_ratio=("spectrum_tail_ratio", "mean")))
    out = []
    for case, g in agg.groupby("case"):
        base = g[g["lambda"] == 0.0]
        b = float(base["rel_l2_mean"].iloc[0]) if len(base) else np.nan
        g = g.copy()
        g["rel_reduction_vs_lambda0"] = (b - g["rel_l2_mean"]) / (b + 1e-12)
        g["tail_distance_to_one"] = (g["spectrum_tail_ratio"] - 1.0).abs()
        out.append(g)
    agg = pd.concat(out)
    agg.to_csv(RESULTS / "tab_pcdps.csv", index=False)
    print(agg.to_string(index=False))

    summary = {}
    for case, g in agg.groupby("case"):
        best = g.loc[g["rel_l2_mean"].idxmin()]
        lam0 = g[g["lambda"] == 0.0]
        tail0 = float(lam0["tail_distance_to_one"].iloc[0]) if len(lam0) else np.nan
        summary[case] = {
            "best_lambda": float(best["lambda"]),
            "best_rel_l2": float(best["rel_l2_mean"]),
            "baseline_rel_l2": float(lam0["rel_l2_mean"].iloc[0]) if len(lam0) else None,
            "relative_reduction": float(best["rel_reduction_vs_lambda0"]),
            "tail_ratio_best": float(best["spectrum_tail_ratio"]),
            "tail_ratio_lambda0": float(lam0["spectrum_tail_ratio"].iloc[0]) if len(lam0) else None,
            "tail_closer_to_one": bool(best["tail_distance_to_one"] < tail0) if len(lam0) else None,
        }
    gate = any(v["relative_reduction"] >= 0.08 and v["tail_closer_to_one"]
               for v in summary.values())
    summary["_gate_pass"] = bool(gate)
    (RESULTS / "p3").mkdir(parents=True, exist_ok=True)
    (RESULTS / "p3" / "gate_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, len(summary) - 1, figsize=(8, 3), squeeze=False)
    for j, (case, g) in enumerate(agg.groupby("case")):
        ax = axes[0, j]
        ax.errorbar(g["lambda"], g["rel_l2_mean"], yerr=g["rel_l2_std"].fillna(0),
                    marker="o", capsize=3)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlabel(r"$\lambda_{phys}$")
        ax.set_ylabel("relative L2")
        ax.set_title(case)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig10_lambda_sweep.png", dpi=300)
    fig.savefig(FIGURES / "fig10_lambda_sweep.pdf")
    print("wrote", FIGURES / "fig10_lambda_sweep.png")


if __name__ == "__main__":
    main()
