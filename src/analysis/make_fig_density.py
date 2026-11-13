"""Aggregate density sweeps and draw the error-vs-observation-density curve."""
from __future__ import annotations

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


def main():
    files = sorted((RESULTS / "p2").glob("density_sweep_seed*.json"))
    if not files:
        raise SystemExit("no density sweep results")
    per = {}
    for f in files:
        d = json.loads(f.read_text())
        for r in d["rows"]:
            per.setdefault(r["density"], []).append(r["rel_l2_mean"])
    rows = []
    for d in sorted(per):
        v = np.array(per[d], dtype=float)
        rows.append({"density": d, "n_obs": int(d) ** 2,
                     "rel_l2_mean": float(v.mean()),
                     "rel_l2_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0})
    (RESULTS / "p2" / "density_summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.array([r["n_obs"] for r in rows])
    y = np.array([r["rel_l2_mean"] for r in rows])
    e = np.array([r["rel_l2_std"] for r in rows])
    ax.errorbar(x, y, yerr=e, marker="o", capsize=3)
    ax.set_xlabel("number of point observations")
    ax.set_ylabel("relative L2")
    ax.set_title("DPS error vs observation density")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig07_error_vs_density.png", dpi=300)
    fig.savefig(FIGURES / "fig07_error_vs_density.pdf")
    print("wrote fig07")


if __name__ == "__main__":
    main()
