"""Recompute P1 metrics with the first 16 ensemble members (paper N_ens=16)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.metrics import summarize_ensemble  # noqa: E402

RESULTS = ROOT / "results"


def main():
    rows = []
    for p in sorted(RESULTS.glob("p1/*/*/seed*/samples.npz")):
        d = np.load(p)
        s = d["samples"]
        if s.shape[0] < 16:
            continue
        m = summarize_ensemble(s[:16], d["truth"])
        (p.parent / "metrics16.json").write_text(json.dumps(m, indent=2))
        parts = p.parts
        case = parts[parts.index("p1") + 1]
        method = parts[parts.index("p1") + 2]
        seed = int(parts[parts.index("p1") + 3].replace("seed", ""))
        rows.append({"case": case, "method": method, "seed": seed,
                     "rel_l2_mean": m["rel_l2_mean"],
                     "rel_l2_ensmean": m["rel_l2_ensmean"],
                     "log_spectrum_rel_error": m["log_spectrum_rel_error"],
                     "spread_error_pearson": m["spread_error"]["pearson"],
                     "spread_skill_ratio": m["spread_error"]["spread_skill_ratio"]})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no samples.npz found")
    agg = (df.groupby(["case", "method"], as_index=False)
             .agg(n_seeds=("seed", "nunique"),
                  rel_l2_mean=("rel_l2_mean", "mean"),
                  rel_l2_seedstd=("rel_l2_mean", "std"),
                  rel_l2_ensmean=("rel_l2_ensmean", "mean"),
                  spread_error_pearson=("spread_error_pearson", "mean"),
                  spread_skill_ratio=("spread_skill_ratio", "mean")))
    ref = pd.read_csv(RESULTS / "paper_reference.csv")
    out = agg.merge(ref, on=["case", "method"], how="left")
    out["rel_dev"] = (out["rel_l2_mean"] - out["paper_rel_l2"]).abs() / out["paper_rel_l2"]
    out["within_10pct"] = out["rel_dev"] < 0.10
    out.sort_values(["case", "method"], inplace=True)
    out.to_csv(RESULTS / "tab_reproduce_16.csv", index=False)
    print(out.to_string(index=False))
    gate = out.groupby("case")["within_10pct"].all()
    summary = {
        "n_cases": int(len(gate)),
        "n_cases_all_methods_within_10pct": int(gate.sum()),
        "case_gate": {k: bool(v) for k, v in gate.items()},
        "method_pass_rate": float(out["within_10pct"].mean()),
        "gate_pass": bool(gate.sum() >= 6),
    }
    (RESULTS / "p1" / "gate_summary_16.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
