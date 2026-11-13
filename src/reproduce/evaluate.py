"""Aggregate P1 runs and compare with the paper's Tables 4 and 5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def collect(run_glob: str) -> pd.DataFrame:
    rows = []
    for p in sorted(ROOT.glob(run_glob)):
        d = json.loads(p.read_text())
        m = d.get("metrics", {})
        rows.append({
            "case": d.get("case"),
            "method": d.get("method"),
            "seed": d.get("seed"),
            "n": m.get("n"),
            "rel_l2_mean": m.get("rel_l2_mean"),
            "rel_l2_std": m.get("rel_l2_std"),
            "rel_l2_ensmean": m.get("rel_l2_ensmean"),
            "log_spectrum_rel_error": m.get("log_spectrum_rel_error"),
            "spread_error_pearson": (m.get("spread_error") or {}).get("pearson"),
            "spread_skill_ratio": (m.get("spread_error") or {}).get("spread_skill_ratio"),
            "skew_error": m.get("skew_error"),
            "kurtosis_error": m.get("kurtosis_error"),
            "runtime_sec": m.get("runtime_sec"),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/p1/*/*/seed*/log.json")
    ap.add_argument("--out", default=str(RESULTS / "tab_reproduce.csv"))
    args = ap.parse_args()

    df = collect(args.pattern)
    if df.empty:
        raise SystemExit(f"no runs matched {args.pattern}")
    agg = (df.groupby(["case", "method"], as_index=False)
             .agg(n_seeds=("seed", "nunique"),
                  rel_l2_mean=("rel_l2_mean", "mean"),
                  rel_l2_seedstd=("rel_l2_mean", "std"),
                  rel_l2_ensmean=("rel_l2_ensmean", "mean"),
                  log_spectrum_rel_error=("log_spectrum_rel_error", "mean"),
                  spread_error_pearson=("spread_error_pearson", "mean"),
                  spread_skill_ratio=("spread_skill_ratio", "mean"),
                  runtime_sec=("runtime_sec", "mean")))
    ref = pd.read_csv(RESULTS / "paper_reference.csv")
    out = agg.merge(ref, on=["case", "method"], how="left")
    out["rel_dev"] = (out["rel_l2_mean"] - out["paper_rel_l2"]).abs() / out["paper_rel_l2"]
    out["within_10pct"] = out["rel_dev"] < 0.10
    out.sort_values(["case", "method"], inplace=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))

    # Gate: a case passes if all four methods are within 10%.
    case_gate = out.groupby("case")["within_10pct"].all()
    n_pass = int(case_gate.sum())
    summary = {
        "n_cases": int(len(case_gate)),
        "n_cases_all_methods_within_10pct": n_pass,
        "case_gate": {k: bool(v) for k, v in case_gate.items()},
        "method_pass_rate": float(out["within_10pct"].mean()),
        "gate_pass": n_pass >= 6,
    }
    (RESULTS / "p1").mkdir(parents=True, exist_ok=True)
    (RESULTS / "p1" / "gate_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
