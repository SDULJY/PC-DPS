"""Collect every run's log.json into results/master.csv."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def main():
    rows = []
    for p in sorted(RESULTS.glob("p*/*/*/seed*/log.json")):
        d = json.loads(p.read_text())
        m = d.get("metrics", {})
        rows.append({
            "stage": d.get("stage"),
            "case": d.get("case"),
            "method": d.get("method"),
            "lambda_phys": d.get("lambda_phys"),
            "seed": d.get("seed"),
            "n": m.get("n"),
            "rel_l2_mean": m.get("rel_l2_mean"),
            "rel_l2_ensmean": m.get("rel_l2_ensmean"),
            "log_spectrum_rel_error": m.get("log_spectrum_rel_error"),
            "runtime_sec": m.get("runtime_sec"),
            "path": str(p),
        })
    df = pd.DataFrame(rows)
    out = RESULTS / "master.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} with {len(df)} rows")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
