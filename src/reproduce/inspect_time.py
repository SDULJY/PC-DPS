"""Inspect enstrophy/vorticity statistics versus snapshot index for one run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import DATA, RESULTS, data_filename  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="jet_re1000_full")
    ap.add_argument("--run-index", type=int, default=-1)
    ap.add_argument("--out", default=str(RESULTS / "p0" / "time_stats.json"))
    args = ap.parse_args()
    regime, re_s, obs = args.case.split("_")
    re = int(re_s.replace("re", ""))
    path = DATA / data_filename(64, "full", regime, re)
    x = np.load(path, mmap_mode="r")
    n_t = x.shape[1]
    rows = []
    for t in range(n_t):
        f = np.asarray(x[args.run_index, t, 0], dtype=np.float64)
        fh = np.fft.fft2(f)
        enst = float((np.abs(fh) ** 2).sum() / f.size)
        rows.append({"t": t, "mean": float(f.mean()), "std": float(f.std()),
                     "enstrophy": enst,
                     "skew": float(((f - f.mean()) ** 3).mean() / (f.std() ** 3 + 1e-30))})
    out = {"case": args.case, "run_index": args.run_index, "shape": list(x.shape),
           "series": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    for t in [0, 50, 81, 100, 150, 182, 195]:
        if t < len(rows):
            r = rows[t]
            print(t, {k: round(r[k], 4) for k in ("mean", "std", "enstrophy", "skew")})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
