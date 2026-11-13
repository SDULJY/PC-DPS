"""Estimate beta from a pair of vorticity fields via QG residual least squares.

For each pair (q_t, q_{t+dt}):
    r(beta) = r0 + beta * b,   b = dpsi/dx
    beta_hat = -<r0, b> / <b, b>

Works on truth pairs (oracle) or on PC-DPS/DPS reconstructions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "third_party/qg-beta-sr")
sys.path.insert(0, "third_party/qg-2d")

from common.config import DATA, RESULTS, TEST_SNAPSHOT, TEST_RUN_INDEX, data_filename  # noqa: E402
from pcdps.qg_ops import QGOps  # noqa: E402


def read_frame(path, run_index, t_index):
    x = np.load(path, mmap_mode="r")
    return np.array(x[run_index, t_index, 0], dtype=np.float32)


def beta_per_pair(q0s, q1s, ops):
    vals = []
    for q0, q1 in zip(q0s, q1s):
        r0 = ops.residual_no_beta(q0, q1, 0.5)
        b = ops.beta_basis(q0, q1)
        vals.append(float(-(r0 * b).sum() / ((b * b).sum() + 1e-12)))
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="jet_re1000_partial")
    ap.add_argument("--source", default="truth", choices=["truth", "pcdps"])
    ap.add_argument("--lambda-phys", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshot", type=int, default=TEST_SNAPSHOT)
    ap.add_argument("--run-index", type=int, default=TEST_RUN_INDEX)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    regime, re_s, obs = args.case.split("_")
    re = int(re_s.replace("re", ""))
    beta_true = 2.5 if regime == "jet" else 0.0

    if args.source == "truth":
        path = DATA / data_filename(64, "full", regime, re)
        q0 = read_frame(path, args.run_index, args.snapshot)
        q1 = read_frame(path, args.run_index, args.snapshot + 1)
        q0s, q1s = [q0], [q1]
    else:
        p = (ROOT.parent / "results" / "p3" / args.case
             / f"lambda{args.lambda_phys:g}" / f"seed{args.seed}" / "samples.npz")
        d = np.load(p)
        s = d["samples"]  # [2N,H,W] physical
        n = s.shape[0] // 2
        q0s, q1s = s[:n], s[n:]

    ops = QGOps(n=64, device="cpu", dtype=torch.float64, mu=2e-2,
                beta=0.0, re=re, forcing=True)
    vals = beta_per_pair([torch.from_numpy(x.astype(np.float64)) for x in q0s],
                         [torch.from_numpy(x.astype(np.float64)) for x in q1s], ops)

    # pooled estimate over all pairs
    r0s, bs = [], []
    for q0, q1 in zip(q0s, q1s):
        r0s.append(ops.residual_no_beta(torch.from_numpy(q0.astype(np.float64)),
                                        torch.from_numpy(q1.astype(np.float64)), 0.5))
        bs.append(ops.beta_basis(torch.from_numpy(q0.astype(np.float64)),
                                 torch.from_numpy(q1.astype(np.float64))))
    r0 = torch.cat([x.reshape(-1) for x in r0s])
    b = torch.cat([x.reshape(-1) for x in bs])
    beta_pooled = float(-(r0 * b).sum() / ((b * b).sum() + 1e-12))

    rng = np.random.default_rng(0)
    boot = [float(np.mean(rng.choice(vals, size=len(vals), replace=True)))
            for _ in range(2000)]
    out = {
        "case": args.case, "source": args.source, "beta_true": beta_true,
        "n_pairs": int(len(vals)),
        "beta_hat_mean": float(vals.mean()),
        "beta_hat_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "beta_hat_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "beta_hat_pooled": beta_pooled,
        "beta_hat_values": vals.tolist(),
    }
    if beta_true > 0:
        out["relative_error_mean"] = abs(out["beta_hat_mean"] - beta_true) / beta_true
        out["relative_error_pooled"] = abs(beta_pooled - beta_true) / beta_true
    else:
        out["absolute_error_mean"] = abs(out["beta_hat_mean"])
        out["absolute_error_pooled"] = abs(beta_pooled)
    out_dir = Path(args.out) if args.out else (
        ROOT.parent / "results" / "p4" / args.case / args.source / f"seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "beta_estimate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
