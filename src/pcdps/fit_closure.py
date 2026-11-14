"""Fit effective linear QG closure coefficients on training runs (P3.2).

Model:
    dq/dt + c_J J + c_mu (mu q) + c_nu (-nu lap q)
          + c_beta (beta dpsi/dx) - c_F F = 0

The coefficients are fit by least squares on pairs from training runs and
validated on held-out runs.  c_J=c_mu=c_nu=c_beta=c_F=1 is the original QG
operator; deviations absorb subgrid/filtering effects.
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

from common.config import DATA, RESULTS, data_filename  # noqa: E402
from pcdps.qg_ops import QGOps  # noqa: E402


def frame(path, run, t):
    x = np.load(path, mmap_mode="r")
    return torch.from_numpy(np.array(x[run, t, 0], dtype=np.float32)).cuda()


def design(ops, q0, q1, dt):
    terms = ops.residual_terms(q0, q1, dt, time_average=True)
    cols = [terms["J"], terms["drag"], terms["diffusion"]]
    names = ["c_J", "c_mu", "c_nu"]
    if ops.beta != 0:
        cols.append(terms["beta_basis"])
        names.append("c_beta")
    cols.append(-ops.biharmonic(0.5 * (q0 + q1)))
    names.append("c_h4")
    cols.append(-terms["F"])
    names.append("c_F")
    X = torch.stack([c.reshape(-1) for c in cols], dim=1)
    b = -terms["dqdt"].reshape(-1)
    return X, b, names


def fit_case(case, train_runs, val_runs, snapshots, dt):
    regime, re_s, _ = case.split("_")
    re = int(re_s.replace("re", ""))
    path = DATA / data_filename(64, "full", regime, re)
    beta = 2.5 if regime == "jet" else 0.0
    ops = QGOps(n=64, device="cuda", dtype=torch.float32, mu=2e-2,
                beta=beta, re=re, forcing=True)
    rng = np.random.default_rng(0)

    def collect(runs, n_per_run):
        Xs, bs = [], []
        for r in runs:
            for _ in range(n_per_run):
                t = int(rng.choice(snapshots))
                q0 = frame(path, r, t)
                q1 = frame(path, r, t + 1)
                q0 = q0 - q0.mean(); q1 = q1 - q1.mean()
                X, b, names = design(ops, q0, q1, dt)
                Xs.append(X.cpu().numpy())
                bs.append(b.cpu().numpy())
        return np.concatenate(Xs), np.concatenate(bs), names

    Xtr, btr, names = collect(train_runs, 5)
    Xva, bva, _ = collect(val_runs, 5)
    # standardize for conditioning, then unscale
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0) + 1e-12
    Z = (Xtr - mu) / sd
    coef_std, *_ = np.linalg.lstsq(Z, btr, rcond=None)
    coef = coef_std / sd
    # intercept is not used in the residual; report it for diagnostics
    intercept = float((btr - Z @ coef_std).mean())
    coef = np.asarray(coef, dtype=float)

    def ratio(X, b, c):
        r = b - X @ c
        # baseline residual is b - X @ [1,1,1,1,1]
        return float(np.linalg.norm(r) / (np.linalg.norm(b) + 1e-12))

    base = np.array([0.0 if n == "c_h4" else 1.0 for n in names])
    out = {
        "case": case, "beta": beta, "nu": 1.0 / re,
        "train_runs": list(train_runs), "val_runs": list(val_runs),
        "coeff_names": names,
        "coeff": coef.tolist(),
        "intercept": intercept,
        "train_ratio_baseline": ratio(Xtr, btr, base),
        "train_ratio_fit": ratio(Xtr, btr, coef),
        "val_ratio_baseline": ratio(Xva, bva, base),
        "val_ratio_fit": ratio(Xva, bva, coef),
    }
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="eddy_re1000_full,eddy_re10000_full,jet_re1000_full")
    ap.add_argument("--train-runs", default="0-399")
    ap.add_argument("--val-runs", default="400-479")
    ap.add_argument("--snapshots", default="101-194")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--out", default=str(RESULTS / "p3" / "closure_coefficients.json"))
    args = ap.parse_args()

    def parse_runs(s):
        out = []
        for part in s.split(","):
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return out

    train = parse_runs(args.train_runs)
    val = parse_runs(args.val_runs)
    snaps = parse_runs(args.snapshots)
    out = {}
    for case in args.cases.split(","):
        regime, re_s, obs = case.split("_")
        re = int(re_s.replace("re", ""))
        if not (DATA / data_filename(64, "full", regime, re)).exists():
            print(f"skip {case}: missing truth")
            continue
        out[case] = fit_case(case, train, val, snaps, args.dt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
