"""Residual audit across runs and wavenumber bands (P3.1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "third_party/qg-beta-sr")
sys.path.insert(0, "third_party/qg-2d")

from common.config import DATA, FIGURES, RESULTS, TEST_SNAPSHOT, data_filename  # noqa: E402
from common.metrics import radial_enstrophy_spectrum  # noqa: E402
from pcdps.qg_ops import QGOps  # noqa: E402


def frame(path, run, t):
    x = np.load(path, mmap_mode="r")
    return torch.from_numpy(np.array(x[run, t, 0], dtype=np.float32)).cuda()


def spatial(ops, q):
    psi = ops.psi(q)
    return (ops.jacobian(psi, q), ops.mu * q,
            -ops.nu * ops.laplacian(q), ops.beta_sign * ops.d_dx(psi))


def rms(x):
    return float(torch.sqrt((x ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="eddy_re1000_full,eddy_re10000_full,jet_re1000_full")
    ap.add_argument("--runs", default="0,100,200,300,400,-6")
    ap.add_argument("--snapshot", type=int, default=TEST_SNAPSHOT)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--out", default=str(RESULTS / "p3" / "residual_audit.csv"))
    args = ap.parse_args()
    rows = []
    spectra = {}
    for case in args.cases.split(","):
        regime, re_s, obs = case.split("_")
        re = int(re_s.replace("re", ""))
        path = DATA / data_filename(64, "full", regime, re)
        if not path.exists():
            continue
        beta = 2.5 if regime == "jet" else 0.0
        ops = QGOps(n=64, device="cuda", dtype=torch.float32, mu=2e-2,
                    beta=beta, re=re, forcing=True)
        for run in [int(x) for x in args.runs.split(",")]:
            qm = frame(path, run, args.snapshot - 1)
            q0 = frame(path, run, args.snapshot)
            q1 = frame(path, run, args.snapshot + 1)
            qm, q0, q1 = qm - qm.mean(), q0 - q0.mean(), q1 - q1.mean()
            terms = ops.residual_terms(q0, q1, args.dt, time_average=True)
            r_fwd = ops.residual(q0, q1, args.dt, time_average=True)
            r0 = ops.residual_no_beta(q0, q1, args.dt, time_average=True)
            dqdt_c = (q1 - qm) / (2 * args.dt)
            Jm, dm, fm, bm = spatial(ops, qm)
            J0, d0, f0, b0 = spatial(ops, q0)
            J1, d1, f1, b1 = spatial(ops, q1)
            w = (0.25, 0.5, 0.25)
            r_c = (dqdt_c + w[0] * Jm + w[1] * J0 + w[2] * J1
                   + w[0] * dm + w[1] * d0 + w[2] * d1
                   + w[0] * fm + w[1] * f0 + w[2] * f1
                   + beta * (w[0] * bm + w[1] * b0 + w[2] * b1) - terms["F"])
            row = {
                "case": case, "run": run, "snapshot": args.snapshot,
                "beta": beta, "nu": 1.0 / re,
                "rms_dqdt": rms(terms["dqdt"]), "rms_J": rms(terms["J"]),
                "rms_r_fwd": rms(r_fwd), "rms_r_centered": rms(r_c),
                "rms_r0": rms(r0),
                "ratio_fwd": rms(r_fwd) / (rms(terms["dqdt"]) + 1e-12),
                "ratio_centered": rms(r_c) / (rms(dqdt_c) + 1e-12),
                "beta_oracle": float(-(r0 * ops.beta_basis(q0, q1)).sum()
                                     / ((ops.beta_basis(q0, q1) ** 2).sum() + 1e-12)),
            }
            rows.append(row)
            if run == -6:
                k, zd = radial_enstrophy_spectrum(dqdt_c.detach().cpu().numpy())
                _, zr = radial_enstrophy_spectrum(r_c.detach().cpu().numpy())
                spectra[case] = {"k": k.tolist(), "dqdt": zd.tolist(),
                                 "residual": zr.tolist()}
            print(row)
    RESULTS.joinpath("p3").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    (RESULTS / "p3" / "residual_audit.json").write_text(json.dumps({
        "rows": rows, "spectra": spectra}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, case in zip(axes, ["eddy_re1000_full", "jet_re1000_full"]):
        s = spectra.get(case)
        if not s:
            continue
        k = np.array(s["k"])
        ax.loglog(k[1:], np.array(s["dqdt"])[1:], label="dq/dt")
        ax.loglog(k[1:], np.array(s["residual"])[1:], label="centered residual")
        ax.set_xlabel("k"); ax.set_ylabel("spectrum")
        ax.set_title(case); ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13_residual_spectrum.png", dpi=300)
    fig.savefig(FIGURES / "fig13_residual_spectrum.pdf")
    print(f"wrote {args.out} and fig13")


if __name__ == "__main__":
    main()
