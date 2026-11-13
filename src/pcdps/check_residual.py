"""Audit the QG residual on truth pairs before trusting the physics gradient."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="jet_re1000_full")
    ap.add_argument("--snapshot", type=int, default=TEST_SNAPSHOT)
    ap.add_argument("--run-index", type=int, default=TEST_RUN_INDEX)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--out", default=str(RESULTS / "p3" / "residual_check.json"))
    args = ap.parse_args()
    regime, re_s, obs = args.case.split("_")
    re = int(re_s.replace("re", ""))
    path = DATA / data_filename(64, "full", regime, re)
    if not path.exists():
        raise SystemExit(f"missing {path}")

    q0 = torch.from_numpy(read_frame(path, args.run_index, args.snapshot)).float().cuda()
    q1 = torch.from_numpy(read_frame(path, args.run_index, args.snapshot + 1)).float().cuda()
    qm = torch.from_numpy(read_frame(path, args.run_index, args.snapshot - 1)).float().cuda()
    # The vorticity mean is zero; remove any residual k=0 mode for the audit.
    q0 = q0 - q0.mean()
    q1 = q1 - q1.mean()
    qm = qm - qm.mean()

    beta = 2.5 if regime == "jet" else 0.0
    ops = QGOps(n=64, device="cuda", dtype=torch.float32, mu=2e-2,
                beta=beta, re=re, forcing=True)
    terms = ops.residual_terms(q0, q1, args.dt, time_average=True)
    r = ops.residual(q0, q1, args.dt, time_average=True)
    r0 = ops.residual_no_beta(q0, q1, args.dt, time_average=True)
    b = ops.beta_basis(q0, q1, time_average=True)

    def rms(x):
        return float(torch.sqrt((x ** 2).mean()))

    # Centered (three-frame) residual: dq/dt at t from q(t-1), q(t+1);
    # spatial terms at t with Simpson weights [1,2,1]/4.
    def spatial(q):
        psi = ops.psi(q)
        return (ops.jacobian(psi, q), ops.mu * q,
                -ops.nu * ops.laplacian(q), ops.d_dx(psi))

    Jm, dragm, diffm, betam = spatial(qm)
    J0, drag0, diff0, beta0 = spatial(q0)
    J1, drag1, diff1, beta1 = spatial(q1)
    w = (0.25, 0.5, 0.25)
    dqdt_c = (q1 - qm) / (2 * args.dt)
    Jc = w[0] * Jm + w[1] * J0 + w[2] * J1
    drag_c = w[0] * dragm + w[1] * drag0 + w[2] * drag1
    diff_c = w[0] * diffm + w[1] * diff0 + w[2] * diff1
    beta_c = w[0] * betam + w[1] * beta0 + w[2] * beta1
    r_c = dqdt_c + Jc + drag_c + diff_c + beta * beta_c - terms["F"]

    out = {
        "case": args.case, "snapshot": args.snapshot, "dt": args.dt,
        "beta": beta, "nu": 1.0 / re, "mu": 2e-2,
        "rms": {k: rms(v) for k, v in terms.items()},
        "residual_rms": rms(r),
        "residual_no_beta_rms": rms(r0),
        "beta_basis_rms": rms(b),
        "centered_residual_rms": rms(r_c),
        "ratios": {
            "residual_over_dqdt": rms(r) / (rms(terms["dqdt"]) + 1e-12),
            "residual_over_J": rms(r) / (rms(terms["J"]) + 1e-12),
            "residual_over_terms_sum": rms(r) / (sum(rms(v) for v in terms.values()) + 1e-12),
            "centered_over_dqdt": rms(r_c) / (rms(dqdt_c) + 1e-12),
        },
    }
    out["beta_oracle_hat"] = float(-(r0 * b).sum() / ((b * b).sum() + 1e-12))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
