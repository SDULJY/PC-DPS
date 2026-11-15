"""Run PC-DPS (or paired DPS at lambda_phys=0) for one case/seed."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "third_party" / "qg-beta-sr"))
sys.path.insert(0, str(ROOT.parent / "third_party" / "qg-2d"))

from common.config import (  # noqa: E402
    DATA, FIELD_STD, N_DIFF, N_CORR, CORR_SCALE, DPS_SCALE, DPS_C, DPS_SIGMA,
    TEST_SNAPSHOT, TEST_RUN_INDEX, data_filename,
)
from common.metrics import summarize_ensemble  # noqa: E402
from common.runlog import write_log  # noqa: E402
from pcdps.qg_ops import QGOps  # noqa: E402
from pcdps.sampler import sample_pcdps  # noqa: E402
from reproduce.run_inference import load_models, read_frame  # noqa: E402
from Grid.grid import Grid  # noqa: E402
from Operators.operators import SpectralDerivatives  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--lambda-phys", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pairs", type=int, default=16)
    ap.add_argument("--snapshot", type=int, default=TEST_SNAPSHOT)
    ap.add_argument("--run-index", type=int, default=TEST_RUN_INDEX)
    ap.add_argument("--field-std", type=float, default=FIELD_STD)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--no-time-average", action="store_true")
    ap.add_argument("--residual-kmax", type=float, default=None)
    ap.add_argument("--dps-c", type=float, default=DPS_C)
    ap.add_argument("--physics-gap-only", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--phys-t-min", type=float, default=0.0)
    ap.add_argument("--phys-t-max", type=float, default=1.0)
    ap.add_argument("--phys-lowpass", type=float, default=None)
    ap.add_argument("--per-sample-scale", action="store_true")
    ap.add_argument("--phys-late-boost", type=float, default=0.0)
    ap.add_argument("--triple", action="store_true",
                    help="reconstruct (t-1,t,t+1) and use a centered residual")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    regime, re_s, obs = args.case.split("_")
    re = int(re_s.replace("re", ""))
    truth_path = DATA / data_filename(64, "full", regime, re)
    obs_path = DATA / data_filename(16, obs, regime, re)
    if not truth_path.exists() or not obs_path.exists():
        raise SystemExit(f"missing data: {truth_path} / {obs_path}")

    device = "cuda"
    grid = Grid(Lx=2 * np.pi, Ly=2 * np.pi, Nx=64, Ny=64, device=device)
    spec = SpectralDerivatives(grid)
    model_uncond, _ = load_models(regime, re, obs, device)

    if args.triple:
        truth_prev = read_frame(truth_path, args.run_index, args.snapshot - 1)
        truth_mid = read_frame(truth_path, args.run_index, args.snapshot)
        truth_next = read_frame(truth_path, args.run_index, args.snapshot + 1)
        obs_prev = read_frame(obs_path, args.run_index, args.snapshot - 1)
        obs_mid = read_frame(obs_path, args.run_index, args.snapshot)
        obs_next = read_frame(obs_path, args.run_index, args.snapshot + 1)
        truth_frames = np.stack([truth_prev, truth_mid, truth_next])
        obs_frames = np.stack([obs_prev, obs_mid, obs_next])
        x_init = torch.from_numpy(truth_mid / args.field_std).float().to(device)
        x_cond = torch.from_numpy(obs_frames / args.field_std).float().to(device)
    else:
        truth0 = read_frame(truth_path, args.run_index, args.snapshot)
        truth1 = read_frame(truth_path, args.run_index, args.snapshot + 1)
        obs0 = read_frame(obs_path, args.run_index, args.snapshot)
        obs1 = read_frame(obs_path, args.run_index, args.snapshot + 1)
        truth_frames = np.stack([truth0, truth1])
        obs_frames = np.stack([obs0, obs1])
        x_init = torch.from_numpy(truth0 / args.field_std).float().to(device)
        x_cond = torch.from_numpy(obs_frames / args.field_std).float().to(device)

    beta = 2.5 if regime == "jet" else 0.0
    ops = QGOps(n=64, device=device, dtype=torch.float32, mu=2e-2,
                beta=beta, re=re, forcing=True)

    phys_mask = None
    if args.physics_gap_only and obs == "partial":
        m = (obs_frames[1 if args.triple else 0] != 0).astype("float32")
        phys_mask = torch.from_numpy(1.0 - m).float().to(device)
    dps = SimpleNamespace(flag=True, scale=DPS_SCALE, C=args.dps_c,
                          sigma_measure=DPS_SIGMA, gappy=(obs == "partial"))
    t0 = time.time()
    traj, diagnostics = sample_pcdps(
        x_init, x_cond, args.seed, args.n_pairs, N_DIFF, N_CORR,
        model_uncond, CORR_SCALE, dps, grid, spec, ops=ops,
        lambda_phys=args.lambda_phys, dt=args.dt,
        pair=not args.triple, triple=args.triple,
        time_average=not args.no_time_average, record_diag=True,
        residual_kmax=args.residual_kmax, field_std=args.field_std,
        phys_mask=phys_mask, grad_clip=args.grad_clip,
        phys_t_min=args.phys_t_min, phys_t_max=args.phys_t_max,
        phys_lowpass=args.phys_lowpass,
        per_sample_scale=args.per_sample_scale,
        phys_late_boost=args.phys_late_boost,
    )
    runtime = time.time() - t0

    samples = traj[..., -1].detach().cpu().numpy().astype(np.float32)
    n = args.n_pairs
    samples_phys = samples * args.field_std
    truth_phys = truth_frames
    if args.triple:
        mid = samples_phys[n:2 * n]
        m_mid = summarize_ensemble(mid, truth_phys[1])
        metrics = {
            "frame_mid": m_mid,
            "rel_l2_mean": m_mid["rel_l2_mean"],
            "rel_l2_ensmean": m_mid["rel_l2_ensmean"],
            "log_spectrum_rel_error": m_mid["log_spectrum_rel_error"],
            "spectrum_tail_ratio": m_mid["spectrum_tail_ratio"],
            "runtime_sec": runtime,
            "diagnostics": diagnostics,
        }
    else:
        m0 = summarize_ensemble(samples_phys[:n], truth_phys[0])
        m1 = summarize_ensemble(samples_phys[n:], truth_phys[1])
        metrics = {
            "frame0": m0,
            "frame1": m1,
            "rel_l2_mean": float(np.mean([m0["rel_l2_mean"], m1["rel_l2_mean"]])),
            "rel_l2_ensmean": float(np.mean([m0["rel_l2_ensmean"], m1["rel_l2_ensmean"]])),
            "log_spectrum_rel_error": float(np.mean([m0["log_spectrum_rel_error"],
                                                    m1["log_spectrum_rel_error"]])),
            "spectrum_tail_ratio": float(np.mean([m0["spectrum_tail_ratio"],
                                                  m1["spectrum_tail_ratio"]])),
            "runtime_sec": runtime,
            "diagnostics": diagnostics,
        }

    out_dir = Path(args.out) if args.out else (
        ROOT.parent / "results" / "p3" / args.case
        / f"lambda{args.lambda_phys:g}" / f"seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "samples.npz",
                        samples=samples_phys,
                        truth=truth_phys,
                        observed=obs_frames.astype(np.float32))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_log(out_dir, {
        "stage": "p3",
        "case": args.case, "regime": regime, "re": re, "obs": obs,
        "lambda_phys": args.lambda_phys, "seed": args.seed, "n_pairs": n,
        "snapshot": args.snapshot, "dt": args.dt, "beta": beta,
        "time_average": not args.no_time_average, "triple": bool(args.triple),
        "residual_kmax": args.residual_kmax,
        "physics_gap_only": bool(args.physics_gap_only), "grad_clip": args.grad_clip, "phys_t_min": args.phys_t_min, "phys_t_max": args.phys_t_max,
        "phys_lowpass": args.phys_lowpass, "per_sample_scale": bool(args.per_sample_scale), "phys_late_boost": args.phys_late_boost,
        "hyperparameters": {"field_std": args.field_std, "n_diff": N_DIFF,
                            "n_corr": N_CORR, "corr_scale": CORR_SCALE,
                            "dps_scale": DPS_SCALE, "dps_C": args.dps_c,
                            "dps_sigma": DPS_SIGMA},
        "metrics": metrics,
        "outputs": [str(out_dir / "samples.npz"), str(out_dir / "metrics.json")],
    })
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
