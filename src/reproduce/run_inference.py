"""Run one baseline inference case with the released weights.

This is a thin wrapper around the third-party code: it imports the original
UNet and sampler, loads the released checkpoints, and writes samples + metrics.
No file under third_party/ is modified.
"""
from __future__ import annotations

import argparse
import json
import random
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
    DATA, WEIGHTS, FIELD_STD, N_DIFF, N_CORR, CORR_SCALE, SDEDIT_TIME,
    DPS_SCALE, DPS_C, DPS_SIGMA, CFG_WEIGHT, TEST_SNAPSHOT, TEST_RUN_INDEX,
    data_filename,
)
from common.metrics import summarize_ensemble  # noqa: E402
from common.runlog import write_log  # noqa: E402
from Models.networks import UNet_large  # noqa: E402
from Models.samplers import get_samples_corrected  # noqa: E402
from Grid.grid import Grid  # noqa: E402
from Operators.operators import SpectralDerivatives  # noqa: E402


def read_frame(path: Path, run_index: int, t_index: int) -> np.ndarray:
    x = np.load(path, mmap_mode="r")
    if x.ndim == 5:       # (batch, time, channel, H, W)
        arr = x[run_index, t_index, 0]
    elif x.ndim == 4:     # (batch, time, H, W)
        arr = x[run_index, t_index]
    elif x.ndim == 3:     # (time, H, W)
        arr = x[t_index]
    else:
        raise ValueError(f"unexpected array shape {x.shape} in {path}")
    return np.array(arr, dtype=np.float32)


def checkpoint_name(regime: str, re: int, obs: str, conditional: bool) -> str:
    if not conditional:
        return f"model_uncond_{regime}_Re_{re}.pt"
    if obs == "partial":
        return f"model_cond_ds16_partial_{regime}_Re_{re}.pt"
    return f"model_cond_ds16_{regime}_Re_{re}.pt"


def load_models(regime: str, re: int, obs: str, device: str = "cuda"):
    uncond = UNet_large(channels=[32, 64, 128, 256], in_channels=1,
                        out_channels=1, dropout_rate=0, attention=False,
                        condition=False).to(device)
    cond = UNet_large(channels=[32, 64, 128, 256], in_channels=2,
                      out_channels=1, dropout_rate=0, attention=False,
                      condition=True).to(device)
    for model, conditional in [(uncond, False), (cond, True)]:
        name = checkpoint_name(regime, re, obs, conditional)
        sd = torch.load(WEIGHTS / name, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        model.eval()
    return uncond, cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="e.g. jet_re1000_full")
    ap.add_argument("--method", required=True, choices=["sdedit", "dps", "cond", "cfg"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--snapshot", type=int, default=TEST_SNAPSHOT)
    ap.add_argument("--run-index", type=int, default=TEST_RUN_INDEX)
    ap.add_argument("--obs-run-index", type=int, default=None,
                    help="separate batch index for a generated observation file")
    ap.add_argument("--obs-snapshot", type=int, default=None,
                    help="separate time index for a generated observation file")
    ap.add_argument("--field-std", type=float, default=FIELD_STD,
                    help="fallback normalization divisor (repo config uses 1.3)")
    ap.add_argument("--truth-std", type=float, default=None,
                    help="explicit target normalizer; overrides --field-std")
    ap.add_argument("--obs-std", type=float, default=None,
                    help="explicit observation normalizer; overrides --field-std")
    ap.add_argument("--normalize-target", action="store_true",
                    help="scale both target and observation by the target-frame std "
                         "(recommended: the observation is a filtered version of the target)")
    ap.add_argument("--normalize-frame", action="store_true",
                    help="standardize target and observation each by their own std (diagnostic)")
    ap.add_argument("--dps-c", type=float, default=DPS_C)
    ap.add_argument("--dps-sigma", type=float, default=DPS_SIGMA)
    ap.add_argument("--dps-scale", type=int, default=DPS_SCALE)
    ap.add_argument("--truth-file", default=None)
    ap.add_argument("--obs-file", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    regime, re_s, obs = args.case.split("_")
    re = int(re_s.replace("re", ""))
    truth_path = Path(args.truth_file) if args.truth_file else DATA / data_filename(64, "full", regime, re)
    obs_path = Path(args.obs_file) if args.obs_file else DATA / data_filename(16, obs, regime, re)
    if not truth_path.exists():
        raise SystemExit(f"missing truth file: {truth_path}")
    if not obs_path.exists():
        raise SystemExit(f"missing observation file: {obs_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    device = "cuda"
    grid = Grid(Lx=2 * np.pi, Ly=2 * np.pi, Nx=64, Ny=64, device=device)
    spec = SpectralDerivatives(grid)
    model_uncond, model_cond = load_models(regime, re, obs, device)

    raw_truth = read_frame(truth_path, args.run_index, args.snapshot)
    obs_run_index = args.run_index if args.obs_run_index is None else args.obs_run_index
    obs_snapshot = args.snapshot if args.obs_snapshot is None else args.obs_snapshot
    raw_obs = read_frame(obs_path, obs_run_index, obs_snapshot)

    if args.normalize_target:
        truth_std = float(raw_truth.std())
        obs_std = truth_std
    elif args.normalize_frame:
        truth_std = float(raw_truth.std())
        if obs == "partial":
            v = raw_obs[raw_obs != 0]
            obs_std = float(v.std()) if v.size else float(raw_obs.std())
        else:
            obs_std = float(raw_obs.std())
    else:
        truth_std = args.field_std
        obs_std = args.field_std
    if args.truth_std is not None:
        truth_std = args.truth_std
    if args.obs_std is not None:
        obs_std = args.obs_std

    truth = raw_truth / truth_std
    observed = raw_obs / obs_std
    x_init = torch.from_numpy(truth).float().to(device)
    x_cond = torch.from_numpy(observed).float().to(device)

    sdedit = SimpleNamespace(flag=(args.method == "sdedit"), time=SDEDIT_TIME)
    dps = SimpleNamespace(flag=(args.method == "dps"), scale=args.dps_scale, C=args.dps_c,
                          sigma_measure=args.dps_sigma, gappy=(obs == "partial"))
    cfg = SimpleNamespace(flag=(args.method == "cfg"), weight=CFG_WEIGHT)
    cond_flag = args.method in ("cond", "cfg")
    n_corr = 0 if args.method == "sdedit" else N_CORR

    t0 = time.time()
    with torch.no_grad():
        traj = get_samples_corrected(
            x_init, 0, x_cond, args.seed, args.n_samples, N_DIFF, n_corr,
            model_uncond, model_cond, CORR_SCALE, sdedit, dps, grid, spec,
            cond_flag, cfg,
        )
    runtime = time.time() - t0

    final_idx = SDEDIT_TIME if args.method == "sdedit" else -1
    samples = traj[..., final_idx].detach().cpu().numpy().astype(np.float32)
    samples_phys = samples * truth_std
    truth_phys = raw_truth

    metrics = summarize_ensemble(samples_phys, truth_phys)
    metrics["runtime_sec"] = runtime
    metrics["final_traj_index"] = int(final_idx)

    out_dir = Path(args.out) if args.out else (
        ROOT.parent / "results" / "p1" / args.case / args.method / f"seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "samples.npz",
                        samples=samples_phys, truth=truth_phys,
                        observed=raw_obs.astype(np.float32))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    payload = {
        "stage": "p1",
        "case": args.case,
        "regime": regime, "re": re, "obs": obs,
        "method": args.method, "seed": args.seed,
        "n_samples": args.n_samples, "snapshot": args.snapshot,
        "run_index": args.run_index,
        "truth_file": str(truth_path), "obs_file": str(obs_path),
        "hyperparameters": {
            "field_std": args.field_std, "truth_std": truth_std,
            "obs_std": obs_std, "normalize_target": bool(args.normalize_target),
            "normalize_frame": bool(args.normalize_frame),
            "n_diff": N_DIFF, "n_corr": n_corr,
            "corr_scale": CORR_SCALE, "sdedit_time": SDEDIT_TIME,
            "dps_scale": args.dps_scale, "dps_C": args.dps_c, "dps_sigma": args.dps_sigma,
            "cfg_weight": CFG_WEIGHT,
        },
        "metrics": metrics,
        "outputs": [str(out_dir / "samples.npz"), str(out_dir / "metrics.json")],
    }
    write_log(out_dir, payload)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
