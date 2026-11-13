"""Binary regime/beta model selection from held-out observations.

For a dot-observation mask, run DPS with the eddy (beta=0) and jet (beta=2.5)
unconditional models.  The model with the lower error on 20% held-out
observations is selected.  This is the no-training part of the beta inversion.
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

from common.config import DATA, RESULTS, FIELD_STD, TEST_SNAPSHOT, TEST_RUN_INDEX, WEIGHTS, data_filename  # noqa: E402
from common.metrics import relative_l2  # noqa: E402
from analysis.dps_mask import sample_dps_mask  # noqa: E402
from Models.networks import UNet_large  # noqa: E402


def read_frame(path, run_index, t_index):
    x = np.load(path, mmap_mode="r")
    return np.array(x[run_index, t_index, 0], dtype=np.float32)


def load_uncond(regime, re, device="cuda"):
    m = UNet_large(channels=[32, 64, 128, 256], in_channels=1, out_channels=1,
                   dropout_rate=0, attention=False, condition=False).to(device)
    m.load_state_dict(torch.load(WEIGHTS / f"model_uncond_{regime}_Re_{re}.pt",
                                 map_location="cpu", weights_only=True))
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="jet_re1000_full")
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshot", type=int, default=TEST_SNAPSHOT)
    ap.add_argument("--run-index", type=int, default=TEST_RUN_INDEX)
    ap.add_argument("--field-std", type=float, default=1.0)
    ap.add_argument("--out", default=str(RESULTS / "p4" / "model_selection.json"))
    args = ap.parse_args()
    regime, re_s, obs = args.case.split("_")
    re = int(re_s.replace("re", ""))
    path = DATA / data_filename(64, "full", regime, re)
    truth = read_frame(path, args.run_index, args.snapshot)
    truth_n = truth / args.field_std

    idx = np.round(np.linspace(0, 63, args.grid)).astype(int)
    pts = [(i, j) for i in idx for j in idx]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pts)
    n_eval = max(1, int(0.2 * len(pts)))
    eval_pts = pts[:n_eval]
    train_pts = pts[n_eval:]
    train_mask = np.zeros((64, 64), np.float32)
    eval_mask = np.zeros((64, 64), np.float32)
    for i, j in train_pts:
        train_mask[i, j] = 1
    for i, j in eval_pts:
        eval_mask[i, j] = 1
    x_obs = torch.from_numpy(truth_n * train_mask).float().cuda()
    x_init = torch.from_numpy(truth_n).float().cuda()
    mask_t = torch.from_numpy(train_mask).float().cuda()
    eval_bool = eval_mask.astype(bool)

    results = {}
    for model_regime in ["eddy", "jet"]:
        model = load_uncond(model_regime, re)
        with torch.no_grad():
            traj = sample_dps_mask(x_init, x_obs, mask_t, args.seed, args.n_samples,
                                   64, 2, model, 0.3)
        samples = traj[..., -1].detach().cpu().numpy() * FIELD_STD
        heldout = np.sqrt(((samples[:, eval_bool] - truth[eval_bool][None]) ** 2).mean(axis=1))
        results[model_regime] = {
            "heldout_rmse_mean": float(heldout.mean()),
            "heldout_rmse_std": float(heldout.std(ddof=1)),
            "rel_l2_mean": float(relative_l2(samples, truth).mean()),
            "n_eval": int(n_eval), "n_train": int(len(train_pts)),
        }
    selected = min(results, key=lambda k: results[k]["heldout_rmse_mean"])
    out = {"case": args.case, "true_regime": regime,
           "selected_regime": selected,
           "correct": selected == regime, "results": results,
           "report": "eddy -> beta=0; jet -> beta=2.5"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
