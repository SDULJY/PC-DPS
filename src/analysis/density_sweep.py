"""Observation-density sweep: d x d point observations on the 64x64 grid."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "third_party/qg-beta-sr")
sys.path.insert(0, "third_party/qg-2d")

from common.config import DATA, RESULTS, FIELD_STD, TEST_SNAPSHOT, data_filename  # noqa: E402
from common.metrics import relative_l2  # noqa: E402
from analysis.dps_mask import sample_dps_mask  # noqa: E402
from reproduce.run_inference import load_models, read_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="eddy_re1000_full")
    ap.add_argument("--densities", default="4,6,8,10")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshot", type=int, default=TEST_SNAPSHOT)
    ap.add_argument("--run-index", type=int, default=-6)
    ap.add_argument("--field-std", type=float, default=1.0)
    ap.add_argument("--dps-c", type=float, default=1e-4)
    ap.add_argument("--score-clip", type=float, default=None)
    ap.add_argument("--out", default=str(RESULTS / "p2" / "density_sweep.json"))
    args = ap.parse_args()
    regime, re_s, obs = args.case.split("_")
    re = int(re_s.replace("re", ""))
    truth_path = DATA / data_filename(64, "full", regime, re)
    if not truth_path.exists():
        raise SystemExit(f"missing {truth_path}")

    device = "cuda"
    model, _ = load_models(regime, re, obs, device)
    truth = read_frame(truth_path, args.run_index, args.snapshot)
    x_init = torch.from_numpy(truth / args.field_std).float().to(device)
    truth_n = truth / args.field_std

    rows = []
    for d in [int(x) for x in args.densities.split(",")]:
        idx = np.round(np.linspace(0, 63, d)).astype(int)
        mask_np = np.zeros((64, 64), np.float32)
        mask_np[np.ix_(idx, idx)] = 1.0
        mask = torch.from_numpy(mask_np).float().to(device)
        obs_np = truth_n * mask_np
        x_obs = torch.from_numpy(obs_np).float().to(device)
        t0 = time.time()
        with torch.no_grad():
            traj = sample_dps_mask(x_init, x_obs, mask, args.seed, args.n_samples,
                                   64, 2, model, 0.3, C=args.dps_c, clip=args.score_clip)
        samples = traj[..., -1].detach().cpu().numpy()
        rel = relative_l2(samples * args.field_std, truth)
        rows.append({"density": d, "dps_c": args.dps_c, "n_obs": int(mask_np.sum()),
                     "rel_l2_mean": float(rel.mean()),
                     "rel_l2_std": float(rel.std(ddof=1)),
                     "runtime_sec": time.time() - t0})
        print(json.dumps(rows[-1]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"case": args.case, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
