"""Audit whether truth and observation files share the same batch/time indexing.

For each available partial case, compare A_les(truth[run,t]) with the released
partial observation at the same index.  If they differ, search the batch/time
axis for the index that actually matches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "third_party" / "qg-beta-sr"))
sys.path.insert(0, str(ROOT.parent / "third_party" / "qg-2d"))

from common.config import DATA, RESULTS, TEST_SNAPSHOT, TEST_RUN_INDEX, data_filename  # noqa: E402
from common.metrics import relative_l2  # noqa: E402
from Models.samplers import A_les_interleaved  # noqa: E402
from Grid.grid import Grid  # noqa: E402
from Operators.operators import SpectralDerivatives  # noqa: E402


def frame(path, run, t):
    x = np.load(path, mmap_mode="r")
    return np.array(x[run, t, 0], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="eddy_re1000_partial,eddy_re10000_partial,jet_re1000_partial")
    ap.add_argument("--run-index", type=int, default=TEST_RUN_INDEX)
    ap.add_argument("--snapshot", type=int, default=TEST_SNAPSHOT)
    ap.add_argument("--out", default=str(RESULTS / "p0" / "alignment_audit.json"))
    args = ap.parse_args()

    grid = Grid(Lx=2 * np.pi, Ly=2 * np.pi, Nx=64, Ny=64, device="cuda")
    spec = SpectralDerivatives(grid)
    dps = SimpleNamespace(scale=4, C=2e-5, sigma_measure=1e-2, gappy=True)

    def A(x):
        t = torch.from_numpy(x).float().cuda()
        with torch.no_grad():
            y, _ = A_les_interleaved(t, dps, torch.tensor(0.5), grid, spec, 0)
        return y.detach().cpu().numpy().astype(np.float32)

    out = {}
    for case in args.cases.split(","):
        regime, re_s, obs = case.split("_")
        re = int(re_s.replace("re", ""))
        truth_path = DATA / data_filename(64, "full", regime, re)
        obs_path = DATA / data_filename(16, obs, regime, re)
        generated = ROOT.parent / "data" / "generated" / f"{regime}{re}_ds16_partial.npy"
        if not obs_path.exists() and generated.exists():
            obs_path = generated
        if not truth_path.exists() or not obs_path.exists():
            out[case] = {"skip": f"missing {truth_path} or {obs_path}"}
            continue
        tr = frame(truth_path, args.run_index, args.snapshot)
        xobs = np.load(obs_path, mmap_mode="r")
        obs_run = args.run_index if xobs.shape[0] > abs(args.run_index) else 0
        obs_t = args.snapshot if xobs.shape[1] > args.snapshot else 0
        ob = frame(obs_path, obs_run, obs_t)
        pred = A(tr)
        rec = {
            "truth": str(truth_path), "obs": str(obs_path),
            "run_index": args.run_index, "snapshot": args.snapshot,
            "obs_run_index": obs_run, "obs_snapshot": obs_t,
            "rel_same_index": float(relative_l2(pred, ob)),
            "obs_zero_frac": float((ob == 0).mean()),
        }
        if rec["rel_same_index"] > 1e-5 and xobs.shape[0] > 1 and xobs.shape[1] > 1:
            # search observation batch at fixed t
            x = xobs
            best = (1e9, None)
            for r in range(x.shape[0]):
                rr = float(relative_l2(pred, np.array(x[r, args.snapshot, 0], dtype=np.float32)))
                if rr < best[0]:
                    best = (rr, r)
            rec["best_obs_run"] = best[1]
            rec["best_obs_run_rel"] = best[0]
            # search observation time at fixed run
            best_t = (1e9, None)
            for t in range(x.shape[1]):
                rr = float(relative_l2(pred, np.array(x[args.run_index, t, 0], dtype=np.float32)))
                if rr < best_t[0]:
                    best_t = (rr, t)
            rec["best_obs_time"] = best_t[1]
            rec["best_obs_time_rel"] = best_t[0]
        out[case] = rec
        print(case, json.dumps(rec, indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
