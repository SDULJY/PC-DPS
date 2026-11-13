"""Build a ds16 observation from a truth frame with the paper's exact operator.

This is used when the released ds16_* file is still in the undownloaded tail,
and to test which observation the conditional model expects.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "third_party/qg-beta-sr")
sys.path.insert(0, "third_party/qg-2d")

from common.config import DATA, TEST_SNAPSHOT, data_filename  # noqa: E402
from Models.samplers import A_les_interleaved  # noqa: E402
from Grid.grid import Grid  # noqa: E402
from Operators.operators import SpectralDerivatives  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--snapshot", type=int, default=TEST_SNAPSHOT)
    ap.add_argument("--run-index", type=int, default=-6)
    ap.add_argument("--gappy", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    regime, re_s, obs = args.case.split("_")
    re = int(re_s.replace("re", ""))
    path = DATA / data_filename(64, "full", regime, re)
    x = np.load(path, mmap_mode="r")
    truth = np.array(x[args.run_index, args.snapshot, 0], dtype=np.float32)
    t = torch.from_numpy(truth).float().cuda()
    grid = Grid(Lx=2 * np.pi, Ly=2 * np.pi, Nx=64, Ny=64, device="cuda")
    spec = SpectralDerivatives(grid)
    dps = SimpleNamespace(scale=4, C=2e-5, sigma_measure=1e-2, gappy=args.gappy)
    with torch.no_grad():
        obs_field, _ = A_les_interleaved(t, dps, torch.tensor(0.5), grid, spec, 0)
    arr = obs_field.detach().cpu().numpy().astype(np.float32)
    out = np.zeros((1, 1, 1, 64, 64), np.float32)
    out[0, 0, 0] = arr
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    print(f"wrote {args.out} shape={out.shape} nonzero={float((arr != 0).mean()):.4f}")


if __name__ == "__main__":
    main()
