"""Run the lambda_phys sweep (and seeds) for the selected P3 cases."""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="eddy_re1000_partial,eddy_re10000_partial")
    ap.add_argument("--lambdas", default="0,0.01,0.1,1,10")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-pairs", type=int, default=50)
    ap.add_argument("--field-std", type=float, default=1.0)
    ap.add_argument("--run-index", type=int, default=-6)
    ap.add_argument("--snapshot", type=int, default=182)
    ap.add_argument("--residual-kmax", type=float, default=None)
    ap.add_argument("--dps-c", type=float, default=1e-4)
    ap.add_argument("--physics-gap-only", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cases = args.cases.split(",")
    lambdas = [float(x) for x in args.lambdas.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    jobs = []
    for ci, case in enumerate(cases):
        for li, lam in enumerate(lambdas):
            for si, seed in enumerate(seeds):
                out = ROOT / "results" / "p3" / case / f"lambda{lam:g}" / f"seed{seed}"
                if (out / "log.json").exists() and not args.force:
                    print("skip", out)
                    continue
                jobs.append((case, lam, seed, out, (ci + li + si) % len(gpus)))
    print(f"{len(jobs)} jobs")

    def run(job):
        case, lam, seed, out, gi = job
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpus[gi]
        cmd = [str(ROOT / "venv" / "bin" / "python"),
               str(ROOT / "src" / "pcdps" / "run_pcdps.py"),
               "--case", case, "--lambda-phys", str(lam),
               "--seed", str(seed), "--n-pairs", str(args.n_pairs),
               "--field-std", str(args.field_std),
               "--run-index", str(args.run_index),
               "--snapshot", str(args.snapshot),
               "--dps-c", str(args.dps_c),
               "--out", str(out)]
        if args.residual_kmax is not None:
            cmd += ["--residual-kmax", str(args.residual_kmax)]
        if args.physics_gap_only:
            cmd += ["--physics-gap-only"]
        if args.grad_clip is not None:
            cmd += ["--grad-clip", str(args.grad_clip)]
        print("RUN", " ".join(cmd), "GPU", gpus[gi], flush=True)
        p = subprocess.run(cmd, env=env, cwd=str(ROOT))
        return case, lam, seed, p.returncode

    fails = []
    with ThreadPoolExecutor(max_workers=max(1, len(gpus))) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for f in as_completed(futs):
            c, l, s, rc = f.result()
            print("DONE", c, l, s, rc)
            if rc != 0:
                fails.append((c, l, s, rc))
    if fails:
        print("FAILURES", fails)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
