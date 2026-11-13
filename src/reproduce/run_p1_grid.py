"""Launch the P1 grid (8 cases x 4 methods x seeds) across the available GPUs."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from reproduce.case_matrix import cases, METHODS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--cases", default=None, help="comma-separated case ids")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    methods = args.methods.split(",")
    wanted = set(args.cases.split(",")) if args.cases else None
    jobs = []
    for c in cases():
        if wanted and c["case"] not in wanted:
            continue
        for method in methods:
            for seed in seeds:
                out = ROOT / "results" / "p1" / c["case"] / method / f"seed{seed}"
                if (out / "log.json").exists() and not args.force:
                    print(f"skip existing {out}")
                    continue
                jobs.append((c["case"], method, seed, out))

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    print(f"{len(jobs)} jobs on GPUs {gpus}")

    def run(job):
        case, method, seed, out = job
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpus[(seed + METHODS.index(method)) % len(gpus)]
        cmd = [str(ROOT / "venv" / "bin" / "python"),
               str(ROOT / "src" / "reproduce" / "run_inference.py"),
               "--case", case, "--method", method, "--seed", str(seed),
               "--n-samples", str(args.n_samples), "--out", str(out)]
        print("RUN", " ".join(cmd), "GPU", env["CUDA_VISIBLE_DEVICES"])
        p = subprocess.run(cmd, env=env, cwd=str(ROOT))
        return case, method, seed, p.returncode

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, len(gpus))) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for f in as_completed(futs):
            case, method, seed, rc = f.result()
            print("DONE", case, method, seed, "rc", rc)
            if rc != 0:
                failures.append((case, method, seed, rc))
    if failures:
        print("FAILURES:", failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
