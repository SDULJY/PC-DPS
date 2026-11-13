"""Run P1 on the cases currently available from the partial download.

Protocol identified from the paper/config and calibration:
- test run 3221 = pooled index -6, snapshot 182.
- jet: normalize target and observation by the target-frame std (unit variance).
- eddy: fixed field_std=1.0 (the conditional checkpoints are stable there;
  the released data are raw vorticity, not pre-normalized).
- eddy full observations are generated from truth with the exact A_les operator
  because the ds16 full files are still in the undownloaded tail.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / "venv" / "bin" / "python"
METHODS = ["sdedit", "dps", "cond", "cfg"]

CASES = {
    "jet_re1000_full": {
        "args": ["--normalize-target"],
        "methods": METHODS,
    },
    "eddy_re1000_full": {
        "args": ["--field-std", "1.0", "--dps-c", "1e-4",
                 "--obs-file", str(ROOT / "data" / "generated" / "eddy1000_ds16_full.npy"),
                 "--obs-run-index", "-1", "--obs-snapshot", "0"],
        "methods": METHODS,
    },
    "eddy_re10000_full": {
        "args": ["--field-std", "1.0", "--dps-c", "2e-4",
                 "--obs-file", str(ROOT / "data" / "generated" / "eddy10000_ds16_full.npy"),
                 "--obs-run-index", "-1", "--obs-snapshot", "0"],
        # DPS diverges numerically for this checkpoint/frame; keep the record.
        "methods": ["sdedit", "cond", "cfg", "dps"],
    },
    "eddy_re1000_partial": {"args": ["--field-std", "1.0", "--dps-c", "1e-4"], "methods": METHODS},
    "eddy_re10000_partial": {"args": ["--field-std", "1.0"], "methods": METHODS},
    # The three cases below need the still-missing Zenodo chunks.
    "jet_re10000_full": {"args": ["--normalize-target"], "methods": METHODS},
    "jet_re1000_partial": {"args": ["--normalize-target", "--dps-c", "1e-3",
        "--obs-file", str(ROOT / "data" / "generated" / "jet1000_ds16_partial.npy"),
        "--obs-run-index", "-1", "--obs-snapshot", "0"], "methods": METHODS},
    "jet_re10000_partial": {"args": ["--normalize-target", "--dps-c", "1e-4"], "methods": METHODS},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    wanted = set(args.cases.split(",")) if args.cases else None

    jobs = []
    for case, cfg in CASES.items():
        if wanted and case not in wanted:
            continue
        for method in cfg["methods"]:
            for seed in seeds:
                out = ROOT / "results" / "p1" / case / method / f"seed{seed}"
                if (out / "log.json").exists() and not args.force:
                    print("skip", out)
                    continue
                jobs.append((case, method, seed, out, cfg["args"]))
    print(f"{len(jobs)} jobs")

    def run(job):
        case, method, seed, out, extra = job
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpus[(seed + METHODS.index(method)) % len(gpus)]
        cmd = [str(PY), str(ROOT / "src" / "reproduce" / "run_inference.py"),
               "--case", case, "--method", method, "--seed", str(seed),
               "--n-samples", str(args.n_samples), "--out", str(out)] + extra
        print("RUN", " ".join(cmd), "GPU", env["CUDA_VISIBLE_DEVICES"], flush=True)
        p = subprocess.run(cmd, env=env, cwd=str(ROOT))
        return case, method, seed, p.returncode

    fails = []
    with ThreadPoolExecutor(max_workers=max(1, len(gpus))) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for f in as_completed(futs):
            c, m, s, rc = f.result()
            print("DONE", c, m, s, rc)
            if rc != 0:
                fails.append((c, m, s, rc))
    if fails:
        print("FAILURES", fails)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
