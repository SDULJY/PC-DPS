#!/usr/bin/env bash
# P2: failure-mode analysis (after P1 partial-case runs exist).
set -euo pipefail
WORK="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$WORK/venv/bin/python"

for seed in 0 1 2; do
  "$PY" "$WORK/src/analysis/density_sweep.py" \
    --case eddy_re1000_full --densities 4,6,8,10 --n-samples 100 --seed "$seed" --dps-c 1e-4 --score-clip 0.999 \
    --out "$WORK/results/p2/density_sweep_seed${seed}.json"
done

"$PY" "$WORK/src/analysis/make_figs_p2.py" \
  --cases eddy_re1000_partial,eddy_re10000_partial --methods dps,cond
