#!/usr/bin/env bash
# Identify the test run/snapshot used by the paper (config says run 3221,
# min_t=101 + snapshot=81 -> pooled index 182 for a 0..195 time axis).
set -euo pipefail
WORK=.
PY="$WORK/venv/bin/python"
OUT="$WORK/results/p0/calibration"
mkdir -p "$OUT"

for run in -1 -7; do
  for snap in 81 182; do
    tag="run${run}_snap${snap}"
    echo "=== $tag ==="
    "$PY" "$WORK/src/reproduce/run_inference.py" \
      --case jet_re1000_full --method dps --seed 0 --n-samples 16 \
      --run-index "$run" --snapshot "$snap" --out "$OUT/$tag" 2>&1 | \
      grep -E 'rel_l2_mean|rel_l2_ensmean' || true
  done
done
