#!/usr/bin/env bash
# Extract only the 12 files needed for P1-P4 (4 truth + 4 ds16 full + 4 ds16 partial).
set -euo pipefail

WORK=.
ZIP="$WORK/data/data.zip"
OUT="$WORK/data"
cd "$WORK"

FILES=(
  data/fields_pooled64_eddy_re_1000.npy
  data/fields_pooled64_eddy_re_10000.npy
  data/fields_pooled64_jet_re_1000.npy
  data/fields_pooled64_jet_re_10000.npy
  data/fields_pooled64_ds16_eddy_re_1000.npy
  data/fields_pooled64_ds16_eddy_re_10000.npy
  data/fields_pooled64_ds16_jet_re_1000.npy
  data/fields_pooled64_ds16_jet_re_10000.npy
  data/fields_pooled64_ds16_partial_eddy_re_1000.npy
  data/fields_pooled64_ds16_partial_eddy_re_10000.npy
  data/fields_pooled64_ds16_partial_jet_re_1000.npy
  data/fields_pooled64_ds16_partial_jet_re_10000.npy
)

echo "[$(date)] extracting ${#FILES[@]} members to $OUT"
unzip -n -j "$ZIP" "${FILES[@]}" -d "$OUT"

echo "[$(date)] extracted files:"
ls -lh "$OUT"/fields_pooled64_*.npy

echo "[$(date)] zip integrity test for extracted members:"
unzip -t "$ZIP" "${FILES[@]}"
