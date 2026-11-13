#!/usr/bin/env bash
# Download the QG dataset from Zenodo using parallel HTTP range requests.
# The Zenodo content endpoint supports GET Range (206); HEAD does not.
set -euo pipefail

WORK=.
URL="https://zenodo.org/api/records/15742146/files/data.zip/content"
TOTAL=9401260503
CHUNK=67108864          # 64 MiB
N=$(( (TOTAL + CHUNK - 1) / CHUNK ))
PROXY="${PROXY:-}"
DIR="$WORK/data/chunks"
OUT="$WORK/data/data.zip"
mkdir -p "$DIR" "$WORK/results/p0"

dl_one() {
  local i="$1"
  local start=$(( i * CHUNK ))
  local end=$(( start + CHUNK - 1 ))
  if [ "$end" -ge "$TOTAL" ]; then end=$(( TOTAL - 1 )); fi
  local want=$(( end - start + 1 ))
  local out
  out=$(printf "%s/chunk_%05d" "$DIR" "$i")
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" -eq "$want" ]; then
    return 0
  fi
  rm -f "$out" "$out.part"
  local attempt
  for attempt in $(seq 1 8); do
    if curl -x "$PROXY" -sSL --fail --retry 3 --retry-delay 2 --max-time 1800 \
         --max-filesize $((want + 1024)) \
         --range "${start}-${end}" -o "$out.part" "$URL"; then
      if [ -f "$out.part" ] && [ "$(stat -c%s "$out.part")" -eq "$want" ]; then
        mv "$out.part" "$out"
        return 0
      fi
    fi
    rm -f "$out.part"
    sleep $(( attempt * 2 ))
  done
  echo "FAILED chunk $i (expected $want bytes)" >&2
  return 1
}

export -f dl_one
export URL TOTAL CHUNK DIR PROXY

echo "[$(date)] downloading $N chunks with 32 workers via $PROXY"
seq 0 $((N-1)) | xargs -P 32 -I{} bash -c 'dl_one {}'

# Verify every chunk size before concatenation.
missing=0
for i in $(seq 0 $((N-1))); do
  start=$(( i * CHUNK )); end=$(( start + CHUNK - 1 ))
  if [ "$end" -ge "$TOTAL" ]; then end=$(( TOTAL - 1 )); fi
  want=$(( end - start + 1 ))
  f=$(printf "%s/chunk_%05d" "$DIR" "$i")
  if [ ! -f "$f" ] || [ "$(stat -c%s "$f")" -ne "$want" ]; then
    echo "missing/short chunk $i" >&2; missing=$((missing+1))
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "ERROR: $missing chunks bad; not concatenating" >&2
  exit 1
fi

echo "[$(date)] concatenating chunks"
cat "$DIR"/chunk_* > "$OUT"
echo "[$(date)] md5:"
md5sum "$OUT"
actual=$(md5sum "$OUT" | awk '{print $1}')
expected="0d4b3282cf8081aecb63efc0845a8405"
if [ "$actual" != "$expected" ]; then
  echo "ERROR: md5 mismatch (expected $expected, got $actual)" >&2
  exit 1
fi
echo "[$(date)] md5 OK, size:"
stat -c '%s bytes' "$OUT"
