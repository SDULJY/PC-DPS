#!/usr/bin/env bash
# Low-concurrency resume for the remaining Zenodo chunks.
# Zenodo returned 403 "unusual traffic" after the 32-way download; wait between
# attempts and use only a few connections.  Run tomorrow:
#   cd .
#   bash src/common/download_missing.sh
# For a direct connection instead of the proxy:
#   PROXY= bash src/common/download_missing.sh
set -uo pipefail

WORK=.
URL="https://zenodo.org/api/records/15742146/files/data.zip/content"
TOTAL=9401260503
CHUNK=67108864
N=$(( (TOTAL + CHUNK - 1) / CHUNK ))
WORKERS="${WORKERS:-4}"
PROXY="${PROXY:-}"
DIR="$WORK/data/chunks"
OUT="$WORK/data/data.zip"
EXPECTED_MD5="0d4b3282cf8081aecb63efc0845a8405"
mkdir -p "$DIR" "$WORK/results/p0"

curl_cmd() {
  if [ -n "$PROXY" ]; then
    echo "curl -x $PROXY"
  else
    echo "curl"
  fi
}

dl_one() {
  local i="$1"
  local start=$(( i * CHUNK ))
  local end=$(( start + CHUNK - 1 ))
  [ "$end" -ge "$TOTAL" ] && end=$(( TOTAL - 1 ))
  local want=$(( end - start + 1 ))
  local out
  out=$(printf "%s/chunk_%05d" "$DIR" "$i")
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" -eq "$want" ]; then
    return 0
  fi
  rm -f "$out" "$out.part"
  local attempt
  for attempt in $(seq 1 8); do
    local px=()
    [ -n "$PROXY" ] && px=(-x "$PROXY")
    if curl "${px[@]}" -sSL --fail --retry 3 --retry-delay 15 --max-time 3600 \
         --max-filesize $((want + 1024)) \
         --range "${start}-${end}" -o "$out.part" "$URL"; then
      if [ -f "$out.part" ] && [ "$(stat -c%s "$out.part")" -eq "$want" ]; then
        mv "$out.part" "$out"
        return 0
      fi
    fi
    rm -f "$out.part"
    echo "[$(date)] chunk $i attempt $attempt failed; sleeping 300s (possible 403 rate limit)"
    sleep 300
  done
  echo "FAILED chunk $i" >&2
  return 1
}

export -f dl_one
export URL TOTAL CHUNK DIR PROXY

echo "[$(date)] resuming missing chunks with $WORKERS workers; proxy='$PROXY'"
seq 0 $((N-1)) | xargs -P "$WORKERS" -I{} bash -c 'dl_one {}'
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "Some chunks failed; not concatenating. Re-run tomorrow." >&2
  exit 1
fi

missing=0
for i in $(seq 0 $((N-1))); do
  start=$(( i * CHUNK )); end=$(( start + CHUNK - 1 ))
  [ "$end" -ge "$TOTAL" ] && end=$(( TOTAL - 1 ))
  want=$(( end - start + 1 ))
  f=$(printf "%s/chunk_%05d" "$DIR" "$i")
  if [ ! -f "$f" ] || [ "$(stat -c%s "$f")" -ne "$want" ]; then
    echo "missing/short chunk $i" >&2; missing=$((missing+1))
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "ERROR: $missing chunks bad" >&2
  exit 1
fi

echo "[$(date)] concatenating"
cat "$DIR"/chunk_* > "$OUT"
actual=$(md5sum "$OUT" | awk '{print $1}')
echo "md5=$actual expected=$EXPECTED_MD5"
if [ "$actual" != "$EXPECTED_MD5" ]; then
  echo "ERROR: md5 mismatch" >&2
  exit 1
fi
echo "[$(date)] download complete and verified"
