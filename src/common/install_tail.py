"""Install a browser/laptop-downloaded tail of the Zenodo zip.

If the last 14 chunks (127..140) cannot be fetched on the server, download only
the tail on a non-blocked network:

  curl -L --range 8522825728-9401260502 -o tail.bin \
    "https://zenodo.org/api/records/15742146/files/data.zip/content"

Expected size: 878434775 bytes.  Upload tail.bin to data/tail.bin, then run:
  python src/common/install_tail.py
It writes chunks 127..140, concatenates all 141 chunks, and verifies md5.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CHUNKS = DATA / "chunks"
TAIL = DATA / "tail.bin"
OUT = DATA / "data.zip"
TOTAL = 9401260503
CHUNK = 67108864
START = 127 * CHUNK
EXPECTED_TAIL = TOTAL - START
EXPECTED_MD5 = "0d4b3282cf8081aecb63efc0845a8405"


def main():
    if not TAIL.exists():
        raise SystemExit(f"missing {TAIL}; upload the 0.88GB tail first")
    size = TAIL.stat().st_size
    if size != EXPECTED_TAIL:
        raise SystemExit(f"tail size {size}, expected {EXPECTED_TAIL}")
    CHUNKS.mkdir(parents=True, exist_ok=True)
    with TAIL.open("rb") as f:
        for i in range(127, 141):
            start = i * CHUNK
            end = min(start + CHUNK, TOTAL)
            want = end - start
            out = CHUNKS / f"chunk_{i:05d}"
            f.seek(start - START)
            with out.open("wb") as g:
                shutil.copyfileobj(f, g, length=want)
            if out.stat().st_size != want:
                raise SystemExit(f"chunk {i} short: {out.stat().st_size} != {want}")
            print(f"wrote {out} ({want} bytes)")
    # verify all chunks before concatenation
    for i in range(141):
        start = i * CHUNK
        end = min(start + CHUNK, TOTAL)
        want = end - start
        p = CHUNKS / f"chunk_{i:05d}"
        if not p.exists() or p.stat().st_size != want:
            raise SystemExit(f"bad/missing chunk {i}")
    with OUT.open("wb") as g:
        for i in range(141):
            with (CHUNKS / f"chunk_{i:05d}").open("rb") as f:
                shutil.copyfileobj(f, g, length=1 << 24)
    h = hashlib.md5()
    with OUT.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    print("md5", h.hexdigest())
    if h.hexdigest() != EXPECTED_MD5:
        raise SystemExit("md5 mismatch")
    print("OK: data/data.zip complete and verified")


if __name__ == "__main__":
    main()
