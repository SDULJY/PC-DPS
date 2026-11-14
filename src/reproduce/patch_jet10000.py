"""Complete `fields_pooled64_jet_re_10000.npy` from a 328MB tail.

The 127 downloaded 64 MiB chunks cover bytes [0, 8522825728).  The ZIP member
`data/fields_pooled64_jet_re_10000.npy` ends at byte 8850684802 (exclusive),
so only bytes [8522825728, 8850684801] are missing.  The user downloads that
range on a non-blocked network and saves it as data/jet10000_tail.bin.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CHUNKS = DATA / "chunks"
TAIL = DATA / "jet10000_tail.bin"
OUT = DATA / "fields_pooled64_jet_re_10000.npy"
META = ROOT / "results" / "p0" / "jet10000_missing_range.json"
CHUNK = 67108864
N_CHUNKS = 127
AVAILABLE = N_CHUNKS * CHUNK


class Reader:
    """Virtual stream over chunks 0..126 followed by the downloaded tail."""

    def __init__(self):
        self.handles = [open(CHUNKS / f"chunk_{i:05d}", "rb") for i in range(N_CHUNKS)]
        self.tail = open(TAIL, "rb")
        self.tail_size = TAIL.stat().st_size

    def close(self):
        for h in self.handles:
            h.close()
        self.tail.close()

    def read_at(self, off: int, n: int) -> bytes:
        out = bytearray()
        while n > 0:
            if off < AVAILABLE:
                ci, co = divmod(off, CHUNK)
                self.handles[ci].seek(co)
                b = self.handles[ci].read(min(n, CHUNK - co, AVAILABLE - off))
            else:
                to = off - AVAILABLE
                if to >= self.tail_size:
                    break
                self.tail.seek(to)
                b = self.tail.read(min(n, self.tail_size - to))
            if not b:
                break
            out += b
            off += len(b)
            n -= len(b)
        return bytes(out)

    def iter_range(self, off: int, n: int, block: int = 1 << 20):
        while n > 0:
            b = self.read_at(off, min(block, n))
            if not b:
                return
            yield b
            off += len(b)
            n -= len(b)


def main():
    if not TAIL.exists():
        raise SystemExit(f"missing {TAIL}; upload the 328MB tail first")
    meta = json.loads(META.read_text())
    if TAIL.stat().st_size != meta["missing_bytes"]:
        raise SystemExit(f"tail size {TAIL.stat().st_size} != {meta['missing_bytes']}")
    if OUT.exists():
        try:
            x = np.load(OUT, mmap_mode="r")
            if x.shape == (500, 196, 1, 64, 64):
                print(f"{OUT} already complete; nothing to do")
                return
        except Exception:
            pass

    data_off = meta["data_offset"]
    csize = meta["compressed_size"]
    usize = meta["uncompressed_size"]
    crc_expected = int(meta["crc32"], 16)
    tmp = OUT.with_suffix(".npy.part")

    r = Reader()
    try:
        print(f"decompressing {csize/1e6:.0f} MB -> {usize/1e9:.3f} GB")
        dec = zlib.decompressobj(-15)
        crc = 0
        written = 0
        with open(tmp, "wb") as f:
            for block in r.iter_range(data_off, csize):
                data = dec.decompress(block)
                if data:
                    f.write(data)
                    crc = zlib.crc32(data, crc)
                    written += len(data)
            tail = dec.flush()
            if tail:
                f.write(tail)
                crc = zlib.crc32(tail, crc)
                written += len(tail)
    finally:
        r.close()

    if written != usize or crc != crc_expected:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"CRC/size mismatch: {written}/{crc:#x} vs {usize}/{crc_expected:#x}")

    x = np.load(tmp, mmap_mode="r")
    if x.shape != (500, 196, 1, 64, 64):
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"unexpected shape {x.shape}")
    tmp.replace(OUT)
    print(f"OK {OUT} shape={x.shape} size={OUT.stat().st_size}")


if __name__ == "__main__":
    main()
