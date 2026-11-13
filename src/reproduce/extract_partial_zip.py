"""Extract complete members from the partially downloaded Zenodo zip.

The download is split into 64 MiB chunks.  The zip local file headers contain
the compressed sizes, so members whose compressed data lies entirely inside the
already-downloaded prefix can be decompressed without the central directory.
This lets P0/P1 start while the last 14 chunks are blocked by Zenodo 403.
"""
from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 67108864
TOTAL = 9401260503
DATA = ROOT / "data"
CHUNKS = DATA / "chunks"


class ChunkReader:
    def __init__(self, n_chunks: int):
        self.paths = [CHUNKS / f"chunk_{i:05d}" for i in range(n_chunks)]
        self.handles = [open(p, "rb") for p in self.paths]
        self.n_chunks = n_chunks
        self.available = n_chunks * CHUNK

    def close(self):
        for h in self.handles:
            h.close()

    def read_at(self, off: int, n: int) -> bytes:
        out = bytearray()
        while n > 0:
            ci, co = divmod(off, CHUNK)
            if ci >= self.n_chunks:
                break
            self.handles[ci].seek(co)
            b = self.handles[ci].read(min(n, CHUNK - co))
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


def parse_members(r: ChunkReader):
    members = {}
    off = 0
    while off + 30 <= r.available:
        h = r.read_at(off, 30)
        if len(h) < 30:
            break
        sig, ver, flags, method, mtime, mdate, crc, csize, usize, fnlen, extralen = \
            struct.unpack("<IHHHHHIIIHH", h)
        if sig != 0x04034B50:
            break
        name = r.read_at(off + 30, fnlen).decode("utf-8", "replace")
        extra = r.read_at(off + 30 + fnlen, extralen)
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF:
            p = 0
            while p + 4 <= len(extra):
                eid, esz = struct.unpack("<HH", extra[p:p + 4])
                p += 4
                if eid == 0x0001:
                    vals = extra[p:p + esz]
                    q = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", vals[q:q + 8])[0]; q += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", vals[q:q + 8])[0]; q += 8
                p += esz
        data_off = off + 30 + fnlen + extralen
        end = data_off + csize
        if end > r.available:
            break
        members[name] = dict(data_off=data_off, csize=csize, usize=usize,
                             method=method, crc=crc)
        off = end
    return members


def extract(name: str, info: dict, r: ChunkReader):
    out = DATA / Path(name).name
    if out.exists() and out.stat().st_size == info["usize"]:
        print(f"skip existing {out}")
        return
    tmp = out.with_suffix(out.suffix + ".part")
    print(f"extract {name}: {info['csize']/1e6:.0f} MB -> {info['usize']/1e9:.3f} GB")
    crc = 0
    written = 0
    if info["method"] == 8:
        dec = zlib.decompressobj(-15)
        with open(tmp, "wb") as f:
            for block in r.iter_range(info["data_off"], info["csize"]):
                data = dec.decompress(block)
                if data:
                    f.write(data); crc = zlib.crc32(data, crc); written += len(data)
            tail = dec.flush()
            if tail:
                f.write(tail); crc = zlib.crc32(tail, crc); written += len(tail)
    elif info["method"] == 0:
        with open(tmp, "wb") as f:
            for block in r.iter_range(info["data_off"], info["csize"]):
                f.write(block); crc = zlib.crc32(block, crc); written += len(block)
    else:
        raise ValueError(f"unsupported compression method {info['method']}")
    if written != info["usize"] or crc != info["crc"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"CRC/size mismatch for {name}: got {written}/{crc:#x}, "
                           f"want {info['usize']}/{info['crc']:#x}")
    tmp.replace(out)
    print(f"  OK {out} ({out.stat().st_size/1e9:.3f} GB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chunks", type=int, default=None)
    ap.add_argument("--names", nargs="*", default=None)
    args = ap.parse_args()
    n = args.n_chunks or len(list(CHUNKS.glob("chunk_*")))
    r = ChunkReader(n)
    try:
        members = parse_members(r)
        print(f"parsed {len(members)} complete members in {r.available/1e9:.3f} GB")
        targets = args.names or [
            "data/fields_pooled64_eddy_re_1000.npy",
            "data/fields_pooled64_eddy_re_10000.npy",
            "data/fields_pooled64_jet_re_1000.npy",
            "data/fields_pooled64_ds16_jet_re_1000.npy",
            "data/fields_pooled64_ds16_jet_re_10000.npy",
            "data/fields_pooled64_ds16_partial_eddy_re_1000.npy",
            "data/fields_pooled64_ds16_partial_eddy_re_10000.npy",
            "data/fields_pooled64_ds16_partial_jet_re_10000.npy",
        ]
        for name in targets:
            if name not in members:
                print(f"MISSING (still in undownloaded tail): {name}")
                continue
            extract(name, members[name], r)
    finally:
        r.close()


if __name__ == "__main__":
    main()
