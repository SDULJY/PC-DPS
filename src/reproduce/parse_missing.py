"""Print the exact byte range needed to complete one ZIP member.

The local ZIP headers are still readable in the 127 downloaded 64 MiB chunks.
For `fields_pooled64_jet_re_10000.npy` we only need the compressed tail that
lies beyond the downloaded prefix; the central directory is not needed.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / "data" / "chunks"
OUT = ROOT / "results" / "p0" / "jet10000_missing_range.json"
TARGET = "fields_pooled64_jet_re_10000.npy"
CHUNK = 67108864
N = 127
AVAILABLE = N * CHUNK


def main():
    paths = [CHUNKS / f"chunk_{i:05d}" for i in range(N)]
    handles = [open(p, "rb") for p in paths]

    def read_at(off, n):
        out = bytearray()
        while n > 0:
            ci, co = divmod(off, CHUNK)
            if ci >= N:
                break
            handles[ci].seek(co)
            b = handles[ci].read(min(n, CHUNK - co))
            if not b:
                break
            out += b
            off += len(b)
            n -= len(b)
        return bytes(out)

    try:
        off = 0
        while off + 30 <= AVAILABLE:
            h = read_at(off, 30)
            if len(h) < 30:
                break
            (sig, ver, flags, method, mt, md, crc, csize, usize,
             fnlen, extralen) = struct.unpack("<IHHHHHIIIHH", h)
            if sig != 0x04034B50:
                break
            name = read_at(off + 30, fnlen).decode("utf-8", "replace")
            extra = read_at(off + 30 + fnlen, extralen)
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
            member_end = data_off + csize          # exclusive
            if TARGET in name:
                missing_start = AVAILABLE
                missing_end = member_end - 1        # inclusive
                rec = {
                    "name": name,
                    "local_header_offset": off,
                    "data_offset": data_off,
                    "compressed_size": csize,
                    "uncompressed_size": usize,
                    "crc32": f"{crc:08x}",
                    "compression_method": method,
                    "member_end_exclusive": member_end,
                    "available_prefix": AVAILABLE,
                    "missing_start": missing_start,
                    "missing_end_inclusive": missing_end,
                    "missing_bytes": member_end - AVAILABLE,
                    "curl_range": f"{missing_start}-{missing_end}",
                }
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(rec, indent=2))
                print(json.dumps(rec, indent=2))
                return
            if member_end > AVAILABLE:
                break
            off = member_end
        raise SystemExit(f"target {TARGET} not found in local headers")
    finally:
        for h in handles:
            h.close()


if __name__ == "__main__":
    main()
