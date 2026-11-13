"""P0 data sanity check: shape, dtype, normalization, and a vorticity frame plot."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DATA, FIGURES, RESULTS, FIELD_STD, TEST_SNAPSHOT


def stats_for_file(path: Path, snapshot: int = TEST_SNAPSHOT):
    x = np.load(path, mmap_mode="r")
    info = {
        "file": path.name,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "size_gb": round(path.stat().st_size / 1e9, 3),
    }
    # npy layout is (batch, time, channel, H, W) per README
    if x.ndim != 5:
        info["error"] = f"unexpected ndim {x.ndim}"
        return info, x
    b, t, c, h, w = x.shape
    info.update(b=b, t=t, c=c, h=h, w=w)

    # Sample a subset for statistics (avoid reading the 1.6 GB file twice).
    bs = np.linspace(0, b - 1, min(20, b), dtype=int)
    ts = np.linspace(0, t - 1, min(20, t), dtype=int)
    sample = np.asarray(x[np.ix_(bs, ts, [0])]).squeeze()
    sample = sample.astype(np.float64)
    info["mean"] = float(sample.mean())
    info["std"] = float(sample.std())
    info["min"] = float(sample.min())
    info["max"] = float(sample.max())
    info["zero_fraction"] = float((sample == 0).mean())
    info["looks_normalized"] = bool(abs(info["mean"]) < 1e-2 and abs(info["std"] - 1.0) < 0.1)
    info["looks_like_field_std_1_3"] = bool(abs(info["std"] - FIELD_STD) < 0.1)

    # A representative frame and its zero mask (for partial observations).
    frame = np.asarray(x[b - 1, snapshot, 0]).astype(np.float32)
    info["test_frame"] = {
        "batch_index": b - 1,
        "time_index": snapshot,
        "mean": float(frame.mean()),
        "std": float(frame.std()),
        "zero_fraction": float((frame == 0).mean()),
    }
    return info, x


def plot_frames(entries, out_path: Path, snapshot: int = TEST_SNAPSHOT):
    n = len(entries)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for row, (info, x) in enumerate(entries):
        b, t, _, _, _ = info["shape"]
        idx = b - 1
        frame0 = np.asarray(x[idx, 0, 0])
        frame81 = np.asarray(x[idx, snapshot, 0])
        mask = (frame81 != 0).astype(float)
        vmax = float(np.percentile(np.abs(frame81), 99)) or 1.0
        for ax, arr, title, cmap in zip(
            axes[row],
            [frame0, frame81, mask],
            [f"{info['file']}: t=0", f"{info['file']}: t={snapshot}", "observed mask"],
            ["RdBu_r", "RdBu_r", "gray"],
        ):
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_title(title, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            if cmap == "gray":
                im.set_clim(0, 1)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", default=None,
                    help="npy file(s); default: all data/fields_pooled64_*.npy")
    ap.add_argument("--snapshot", type=int, default=TEST_SNAPSHOT)
    ap.add_argument("--out-json", default=str(RESULTS / "p0" / "data_stats.json"))
    ap.add_argument("--out-fig", default=str(FIGURES / "fig00_data_sample.png"))
    args = ap.parse_args()

    files = [Path(f) for f in args.file] if args.file else sorted(DATA.glob("fields_pooled64_*.npy"))
    if not files:
        raise SystemExit("no input files; pass --file or download the dataset first")
    entries, all_info = [], []
    for f in files:
        print(f"=== {f} ===")
        info, x = stats_for_file(f, args.snapshot)
        print(json.dumps(info, indent=2))
        all_info.append(info)
        entries.append((info, x))
        if info["shape"][0] != 500 or info["shape"][1] < TEST_SNAPSHOT + 1:
            print(f"WARNING: unexpected shape {info['shape']}")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(all_info, indent=2))
    print(f"wrote {out_json}")
    plot_frames(entries, Path(args.out_fig), args.snapshot)


if __name__ == "__main__":
    main()
