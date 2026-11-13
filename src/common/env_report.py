"""Record the environment used for the experiments (P0)."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "p0" / "env.json"


def git_hash(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "na"


def main():
    usage = shutil.disk_usage(ROOT)
    out = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "disk_total_gb": usage.total / 1e9,
        "disk_free_gb": usage.free / 1e9,
        "git": {
            "qg_beta_sr": git_hash(ROOT / "third_party" / "qg-beta-sr"),
            "qg_2d": git_hash(ROOT / "third_party" / "qg-2d"),
        },
        "third_party_status": {
            "qg_beta_sr": subprocess.run(
                ["git", "-C", str(ROOT / "third_party" / "qg-beta-sr"),
                 "status", "--porcelain"], capture_output=True, text=True).stdout.strip(),
            "qg_2d": subprocess.run(
                ["git", "-C", str(ROOT / "third_party" / "qg-2d"),
                 "status", "--porcelain"], capture_output=True, text=True).stdout.strip(),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
