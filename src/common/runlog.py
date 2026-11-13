"""Run logging: every formal run writes a JSON record next to its outputs."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch


def _git_hash(path: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return out
    except Exception:
        return "na"


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def write_log(out_dir: Path, payload: dict):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    payload.setdefault("command", " ".join(sys.argv))
    payload.setdefault("env", environment())
    payload.setdefault("git", {
        "qg_beta_sr": _git_hash(Path("third_party/qg-beta-sr")),
        "qg_2d": _git_hash(Path("third_party/qg-2d")),
        "ours": _git_hash(Path(".")),
    })
    path = out_dir / "log.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {path}")
    return path
