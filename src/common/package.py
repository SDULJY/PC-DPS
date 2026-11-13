"""Package code, figures, tables and logs for submission (excludes data/weights)."""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "题目(XX).zip"

INCLUDE_DIRS = ["src", "tests", "figures", "results", "report"]
INCLUDE_FILES = ["README.md", "requirements.txt", "STATUS.md"]
EXCLUDE_SUFFIX = {".npz", ".pt", ".pth", ".npy", ".zip"}
EXCLUDE_PARTS = {"data", "weights", "venv", "third_party", "__pycache__", ".git"}


def ok(p: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in p.parts):
        return False
    if p.suffix in EXCLUDE_SUFFIX:
        return False
    if p.name in {"paper.txt"} or p.name.startswith("paper_") and p.suffix == ".pdf":
        return False
    return p.is_file()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in INCLUDE_FILES:
            p = ROOT / name
            if p.exists():
                z.write(p, p.relative_to(ROOT))
        for d in INCLUDE_DIRS:
            base = ROOT / d
            if not base.exists():
                continue
            for p in sorted(base.rglob("*")):
                if ok(p):
                    z.write(p, p.relative_to(ROOT))
        # A placeholder the user replaces with the final docx.
        z.writestr("report/请放入最终docx.txt",
                   "请将最终 docx/pdf 放入本压缩包，并把文件名改为 题目(姓名学号)。\n")
    size = OUT.stat().st_size / 1e6
    print(f"wrote {OUT} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
