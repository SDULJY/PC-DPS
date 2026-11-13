"""Paper Table 3: eight test cases (4 regime/Re x 2 observation operators)."""
from __future__ import annotations

from common.config import PAPER_CASES, case_id, data_filename

METHODS = ["sdedit", "dps", "cond", "cfg"]
METHOD_LABEL = {"sdedit": "M1 SDEdit", "dps": "M2 DPS",
                "cond": "M3 conditional", "cfg": "M4 CFG"}


def cases() -> list[dict]:
    out = []
    for num, regime, re, obs in PAPER_CASES:
        out.append({
            "number": num,
            "case": case_id(regime, re, obs),
            "regime": regime,
            "re": re,
            "obs": obs,
            "truth_file": data_filename(64, "full", regime, re),
            "obs_file": data_filename(16, obs, regime, re),
        })
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(cases(), indent=2))
