"""Shared paths and paper hyperparameters.

All values below are taken from the baseline repository and the JAMES 2025
paper (arXiv:2507.00719).  They are deliberately centralized so every run log
records the same configuration.
"""
from pathlib import Path

WORK = Path(".")
THIRD = WORK / "third_party"
QG_SR = THIRD / "qg-beta-sr"
QG2D = THIRD / "qg-2d"
DATA = WORK / "data"
WEIGHTS = QG_SR / "Weights"
RESULTS = WORK / "results"
FIGURES = WORK / "figures"

# ---------------------------------------------------------------------------
# Paper / repository constants
# ---------------------------------------------------------------------------
FIELD_STD = 1.3          # fields are divided by this value (Dataloaders/dataloader.py)
N_DIFF = 64              # diffusion pseudo-time steps
N_CORR = 2               # Langevin corrections per step
CORR_SCALE = 0.3         # Langevin correction scale
SDEDIT_TIME = 16         # t_i = 0.25 -> 16/64
DPS_SCALE = 4            # 64 / 16 = 4 (ds16 observations)
DPS_C = 2e-5             # DPS guidance strength (paper Sect. 5.3.2)
DPS_SIGMA = 1e-2         # measurement noise
CFG_WEIGHT = 1.25        # classifier-free guidance weight (paper Sect. 5.3.4)

# Test protocol (Config/Run00041_Gen0000*.py and paper Sect. 4.4)
# The driver indexes the full per-run array at min_t(101) + snapshot(81) = 182.
# The pooled npy keeps the full 0..195 time axis, so the pooled index is 182.
TEST_SNAPSHOT = 182
TEST_RUN_INDEX = -6      # 500 runs start at 2727; run 3221 is index 494 = -6

SEEDS = [0, 1, 2]
REGIMES = ["eddy", "jet"]
REYNOLDS = [1000, 10000]
OBS_TYPES = ["full", "partial"]

# Paper Table 3 case numbering
PAPER_CASES = [
    (1, "jet", 1000, "full"),
    (2, "jet", 10000, "full"),
    (3, "eddy", 1000, "full"),
    (4, "eddy", 10000, "full"),
    (5, "jet", 1000, "partial"),
    (6, "jet", 10000, "partial"),
    (7, "eddy", 1000, "partial"),
    (8, "eddy", 10000, "partial"),
]


def case_id(regime: str, re: int, obs: str) -> str:
    return f"{regime}_re{re}_{obs}"


def data_filename(res: int, obs: str, regime: str, re: int) -> str:
    """Released pooled filename (confirmed from the Zenodo zip central directory).

    - 64x64 truth:   fields_pooled64_{regime}_re_{re}.npy
    - ds16 full:     fields_pooled64_ds16_{regime}_re_{re}.npy
    - ds16 partial:  fields_pooled64_ds16_partial_{regime}_re_{re}.npy
    """
    if res == 64:
        return f"fields_pooled64_{regime}_re_{re}.npy"
    if obs == "full":
        return f"fields_pooled64_ds{res}_{regime}_re_{re}.npy"
    return f"fields_pooled64_ds{res}_partial_{regime}_re_{re}.npy"
