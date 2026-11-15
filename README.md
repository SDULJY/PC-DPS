# QG Turbulence Diffusion — Reproduction and Physics-Constrained DPS

Reproduction of

> **Guided Unconditional and Conditional Generative Models for Super-Resolution
> and Inference of Quasi-Geostrophic Turbulence**
> A. N. Suresh Babu, A. Sadam, P. F. J. Lermusiaux, JAMES 2025.
> [arXiv:2507.00719](https://arxiv.org/abs/2507.00719) ·
> [DOI 10.48550/arXiv.2507.00719](https://doi.org/10.48550/arXiv.2507.00719)

plus two extensions:

- **PC-DPS** — physics-constrained DPS: a differentiable QG residual used as a
  soft guidance term during diffusion sampling (`src/pcdps/`).
- **β inversion** — system identification of the beta-plane coefficient from a
  reconstructed field (`src/beta_invert/`).

The upstream diffusion code (`qg-beta-sr`) and QG simulator (`qg-2d`) are used
**unmodified**; they are cloned into `third_party/` by `setup_third_party.sh`.
All new code lives in `src/`.

This repository reports what actually ran. Some gates were not met; the
negative and marginal results are kept in the tables below and in
[`report/FINAL_RESULTS.md`](report/FINAL_RESULTS.md).

---

## Results at a glance

### P1 — reproduction of the paper (8 cases × 4 methods × 3 seeds)

Protocol: test run `3221` = pooled index **-6**, snapshot **182**, ensemble
size `N_ens=16` (the paper's Table 4/5 protocol). Numbers are the mean over
three seeds; a full 200-sample × 3-seed run gives the same conclusions
([`results/tab_reproduce_16.csv`](results/tab_reproduce_16.csv),
[`results/tab_reproduce.csv`](results/tab_reproduce.csv)).

Deviation from the paper value (lower is better):

| Case | SDEdit | DPS | cond | CFG | within 10% |
|---|---:|---:|---:|---:|---:|
| eddy Re=10⁴, full | 3.5% | 1.8% | 5.1% | 3.8% | **4/4** |
| eddy Re=10⁴, sparse | 0.3% | 27.0% | 9.4% | 8.8% | 3/4 |
| eddy Re=10³, full | 11.5% | 16.5% | 4.7% | 8.8% | 2/4 |
| eddy Re=10³, sparse | 3.6% | 12.4% | 64.6% | 64.8% | 1/4 |
| jet Re=10⁴, full | 19.8% | 14.9% | 12.9% | 11.2% | 0/4 |
| jet Re=10⁴, sparse | 0.2% | 68.8% | 16.6% | 15.1% | 1/4 |
| jet Re=10³, full | 3.0% | 11.8% | 35.8% | 8.0% | 2/4 |
| jet Re=10³, sparse | 0.7% | 4.8% | 19.1% | 22.6% | 2/4 |

**Summary: 15/32 case–method combinations are within 10% (46.9%). Only
`eddy Re=10⁴ full` passes with all four methods.** The strict 6/8-case gate is
not met. The largest anomaly is `eddy Re=10³ sparse` with cond/CFG ~65% high;
indexing, observation operator, and normalization were audited and excluded as
causes — the evidence points to the released partial checkpoint / test
trajectory, documented in
[`results/p1/checkpoint_audit.md`](results/p1/checkpoint_audit.md).

### P2 — failure modes of sparse reconstruction

- Observation density (eddy Re=10³ full, DPS): 4×4 → 10×10 grid gives
  relative L2 **2.546 → 1.538 → 1.223 → 0.985**, monotonic.
- Far-field error on `eddy Re=10⁴ sparse`: DPS **1.733**× the target
  variability vs **0.524** for the conditional model; the spectral tail ratio
  is 45.8 (DPS) vs 10.1 (cond). DPS does not propagate information into
  unobserved regions.

### P3 — PC-DPS and subgrid closure

Pair sampling, QG residual gradient, 100 pairs × 3 seeds:

| Case | λ=0 | best λ | best rel L2 | reduction | spectral tail |
|---|---:|---:|---:|---:|---:|
| eddy Re=10⁴ sparse | 0.4691 | 2.0 | 0.4329 | **−7.7%** | 44.2 → 42.5 |
| eddy Re=10³ sparse | 0.5009 | 1.0 | 0.4819 | −3.8% | 97.1 → 186.3 (worse) |
| jet Re=10³ full | 0.0533 | 0.1 | 0.0533 | ≈0 | no effect |

The 8% gate is **not met** (7.7% best). A three-frame centered residual,
per-sample scaling, λ(t), and residual low-pass variants all stayed at or
below this result. The 64×64 fields are filtered from 512×512 simulations, so
the QG residual is a soft constraint; an effective linear closure reduces the
jet residual by 33% but does not improve eddy Re=10⁴
([`results/p3/closure_findings.md`](results/p3/closure_findings.md)).

### P4 — β inversion (system identification)

| Data | β true | β̂ | error |
|---|---:|---:|---:|
| jet Re=10³ truth (oracle) | 2.5 | 2.7388 | 9.6% |
| jet Re=10³ PC-DPS reconstruction | 2.5 | **2.6565** | **6.3%** |
| eddy Re=10³ truth (oracle) | 0 | 0.0009 | — |
| eddy Re=10³ sparse PC-DPS | 0 | 0.0254 | — |

Binary model selection correctly identifies the jet case (held-out RMSE 0.925
vs 2.016). The <15% gate **passes**. ν is not identifiable at 64²; only its
order of magnitude is reported.

Figures: `figures/fig00` … `figures/fig15` (PNG + PDF), listed below.

---

## Repository layout

| Path | Contents |
|---|---|
| `src/common/` | config, metrics, run logging, data download, aggregation |
| `src/reproduce/` | P0 data checks, baseline inference, P1 evaluation and figures |
| `src/analysis/` | P2 density sweep and failure-mode analysis |
| `src/pcdps/` | differentiable QG residual, PC-DPS sampler, residual audit, closure fit |
| `src/beta_invert/` | P4 β estimation and model selection |
| `tests/` | spectral-operator, QG-residual and metric self-tests (8 tests) |
| `results/` | per-run `log.json` / metrics, CSV tables, protocol notes |
| `figures/` | figures 00–15, PNG + PDF |
| `report/` | final results, Chinese program notes, summary |
| `setup_third_party.sh` | clone the two upstream repositories at pinned commits |
| `third_party/` | upstream code and weights, **not tracked** (created by setup) |
| `data/` | dataset and generated observations, **not tracked** |
| `venv/`, `weights/` | local environment / optional local weights, **not tracked** |

Every number in this README is produced by scripts from stored logs; no value
was edited by hand. `report/FINAL_RESULTS.md` is the full result document.

---

## Installation

Requirements: Linux, Python ≥ 3.10, CUDA GPU (the sampler calls CUDA directly),
and about 30 GB of free disk space for the dataset.

```bash
git clone https://github.com/SDULJY/qg-diffusion-repro.git
cd qg-diffusion-repro

bash setup_third_party.sh          # qg-beta-sr + qg-2d at pinned commits (MIT)

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`setup_third_party.sh` also brings the 12 pretrained checkpoints that ship in
the upstream `qg-beta-sr/Weights/` directory (~167 MB).

The reference runs used 2× RTX 4090 (24 GB), CUDA 12.4, torch 2.5.1+cu124.
A single GPU is sufficient; pass `--gpus 0` to the grid drivers.

---

## Data and pretrained weights

The dataset is **not** included in this repository.

- Dataset: Zenodo record [15742146](https://doi.org/10.5281/zenodo.15742146)
  (CC BY 4.0), file `data.zip`, 9,401,260,503 bytes,
  MD5 `0d4b3282cf8081aecb63efc0845a8405`.
- Code and weights archive: Zenodo record
  [15750244](https://doi.org/10.5281/zenodo.15750244) (MIT).
- Paper: [arXiv:2507.00719](https://arxiv.org/abs/2507.00719).

Download and extract the 12 arrays used here:

```bash
# Option A: the parallel range downloader
bash src/common/download_data.sh

# Option B: download data.zip manually from Zenodo, then place it in data/
#   data/data.zip

bash src/reproduce/extract_data.sh
python src/reproduce/check_data.py --file data/fields_pooled64_jet_re_1000.npy
```

Expected shapes are `(500, 196, 1, 64, 64)`; each array is about 1.5 GB.

### Observation files

Three released `ds16` observation members were not available in the recovered
ZIP subset used for this work: the two `eddy` **full** observations and the
`jet Re=10³` **sparse** observation. They are regenerated from the truth with
the paper's exact `A_les_interleaved` operator (Gaussian + cutoff + average
pooling, scale 4, gappy mask). The same operator reproduces released `ds16`
files to ≈1.2×10⁻⁷.

```bash
mkdir -p data/generated

# each command uses the test frame run=-6, snapshot=182 by default
python src/reproduce/make_obs_from_truth.py \
  --case eddy_re1000_full  --out data/generated/eddy1000_ds16_full.npy
python src/reproduce/make_obs_from_truth.py \
  --case eddy_re10000_full --out data/generated/eddy10000_ds16_full.npy
python src/reproduce/make_obs_from_truth.py \
  --case jet_re1000_full --gappy --out data/generated/jet1000_ds16_partial.npy
```

The generated files store one frame at index `[0, 0, 0]`; the P1 driver selects
it with `--obs-run-index -1 --obs-snapshot 0`. A complete download of
`data.zip` contains the official `ds16` files, but the driver keeps using these
generated observations so that the stored results are exactly reproducible
from this repository.

---

## Running the experiments

All commands are run from the repository root with the virtual environment
activated.

### P0 — data sanity checks

```bash
python src/reproduce/check_data.py --file data/fields_pooled64_jet_re_1000.npy
python src/reproduce/audit_alignment.py       # A_les(truth) vs released obs
```

### P1 — baseline reproduction grid

```bash
# 8 cases x 4 methods (SDEdit, DPS, cond, CFG) x seeds 0,1,2
python src/reproduce/run_p1_available.py --n-samples 200 --seeds 0,1,2 --gpus 0,1

# aggregate to CSV and make figures
python src/reproduce/evaluate.py
python src/reproduce/evaluate_first16.py       # N_ens=16, paper protocol
python src/reproduce/make_figs_p1.py
```

Use `--cases eddy_re10000_full,jet_re1000_full` to run a subset. The driver
skips runs whose `log.json` already exists unless `--force` is given. A
16-sample run is enough to reproduce the paper table; the 200-sample runs are
the stronger check.

### P2 — failure modes

```bash
bash src/analysis/run_p2.sh
# density sweep (4x4, 6x6, 8x8, 10x10) + spectrum/distance figures
```

### P3 — PC-DPS and closure

```bash
pytest tests/ -q
python src/pcdps/check_residual.py --case jet_re1000_full --snapshot 182
python src/pcdps/residual_audit.py --cases eddy_re1000,eddy_re10000,jet_re1000
python src/pcdps/fit_closure.py --train-runs 0-399 --val-runs 400-479

# single run (pair sampling + gap-only physics + quantile clipping)
python src/pcdps/run_pcdps.py --case eddy_re10000_partial \
  --lambda-phys 2 --dps-c 1e-4 --physics-gap-only --grad-clip 0.999 \
  --n-pairs 100 --seed 0

# full lambda sweep over 3 seeds
python src/pcdps/sweep.py --n-pairs 100 --seeds 0,1,2 --gpus 0,1
python src/pcdps/evaluate_p3.py
python src/pcdps/make_figs_p3.py
```

`run_pcdps.py --triple` enables the three-frame centered residual variant.
`--per-sample-scale`, `--phys-t-min/max`, `--phys-lowpass` and
`--phys-late-boost` are the other ablations reported in the results.

### P4 — β inversion

```bash
python src/beta_invert/estimate_beta.py --case jet_re1000_full --source truth
python src/beta_invert/estimate_beta.py --case jet_re1000_full --source pcdps
python src/beta_invert/model_selection.py --case jet_re1000_full
python src/beta_invert/make_figs_p4.py
```

### Aggregation

```bash
python src/common/collect.py        # results/master.csv
python src/common/make_summary.py   # report/results_summary.md
```

---

## Figures

| Figure | Content |
|---|---|
| `fig00_data_sample` | sample fields from the dataset |
| `fig01_recon_panels` | reconstruction panels (truth / observations / four methods) |
| `fig02_rel_l2_vs_paper` | relative L2 vs the paper, 8 cases × 4 methods |
| `fig03_spectra` | reconstructed vs true energy spectra |
| `fig04_stats_spread` | vorticity statistics and ensemble spread–skill |
| `fig05_dps_spectrum_smoothing` | DPS spectrum smoothing in sparse cases |
| `fig06_error_vs_distance` | error vs distance from observations |
| `fig07_error_vs_density` | error vs observation density |
| `fig08_pcdps_panels` | PC-DPS reconstruction panels |
| `fig09_pcdps_spectra` | PC-DPS spectra |
| `fig10_lambda_sweep` | λ sweep for PC-DPS |
| `fig11_beta_estimate` | β estimates |
| `fig12_beta_convergence` | β convergence / selection |
| `fig13_residual_spectrum` | subgrid QG residual spectrum |
| `fig15_pcdps_diagnosis` | PC-DPS error by scale and by region |

All figures are generated by the scripts above from `results/`; no plotting
value is typed in by hand.

---

## Protocol notes

- **Test protocol**: pooled index `-6` (run 3221), snapshot `182`
  (`min_t=101` + `snapshot=81`), `N_ens=16` for the paper comparison and 200
  samples for the seed study.
- **Normalization**: released fields are raw vorticity, not pre-normalized.
  Jet cases normalize target and observation by the target-frame std; eddy
  cases use `field_std=1.0`. Target and observation must share the same scalar.
- **Diffusion**: `N_diff=64`, `N_corr=2`, Langevin scale 0.3, SDEdit
  `t=16/64`; DPS `scale=4`, `sigma=1e-2`; CFG weight 1.25.
- **DPS guidance `C`** is per case (e.g. jet full `2e-5`, eddy Re=10³ full
  `1e-4`, eddy Re=10⁴ full `2e-4`, sparse eddy `1e-4`); these values are in
  `src/reproduce/run_p1_available.py` and in each run's `log.json`.
- **QG sign convention**: the released jet data is consistent with the
  `−β ∂ψ/∂x` term (`beta_sign=-1`); eddy cases have β=0 and are unaffected.
- **Evaluation** uses physical (unnormalized) fields; metrics are relative L2,
  log-spectrum error, spectral tail ratio, and spread–skill.

---

## Known limitations

1. **P1 is not a full pass.** 15/32 combinations are within 10%; the strict
   6/8-case gate is not met. `eddy Re=10³ sparse` cond/CFG are ~65% high after
   excluding indexing, mask, operator, and normalization bugs. The released
   partial checkpoint is the most likely cause.
2. **PC-DPS is marginal.** Best reduction is 7.7%, below the 8% target;
   eddy Re=10³ improves 3.8% but worsens the spectral tail; jet full is
   unaffected.
3. **The QG residual is a soft constraint.** The 64×64 fields are filtered
   from 512×512 simulations, so subgrid terms remain. A linear effective
   closure helps the jet case (residual −33%) but not eddy Re=10⁴.
4. **ν is not identifiable** at 64²; only β is reported quantitatively.

These limitations are part of the result, not something hidden by tuning on
the test run. Hyperparameters were selected on validation/training runs; the
test run `-6` was never used for tuning.

---

## License, attribution, and citation

- **This repository's original code**: MIT — see [`LICENSE`](LICENSE).
- **Upstream code**: `qg-beta-sr` and `qg-2d`, both MIT, used unmodified at
  pinned commits; see [`NOTICE`](NOTICE) and the `LICENSE` files created under
  `third_party/` by `setup_third_party.sh`.
- **Dataset**: CC BY 4.0, Zenodo
  [10.5281/zenodo.15742146](https://doi.org/10.5281/zenodo.15742146). The data
  is not redistributed here.
- **Weights**: MIT, Zenodo
  [10.5281/zenodo.15750244](https://doi.org/10.5281/zenodo.15750244), shipped
  with the upstream repository.

If you use this code, please cite the paper and this repository:

```bibtex
@article{sureshbabu2025guided,
  title   = {Guided Unconditional and Conditional Generative Models for
             Super-Resolution and Inference of Quasi-Geostrophic Turbulence},
  author  = {Suresh Babu, A. N. and Sadam, A. and Lermusiaux, P. F. J.},
  journal = {Journal of Advances in Modeling Earth Systems},
  year    = {2025},
  doi     = {10.48550/arXiv.2507.00719}
}

@software{sduljy_qg_diffusion_repro,
  title  = {QG Turbulence Diffusion: Reproduction and Physics-Constrained DPS},
  author = {SDULJY},
  year   = {2026},
  url    = {https://github.com/SDULJY/qg-diffusion-repro}
}
```

A Chinese version of this document is available at
[`README.zh-CN.md`](README.zh-CN.md). The full result report (Chinese) is
[`report/FINAL_RESULTS.md`](report/FINAL_RESULTS.md).
