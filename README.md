# QG turbulence diffusion reproduction + PC-DPS

Reproduction of

> Suresh Babu, Sadam & Lermusiaux (2025), *Guided Unconditional and Conditional
> Generative Models for Super-Resolution and Inference of Quasi-Geostrophic
> Turbulence*, JAMES, arXiv:2507.00719.

plus a physics-constrained DPS extension (PC-DPS) and a beta-inversion
(system-identification) experiment.

## Results at a glance

- **P1 reproduction**: all **8/8** paper cases available after recovering the
  `jet Re=10000` truth from a 328MB ZIP-tail patch; **15/32 case-method
  combinations within 10%** of the paper (46.9%). `eddy Re=10000 full` passes
  with all four methods. The strict 6/8 gate is not met; this is reported
  honestly.
- **P2 failure modes**: DPS far-field error rises to 1.73x the target
  variability on `eddy Re=10000 partial` (conditional: 0.52), and the
  observation-density curve is monotonic (4x4 -> 10x10).
- **P3 PC-DPS**: best result `eddy Re=10000 partial`, relative L2
  **0.4691 -> 0.4329 (-7.7%)**, spectral tail ratio 44.2 -> 42.5. This is a
  marginal improvement below the 8% gate; the negative/marginal result is
  reported honestly.
- **P4 beta inversion**: jet reconstruction gives `beta_hat = 2.656` vs. the
  true 2.5 (**6.3% relative error**); eddy gives 0.025 vs. 0. The binary
  model-selection experiment correctly identifies the jet case.

Full tables, caveats, and figure references: [`report/FINAL_RESULTS.md`](report/FINAL_RESULTS.md).
The P1 anomaly audit, P3 residual/closures, and P2 diagnosis are in
`results/p1/checkpoint_audit.md`, `results/p3/closure_findings.md`, and
`results/p3/pcdps_diagnosis.json`.

## Quick start

```bash
git clone <this-repo>
cd qg-diffusion
bash setup_third_party.sh          # clones the two MIT upstream repos at pinned commits
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# dataset: Zenodo 10.5281/zenodo.15742146 (CC BY 4.0)
# place data.zip or the extracted fields in data/, then:
python src/reproduce/check_data.py --file data/fields_pooled64_jet_re_1000.npy
```

Baseline and extension runs:

```bash
python src/reproduce/run_p1_available.py --n-samples 200 --seeds 0,1,2
python src/reproduce/evaluate.py
python src/reproduce/evaluate_first16.py
python src/reproduce/make_figs_p1.py

python src/analysis/run_p2.sh
python src/pcdps/check_residual.py --case eddy_re1000_full --snapshot 182
python src/pcdps/sweep.py --n-pairs 100 --seeds 0,1,2 --gpus 0,1
python src/pcdps/evaluate_p3.py

python src/beta_invert/estimate_beta.py --case jet_re1000_full --source truth
python src/beta_invert/model_selection.py --case jet_re1000_full
python src/common/make_summary.py
```

## Layout

- `src/reproduce`: P0/P1 data checks, baseline inference, evaluation.
- `src/analysis`: P2 failure modes and density sweep.
- `src/pcdps`: differentiable QG residual and PC-DPS (new).
- `src/beta_invert`: P4 beta inversion (new).
- `tests`: spectral operator / QG residual / metric self-tests (8 tests).
- `results`: per-run `log.json`, `metrics.json`, CSV tables, protocol notes.
- `figures`: figures 00-12 in PNG + PDF.
- `report`: `FINAL_RESULTS.md`, `results_summary.md`, `程序说明.txt`.

`third_party/`, `data/`, `venv/`, and per-run sample tensors (`*.npz`) are
excluded from git. Run `setup_third_party.sh` to restore the upstream code.

## Data / weights / licenses

- Dataset: Zenodo `10.5281/zenodo.15742146` (CC BY 4.0).
- Code/weights: Zenodo `10.5281/zenodo.15750244` (MIT); checkpoints live in
  the upstream `qg-beta-sr/Weights/` after `setup_third_party.sh`.
- Paper: `10.48550/arXiv.2507.00719`.
- Upstream code: `qg-beta-sr` and `qg-2d`, both MIT, unmodified.

## Notes

- The 64x64 fields are filtered from 512x512 QG simulations; the QG residual
  on these fields is not exactly zero (subgrid terms). PC-DPS uses it as a
  soft physical constraint; this is documented in the report.
- The released fields are raw vorticity, not pre-normalized. The reproduction
  uses the target-frame std for jet and `field_std=1.0` for eddy; see
  `results/p0/calibration_findings.md`.
- All 8 official cases are now present; no numbers are fabricated.
