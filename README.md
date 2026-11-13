# QG turbulence diffusion reproduction + PC-DPS

Reproduction of

> Suresh Babu, Sadam & Lermusiaux (2025), *Guided Unconditional and Conditional
> Generative Models for Super-Resolution and Inference of Quasi-Geostrophic
> Turbulence*, JAMES, arXiv:2507.00719.

and a physics-constrained DPS extension (PC-DPS) plus an optional beta inversion.

## Quick start

```bash
cd .
source venv/bin/activate

bash src/common/download_data.sh       # 9.4 GB, parallel Range download, md5
bash src/reproduce/extract_data.sh     # only the 12 npy files needed

python src/reproduce/check_data.py --file data/fields_pooled64_jet_re_1000.npy
python src/reproduce/run_inference.py --case jet_re1000_full --method dps \
    --seed 0 --n-samples 16 --snapshot 182
python src/reproduce/run_p1_grid.py --n-samples 200 --seeds 0,1,2 --gpus 0,1
python src/reproduce/evaluate.py
python src/reproduce/make_figs_p1.py

pytest tests/test_qg_ops.py -q
python src/pcdps/check_residual.py --case jet_re1000_full --snapshot 182
python src/pcdps/run_pcdps.py --case eddy_re1000_partial --lambda-phys 0.1 \
    --seed 0 --n-pairs 100
python src/pcdps/sweep.py --n-pairs 100 --seeds 0,1,2
python src/pcdps/evaluate_p3.py
python src/pcdps/make_figs_p3.py

python src/common/collect.py
python src/common/make_summary.py
```

## Layout

- `third_party/qg-beta-sr`, `third_party/qg-2d`: upstream code, unmodified (MIT).
- `src/reproduce`: P0/P1 data checks and baseline inference.
- `src/analysis`: P2 failure modes.
- `src/pcdps`: differentiable QG residual and PC-DPS (new).
- `src/beta_invert`: P4 beta inversion (optional).
- `results`: per-run `log.json`, metrics, CSV tables.
- `figures`: report figures.
- `report`: `程序说明.txt`, `results_summary.md`, submission zip.

## Data / weights

- Dataset: Zenodo `10.5281/zenodo.15742146` (CC BY 4.0).
- Code/weights: Zenodo `10.5281/zenodo.15750244` (MIT); the 12 checkpoints are
  also in `third_party/qg-beta-sr/Weights/`.
- Paper: `10.48550/arXiv.2507.00719`.

## Notes

- The 64x64 fields are filtered from 512x512 QG simulations; the QG residual on
  these fields is not exactly zero (subgrid terms). PC-DPS uses it as a soft
  physical constraint; this is documented in the report.
- `third_party/` is never modified. All extensions live in `src/`.
