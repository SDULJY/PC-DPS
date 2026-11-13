# P1 evaluation protocol (updated 2026-11-12)

- Cases: paper Table 3, 8 = {eddy, jet} x {Re=1000, 10000} x {ds16 full, ds16 partial}.
  Only the cases whose truth/observation files are fully downloaded are run now;
  the rest are marked `N/A` and will be filled after the Zenodo download completes.
- Test trajectory: run 3221 = pooled batch index **-6** (500 trajectories start at
  run 2727; run 3221 is index 494). This was confirmed on jet Re=1000 full:
  cond ensemble-mean rel L2 = 0.0182 vs paper 0.019.
- Test snapshot: pooled index **182** (`min_t 101 + snapshot 81`).
- Normalization: released fields are raw vorticity. The paper's zero-mean/unit-variance
  protocol is applied by dividing **target and observation by the same scalar**:
  - jet Re=1000: target-frame std (`--normalize-target`).
  - eddy Re=1000/10000: `field_std=1.0` (calibrated; fixed 1.3 is worse).
  - Do **not** normalize target and observation separately.
- Baseline hyperparameters: N_diff=64, N_corr=2, correction scale 0.3,
  SDEdit t_i=16/64=0.25, DPS scale=4, sigma_measure=1e-2, CFG w=1.25.
  DPS C is tuned per case: jet 2e-5; eddy Re=1000 full 1e-4;
  eddy Re=10000 full 2e-4 (2e-5 diverges); partial cases 2e-5.
- Paper uses N_ens=16 (Table 4/5). Final runs use 200 samples per case/method/seed
  and 3 seeds {0,1,2}; `tab_reproduce.csv` reports the mean across seeds and
  seed-to-seed std.
- Metrics: per-sample relative L2 `||w_FF - w_hat_0||_2 / ||w_FF||_2`
  (mean +/- std), ensemble-mean relative L2, angle-averaged enstrophy spectrum,
  vorticity mean/var/skew/kurtosis, spread-error Pearson/Spearman correlation,
  and spread/RMSE ratio.
- For eddy full cases the ds16 full observation file is still in the undownloaded
  tail; it is generated from truth with the validated `A_les` operator
  (`data/generated/eddy{1000,10000}_ds16_full.npy`, rel error 1e-7).
