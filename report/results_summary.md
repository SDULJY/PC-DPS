# 仿真结果摘要（自动生成，数字均来自 results/）

> 注意：目前只解压了 127/141 个数据块，5 个算例可跑；jet10000 full、jet1000 partial、eddy 全观测文件缺失。论文 Table 4/5 的官方 N_ens=16；我们同时给 16 和 200 样本两套结果。

## P1 复现（N_ens=16，与论文同口径）

| case | method | rel_l2_mean | paper_rel_l2 | rel_dev | within_10pct |
|---|---|---|---|---|---|
| eddy_re10000_full | cfg | 0.1472 | 0.153 | 0.03803 | True |
| eddy_re10000_full | cond | 0.1442 | 0.152 | 0.05109 | True |
| eddy_re10000_full | dps | 0.3163 | 0.322 | 0.01764 | True |
| eddy_re10000_full | sdedit | 0.3281 | 0.34 | 0.0351 | True |
| eddy_re10000_partial | cfg | 0.3373 | 0.31 | 0.08794 | True |
| eddy_re10000_partial | cond | 0.3424 | 0.313 | 0.0939 | True |
| eddy_re10000_partial | dps | 0.4685 | 0.642 | 0.2703 | False |
| eddy_re10000_partial | sdedit | 0.6851 | 0.683 | 0.003117 | True |
| eddy_re1000_full | cfg | 0.1099 | 0.101 | 0.08776 | True |
| eddy_re1000_full | cond | 0.1047 | 0.1 | 0.04749 | True |
| eddy_re1000_full | dps | 0.3086 | 0.265 | 0.1647 | False |
| eddy_re1000_full | sdedit | 0.3367 | 0.302 | 0.1151 | False |
| eddy_re1000_partial | cfg | 0.3938 | 0.239 | 0.6475 | False |
| eddy_re1000_partial | cond | 0.4067 | 0.247 | 0.6465 | False |
| eddy_re1000_partial | dps | 0.4981 | 0.443 | 0.1243 | False |
| eddy_re1000_partial | sdedit | 0.7472 | 0.775 | 0.03591 | True |
| jet_re1000_full | cfg | 0.02159 | 0.02 | 0.07975 | True |
| jet_re1000_full | cond | 0.02581 | 0.019 | 0.3584 | False |
| jet_re1000_full | dps | 0.05033 | 0.045 | 0.1184 | False |
| jet_re1000_full | sdedit | 0.1018 | 0.105 | 0.0302 | True |
| jet_re1000_partial | cfg | 0.03715 | 0.048 | 0.2261 | False |
| jet_re1000_partial | cond | 0.03643 | 0.045 | 0.1905 | False |
| jet_re1000_partial | dps | 0.3541 | 0.338 | 0.04773 | True |
| jet_re1000_partial | sdedit | 0.6028 | 0.607 | 0.006881 | True |

Gate: 1/6 cases all methods within 10% (pass=False); method pass rate=0.58.


## P1（200 样本 × 3 seeds）

| case | method | rel_l2_mean | rel_l2_seedstd | rel_l2_ensmean | paper_rel_l2 | rel_dev |
|---|---|---|---|---|---|---|
| eddy_re10000_full | cfg | 0.1478 | 0.0006823 | 0.1181 | 0.153 | 0.03381 |
| eddy_re10000_full | cond | 0.1455 | 0.001168 | 0.1163 | 0.152 | 0.04257 |
| eddy_re10000_full | dps | 0.3147 | 0.0007574 | 0.2484 | 0.322 | 0.0226 |
| eddy_re10000_full | sdedit | 0.3275 | 0.0003222 | 0.2972 | 0.34 | 0.03669 |
| eddy_re10000_partial | cfg | 0.3398 | 0.0004635 | 0.309 | 0.31 | 0.09604 |
| eddy_re10000_partial | cond | 0.3453 | 0.0005759 | 0.3164 | 0.313 | 0.1032 |
| eddy_re10000_partial | dps | 0.5191 | 0.03198 | 0.3855 | 0.642 | 0.1915 |
| eddy_re10000_partial | sdedit | 0.6845 | 0.0007832 | 0.6729 | 0.683 | 0.002151 |
| eddy_re1000_full | cfg | 0.1093 | 0.0004048 | 0.08574 | 0.101 | 0.08215 |
| eddy_re1000_full | cond | 0.1041 | 0.0004783 | 0.08249 | 0.1 | 0.0408 |
| eddy_re1000_full | dps | 0.3064 | 0.0008549 | 0.2576 | 0.265 | 0.1563 |
| eddy_re1000_full | sdedit | 0.3359 | 0.001321 | 0.3119 | 0.302 | 0.1122 |
| eddy_re1000_partial | cfg | 0.3963 | 0.0003333 | 0.3682 | 0.239 | 0.6581 |
| eddy_re1000_partial | cond | 0.4066 | 0.003226 | 0.3834 | 0.247 | 0.6462 |
| eddy_re1000_partial | dps | 0.4997 | 0.003406 | 0.4371 | 0.443 | 0.128 |
| eddy_re1000_partial | sdedit | 0.7474 | 0.000885 | 0.7396 | 0.775 | 0.03565 |
| jet_re1000_full | cfg | 0.02302 | 0.0006741 | 0.01405 | 0.02 | 0.151 |
| jet_re1000_full | cond | 0.02796 | 0.003451 | 0.0136 | 0.019 | 0.4716 |
| jet_re1000_full | dps | 0.05002 | 0.0003476 | 0.03486 | 0.045 | 0.1116 |
| jet_re1000_full | sdedit | 0.1026 | 0.0002792 | 0.08337 | 0.105 | 0.02238 |
| jet_re1000_partial | cfg | 0.03787 | 0.0001361 | 0.02377 | 0.048 | 0.211 |
| jet_re1000_partial | cond | 0.04014 | 0.001327 | 0.02044 | 0.045 | 0.1079 |
| jet_re1000_partial | dps | 0.3706 | 0.007909 | 0.2654 | 0.338 | 0.09635 |
| jet_re1000_partial | sdedit | 0.6056 | 0.001251 | 0.5649 | 0.607 | 0.002296 |

## P2 观测密度扫描（eddy Re=1000 full，DPS，C=1e-4 + clip）

| density | n_obs | rel_l2_mean | rel_l2_std |
|---|---|---|---|
| 4 | 16 | 2.546 | 1.117 |
| 6 | 36 | 1.538 | 0.02379 |
| 8 | 64 | 1.223 | 0.006324 |
| 10 | 100 | 0.9849 | 0.004819 |

失败模式：DPS 远场 RMSE/std 随距离升高并平台化，条件模型显著更低（见 fig06）。


## P3 PC-DPS λ 扫描（成对采样，100 对，3 seeds）

| case | lambda | rel_l2_mean | rel_l2_std | rel_reduction_vs_lambda0 | spectrum_tail_ratio |
|---|---|---|---|---|---|
| eddy_re10000_partial | 0 | 0.4691 | 0.001906 | 0 | 44.2 |
| eddy_re10000_partial | 0.01 | 0.4687 | 0.001924 | 0.0009473 | 44.2 |
| eddy_re10000_partial | 0.1 | 0.4653 | 0.001875 | 0.008063 | 44.21 |
| eddy_re10000_partial | 0.5 | 0.455 | 0.002022 | 0.03013 | 44.12 |
| eddy_re10000_partial | 1 | 0.4455 | 0.002044 | 0.05022 | 43.63 |
| eddy_re10000_partial | 2 | 0.4329 | 0.002137 | 0.07712 | 42.47 |
| eddy_re10000_partial | 2.5 | 0.4342 | 0.001637 | 0.07451 | 42.34 |
| eddy_re1000_partial | 0 | 0.5009 | 0.003972 | 0 | 97.09 |
| eddy_re1000_partial | 0.01 | 0.5001 | 0.003827 | 0.001544 | 97.81 |
| eddy_re1000_partial | 0.1 | 0.4938 | 0.003581 | 0.01413 | 105 |
| eddy_re1000_partial | 0.5 | 0.4836 | 0.003506 | 0.03437 | 143.6 |
| eddy_re1000_partial | 1 | 0.4819 | 0.00401 | 0.03785 | 186.3 |
| eddy_re1000_partial | 2 | 0.4882 | 0.003467 | 0.02526 | 227.2 |
| jet_re1000_full | 0 | 0.05328 | nan | 0 | 802 |
| jet_re1000_full | 0.1 | 0.05331 | nan | -0.0004997 | 809.8 |

```json
{
  "eddy_re10000_partial": {
    "best_lambda": 2.0,
    "best_rel_l2": 0.43292972881842773,
    "baseline_rel_l2": 0.4691076059866048,
    "relative_reduction": 0.07712063651582102,
    "tail_ratio_best": 42.47026136326581,
    "tail_ratio_lambda0": 44.19714168110101,
    "tail_closer_to_one": true
  },
  "eddy_re1000_partial": {
    "best_lambda": 1.0,
    "best_rel_l2": 0.48189935044170645,
    "baseline_rel_l2": 0.5008543144690032,
    "relative_reduction": 0.03784526454035763,
    "tail_ratio_best": 186.2538750862076,
    "tail_ratio_lambda0": 97.09177058509783,
    "tail_closer_to_one": false
  },
  "jet_re1000_full": {
    "best_lambda": 0.0,
    "best_rel_l2": 0.05328327690860514,
    "baseline_rel_l2": 0.05328327690860514,
    "relative_reduction": 0.0,
    "tail_ratio_best": 802.0266150846555,
    "tail_ratio_lambda0": 802.0266150846555,
    "tail_closer_to_one": false
  },
  "_gate_pass": false
}
```


说明：eddy10000 partial 最佳 λ=2（clip=0.999）rel L2 降 7.7%，接近但未达 8% Gate；谱尾比值指标对真值近零桶敏感，log-spectrum 误差未改善。eddy1000 partial 降 3.8%，属边缘提升。


## P4 β 反演（系统辨识）

- p4/eddy_re1000_full/truth/seed0: β_true=0.0, β̂=0.0009, absolute_error_mean=0.0009, n=1
- p4/eddy_re1000_partial/pcdps/seed0: β_true=0.0, β̂=0.0254, absolute_error_mean=0.0254, n=100
- p4/jet_re1000_full/pcdps/seed0: β_true=2.5, β̂=2.6565, relative_error_mean=0.0626, n=50
- p4/jet_re1000_full/truth/seed0: β_true=2.5, β̂=2.7388, relative_error_mean=0.0955, n=1
- model selection: true=jet, selected=jet, correct=True; heldout RMSE eddy=2.016, jet=0.925

## 图

- figures/fig00_data_sample.png
- figures/fig01_recon_panels.png
- figures/fig02_rel_l2_vs_paper.png
- figures/fig03_spectra.png
- figures/fig04_stats_spread.png
- figures/fig05_dps_spectrum_smoothing.png
- figures/fig06_error_vs_distance.png
- figures/fig07_error_vs_density.png
- figures/fig08_pcdps_panels.png
- figures/fig09_pcdps_spectra.png
- figures/fig10_lambda_sweep.png
- figures/fig11_beta_estimate.png
- figures/fig12_beta_convergence.png