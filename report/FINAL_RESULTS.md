# 最终运行结果汇总

> 所有数字来自 `results/**/log.json`、`results/*.csv`、`results/**/*.json`，没有手工编造。
> 数据只完成 127/141 个 Zenodo 分块；当前可跑 **6/8** 个论文算例。
> 缺失：`jet Re=10000 full`、`jet Re=10000 partial`（缺 jet10000 真值文件）。

## 0. 实验设置

| 项 | 值 |
|---|---|
| 硬件 | 2× RTX 4090 24GB，CUDA 12.4，torch 2.5.1+cu124 |
| 测试轨迹 | run 3221 = 池化索引 **-6** |
| 测试快照 | **182**（min_t 101 + snapshot 81） |
| 归一化 | jet：目标帧 std；eddy：field_std=1.0（发布数据是原始涡度） |
| 论文口径 | N_ens=16；另做 200 样本 × 3 seeds |
| 观测算子 | `A_les`（Gaussian+cutoff+avg_pool，scale=4）；与发布 ds16 误差 ≈1.2e-7 |
| 缺失观测的替代 | eddy 全观测、jet1000 稀疏观测由真值经 `A_les` 生成（数学等价，非编造） |

## 1. P1 复现（N_ens=16，与论文 Table 4/5 对比）

| 算例 | 方法 | 本文 rel L2 | 论文 | 相对偏差 | <10% |
|---|---|---:|---:|---:|:---:|
| eddy10000 full | CFG | 0.1472 | 0.153 | 3.8% | ✓ |
| eddy10000 full | cond | 0.1442 | 0.152 | 5.1% | ✓ |
| eddy10000 full | DPS | 0.3163 | 0.322 | 1.8% | ✓ |
| eddy10000 full | SDEdit | 0.3281 | 0.340 | 3.5% | ✓ |
| eddy10000 partial | CFG | 0.3373 | 0.310 | 8.8% | ✓ |
| eddy10000 partial | cond | 0.3424 | 0.313 | 9.4% | ✓ |
| eddy10000 partial | DPS | 0.4685 | 0.642 | 27.0% | ✗ |
| eddy10000 partial | SDEdit | 0.6851 | 0.683 | 0.3% | ✓ |
| eddy1000 full | CFG | 0.1099 | 0.101 | 8.8% | ✓ |
| eddy1000 full | cond | 0.1047 | 0.100 | 4.7% | ✓ |
| eddy1000 full | DPS | 0.3086 | 0.265 | 16.5% | ✗ |
| eddy1000 full | SDEdit | 0.3367 | 0.302 | 11.5% | ✗ |
| eddy1000 partial | CFG | 0.3938 | 0.239 | 64.8% | ✗ |
| eddy1000 partial | cond | 0.4067 | 0.247 | 64.6% | ✗ |
| eddy1000 partial | DPS | 0.4981 | 0.443 | 12.4% | ✗ |
| eddy1000 partial | SDEdit | 0.7472 | 0.775 | 3.6% | ✓ |
| jet1000 full | CFG | 0.0216 | 0.020 | 8.0% | ✓ |
| jet1000 full | cond | 0.0258 | 0.019 | 35.8% | ✗ |
| jet1000 full | DPS | 0.0503 | 0.045 | 11.8% | ✗ |
| jet1000 full | SDEdit | 0.1018 | 0.105 | 3.0% | ✓ |
| jet1000 partial | CFG | 0.0371 | 0.048 | 22.6% | ✗ |
| jet1000 partial | cond | 0.0364 | 0.045 | 19.1% | ✗ |
| jet1000 partial | DPS | 0.3541 | 0.338 | 4.8% | ✓ |
| jet1000 partial | SDEdit | 0.6028 | 0.607 | 0.7% | ✓ |

**汇总**
- 24 个“算例×方法”组合中 **14 个 <10%**，方法通过率 **58.3%**。
- 四个方法全部 <10% 的算例：**eddy10000 full**（4/4）。
- 3/4 通过：eddy10000 partial；2/4：eddy1000 full、jet1000 full、jet1000 partial；1/4：eddy1000 partial。
- **未达到任务书 6/8 Gate**：只有 6 个算例可跑，且仅 1 个算例四方法全过。
- 200 样本 × 3 seeds 的 `tab_reproduce.csv` 与 16 样本结论一致，seed 间标准差很小。

**主要偏差来源（已排查）**
1. `eddy1000 partial` 的 released partial checkpoint 与论文 Table 5 明显不一致：
   观测算子/掩码已验证；全观测模型+稀疏输入更差（0.79），partial 权重是正确权重；
   run=-6 下 cond=0.407 vs 论文 0.247，扫描 run 也找不到同时匹配四方法的轨迹。
   判断为 checkpoint 训练配置与论文存在差异，报告中如实说明。
2. jet1000 的 cond 略差（0.0258 vs 0.019），但 CFG/SDEdit 通过；属于模型/归一化细节差异。
3. DPS 对 C 很敏感，已按算例调参：
   jet 2e-5；eddy1000 full 1e-4；eddy10000 full 2e-4；partial 1e-4。

## 2. P2 失败模式

### 2.1 观测密度扫描（eddy1000 full，DPS，C=1e-4 + score clip）

| 网格 | 观测点数 | rel L2 |
|---|---:|---:|
| 4×4 | 16 | 2.546 |
| 6×6 | 36 | 1.538 |
| 8×8 | 64 | 1.223 |
| 10×10 | 100 | 0.985 |

- 曲线**单调下降**，但低密度下 rel L2>1，说明 4×4/6×6 点观测不足以恢复场。
- 图：`fig07_error_vs_density.png`

### 2.2 误差-离观测距离（eddy partial）

| 算例 | 方法 | 远端 RMSE/std | 能谱尾比 k≥16 |
|---|---|---:|---:|
| eddy1000 partial | DPS | 0.691 | 115.9 |
| eddy1000 partial | cond | 0.711 | 56.4 |
| eddy10000 partial | DPS | **1.733** | 45.8 |
| eddy10000 partial | cond | **0.524** | 10.1 |

- eddy10000 上 DPS 远端误差升到 1.73（超过目标场自身变率），cond 只有 0.52；
  清楚显示“DPS 的观测信息不向未观测区传播”。
- eddy1000 上两者远端接近（0.69 vs 0.71），但近端 cond 误差低得多（0.11 vs 0.29）。
- 图：`fig05_dps_spectrum_smoothing.png`、`fig06_error_vs_distance.png`

## 3. P3 PC-DPS

成对采样（t, t+1），物理项 = QG 残差梯度；λ 扫描，100 对，3 seeds。
`eddy10000 partial` 的 λ=2/2.5 使用梯度分位裁剪 0.999 + 只约束缺口区域（gap-only）。

| 算例 | λ | rel L2 | 相对 λ=0 下降 | 谱尾比 |
|---|---:|---:|---:|---:|
| eddy10000 partial | 0 | 0.4691 | — | 44.20 |
| eddy10000 partial | 0.1 | 0.4653 | 0.8% | 44.21 |
| eddy10000 partial | 0.5 | 0.4550 | 3.0% | 44.12 |
| eddy10000 partial | 1.0 | 0.4456 | 5.0% | 43.63 |
| **eddy10000 partial** | **2.0** | **0.4329** | **7.7%** | **42.47** |
| eddy10000 partial | 2.5 | 0.4342 | 7.5% | 42.34 |
| eddy1000 partial | 0 | 0.5009 | — | 97.09 |
| eddy1000 partial | 0.1 | 0.4938 | 1.4% | 105.01 |
| eddy1000 partial | 0.5 | 0.4836 | 3.4% | 143.64 |
| **eddy1000 partial** | **1.0** | **0.4819** | **3.8%** | 186.25 |
| eddy1000 partial | 2.0 | 0.4882 | 2.5% | 227.24 |
| jet1000 full | 0 | 0.05328 | — | 802.0 |
| jet1000 full | 0.1 | 0.05331 | -0.05% | 809.8 |

**结论**
- eddy10000 partial：**-7.7%**，差 0.3 个百分点未达 8% Gate；谱尾比值更接近 1。
- eddy1000 partial：**-3.8%**，谱尾变差。
- jet1000 full：物理项基本无效（jet 64² 残差/时间导数≈0.44，亚格子项仍大）。
- **P3 Gate 未通过**（8% + 谱尾双条件）。属任务书定义的“3%–8% 边缘提升”，
  已如实报告，未修改数字。
- 图：`fig08_pcdps_panels.png`、`fig09_pcdps_spectra.png`、`fig10_lambda_sweep.png`

## 4. P4 β 反演（系统辨识）

| 数据 | β_true | β̂ | 误差 |
|---|---:|---:|---:|
| jet1000 真值（oracle） | 2.5 | 2.7388 | 9.6% |
| jet1000 PC-DPS 重建 | 2.5 | **2.6565** | **6.3%** |
| eddy1000 真值（oracle） | 0 | 0.0009 | 0.0009 |
| eddy1000 partial PC-DPS | 0 | 0.0254 | 0.0254 |

- **P4 Gate 通过**：jet 重建场 β̂ 相对误差 6.3% <15%。
- 二分类模型选择：jet 观测上正确选中 jet；
  留出点 RMSE：eddy 模型 2.016，jet 模型 0.925。
- ν 在 64² 下不可辨识，只做量级估计；报告中写明这一边界。
- 图：`fig11_beta_estimate.png`、`fig12_beta_convergence.png`

## 5. 对报告写作的建议

- **仿真结果与程序**：用 P1 的 24 行表（或精简为 6 算例 × 4 方法的均值表），
  明确写“6/8 算例，jet Re=10000 两例因 Zenodo 403 缺真值标 N/A”；不要写“8 个全跑完”。
- **算法比较**：用 P2 密度曲线 + P3 的 λ 表和 fig08–fig10；
  结论写“PC-DPS 在 eddy10000 partial 上 -7.7%，未达 8% Gate，谱尾改善/恶化分算例”。
- **系统辨识**：P4 的 β 表 + 模型选择；这是课程“系统辨识”的直接落点。
- **课程口径**：场重建 = 稀疏观测下的状态估计；β 反演 = 系统辨识。
- **诚实边界**：
  1. 数据缺 2 个算例；
  2. P3 未过 8%；
  3. eddy1000 partial 的 M3/M4 与论文差约 65%，原因未完全定位（checkpoint 差异）；
  4. 64² 滤波场 QG 残差含亚格子项，物理项是软约束。

## 6. 文件索引

| 内容 | 路径 |
|---|---|
| P1 论文口径表 | `results/tab_reproduce_16.csv` |
| P1 200 样本表 | `results/tab_reproduce.csv` |
| P1 Gate | `results/p1/gate_summary_16.json` |
| P2 密度 | `results/p2/density_summary.json` |
| P2 失败模式 | `results/p2/failure_modes.json` |
| P3 表 | `results/tab_pcdps.csv` |
| P3 Gate | `results/p3/gate_summary.json` |
| P4 β | `results/p4/**/beta_estimate.json` |
| 图 | `figures/fig00..fig12`（png+pdf） |
| 说明 | `report/程序说明.txt` |
| 自动汇总 | `report/results_summary.md` |
| 检查点包 | `report/checkpoint_2026-11-13_final.zip` |

---

# 补充：四个局限的定位结果（2026-11-14）

## A. P1 eddy Re=1000 sparse +65% 偏差：根因定位

1. **索引对齐已排除**：`A_les(truth[-6,182], gappy=True)` 与发布 partial 的
   rel L2 = **0.0**（eddy1000、eddy10000 都是 0.0）。不是 batch/time 错位。
2. **checkpoint 并非整体失效**：同一 partial cond 权重在训练 run 上
   rel L2 = 0.246（run 0）、0.263（100）、0.135（200）、0.133（300）、0.381（400）；
   测试 run -6 = 0.446。模型对场非常敏感。
3. **不是多快照平均**：run -6 在 101–195 多个快照上的 cond 平均约 0.359，不是 0.247。
4. **候选轨迹**：run 0 的 M1/M3/M4 = 0.703/0.246/0.232，与论文 0.775/0.247/0.239
   差 <10%；M2=0.489 差 10.3%。run -19 的 M1/M3/M4 也接近。
   但没有 run 在固定 C 下同时匹配四方法，且 run 0/-19 很可能是训练轨迹。
5. **权重历史**：`git log --all` 只有 Initial commit，无旧 partial 权重版本。
6. **结论**：65% 不是代码/索引 bug，最可能是**发布 partial checkpoint 与论文训练配置/
   测试轨迹不一致**。正式结果仍用 run=-6、snapshot=182；eddy1000 partial 的 M3/M4
   标为“复现异常”，不挑 run、不改数字。证据：`results/p1/checkpoint_audit.md`、
   `results/p0/alignment_audit.json`。

## B. P3 亚格子项：残差审计与有效闭合

- 前向/中心残差 `rms(r)/rms(dqdt)`：
  - eddy1000：0.13–0.24 / 0.11–0.20
  - eddy10000：0.28–0.62 / 0.21–0.45
  - jet1000：0.33–0.45 / 0.26–0.36；β oracle 2.74–2.79
- 有效线性闭合（训练 0–399，验证 400–479）：

| case | c_J | c_mu | c_nu | c_beta | c_h4 | c_F | val ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| eddy1000 | 1.09 | 0.91 | 2.03 | — | 0 | 1.07 | 0.276→0.260 |
| eddy10000 | 0.99 | 0.82 | 8.48 | — | 0 | 0.97 | 0.523→0.522 |
| jet1000 | 0.90 | 1.10 | 1.41 | 2.78 | 0 | 1.11 | 0.565→0.380 |

- jet 闭合降残差 **33%**；eddy10000 **无改善**；eddy1000 仅 6%。双调和 ∇⁴q 系数≈0。
- 结论：eddy10000 的缺失项不在 J/μq/∇²q/∇⁴q/β/F 张成空间内，线性闭合不足；
  物理项只能作软约束。证据：`results/p3/residual_audit.csv`、
  `results/p3/closure_coefficients.json`、`results/p3/closure_findings.md`、
  `figures/fig13_residual_spectrum.png`。

## C. P2 PC-DPS：为什么停在 7.7%

- 误差分解（eddy10000 partial）：DPS→PC-DPS
  - k0–8：0.378→0.358
  - k9–16：0.976→0.968
  - k17–32：1.005→1.000
  - 观测区 RMSE/std：0.196→0.195；缺口区：0.394→0.378
  大尺度/缺口有改善，细尺度几乎不动。
- 负结果：
  - 三帧中心差分（20 对）：0.4624→0.4491（-2.9%），大 λ 爆炸；
  - 逐样本梯度归一化：~0.1% 额外改善；
  - λ 随 t 增强：~0.4% 额外改善；
  - 残差低通、x0 低通、t_min/t_max 门控：均无改善或更差。
- 正式最佳仍是 **eddy10000 partial λ=2 + clip 0.999 + gap-only：-7.7%**；
  差 0.3% 未过 8% Gate。结论：要过 8% 需要尺度依赖闭合或学习型亚格子模型，
  不是继续调 λ。证据：`results/p3/pcdps_diagnosis.json`、
  `figures/fig15_pcdps_diagnosis.png`。

## D. P0 最小补数据方案（未执行，等待用户下载）

`parse_missing.py` 已精确算出：只需 zip 尾部
**327,859,074 字节**，范围 `8522825728-8850684801`，即可补全
`fields_pooled64_jet_re_10000.npy`；不需要 0.88GB 全尾部/中央目录。
用户侧命令与服务器补丁见 `src/reproduce/parse_missing.py`、
`src/reproduce/patch_jet10000.py` 与 `STATUS.md`。
