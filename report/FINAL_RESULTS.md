# 最终运行结果汇总（8/8 算例）

> 所有数字来自 `results/**/log.json`、`results/*.csv`、`results/**/*.json`，没有手工编造。
> 数据已补齐：127 个分块 + 328MB 最小尾部补丁，官方 8 个算例全部可跑。

## 0. 实验设置

| 项 | 值 |
|---|---|
| 硬件 | 2× RTX 4090 24GB，CUDA 12.4，torch 2.5.1+cu124 |
| 测试轨迹 | run 3221 = 池化索引 **-6** |
| 测试快照 | **182**（min_t 101 + snapshot 81） |
| 归一化 | jet：目标帧 std；eddy：field_std=1.0（发布数据是原始涡度） |
| 论文口径 | N_ens=16；另做 200 样本 × 3 seeds |
| 观测算子 | `A_les`（Gaussian+cutoff+avg_pool，scale=4）；与发布 ds16 误差 ≈1.2e-7 |
| 数据补齐 | `jet_re_10000` 真值用 327,859,074 字节尾部补丁恢复（CRC 校验） |
| 替代观测 | eddy 全观测、jet1000/jet10000 稀疏观测由真值经 `A_les` 生成（数学等价） |

## 1. P1 复现（N_ens=16，论文 Table 4/5）

| 算例 | 方法 | 本文 | 论文 | 偏差 | <10% |
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
| jet10000 full | CFG | 0.0631 | 0.071 | 11.2% | ✗ |
| jet10000 full | cond | 0.0601 | 0.069 | 12.9% | ✗ |
| jet10000 full | DPS | 0.1098 | 0.129 | 14.9% | ✗ |
| jet10000 full | SDEdit | 0.1708 | 0.213 | 19.8% | ✗ |
| jet10000 partial | CFG | 0.0866 | 0.102 | 15.1% | ✗ |
| jet10000 partial | cond | 0.0825 | 0.099 | 16.6% | ✗ |
| jet10000 partial | DPS | 0.1358 | 0.435 | 68.8% | ✗ |
| jet10000 partial | SDEdit | 0.6665 | 0.668 | 0.2% | ✓ |
| jet1000 full | CFG | 0.0216 | 0.020 | 8.0% | ✓ |
| jet1000 full | cond | 0.0258 | 0.019 | 35.8% | ✗ |
| jet1000 full | DPS | 0.0503 | 0.045 | 11.8% | ✗ |
| jet1000 full | SDEdit | 0.1018 | 0.105 | 3.0% | ✓ |
| jet1000 partial | CFG | 0.0371 | 0.048 | 22.6% | ✗ |
| jet1000 partial | cond | 0.0364 | 0.045 | 19.1% | ✗ |
| jet1000 partial | DPS | 0.3541 | 0.338 | 4.8% | ✓ |
| jet1000 partial | SDEdit | 0.6028 | 0.607 | 0.7% | ✓ |

**汇总**
- 32 个“算例×方法”组合中 **15 个 <10%**，方法通过率 **46.9%**。
- 四方法全过的算例：**eddy10000 full**（4/4）。
- 3/4：eddy10000 partial；2/4：eddy1000 full、jet1000 full、jet1000 partial；
  1/4：eddy1000 partial、jet10000 partial；0/4：jet10000 full。
- **未达 6/8 Gate**（只有 1 个算例四方法全过；任务书的 Gate 是“至少 6/8 算例
  所有方法 <10%”，不是方法级通过率）。
- 200 样本 × 3 seeds 的 `tab_reproduce.csv` 与 16 样本结论一致，seed 间标准差很小。

**主要偏差**
1. `jet10000 full` 四方法一致偏低 11–20%（不是单个方法异常），更像测试帧/评估协议差异；
   索引已校验：`A_les(truth[-6,182])` 与 ds16 全观测 rel=0.0。
2. `jet10000 partial` 的 DPS 为 0.136（论文 0.435，低 69%）；C=1e-4 下 DPS 太强，
   论文该工况的 tuned C 可能更大；cond/CFG 低 15–17%。
3. `eddy1000 partial` 的 cond/CFG 高约 65%：已排除索引/掩码/算子 bug；
   同一 checkpoint 在训练 run 上 0.133–0.381，测试 -6 为 0.446；
   判定为发布 partial checkpoint/测试轨迹与论文配置不一致，详见
   `results/p1/checkpoint_audit.md`。
4. jet1000 full cond 高 35.8%，DPS 高 11.8%；其余方法通过。

## 2. P2 失败模式

### 2.1 观测密度扫描（eddy1000 full，DPS，C=1e-4 + score clip）

| 网格 | 观测点数 | rel L2 |
|---|---:|---:|
| 4×4 | 16 | 2.546 |
| 6×6 | 36 | 1.538 |
| 8×8 | 64 | 1.223 |
| 10×10 | 100 | 0.985 |

曲线单调下降；低密度下 rel L2>1，说明点观测不足以恢复场。

### 2.2 误差-离观测距离

| 算例 | 方法 | 远端 RMSE/std | 能谱尾比 k≥16 |
|---|---|---:|---:|
| eddy1000 partial | DPS | 0.691 | 115.9 |
| eddy1000 partial | cond | 0.711 | 56.4 |
| eddy10000 partial | DPS | **1.733** | 45.8 |
| eddy10000 partial | cond | **0.524** | 10.1 |

eddy10000 上 DPS 远端误差升到 1.73，cond 仅 0.52，清楚显示 DPS 不向未观测区传播信息。

图：`fig05`、`fig06`、`fig07`。

## 3. P3 PC-DPS

成对采样，物理项 = QG 残差梯度；100 对，3 seeds。
`eddy10000 partial` 的 λ=2 使用梯度分位裁剪 0.999 + gap-only。

| 算例 | λ=0 | 最佳 λ | 最佳 rel L2 | 下降 | 谱尾 |
|---|---:|---:|---:|---:|---:|
| eddy10000 partial | 0.4691 | 2.0 | 0.4329 | **-7.7%** | 44.2→42.5 |
| eddy1000 partial | 0.5009 | 1.0 | 0.4819 | -3.8% | 97.1→186.3 |
| jet1000 full | 0.0533 | 0.1 | 0.0533 | ≈0 | 基本无效 |

**结论**：eddy10000 差 0.3% 未达 8%；eddy1000 边缘提升但谱尾变差；
jet full 无效。P3 Gate 未通过，按任务书如实报告。

**为什么停在 7.7%**：误差分解显示 k0–8 0.378→0.358、k9–16 0.976→0.968、
k17–32 1.005→1.000；缺口区 0.394→0.378。大尺度/缺口改善，细尺度不动。
三帧中心差分（-2.9%）、逐样本归一化（~0.1%）、λ(t)（~0.4%）、残差/x0 低通、
t_min/t_max 门控均未超过 7.7%。要过 8% 需要尺度依赖闭合或学习型亚格子模型。

图：`fig08`、`fig09`、`fig10`、`fig15`。

## 4. P3 亚格子残差与有效闭合（补充）

- 残差/dqdt：eddy1000 0.11–0.20；eddy10000 0.21–0.45；jet1000 0.26–0.36。
- 有效线性闭合（训练 0–399，验证 400–479）：

| case | c_J | c_mu | c_nu | c_beta | c_h4 | val ratio |
|---|---:|---:|---:|---:|---:|---:|
| eddy1000 | 1.09 | 0.91 | 2.03 | — | 0 | 0.276→0.260 |
| eddy10000 | 0.99 | 0.82 | 8.48 | — | 0 | 0.523→0.522 |
| jet1000 | 0.90 | 1.10 | 1.41 | 2.78 | 0 | 0.565→0.380 |

jet 降残差 33%；eddy10000 无改善；∇⁴q 系数≈0。结论：eddy10000 的缺失项不在
J/μq/∇²q/∇⁴q/β/F 张成空间内，物理项只能作软约束。
证据：`results/p3/closure_findings.md`、`fig13`。

## 5. P4 β 反演（系统辨识）

| 数据 | β_true | β̂ | 误差 |
|---|---:|---:|---:|
| jet1000 真值 oracle | 2.5 | 2.7388 | 9.6% |
| jet1000 PC-DPS 重建 | 2.5 | **2.6565** | **6.3%** |
| eddy1000 真值 oracle | 0 | 0.0009 | — |
| eddy1000 partial PC-DPS | 0 | 0.0254 | — |

模型选择：jet 观测正确选中 jet（留出 RMSE 0.925 vs eddy 2.016）。
**P4 Gate 通过**（jet 重建场 6.3% <15%）。ν 在 64² 下不可辨识，只做量级估计。
图：`fig11`、`fig12`。

## 6. 对报告写作的建议

- P1：用 32 行表或精简的 8×4 均值表；必须写“方法通过率 46.9%，未达 6/8 Gate”。
  不要写“8 个算例全部复现成功”。
- P2：密度曲线 + 误差-距离图，突出 eddy10000 的 DPS 远端 1.73 vs cond 0.52。
- P3：7.7% 边缘提升 + 负结果 + 亚格子闭合诊断；不要包装成“显著改进”。
- P4：β 反演 6.3% 是明确的系统辨识结果；ν 只报量级。
- 课程口径：场重建 = 稀疏观测下的状态估计；β 反演 = 系统辨识。

## 7. 文件索引

| 内容 | 路径 |
|---|---|
| P1 论文口径表 | `results/tab_reproduce_16.csv` |
| P1 200 样本表 | `results/tab_reproduce.csv` |
| P1 Gate | `results/p1/gate_summary_16.json` |
| P1 异常审计 | `results/p1/checkpoint_audit.md` |
| P2 密度/失败模式 | `results/p2/density_summary.json`、`failure_modes.json` |
| P3 表/Gate | `results/tab_pcdps.csv`、`results/p3/gate_summary.json` |
| P3 残差/闭合 | `results/p3/residual_audit.csv`、`closure_coefficients.json` |
| P4 β | `results/p4/**/beta_estimate.json` |
| 图 | `figures/fig00..fig15`（png+pdf） |
| 说明 | `report/程序说明.txt` |
| 检查点包 | `report/checkpoint_2026-11-14_final.zip` |
