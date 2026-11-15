# 准地转湍流扩散模型：论文复现与物理约束 DPS

复现论文

> **Guided Unconditional and Conditional Generative Models for Super-Resolution
> and Inference of Quasi-Geostrophic Turbulence**
> A. N. Suresh Babu, A. Sadam, P. F. J. Lermusiaux, JAMES 2025.
> [arXiv:2507.00719](https://arxiv.org/abs/2507.00719) ·
> [DOI 10.48550/arXiv.2507.00719](https://doi.org/10.48550/arXiv.2507.00719)

并增加两个扩展：

- **PC-DPS**：把可微 QG 残差作为软约束加入扩散采样（`src/pcdps/`）。
- **β 反演**：从重建场辨识 β 平面系数，属于系统辨识（`src/beta_invert/`）。

上游扩散代码（`qg-beta-sr`）和 QG 模拟器（`qg-2d`）**未做任何修改**，
由 `setup_third_party.sh` 克隆到 `third_party/`；所有新代码都在 `src/` 中。

本仓库只报告实际跑出的结果。部分 Gate 没有通过，负结果和边缘结果都保留在
下面的表格与 [`report/FINAL_RESULTS.md`](report/FINAL_RESULTS.md) 中。

---

## 主要结果

### P1 — 论文复现（8 算例 × 4 方法 × 3 seeds）

测试协议：测试轨迹 `3221` = 池化索引 **-6**，快照 **182**，
`N_ens=16`（论文 Table 4/5 口径）。表中数字为 3 个 seed 的均值；
200 样本 × 3 seeds 的完整结果结论一致
（[`results/tab_reproduce_16.csv`](results/tab_reproduce_16.csv)、
[`results/tab_reproduce.csv`](results/tab_reproduce.csv)）。

与论文值的相对偏差（越低越好）：

| 算例 | SDEdit | DPS | cond | CFG | 偏差 <10% |
|---|---:|---:|---:|---:|---:|
| eddy Re=10⁴，全观测 | 3.5% | 1.8% | 5.1% | 3.8% | **4/4** |
| eddy Re=10⁴，稀疏 | 0.3% | 27.0% | 9.4% | 8.8% | 3/4 |
| eddy Re=10³，全观测 | 11.5% | 16.5% | 4.7% | 8.8% | 2/4 |
| eddy Re=10³，稀疏 | 3.6% | 12.4% | 64.6% | 64.8% | 1/4 |
| jet Re=10⁴，全观测 | 19.8% | 14.9% | 12.9% | 11.2% | 0/4 |
| jet Re=10⁴，稀疏 | 0.2% | 68.8% | 16.6% | 15.1% | 1/4 |
| jet Re=10³，全观测 | 3.0% | 11.8% | 35.8% | 8.0% | 2/4 |
| jet Re=10³，稀疏 | 0.7% | 4.8% | 19.1% | 22.6% | 2/4 |

**汇总：32 个“算例 × 方法”组合中 15 个偏差 <10%（46.9%）；只有
`eddy Re=10⁴ 全观测` 的四个方法全部通过。** 严格的 6/8 算例 Gate 未通过。
最大异常是 `eddy Re=10³ 稀疏` 的 cond/CFG 偏高约 65%；索引、观测算子、
归一化均已审计并排除，证据指向发布的 partial checkpoint / 测试轨迹差异，
详见 [`results/p1/checkpoint_audit.md`](results/p1/checkpoint_audit.md)。

### P2 — 稀疏重建的失败模式

- 观测密度（eddy Re=10³ 全观测，DPS）：4×4 → 10×10 网格，
  rel L2 依次为 **2.546 → 1.538 → 1.223 → 0.985**，单调下降。
- `eddy Re=10⁴ 稀疏` 远端误差：DPS 为**目标场标准差的 1.733 倍**，
  条件模型为 **0.524**；谱尾比 45.8（DPS）vs 10.1（cond）。
  说明 DPS 不向未观测区传播信息。

### P3 — PC-DPS 与亚格子闭合

成对采样、QG 残差梯度、100 对 × 3 seeds：

| 算例 | λ=0 | 最佳 λ | 最佳 rel L2 | 下降 | 谱尾 |
|---|---:|---:|---:|---:|---:|
| eddy Re=10⁴ 稀疏 | 0.4691 | 2.0 | 0.4329 | **−7.7%** | 44.2 → 42.5 |
| eddy Re=10³ 稀疏 | 0.5009 | 1.0 | 0.4819 | −3.8% | 97.1 → 186.3（变差） |
| jet Re=10³ 全观测 | 0.0533 | 0.1 | 0.0533 | ≈0 | 基本无效 |

8% Gate **未通过**（最好 7.7%）。三帧中心差分、逐样本归一化、λ(t)、
残差低通等变体都没有超过这一结果。64×64 场是 512×512 模拟滤波后的场，
QG 残差只能作软约束；有效线性闭合把 jet 残差降低 33%，但对 eddy Re=10⁴
没有改善（[`results/p3/closure_findings.md`](results/p3/closure_findings.md)）。

### P4 — β 反演（系统辨识）

| 数据 | β 真值 | β̂ | 误差 |
|---|---:|---:|---:|
| jet Re=10³ 真值（oracle） | 2.5 | 2.7388 | 9.6% |
| jet Re=10³ PC-DPS 重建 | 2.5 | **2.6565** | **6.3%** |
| eddy Re=10³ 真值（oracle） | 0 | 0.0009 | — |
| eddy Re=10³ 稀疏 PC-DPS | 0 | 0.0254 | — |

二分类模型选择正确选中 jet（留出 RMSE 0.925 vs 2.016）。<15% 的 Gate
**通过**。ν 在 64² 下不可辨识，只报告量级。

图：`figures/fig00` … `figures/fig15`（PNG + PDF），见下方列表。

---

## 目录结构

| 路径 | 内容 |
|---|---|
| `src/common/` | 配置、指标、运行日志、数据下载、汇总 |
| `src/reproduce/` | P0 数据检查、基线推理、P1 评估与作图 |
| `src/analysis/` | P2 观测密度扫描与失败模式分析 |
| `src/pcdps/` | 可微 QG 残差、PC-DPS 采样器、残差审计、闭合拟合 |
| `src/beta_invert/` | P4 β 估计与模型选择 |
| `tests/` | 谱算子、QG 残差、指标自检（8 个测试） |
| `results/` | 每次运行的 `log.json` / 指标、CSV 表、协议记录 |
| `figures/` | 图 00–15，PNG + PDF |
| `report/` | 最终结果、程序说明、结果摘要 |
| `setup_third_party.sh` | 按固定 commit 克隆两个上游仓库 |
| `third_party/` | 上游代码与权重，**不入库**（由 setup 生成） |
| `data/` | 数据集与生成的观测，**不入库** |
| `venv/`、`weights/` | 本地环境 / 可选权重，**不入库** |

README 中的每个数字都由脚本从已保存的日志生成，没有手工修改；
`report/FINAL_RESULTS.md` 是完整结果文档。

---

## 安装

环境要求：Linux、Python ≥ 3.10、CUDA GPU（采样器直接调用 CUDA），
数据集需要约 30 GB 磁盘空间。

```bash
git clone https://github.com/SDULJY/qg-diffusion-repro.git
cd qg-diffusion-repro

bash setup_third_party.sh          # 克隆 qg-beta-sr + qg-2d（MIT，固定 commit）

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`setup_third_party.sh` 同时会得到上游 `qg-beta-sr/Weights/` 中的 12 个
预训练权重（约 167 MB）。

参考运行环境为 2× RTX 4090（24 GB）、CUDA 12.4、torch 2.5.1+cu124。
单卡也可以运行，给批量脚本传 `--gpus 0` 即可。

---

## 数据与预训练权重

数据集**不在本仓库中**。

- 数据集：Zenodo record [15742146](https://doi.org/10.5281/zenodo.15742146)
  （CC BY 4.0），文件 `data.zip`，9,401,260,503 字节，
  MD5 `0d4b3282cf8081aecb63efc0845a8405`。
- 代码与权重归档：Zenodo record
  [15750244](https://doi.org/10.5281/zenodo.15750244)（MIT）。
- 论文：[arXiv:2507.00719](https://arxiv.org/abs/2507.00719)。

下载并解压本项目使用的 12 个数组：

```bash
# 方式 A：并行 Range 下载脚本
bash src/common/download_data.sh

# 方式 B：从 Zenodo 手动下载 data.zip，放到 data/
#   data/data.zip

bash src/reproduce/extract_data.sh
python src/reproduce/check_data.py --file data/fields_pooled64_jet_re_1000.npy
```

数组形状为 `(500, 196, 1, 64, 64)`，每个约 1.5 GB。

### 观测文件

本项目恢复的 ZIP 子集中缺少 3 个发布的 `ds16` 观测成员：
两个 `eddy` **全观测**和 `jet Re=10³` **稀疏观测**。它们用论文完全一致的
`A_les_interleaved` 算子（Gaussian + cutoff + average pooling，scale=4，
gappy mask）从真值生成；同一算子对已发布的 `ds16` 文件复现误差约
1.2×10⁻⁷。

```bash
mkdir -p data/generated

# 默认使用测试帧 run=-6, snapshot=182
python src/reproduce/make_obs_from_truth.py \
  --case eddy_re1000_full  --out data/generated/eddy1000_ds16_full.npy
python src/reproduce/make_obs_from_truth.py \
  --case eddy_re10000_full --out data/generated/eddy10000_ds16_full.npy
python src/reproduce/make_obs_from_truth.py \
  --case jet_re1000_full --gappy --out data/generated/jet1000_ds16_partial.npy
```

生成文件只存一帧，索引为 `[0, 0, 0]`；P1 驱动用
`--obs-run-index -1 --obs-snapshot 0` 读取它。完整下载 `data.zip` 会包含
官方 `ds16` 文件，但驱动仍使用这些生成观测，以保证仓库中保存的结果可以
原样复现。

---

## 运行实验

以下命令都在仓库根目录、激活虚拟环境后执行。

### P0 — 数据检查

```bash
python src/reproduce/check_data.py --file data/fields_pooled64_jet_re_1000.npy
python src/reproduce/audit_alignment.py       # A_les(truth) 与发布观测对齐
```

### P1 — 基线复现网格

```bash
# 8 算例 x 4 方法（SDEdit、DPS、cond、CFG）x seeds 0,1,2
python src/reproduce/run_p1_available.py --n-samples 200 --seeds 0,1,2 --gpus 0,1

# 汇总为 CSV 并作图
python src/reproduce/evaluate.py
python src/reproduce/evaluate_first16.py       # N_ens=16，论文口径
python src/reproduce/make_figs_p1.py
```

用 `--cases eddy_re10000_full,jet_re1000_full` 可以只跑部分算例；
已有 `log.json` 的运行默认跳过，`--force` 可强制重跑。16 样本足以复现
论文表，200 样本是更强的检验。

### P2 — 失败模式

```bash
bash src/analysis/run_p2.sh
# 观测密度扫描（4x4、6x6、8x8、10x10）+ 能谱/误差-距离图
```

### P3 — PC-DPS 与闭合

```bash
pytest tests/ -q
python src/pcdps/check_residual.py --case jet_re1000_full --snapshot 182
python src/pcdps/residual_audit.py --cases eddy_re1000,eddy_re10000,jet_re1000
python src/pcdps/fit_closure.py --train-runs 0-399 --val-runs 400-479

# 单次运行（成对采样 + 仅缺口物理项 + 分位裁剪）
python src/pcdps/run_pcdps.py --case eddy_re10000_partial \
  --lambda-phys 2 --dps-c 1e-4 --physics-gap-only --grad-clip 0.999 \
  --n-pairs 100 --seed 0

# 完整 λ 扫描（3 seeds）
python src/pcdps/sweep.py --n-pairs 100 --seeds 0,1,2 --gpus 0,1
python src/pcdps/evaluate_p3.py
python src/pcdps/make_figs_p3.py
```

`run_pcdps.py --triple` 启用三帧中心差分；
`--per-sample-scale`、`--phys-t-min/max`、`--phys-lowpass`、
`--phys-late-boost` 对应结果中的其他消融。

### P4 — β 反演

```bash
python src/beta_invert/estimate_beta.py --case jet_re1000_full --source truth
python src/beta_invert/estimate_beta.py --case jet_re1000_full --source pcdps
python src/beta_invert/model_selection.py --case jet_re1000_full
python src/beta_invert/make_figs_p4.py
```

### 汇总

```bash
python src/common/collect.py        # results/master.csv
python src/common/make_summary.py   # report/results_summary.md
```

---

## 图

| 图 | 内容 |
|---|---|
| `fig00_data_sample` | 数据集样例场 |
| `fig01_recon_panels` | 重建面板（真值 / 观测 / 四种方法） |
| `fig02_rel_l2_vs_paper` | 8 算例 × 4 方法与论文的相对 L2 |
| `fig03_spectra` | 重建场与真值的能谱 |
| `fig04_stats_spread` | 涡度统计与集合 spread–skill |
| `fig05_dps_spectrum_smoothing` | 稀疏工况下 DPS 的能谱抹平 |
| `fig06_error_vs_distance` | 误差随离观测距离的变化 |
| `fig07_error_vs_density` | 误差随观测密度 |
| `fig08_pcdps_panels` | PC-DPS 重建面板 |
| `fig09_pcdps_spectra` | PC-DPS 能谱 |
| `fig10_lambda_sweep` | PC-DPS 的 λ 扫描 |
| `fig11_beta_estimate` | β 估计 |
| `fig12_beta_convergence` | β 收敛 / 模型选择 |
| `fig13_residual_spectrum` | 亚格子 QG 残差谱 |
| `fig15_pcdps_diagnosis` | PC-DPS 分尺度、分区域误差 |

所有图都由上述脚本从 `results/` 生成，没有手填数值。

---

## 协议细节

- **测试协议**：池化索引 `-6`（run 3221），快照 `182`
  （`min_t=101` + `snapshot=81`）；论文对比用 `N_ens=16`，seed 研究用 200 样本。
- **归一化**：发布数据是原始涡度，不是已归一化场。jet 用目标帧 std 同时
  归一化真值和观测；eddy 用 `field_std=1.0`。真值与观测必须共用同一尺度。
- **扩散设置**：`N_diff=64`、`N_corr=2`、Langevin 步长 0.3、SDEdit
  `t=16/64`；DPS `scale=4`、`sigma=1e-2`；CFG 权重 1.25。
- **DPS 引导强度 `C`** 逐算例选择（如 jet full `2e-5`、eddy Re=10³ full
  `1e-4`、eddy Re=10⁴ full `2e-4`、稀疏 eddy `1e-4`），写在
  `src/reproduce/run_p1_available.py` 和每次运行的 `log.json` 中。
- **QG 符号约定**：发布 jet 数据与 `−β ∂ψ/∂x` 项一致（`beta_sign=-1`）；
  eddy 的 β=0，不受影响。
- **评估**使用物理（未归一化）场；指标包括相对 L2、对数能谱误差、
  谱尾比和 spread–skill。

---

## 已知局限

1. **P1 没有全过。** 15/32 个组合偏差 <10%，严格 6/8 算例 Gate 未通过。
   `eddy Re=10³ 稀疏` 的 cond/CFG 偏高约 65%，已排除索引、掩码、算子和
   归一化 bug；最可能的原因是发布的 partial checkpoint。
2. **PC-DPS 提升有限。** 最好 7.7%，低于 8% 目标；eddy Re=10³ 提升 3.8%
   但谱尾变差；jet 全观测基本无效。
3. **QG 残差只能作软约束。** 64×64 场是 512×512 模拟滤波后的结果，
   亚格子项仍在。线性有效闭合对 jet 有帮助（残差 −33%），对 eddy Re=10⁴
   没有改善。
4. **ν 在 64² 下不可辨识**，只定量报告 β。

这些局限是结果的一部分，没有通过调测试集来掩盖。超参数只在验证/训练 run
上选择，测试 run `-6` 未参与调参。

---

## 许可、署名与引用

- **本仓库原创代码**：MIT，见 [`LICENSE`](LICENSE)。
- **上游代码**：`qg-beta-sr` 与 `qg-2d`，均为 MIT，按固定 commit 使用且
  未修改；见 [`NOTICE`](NOTICE) 以及 `setup_third_party.sh` 生成的
  `third_party/*/LICENSE`。
- **数据集**：CC BY 4.0，Zenodo
  [10.5281/zenodo.15742146](https://doi.org/10.5281/zenodo.15742146)；
  本仓库不再分发数据。
- **权重**：MIT，Zenodo
  [10.5281/zenodo.15750244](https://doi.org/10.5281/zenodo.15750244)，
  随上游仓库提供。

如果使用本代码，请同时引用论文和本仓库：

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

英文版见 [`README.md`](README.md)。完整中文结果报告见
[`report/FINAL_RESULTS.md`](report/FINAL_RESULTS.md)。
