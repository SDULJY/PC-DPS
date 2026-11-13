# 校准结论（2026-11-12）

## 测试轨迹 / 快照
- 数据集 500 条轨迹从 run 2727 开始：训练 2727–3220（494 条），测试 run 3221 是第 494 条，
  即池化 batch 索引 **-6**（不是 -1）。
- 池化 npy 保留完整 0..195 时间轴；测试快照 = `min_t 101 + snapshot 81` = **182**。
- 验证：jet Re=1000 全观测、run=-6、snap=182、目标场 std 归一化，
  条件模型 ensemble-mean rel L2=0.0182（论文 0.019±0.001），DPS=0.0466（论文 0.045±0.007）。

## 归一化（关键）
- 发布的数据是**原始涡度**，不是已归一化场。全数据 std 约 1.7（jet）/~0.9（eddy），
  测试帧 std 会随帧变化。
- 代码里的 `field_std=1.3` 不能直接用于所有算例：
  - jet Re=1000：用**目标帧 std**（`--normalize-target`，帧 std≈2.086）最稳；
    固定 1.3 时 cond 从 0.022 恶化到 0.131。
  - eddy Re=1000/10000：`--field-std 1.0` 最稳；eddy10000 全观测 cond=0.148（论文 0.152）。
- 结论：真值与观测必须共用**同一个尺度**；不要分别按各自 std 归一化（per-sample 会从 0.03 恶化到 0.22）。
- 论文“zero mean/unit variance”应按目标场的统计量执行；released data 需要重新标准化。

## 观测算子
- 用 `A_les_interleaved`（Gaussian+cutoff+avg_pool，scale=4）从真值生成的 ds16 场，
  与发布 ds16 文件的相对误差 ≈1.2e-7（jet full 和 eddy partial 都验证）。
- partial 零值比例 0.5625 = 三段 20% 缺口（36/64），与代码 `gappy=True` 完全一致。

## DPS 引导强度 C
- 论文/配置 C=2e-5 只对 jet 合适；eddy 需要按算例调：
  - jet Re=1000 full：C=2e-5，DPS=0.049（论文 0.045）。
  - eddy Re=1000 full：C=1e-4，DPS=0.309（论文 0.265，约 17% 高）。
  - eddy Re=10000 full：C=2e-5 会数值发散（rel L2 2.4~1e22）；
    C=2e-4 稳定到 **0.318（论文 0.322）**。
  - eddy Re=1000 partial：C=2e-5，DPS=0.458（论文 0.443，3.4%）。
  - eddy Re=10000 partial：C=2e-5，DPS=0.449（论文 0.642；我们更低）。
- 这说明论文“best-tuned guidance strength”确实是逐算例的。

## 尚未解释的偏差
- **partial 条件模型**（M3/M4）明显差于论文：
  eddy Re=1000 partial cond=0.408（论文 0.247），eddy Re=10000 partial cond=0.342（论文 0.313）。
  - 观测算子/掩码已验证一致；全观测模型+稀疏输入更差（0.79），partial 模型是正确权重。
  - 扫描 run=-20..-1：eddy1000 partial cond 在 0.17–0.40 间波动，run=-6 为 0.399；
    没有一个 run 同时匹配论文的四个方法值。
  - 结论：很可能是 released partial checkpoint 与论文 Table 5 的训练配置存在差异，
    或论文 partial 测试帧不同。报告中如实标注，不强行调参。
