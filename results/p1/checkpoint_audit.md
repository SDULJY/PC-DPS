# P1 checkpoint / 测试轨迹审计（eddy Re=1000 partial）

## 1. 索引对齐：已排除

`src/reproduce/audit_alignment.py`，run=-6、snapshot=182：

| case | rel(A_les(truth), released partial) |
|---|---:|
| eddy1000 partial | **0.0** |
| eddy10000 partial | **0.0** |

结论：truth 与 partial 文件的 batch/time 索引完全一致，65% 偏差不是索引错位。

## 2. 同一 checkpoint 在训练 run 上的误差（cond，16 样本，normalize-target）

| run | rel L2 (per-sample) | ens-mean |
|---:|---:|---:|
| 0 | 0.2458 | 0.2109 |
| 100 | 0.2630 | 0.2459 |
| 200 | 0.1353 | 0.1017 |
| 300 | 0.1329 | 0.0976 |
| 400 | 0.3813 | 0.3677 |
| -6（测试） | 0.4458 | 0.4309 |

- 同一权重在训练 run 上可好到 0.13，也可差到 0.38；模型对场非常敏感。
- 论文 0.247 落在训练 run 的范围内（run 0/100 接近），但测试 run -6 是较难的场。

## 3. 多快照平均（run -6，cond，normalize-target）

snapshots 101,110,...,195 的 rel L2 在 0.286–0.467，平均约 **0.359**。
不是论文 0.247；因此“论文对多个快照平均”也不能解释。

## 4. 候选测试轨迹（run 0 / -2 / -19，snapshot 182，16 样本）

论文 Table 5（eddy1000 partial）：M1 0.775，M2 0.443，M3 0.247，M4 0.239。

| run | M1 SDEdit | M2 DPS | M3 cond | M4 CFG |
|---:|---:|---:|---:|---:|
| 0 | 0.7028 | 0.4887 (C=2e-5) | 0.2458 | 0.2322 |
| -2 | 0.7117 | 0.3010 (C=1e-4) | 0.1790 | 0.1770 |
| -19 | 0.7070 | 0.3438 (C=2e-5) | 0.2614 | 0.2300 |

- run 0：M1/M3/M4 与论文差 <10%，M2 差 10.3%（C=2e-5）；非常接近。
- run -19：M1/M3/M4 差 <10%，M2 差 15–22%。
- 但没有任何 run 在固定 C 下同时匹配四方法；且 run 0/-19 很可能是训练 run，
  没有独立证据表明它们是论文测试轨迹。

## 5. 权重版本

`git log --all -- Weights/model_cond_ds16_partial_eddy_Re_1000.pt` 只有 `Initial commit`；
tag v1.0.0 也没有其他权重版本。无法通过旧 checkpoint 解释。

## 6. 结论（诚实边界）

1. 不是索引/掩码/观测算子 bug：对齐误差为 0，算子误差 1e-7。
2. 不是整体 checkpoint 失效：训练 run 上可达 0.13。
3. 65% 偏差最可能来自 **测试轨迹/发布 partial checkpoint 与论文训练配置的差异**：
   - 论文的测试场若不在当前 500 条轨迹的 -6 索引，或论文使用了不同的 partial 权重，
     我们无法从发布数据中唯一恢复；
   - run 0/-19 的巧合说明“换一条轨迹”可以接近论文值，但这属于挑测试集，不采用。
4. 正式结果仍以 run=-6、snapshot=182 为准，eddy1000 partial 的 M3/M4 标为**复现异常**，
   不修改、不挑选。

## 7. 复现命令

```bash
python src/reproduce/audit_alignment.py
python src/reproduce/run_inference.py --case eddy_re1000_partial --method cond \
  --run-index 0 --snapshot 182 --normalize-target --n-samples 16
```
