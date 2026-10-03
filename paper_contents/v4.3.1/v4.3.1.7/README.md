# v4.3.1.7 — ILP 基准改为真正的 AR 最优（Dinkelbach），重新评估 + 求解耗时对比

v4.3.1 的子版本（tag `v4.3.1.7`）。**不重新训练**：v4.3.1.6 的 108 个模型（12 模型 × 3 场景 × 3 种子 × 5M）全部有效，只更换 ILP 基准并重新评估。

## 1. 发现的问题

此前所有"ILP AR"（含数据集中的 `ar_star`、各训练脚本输出的 ILP AR、v4.2.0 的 `ilp_timing.py`）都来自 `shared.ilp_utils.solve_ilp`。它最大化的是**总利用率** $\sum_{ij}x_{ij}n_i/e_j$，再除以启用 ECU 数，**不是 AR 的最大值**。真正最大化 AR 的 Dinkelbach 实现一直在 `scenarios/*/ilp/optimal_solution/main.py`，但未被调用（v4.2.0 `formula.md` 描述的是 Dinkelbach，与实际使用的求解器不符）。

| 场景 | 旧 AR\*（总利用率 ILP） | 真正最优 AR\*（Dinkelbach） | 差 |
|---|---|---|---|
| LT | 0.8046 | 0.8800 | +0.075 |
| EQ | 0.7602 | 0.8580 | +0.098 |
| GT | 0.7919 | 0.9100 | +0.118 |

（全部 2000 个实例的均值；81%～94% 的实例上 Dinkelbach 更高，单实例最大差 0.32。）

影响：v4.3.1.x 中所有 AR/AR\* 偏高（例：LT Maskable-PPO 0.895 → 0.814）；训练本身不受影响（legacy 奖励不使用 AR\*）。

## 2. 修正

- `shared.ilp_utils.solve_ilp_max_ar(caps, reqs, conflict_sets)`：Dinkelbach，每轮解 $\max\sum x_{ij}n_i/e_j-\lambda\sum_j y_j$（约束与原 ILP 相同，另加 $x_{ij}\le y_j$、$\sum_i x_{ij}n_i\le e_jy_j$），$\lambda\leftarrow F/G$ 直到 $|\Delta\lambda|<10^{-6}$。与原 `main.py` 实现对照 45 个实例，0 处不一致。`solve_ilp` 保留并加注释说明其不是 AR 最优。
- `scripts/recompute_ar_star.py v4.3.1.4`：`data/v4.3.1.4/*.yaml` 的 `ar_star` 改为 Dinkelbach 值，旧值保留为 `ar_star_total_util`；实例内容不变。
- `scripts/eval_v4.3.1.7.py`：108 个模型在各自种子测试集（400 实例）上重新评估（与存储的 success_rate 逐一一致），并做 ILP / RL 计时；两者都在 16 个单线程进程并行的相同条件下测量。

输出：`report.md`（完整表格）、`summary.json`、`eval_raw.jsonl`（每个实例的 AR、AR\*、成功、违规、死局、耗时）、`ilp_timing.json`。

## 3. 结果（3 种子均值）

| 场景 | ILP 最优 AR\* | ILP 耗时 ms/实例（均值 / 中位数 / 最大） |
|---|---|---|
| LT | 0.8792 | 3137 / 1440 / 45500 |
| EQ | 0.8529 | 475 / 283 / 6021 |
| GT | 0.9094 | 696 / 429 / 5194 |

综合 AR/AR\*×success（失败计 0）与耗时：

| 模型 | 约束处理 | LT | EQ | GT | 耗时 ms（LT / EQ / GT） |
|---|---|---|---|---|---|
| PPO | 无约束 | 0.682 | 0.827 | 0.817 | 11.6 / 7.3 / 7.6 |
| PPO | Lagrangian | 0.697 | 0.820 | 0.814 | 11.6 / 7.3 / 7.6 |
| PPO | Maskable | **0.708** | 0.829 | **0.835** | 16.5 / 11.1 / 10.8 |
| PPO | Repair | 0.659 | 0.826 | 0.832 | 11.7 / 7.7 / 7.6 |
| DQN | 无约束 / Lagrangian | 0.220 / 0.147 | 0.470 / 0.377 | 0.411 / 0.348 | ~8.6 / 5.3 / 5.6 |
| DQN | Maskable / Repair | 0.687 / 0.641 | 0.817 / **0.832** | 0.814 / 0.831 | ~9.2 / 5.7 / 6.1 |
| DDQN | 无约束 / Lagrangian | 0.225 / 0.068 | 0.468 / 0.386 | 0.458 / 0.287 | ~8.6 / 5.3 / 5.5 |
| DDQN | Maskable / Repair | 0.675 / 0.648 | 0.815 / 0.828 | 0.810 / 0.829 | ~8.9 / 5.8 / 5.7 |

成功 episode 上的 AR/AR\*：所有带硬性机制的方法与 PPO 系在 0.79～0.83 之间，即 AR 比 ILP 最优低约 0.15～0.18（AR gap）；DQN/DDQN 的 Lagrangian 只有 0.54～0.71。

## 4. 结论

1. **质量**：最好的 RL（Maskable-PPO）在成功实例上达到 ILP 最优 AR 的 81%（LT）～83%（EQ/GT），AR gap 约 0.15～0.17。之前报告的 0.89～0.96 是相对错误基准的结果，应作废。
2. **速度**：RL 每个实例 5～17 ms，ILP 475～3137 ms（LT 最长 45 s）；RL 快 **43～370 倍**（LT 190～370×，EQ 43～90×，GT 65～128×）。ILP 耗时随问题规模（LT：M = 15）显著上升且方差极大，RL 耗时稳定。
3. 方法间的排序与 v4.3.1.6 结论一致（排序不受基准变化影响，因为同一实例的 AR\* 对所有方法相同）：Maskable-PPO 综合最优且零违规；无硬性机制时 PPO ≫ DQN/DDQN，有 Maskable / Repair 时三者接近；Lagrangian 对 DQN/DDQN 有害。
4. 违规率口径：本报告按"至少一项违规"计（LT 无约束 PPO 15.6%，v4.3.1.6 报告中取 max(容量, 冲突) 为 13.2%，属下界）。

## 5. 仍需注意

- v4.3.1.3 的数据（`data/v4.3.1.3/`）中的 `ar_star` 未重算（该版本已不用于论文）。
- v4.2.0 论文中的 ILP 数字同样来自总利用率 ILP，引用前需按本版本口径替换。
