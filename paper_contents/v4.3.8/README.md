# v4.3.8 Maskable 加 EXIT 动作

在 v4.3.7（统一奖励、不提前结束）基础上，只改 Maskable：

- 动作空间为 [ECU1, …, ECUn, EXIT]。
- 只要还有合法 ECU，EXIT 被屏蔽，智能体只能在合法 ECU 中选，**永不违约**。
- 一个合法 ECU 都没有时，EXIT 是唯一合法动作；选后回合结束，得失败奖励 −(1 − valid/M) ∈ [−1, 0)。负奖励沿轨迹回传到前面"当时合法、但把后面逼进死路"的选择，模型要学会避开。
- mask 永远至少有一个合法动作，不再有全 False 的 mask，也不再强放。
- 测试：Maskable 的 capacity / privacy 违约率恒为 0，另报 EXIT 率（选了 EXIT 的测试实例比例）；AR / ILP AR / AR gap 在无违约且未 EXIT 的实例上算。
- 其他机制（无约束、Lagrangian、Repair）与 v4.3.7 完全相同。

运行：`scripts/run_v4.3.8.py --pilot`（只重训 Maskable PPO / DQN × 3 场景 × 种子 1 × 1M 步，其余 4 个模型取自 v4.3.7 试跑），报告 `pilot_report.md`。

## 状态（观测）

v4.3.8 用原观测（`obs = base`，不含 v4.3.4 的冲突图），与 v4.3.1.6 起各版本相同，EXIT 动作没有改观测。代码：`paper_rl/env.py` 的 `_obs()`。

记号：N 个 ECU、M 个服务（按需求降序排好，第 t 步放第 t 个），c_j 为 ECU j 容量，r_j 为剩余容量，d_i 为服务 i 需求，c_max = max_j c_j，当前服务 i = t。

维数 = 6 + 5N + 2M + 1：LT（N=10, M=15）87 维，EQ（N=10, M=10）77 维，GT（N=15, M=10）102 维。

| 序号 | 分量 | 长度 | 含义 |
|---|---|---|---|
| 1 | d_i / c_max | 1 | 当前服务的需求 |
| 2 | AR | 1 | 目前已合法放置部分的 AR |
| 3 | Σ_j max(r_j, 0) / Σ_j c_j | 1 | 全部 ECU 的剩余容量占比 |
| 4 | Σ_{k≥t} d_k / Σ_j c_j | 1 | 还没放的服务总需求占比 |
| 5 | 可行 ECU 比例 | 1 | 当前服务能合法放进去的 ECU 占 N 的比例 |
| 6 | (M − t) / M | 1 | 还剩多少服务没放 |
| 7 | c_j / c_max | N | 每个 ECU 的容量 |
| 8 | clip(r_j / c_max, −1, 1) | N | 每个 ECU 的剩余容量 |
| 9 | 冲突标志 | N | ECU j 上已有和当前服务冲突的服务为 1 |
| 10 | 允许度 | N | 1 − （与 ECU j 上已放服务冲突、且未放在 j 上的服务数）/ M；空 ECU 为 1 |
| 11 | 可行标志 | N | 当前服务放进 ECU j 既不超容量也不冲突为 1（即 mask；MaskableDQN 从这里读 mask） |
| 12 | 剩余服务需求 | M | 第 k 个服务未放则为 d_k / c_max，已放为 0 |
| 13 | 剩余服务可行度 | M | 第 k 个服务未放时，它在当前状态下可合法放入的 ECU 比例；已放为 0 |
| 14 | λ / λ_max | 1 | 仅 Lagrangian 有值（λ_max = 50），其他机制为 0 |

所有服务放完（t = M）或选 EXIT 后的终止状态里，1、5 和 9、11 两组为 0。

EXIT 的 mask 不在观测里单独占一维：Maskable PPO 从 `action_masks()` 取 [可行标志, 无可行 ECU]，MaskableDQN 用第 11 组可行标志，并在全为 0 时打开 EXIT。
