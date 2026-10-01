# v4.3.1.2 — 补齐 12 个模型 + 奖励选型试跑

v4.3.1 的子版本（tag `v4.3.1.2`）。目的：在跑 12 个模型的 5M × 3 种子正式实验之前，先用 1M 步试跑选定奖励公式。

## 1. 12 个模型（约束机制 × 学习算法）

| | 无约束 | Mask | Lagrange | Repair |
|---|---|---|---|---|
| PPO | `ppo` | `ppo_mask` | `ppo_lagrangian` | `ppo_opt` |
| DQN | `dqn` | `mask_dqn` | `lagrange_dqn`（新增） | `repair_dqn` |
| DDQN | `ddqn` | `mask_ddqn` | `lagrange_ddqn`（新增） | `repair_ddqn` |

同一列复用同一个环境（Mask → `ppo_mask/env.py`，Lagrange → `ppo_lagrangian/env.py`，Repair → `ppo_opt/env.py`），所以同一约束机制下 PPO / DQN / DDQN 之间只差学习算法；DQN 与 DDQN 的超参数相同。

**Lagrange-DQN / Lagrange-DDQN**（`shared/dqn_variant_runner.py`，`variant="lagrange"`）：

- 环境为 Lagrange-PPO 的 `LagrangeEnv`，$\lambda$ 归一化后进入观测。
- 对偶上升与 Lagrange-PPO 相同，常数从同场景 `ppo_lagrangian/config.py` 复制（`LAMBDA_INIT / LR / TARGET / MAX / UPDATE_WINDOW / WARMUP_EPISODES`）：预热 $E_w$ 个 episode 后，每 $W$ 个 episode
  $\lambda\leftarrow\mathrm{clip}\big(\lambda+\eta_\lambda(\bar\nu-\nu^*),0,\lambda_{\max}\big)$，$\bar\nu$ 为最近 $W$ 个 episode 的平均违规率。
- 评估时 $\lambda$ 固定为训练结束时的值（`results.json` 的 `training.final_lambda`）。
- off-policy 的特点：经验池里的旧样本保留采集时 $\lambda$ 下的奖励，不随 $\lambda$ 更新重算。

## 2. 本版本的代码修改

1. **Lagrange-PPO / Repair-PPO 接入 `REWARD_MODE`**：v4.3.1 只给 PPO、Mask-PPO、DQN、DDQN 接了 `ar` / `ratio`，Lagrange-PPO 和 Repair-PPO 的终局奖励仍是固定的 $M(2AR-1)$。v4.3.1 的试跑不涉及这两个算法，所以不受影响；本版本补齐，现在 6 类环境 × 3 场景都可切换。
2. **directional 的 $\Delta AR$ 只按合法执行的放置计算**：冒烟测试中 Lagrange-DQN 的测试 AR 出现 4.9～5.9（>1）。原因是 PPO / Lagrange 环境把违规（超容量）放置的利用率也计入了环境自身的 AR，而 DQN 环境计为 0，各环境口径不一致；directional 原先用环境自身的 AR 求 $\Delta AR$，会直接奖励超容量放置。现在由 `directional_step()` 自行维护"合法放置 AR"：
   - 违规步 $\Delta AR=0$，只由各方法的约束机制处理，对应 Reward-Discussion"Reward 决定合法决策做得好不好"；
   - Repair-* 执行的（修复后的）放置一律算合法；
   - 已验证：18 个环境随机策略下该 AR 恒 ≤ 1，且在无违规 episode 上与环境自身的 AR 完全一致。
3. 面板中"变体"改称"奖励模式"；12 个模型按 约束机制 × 学习算法 排列。

## 3. 奖励选型试跑

- 脚本：`scripts/run_v4.3.1.2_reward_pilot.py`（支持 `--resume`）
- 规模：12 个模型 × 3 场景 × {`legacy`, `ar`, `directional`} × 种子 1，1M 步，共 108 个任务。不再跑 `ratio`：v4.3.1 试跑中它与 `ar` 几乎没有差别，且训练时需要 ILP 最优值。
- directional 参数为默认值：$\beta=10$，$\lambda_d=1$，$\epsilon=10^{-3}$，$B=C=M$。
- 记录：manifest `scripts/logs/v4.3.1.2_reward_pilot/manifest.json`，每个任务日志在同一目录；结果在 `results/final_paper_experiments/<scen>/<algo>/<exp_id>/`；模型文件名带 `v4.3.1.2-pilot-<奖励>`。
- 报告：`reward_pilot_report.md`（本目录，任务全部结束后自动生成）。

**选型标准**（用于决定 5M 正式实验的奖励）：

1. 带约束机制的方法 success_rate 不下降；
2. Mask / Repair 方法的 AR 不明显下降（这两类执行的放置都合法，AR 可比）；
3. 4 种约束机制之间的排序在三种奖励下是否稳定；若不稳定，需要在论文中说明奖励对结论的影响。

## 4. 结果

（试跑结束后补充）
