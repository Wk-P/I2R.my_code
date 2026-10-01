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

## 4. 结果：试跑已作废（2026-10-01 23:30 停止，108 个任务完成 6 个，均为 LT）

停止原因（审计结论）：

1. **三个场景的代码和配置不一致**，违反"LT / EQ / GT 只有 SVC、ECU 数量不同"的要求。各算法 `env.py` / `run_all.py` 在场景之间相差数十到数百行，主要差异：
   - 服务降序重排：LT 的 Mask/Lagrange/Repair-PPO 与 DQN/DDQN 有，EQ 只有 Mask/Lagrange/Repair-PPO，GT 全无；
   - 观测：`ecu_allowed_frac` 只在 LT 全部算法 + EQ/GT 的 Mask-PPO 中出现；
   - Mask-PPO：一步前瞻掩码、AR 课程权重 $w$ 只在 LT；熵系数 LT 0.02→0.002、EQ/GT 0.005；EQ 价值网络 [512,512]，其余 [256,256]；
   - Lagrange-PPO：$\lambda$ 的初值/步长/上限/窗口/预热三场景全不同；逐步奖励 LT 为 0、EQ 含利用率增益 + 容量惩罚 + $\lambda$ 冲突惩罚、GT 含容量惩罚 + $\lambda$ 冲突惩罚；
   - Repair-PPO：$\gamma$ LT/GT 0.999、EQ 0.99；LT 触发过修复时终局奖励为 0，EQ/GT 不是；
   - DQN/DDQN：`learning_starts` LT/GT 2000、EQ 64；探索比例 LT/GT 0.5、EQ 0.1。
2. **EQ / GT 的场景数据几乎不允许服务共用 ECU**。三个场景用同一个生成器（每个实例 10 个冲突集、集大小在 2～M 之间随机），但冲突集大小与 M 挂钩：

   | 场景 | N / M | 平均冲突集大小 | 服务两两之间存在冲突的比例 |
   |---|---|---|---|
   | LT | 10 / 15 | 5.2 | 74% |
   | EQ | 10 / 10 | 6.0 | 99% |
   | GT | 15 / 10 | 6.0 | 99% |

   EQ/GT 中几乎任意两个服务都冲突，实际上退化为"一个 ECU 只能放一个服务"，与"非 privacy conflict 的服务可以部署在同一 ECU"的问题设定不符。这也解释了 EQ/GT 上 Mask/Repair 方法 success_rate 恒为 100%、AR 只有 0.54～0.62。
3. 冲突判定本身三场景一致且与 ILP 相同（同一冲突集内的服务不能共用 ECU，其余服务在容量允许时可共用），已用随机放置对 15 个环境逐步核对，0 处不一致。

后续：统一实现与场景数据后重新试跑（见 v4.3.1.3）。
