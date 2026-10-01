# v4.3.1.4 — 冲突密度 0.6、统一 objective 奖励、死局即失败

v4.3.1 的子版本（tag `v4.3.1.4`），在 v4.3.1.3 统一实现（`paper_rl/`）基础上改三处。依据：v4.3.1.3 试跑结论（`../v4.3.1.3/README.md` 第 8 节）及用户决定。

## 1. 冲突密度 p：0.3 → 0.6

`python -m paper_rl.data --p 0.6 --version v4.3.1.4` → `data/v4.3.1.4/{lt,eq,gt}.yaml`（v4.3.1.3 的 p = 0.3 数据保留在 `data/v4.3.1.3/`；`paper_rl/config.DATA_VERSION` 选择数据集）。其余分布不变，三场景仍只差 N / M。

| 场景 | 平均 AR\*（p = 0.3） | 平均 AR\*（p = 0.6） |
|---|---|---|
| LT | 0.9026 | 0.8046 |
| EQ | 0.8485 | 0.7602 |
| GT | 0.8547 | 0.7919 |

原因：p = 0.3 时 LT 上无约束 PPO 的 success_rate 已达 0.97～0.985，与带约束机制的方法几乎相同，对照拉不开。

## 2. 统一 objective 奖励（新的默认 `REWARD_MODE=objective`）

所有模型共用同一个目标奖励，约束只由约束处理机制负责（Reward-Discussion："Constraint 决定动作能不能执行，Reward 决定合法决策做得好不好"）。

| 约束处理 | 中间步 | 放完 M 个服务 | 死局 |
|---|---|---|---|
| 无约束（对照） | 0 | $M\cdot AR_{exec}$ | — |
| Lagrangian | $-\lambda c_t$ | $M\cdot AR_{exec}$ | — |
| Maskable | 0 | $M\cdot AR_{exec}$ | 失败，$-M(1-\text{valid}/M)$，终止 |
| Repair | 0 | $M\cdot AR_{exec}$ | 失败，$-M(1-\text{valid}/M)$，终止 |

- $AR_{exec}$ 计入**所有执行的放置**（违规的也算，超容量时单个 ECU 利用率可 > 1），不对违规做任何惩罚。Maskable / Repair 执行的放置都合法，$AR_{exec}$ 即通常的 AR。
- 无约束对照因此"只有 objective"，与设计本意一致。**更正**：v4.2.0 的无约束 PPO 实际并非如此——其代码（tag `v4.0.0_final_1`，`scenarios/*/ppo/env.py`）在有违规时终局走失败分支 $-M(1-\text{valid}/M)$，v4.2.0 `formula.md` 4.1 节"违规只记录不惩罚"只对中间步成立。
- 评估指标不变：AR 只计合法放置；违规单独报告。
- `legacy / ar / directional` 仍保留，仅用于复现 v4.3.1.3。

## 3. 死局即失败

Maskable（下一服务无可行 ECU）与 Repair（无可修复 ECU）在**所有**奖励模式下立即终止、success = 0、给失败惩罚（directional 为 $-C$），不再执行违规放置。v4.3.1.3 中 legacy / ar 下 Maskable 死局时放开掩码执行一次违规放置并继续（LT 上 2%～4% 的回合因此出现违规），该行为取消。

## 4. 自检

- 3 场景 × 4 机制 × {objective, ar} 各 200 个随机 episode：Maskable / Repair 违规回合 0；无约束 objective 回合回报恰好等于 $M\cdot AR_{exec}$（0 处不一致）；违规放置可使 $AR_{exec}$ 达 1.89。
- p = 0.6 下随机策略：LT 的 Maskable / Repair 死局率 44% / 55%（约束明显变紧）；EQ / GT 仍几乎不出现死局。
- 12 个模型在 LT 上各训练 2 万步跑通全流程。

## 5. 试跑

- 脚本：`scripts/run_v4.3.1.4_pilot.py`；12 个模型 × 3 场景 × objective × 种子 1 × 1M 步 = 36 个任务。
- 记录：`scripts/logs/v4.3.1.4_pilot/manifest.json`（含代码提交）；结果 `results/unified/<scen>/<algo>/<exp_id>/`；报告 `pilot_report.md`（本目录）。

（结果待补充）
