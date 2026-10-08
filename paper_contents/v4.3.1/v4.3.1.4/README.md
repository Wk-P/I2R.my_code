# v4.3.1.4 — 冲突密度 0.6、统一 objective 奖励、死局即失败

v4.3.1 的子版本（tag `v4.3.1.4`），在 v4.3.1.3 统一实现（`src/paper_rl/`）基础上改三处。依据：v4.3.1.3 试跑结论（`../v4.3.1.3/README.md` 第 8 节）及用户决定。

## 1. 冲突密度 p：0.3 → 0.6

`python -m paper_rl.data --p 0.6 --version v4.3.1.4` → `data/v4.3.1.4/{lt,eq,gt}.yaml`（v4.3.1.3 的 p = 0.3 数据保留在 `data/v4.3.1.3/`；`src/paper_rl/config.DATA_VERSION` 选择数据集）。其余分布不变，三场景仍只差 N / M。

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

- 脚本：`src/scripts/run_v4.3.1.4_pilot.py`；12 个模型 × 3 场景 × objective × 种子 1 × 1M 步 = 36 个任务。
- 记录：`logs/v4.3.1.4_pilot/manifest.json`（含代码提交）；结果 `results/unified/<scen>/<algo>/<exp_id>/`；报告 `pilot_report.md`（本目录）。

**已完成**（2026-10-02 05:10 – 05:35，36/36，无失败）。完整表格见 `pilot_report.md`。

### 结果摘要（单种子、1M 步）

success_rate（括号内为 AR/AR\*，只在成功 episode 上计算）：

| 模型 | 约束处理 | LT | EQ | GT |
|---|---|---|---|---|
| PPO | 无约束 | 0.000 | 0.000 | 0.000 |
| PPO | Lagrangian | **0.000** | 0.998（0.940） | 0.995（0.955） |
| PPO | Maskable | **0.907**（0.845） | 0.998（0.938） | 1.000（0.957） |
| PPO | Repair | 0.752（0.898） | 0.995（0.929） | 1.000（0.953） |
| DQN | 无约束 | 0.000 | 0.000 | 0.000 |
| DQN | Lagrangian | 0.830（0.868） | 1.000（0.922） | 0.995（0.923） |
| DQN | Maskable | 0.820（0.874） | 1.000（0.926） | 1.000（0.923） |
| DQN | Repair | 0.623（0.888） | 0.998（0.926） | 1.000（0.934） |
| DDQN | 无约束 | 0.000 | 0.000 | 0.000 |
| DDQN | Lagrangian | 0.812（0.870） | 1.000（0.926） | 0.978（0.928） |
| DDQN | Maskable | 0.812（0.864） | 1.000（0.916） | 1.000（0.925） |
| DDQN | Repair | 0.578（0.887） | 0.998（0.916） | 1.000（0.937） |

### 分析

1. **无约束对照：三个模型、三个场景全部 100% 违规**（容量与冲突违规率均为 1.000）。用模型实测：所有服务都放到**同一个 ECU** 上（测试集前 100 个实例平均使用 1.00 个 ECU），$AR_{exec}$ 达 5.0～11.7。只优化 objective 时，"全部堆到一个 ECU"使 $\sum n_i/e_j$ 最大——这正是不做约束处理的后果，可作为 Safe RL 必要性的直接证据。
2. **Lagrangian-PPO 在 LT 上失败**（success 0，100% 违规，同样退化为单 ECU 堆叠），$\lambda$ 卡在上限 $\lambda_{max}=5$。原因是尺度：违规堆叠的终局奖励可达 $15\times11.7\approx175$，而每步违规代价至多 $\lambda\cdot2=10$，λ 封顶后惩罚不足以抵消。同样设定下 Lagrangian-DQN / DDQN 在 LT 上成功（0.83 / 0.81，但仍有 8%～14% 违规回合，最终 λ≈3.1），EQ / GT 上 Lagrangian-PPO 也正常（λ 5.0 / 3.35）。→ $\lambda_{max}$ 需要按 objective 尺度设定，或对代价做归一化。
3. **LT 现在能区分约束处理机制**：Maskable 的 success_rate 最高（PPO 0.907，DQN/DDQN 0.82/0.81），Repair 最低（0.58～0.75）但 AR/AR\* 最高（0.887～0.898）。Repair 的最佳适应（选利用率最高的 ECU）把 ECU 填得更满，单步更优，但更容易把后面的服务逼进死局（死局率 25%～42%，Maskable 9%～19%）。
4. **EQ / GT 仍然区分度很低**：所有带约束机制的方法 success_rate 0.978～1.000，AR/AR\* 在 0.916～0.957 之间，同一场景内差距不到 2.5 个百分点，单种子噪声量级。原因是 N ≥ M 且总容量约为需求的 2 倍。
5. **学习算法**：LT 上 Maskable-PPO 明显好于 Maskable-DQN/DDQN（0.907 vs 0.82）；EQ / GT 上 PPO 的 AR/AR\* 略高于 DQN 系（0.94～0.96 vs 0.92～0.94）。DQN 与 DDQN 之间差别很小。
6. **口径注意**：AR/AR\* 只在成功 episode 上算，不同方法成功的实例集合不同（LT 上 Repair 只在较容易的 58%～75% 实例上成功），直接比较会偏向成功率低的方法。正式实验建议增加"失败计 0"的口径（AR/AR\* × success）。

### 待决定

- Lagrangian 的 $\lambda_{max}$ / 代价归一化（第 2 点）。
- EQ / GT 是否要收紧（例如降低容量或提高 p），否则这两个场景只能用来说明"资源充足时各机制差别不大"。
- 之后进入 12 个模型 × 3 场景 × 3 种子 × 5M 的正式实验。
