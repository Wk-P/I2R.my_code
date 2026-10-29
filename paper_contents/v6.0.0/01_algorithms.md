# 1. 12 个变种：strategy × RL_algorithm

## 1.1 共同的问题设定

- 任务：把 M 个服务放到 N 个 ECU 上。每个实例给定容量 c_n、需求 e_m 和隐私冲突集（K = 10 个）。
- 约束：容量 —— 放在 ECU n 上的服务需求之和 ≤ c_n；隐私 —— 同一冲突集中的两个服务不能放在同一个 ECU 上。
- 目标 AR（平均资源利用率）= Σ_{合法放置 (m,n)} e_m / c_n ÷ 至少承载一个合法放置的 ECU 数。违约的放置不计入 AR。
- 序贯决策：服务按需求降序排列（`order = desc`），第 t 步为第 t 个服务选择一个 ECU；一个 episode 最多 M 步。
- `full_episode = True`（所有 run）：不提前结束；出现死局（当前服务没有可行 ECU）时：maskable 只能选 EXIT；其余 strategy 照常执行并记录违约。唯一的提前结束是 maskable 的 EXIT 动作。
- 场景：LT（N=10, M=15，资源紧张）、EQ（N=10, M=10）、GT（N=15, M=10，资源充足）。
- 折扣 γ = 1.0（所有 run），所以回报 = 一个 episode 的奖励总和。

## 1.2 四种 strategy（约束处理方式）

**maskable（论文名 Mask）**

- 动作掩码 = 对当前服务 m，同时满足容量（剩余容量 ≥ e_m）和隐私（ECU 上没有与 m 冲突的服务）的 ECU。
- 额外的 EXIT 动作（下标 N）：仅当没有任何可行 ECU 时它才有效，此时它也是唯一有效动作。选择 EXIT 会以失败奖励立即结束 episode。因此 Mask 在构造上永不违约。
- 奖励：`ar_pen`（见 1.3）。

**repair**

- 智能体选择的动作若不可行，环境把它替换为可行 ECU 中 e_m / c_n 最大者（即容量最小的可行 ECU，并列取下标最小），并计入 `repairs`。
- 若没有任何可行 ECU，则执行智能体自己的动作并记录违约（full_episode）。所以 repair 只在死局时违约。
- 替换发生在环境内部，回放缓冲 / rollout 中存的是智能体原本选的动作。奖励：`ar_pen`。

**lagrangian（标准 Lagrangian）**

- 动作按选择执行，违约照常记录。每步代价 c_t = 1[容量违约] + 1[隐私违约] ∈ {0, 1, 2}。
- 每步奖励 r_t = r_t^obj − λ·c_t，其中 r_t^obj 是 `ar_obj`（见 1.3）；λ 是对偶变量，用 Adam 更新（参数见 03_training），在一个 episode 内固定，并作为观测的最后一维（λ / (1 + λ)）输入网络。

**unconstraint（论文名 Unconstraint）**

- 不做任何约束处理：动作按选择执行，违约只被记录。奖励：完整的 `ar_pen`（v6.0.0 定义），违约导致 episode 失败、得到负奖励。
- 状态中包含可行性特征（与其他 strategy 相同，feas_in = True），但没有任何机制强制可行。

## 1.3 奖励

记 valid = 合法放置的服务数。所有中间步奖励为 0，只在最后一步给出：

- `ar_pen`（maskable / repair / unconstraint）：全部 M 个服务都合法放置时为 AR ∈ (0, 1]；否则（含 EXIT）为 −(1 − valid / M) ∈ [−1, 0)。
- `ar_obj`（lagrangian）：AR_obj = 在所有执行过放置（含违约放置）的 ECU 上，取 min(负载_n / c_n, 1) 的平均；没有失败惩罚，违约只通过每步的 −λ·c_t 体现。

## 1.4 三种 RL_algorithm

- **PPO**：on-policy 的 actor-critic（裁剪的概率比目标 + GAE）。maskable 用 sb3-contrib 的 `MaskablePPO`，其余用 stable-baselines3 的 `PPO`；策略网络都是我们的 `GraphMaskablePolicy`（02_network），actor 与 critic 共用一个编码器。
- **DQN**：off-policy 的值学习，经验回放 + 目标网络，TD 目标 r + γ·max_a' Q_target(s', a')。maskable 用我们的 `MaskableDQN`，其余用 stable-baselines3 的 `DQN`。
- **DDQN = Double DQN**（van Hasselt et al., 2016）：在线网络选出下一状态的动作 a* = argmax_a' Q_online(s', a')，目标网络给它估值，TD 目标 r + γ·Q_target(s', a*)，以减小 DQN 的 Q 值高估。maskable 用我们的 `MaskableDDQN`，其余用我们的 `DoubleDQN`（`src/shared/dqn_variants.py`，继承 SB3 `DQN`，只改 `train()`）。
- Q 网络与 PPO 策略是同一个结构（02_network）：每个 ECU token 给出 Q(s, ECU n)；maskable 另由全局 token 给出 Q(s, EXIT)。

## 1.5 12 个变种一览

| strategy | RL_algorithm | 类 | 奖励 | 动作数 | 约束处理 |
|---|---|---|---|---|---|
| maskable | PPO | `sb3_contrib.MaskablePPO` | `ar_pen` | N + 1 (EXIT) | 环境 `action_masks()` 每步给出，存入 rollout 并用于更新；网络内部也按同样的可行性把不可行 logit 置为 −1e9 |
| maskable | DQN | `MaskableDQN` | `ar_pen` | N + 1 (EXIT) | 由网络从原始观测计算（`graph_net.build`，与环境同一可行性）；ε-贪心只在可行动作中随机，贪心与 TD 目标的 max 只在可行动作上取 |
| maskable | DDQN (Double DQN) | `MaskableDDQN` | `ar_pen` | N + 1 (EXIT) | 同 DQN；TD 目标中在线网络的 argmax 只在下一状态的可行动作上取 |
| repair | PPO | `stable_baselines3.PPO` | `ar_pen` | N | 无掩码；环境替换不可行动作 |
| repair | DQN | `stable_baselines3.DQN` | `ar_pen` | N | 无掩码；环境替换不可行动作 |
| repair | DDQN (Double DQN) | `DoubleDQN` | `ar_pen` | N | 无掩码；环境替换不可行动作 |
| lagrangian | PPO | `stable_baselines3.PPO` | `ar_obj` − λ·c_t | N | 无掩码；−λ·c_t 惩罚 |
| lagrangian | DQN | `stable_baselines3.DQN` | `ar_obj` − λ·c_t | N | 无掩码；−λ·c_t 惩罚 |
| lagrangian | DDQN (Double DQN) | `DoubleDQN` | `ar_obj` − λ·c_t | N | 无掩码；−λ·c_t 惩罚 |
| unconstraint | PPO | `stable_baselines3.PPO` | `ar_pen` | N | 无 |
| unconstraint | DQN | `stable_baselines3.DQN` | `ar_pen` | N | 无 |
| unconstraint | DDQN (Double DQN) | `DoubleDQN` | `ar_pen` | N | 无 |

## 1.6 训练与评估

- 每个 run 训练 20,000,000 步（环境交互步数），训练实例为 `results/sup_diag/<场景>.json` 的 50,000 个 ILP 验证实例（p = 0.6，K = 10），每次 reset 随机抽一个。
- 总图的评估：最终模型在 v6.0.4 固定测试集（`data/testset_v6.0.4/<场景>.json`，每场景 5000 个 ILP 最优实例，与训练池不重叠）上确定性评估（`src/scripts/eval_testset_v6_0_4.py`），设置与该 run 训练时相同（Lagrangian 用训练结束时的 λ）。
- gap = 1 − mean(AR) / mean(ILP AR)；ILP AR 由 Dinkelbach 方法求得（`shared/ilp_utils.solve_ilp_max_ar`）。

### Lagrangian 训练结束时的 λ

| RL_algorithm | 场景 | 种子 1 | 种子 2 | 种子 3 |
|---|---|---|---|---|
| PPO | LT | 2.0258 | 2.3696 | 2.2095 |
| PPO | GT | 1.6404 | 0.9867 | 1.0642 |
| PPO | EQ | 1.4388 | 1.6365 | 1.7882 |
| DQN | LT | 4.6051 | 4.5378 | 4.4091 |
| DQN | GT | 4.8633 | 4.8822 | 4.9638 |
| DQN | EQ | 5.0588 | 5.1759 | 4.8931 |
| DDQN (Double DQN) | LT | 4.5790 | 5.4718 | 5.0258 |
| DDQN (Double DQN) | GT | 5.1293 | 4.9786 | 5.0154 |
| DDQN (Double DQN) | EQ | 5.0747 | 5.4400 | 5.1392 |
