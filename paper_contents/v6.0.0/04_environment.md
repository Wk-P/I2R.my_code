# 4. 环境：变量、观测（状态）、动作空间

代码：`src/paper_rl/env.py` 的 `PlacementEnv`（gymnasium.Env），12 个变种共用。

## 4.1 实例（每次 reset 抽一个）

| 变量 | 含义 | 分布（数据生成） |
|---|---|---|
| c_n | ECU n 的容量（容器数），N 个 | uniform{50, 55, …, 195} |
| e_m | 服务 m 的需求（容器数），M 个 | uniform{10, 15, …, 95} |
| 冲突集 | K = 10 个；同一集合中的两个服务不能同 ECU | 每个服务以概率 q 独立加入每个集合，少于 2 个成员的集合丢弃；两服务冲突概率 p = 1 − (1 − q²)^K = 0.6 |
| ar_star | 该实例的 ILP AR（Dinkelbach），只用于记录与评估，不进入观测 |  |

所有实例都经 ILP 解出最优（Optimal）才保留。

## 4.2 环境内部变量（每个 episode）

| 变量 | 含义 |
|---|---|
| t | 当前步 = 当前要放置的服务（按需求降序后的位置），0 … M |
| remaining_n | ECU n 的剩余容量 c_n − 负载（违约时可为负） |
| hosted_n | ECU n 上已执行放置的服务集合（含违约放置） |
| partners_m | 与服务 m 至少共处一个冲突集的服务 |
| legal_ru / legal_ecus | 合法放置的 e_m / c_n 之和、承载合法放置的 ECU 集合 → AR |
| exec_ru / exec_ecus | 所有执行放置（含违约）→ AR_obj（lagrangian 奖励） |
| valid_placed | 合法放置的服务数 |
| cap_violations / conflict_violations | 容量 / 隐私违约次数 |
| repairs | repair 替换动作的次数 |
| dead_end | 是否出现过「当前服务没有可行 ECU」 |
| exited | 是否以 EXIT 结束（只有 maskable） |
| λ | 对偶变量（只有 lagrangian），episode 内固定 |

## 4.3 观测（observation）与状态（state）

- 环境输出的观测就是完整状态（完全可观测、马尔可夫）：`obs_mode = raw`（命令行虽写 `--obs base`，`--net graph` 会强制改为 raw，results.json 记录为 raw）。
- 观测向量 = [c (N) | e，按放置顺序 (M) | 冲突图邻接矩阵 (M × M) | 每个服务的 ECU 编号 + 1，0 = 未放置 (M) | t (1)]，lagrangian 再在末尾加 λ / (1 + λ) (1)。
- 观测空间：`gymnasium.spaces.Box(low=-1, high=1e4, float32)`，数值未归一化（容量与需求是原始容器数）；归一化在网络内部做（02_network §2.2）。
- 关于「observer」：代码里没有单独的 observer 类。观测由 `PlacementEnv._raw_obs()` 生成；特征提取（可行性、负载、关系矩阵等）由网络的 `build()` 完成。PPO 与 DQN / DDQN 收到的观测和做的特征提取完全相同。

## 4.4 动作空间

- 离散动作空间 `gymnasium.spaces.Discrete`（不是连续的）：动作 a ∈ {0, …, N − 1} = 把当前服务放到 ECU a；maskable 多一个 EXIT 动作 a = N。
- 每步只决定一个服务放哪里（`action_mode = ecu`），服务顺序固定为需求降序。

## 4.5 各场景、各 strategy 的维度（由环境实例化得到）

| 场景 | strategy | N | M | 观测维度 | 动作空间 |
|---|---|---|---|---|---|
| LT | maskable | 10 | 15 | 266 | Discrete(11) |
| LT | repair | 10 | 15 | 266 | Discrete(10) |
| LT | lagrangian | 10 | 15 | 267 | Discrete(10) |
| LT | unconstraint | 10 | 15 | 266 | Discrete(10) |
| GT | maskable | 15 | 10 | 136 | Discrete(16) |
| GT | repair | 15 | 10 | 136 | Discrete(15) |
| GT | lagrangian | 15 | 10 | 137 | Discrete(15) |
| GT | unconstraint | 15 | 10 | 136 | Discrete(15) |
| EQ | maskable | 10 | 10 | 131 | Discrete(11) |
| EQ | repair | 10 | 10 | 131 | Discrete(10) |
| EQ | lagrangian | 10 | 10 | 132 | Discrete(10) |
| EQ | unconstraint | 10 | 10 | 131 | Discrete(10) |

## 4.6 一步的执行顺序（`step`）

1. maskable 选 EXIT：标记 exited，给失败奖励，episode 结束。
2. repair：动作不可行且存在可行 ECU 时替换为 e_m / c_n 最大的可行 ECU；没有可行 ECU 时保留原动作（记 dead_end）。
3. 判定违约：容量（remaining_n < e_m）、隐私（ECU 上有冲突伙伴）；执行放置并更新 remaining、hosted、AR 相关量。
4. t ← t + 1；t = M 时 episode 结束并给终局奖励（01_algorithms §1.3）。lagrangian 每步再减 λ·c_t。

## 4.7 12 个变种的区别

| 方面 | maskable | repair | lagrangian | unconstraint |
|---|---|---|---|---|
| 观测 | 原始状态 | 原始状态 | 原始状态 + λ / (1 + λ) | 原始状态 |
| 动作空间 | Discrete(N + 1) | Discrete(N) | Discrete(N) | Discrete(N) |
| 不可行动作 | 被掩码；无可行时只能 EXIT | 环境替换；无可行时执行并违约 | 执行，违约受 −λ·c_t 惩罚 | 执行，违约只记录 |
| 奖励 | ar_pen | ar_pen | ar_obj − λ·c_t | ar_pen |
| 可能违约 | 否 | 仅死局 | 是 | 是 |

| 方面 | PPO | DQN | DDQN (Double DQN) |
|---|---|---|---|
| 类型 | on-policy actor-critic | off-policy, replay + target network | off-policy, replay + target network |
| 输出 | 动作 logit（分类分布）+ V(s) | Q(s, a) | Q(s, a) |
| 训练时探索 | 从分类分布采样（熵系数 0.005） | ε-贪心，ε 在前 2M 步从 1 线性降到 0 | 同 DQN |
| 评估时 | 确定性：取最大 logit | argmax Q | argmax Q |
| maskable 的掩码来源 | 环境 `action_masks()`（+ 网络内同一掩码） | 网络从观测计算 | 网络从观测计算 |
| 并行环境数 | 40 | 12 | 12 |

## 4.8 info 字段（每步返回，用于日志与评估）

`ar`, `ar_obj`, `ar_star`, `valid_placed`, `services_placed`, `capacity_violations`, `conflict_violations`, `repairs`, `cap_violated`, `conflict_violated`, `dead_end`, `exited`, `ecus_used`, `inst_idx`, `viol_rate_ep`, `cost_ep`
