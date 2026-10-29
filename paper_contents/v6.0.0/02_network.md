# 2. 网络：关系偏置的 Transformer 编码器

代码：`src/paper_rl/graph_net.py`（`GraphPolicyNet`、`build`、`Block`），PPO 包装 `src/paper_rl/graph_policy.py`，DQN 包装 `src/paper_rl/graph_dqn.py`。12 个变种全部使用同一个结构（`--net graph`），没有使用 MLP。

## 2.1 结构参数

| 项 | 值 |
|---|---|
| 隐藏维度 d | 128 |
| 注意力头数 | 4（d_k = 32） |
| Transformer 层数 | 3 |
| 前馈层 | 128 → 256 → 128, GELU |
| 归一化 | Pre-LN（每个子层前 LayerNorm）+ 编码器末尾 LayerNorm |
| 位置编码 | 无（对 ECU、服务的排列等变）；当前服务由特征「是否当前」标出 |
| token 数 | 1 + N + M |
| 参数量（PPO 一个网络；DQN 在线 / 目标网络各一个） | 450,227（lagrangian：450,355，全局输入多 1 维 λ） |

## 2.2 输入：token 与特征

网络读原始观测（04_environment），在 `build()` 里计算全部特征，输入只是这些特征（不读其他东西）：

- **全局 token（4 维）**：当前 AR、未放置服务占比、未放置需求 / 总容量、剩余容量 / 总容量。lagrangian 再加 1 维 λ / (1 + λ)。
- **ECU token（每个 6 维）**：c_n / c_max、max(剩余容量, 0) / c_max、是否已启用、对当前服务是否可行、承载服务数 / M、负载 / c_n。
- **服务 token（每个 5 维）**：e_m / c_max、是否已放置、是否当前服务、可行 ECU 数 / N、与其他未放置服务的冲突数 / M（已放置的服务为 0）。c_max = 该实例最大容量。
- 每种 token 先经各自的线性层投影到 d 维（4→128、6→128、5→128；lagrangian 全局 5→128），再加上该类型的可学习类型嵌入（3 × d，零初始化）。

## 2.3 关系偏置注意力

每层的注意力 logit = QKᵀ / √d_k + Σ_r R_r · w_{r,h}，R_r 是 4 种 0/1 关系矩阵，w 是每层每头可学习的权重（4 × heads，零初始化）：

1. 服务–服务冲突（两者都未放置）
2. 服务承载于 ECU（对称）
3. 未放置服务与 ECU 上的某个服务冲突（对称）
4. 未放置服务的需求 ≤ ECU 剩余容量（对称）

## 2.4 输出头

| 头 | 结构 | 用途 |
|---|---|---|
| ecu_head | Linear(128→128) → GELU → Linear(128→1) | 每个 ECU token 一个分数，所有 ECU 共用同一个头 |
| exit_head | Linear(128→1) | 由全局 token 给 EXIT 打分（只有 maskable 用到） |
| value_head | Linear(256→128) → GELU → Linear(128→1) | 输入 [全局 token, 其余 token 的均值]，给出 V(s)（只有 PPO 用到） |

## 2.5 在不同算法中的用法

- PPO：actor 与 critic 共用编码器（`separate_critic = False`）。maskable 用 N + 1 个 logit（含 EXIT），并在网络内按可行性把不可行 logit 置为 −1e9；其余 strategy 用不加掩码的分数，截取前 N 个（无 EXIT）。动作分布为分类分布（Categorical）。
- DQN / DDQN：不加掩码的分数直接作为 Q 值（maskable 为 N + 1 个，其余为 N 个）；value_head 不参与，不产生梯度。在线网络和目标网络各是一个完整的 `GraphPolicyNet`。
- 关闭的可选结构（所有 run）：glob_std、glob3、pair_head、plan、plan_decay、separate_critic 全部为 False。
- 初始化：PyTorch 默认初始化（`_build` 被覆盖，SB3 的正交初始化不起作用）；关系权重 w 和类型嵌入为零初始化。
- config.py 中的 `PPO_NET_ARCH`（256-256）与 `DQN_NET_ARCH`（128-128）只用于 MLP 网络，这些 run 不使用。

## 2.6 参数量明细（不含 λ 输入）

| 部分 | 参数 |
|---|---|
| inp | 2,304 |
| blocks | 397,488 |
| ln | 256 |
| ecu_head | 16,641 |
| exit_head | 129 |
| value_head | 33,025 |
| type_emb | 384 |
| 合计 | 450,227 |

每个 Transformer 层 132,496 个参数：

| 部分 | 参数 |
|---|---|
| ln1 + ln2 | 512 |
| qkv | 49,536 |
| o | 16,512 |
| ff | 65,920 |
| rel_w | 16 |
