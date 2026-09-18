# 核心公式汇总 / Core Formula Reference

**说明 / Note**：本文档只记录公式本身及其在代码中的实际实现状态，不涉及论文的论证方式或叙事框架。所有公式均已与代码逐行核对（截至 commit `82513ac` / tag `v4.0.0_final_1`），若发现文档描述与实际运行代码不符，会在对应小节以"⚠️ 实现说明"标出。

符号统一约定见文末《符号表 / Notation Table》。

---

## 1. 问题定义与 AR 指标 / Problem Definition and AR Metric

### 1.1 场景变量 / Scenario Variables

| 符号 | 中文含义 | English meaning |
|------|---------|------------------|
| $N$ | ECU（部署节点）数量 | number of ECUs (deployment nodes) |
| $M$ | 服务（待部署任务）数量 | number of services (jobs to place) |
| $e_j$ | 第 $j$ 个 ECU 的容量 | capacity of ECU $j$ |
| $n_i$ | 第 $i$ 个服务的资源需求 | resource requirement of service $i$ |
| $x_{ij} \in \{0,1\}$ | 服务 $i$ 是否分配到 ECU $j$ | indicator: service $i$ assigned to ECU $j$ |
| $y_j \in \{0,1\}$ | ECU $j$ 是否被启用（至少承载一个服务） | indicator: ECU $j$ is activated (hosts ≥1 service) |
| $\mathcal{J}^*$ | 被启用的 ECU 集合 | set of activated ECUs |

三场景（lt/eq/gt）区别仅在于 $N$ 与 $M$ 的相对大小关系（资源紧缺/均衡/充裕），核心公式结构不变。
*The three scenarios (lt/eq/gt) differ only in the relative size of $N$ vs. $M$ (scarce / balanced / abundant resources); the core formulas below are identical across all three.*

### 1.2 资源利用率 AR / Allocation Ratio (AR)

$$
AR = \frac{1}{|\mathcal{J}^*|} \sum_{j \in \mathcal{J}^*} \; \sum_{i:\, x_{ij}=1} \frac{n_i}{e_j}
$$

**中文**：AR 是"被启用的 ECU 上，已分配服务需求量占该 ECU 容量的比例"，在所有被启用 ECU 上取平均——即平均资源利用率，取值范围 $(0, 1]$。

**English**: AR is the average, over all *activated* ECUs, of the fraction of that ECU's capacity consumed by its assigned services — i.e. mean resource-utilisation ratio, range $(0,1]$.

**⚠️ 方法论说明**：跨场景（eq/gt/lt）平均 ILP 最优 AR 没有意义，因为三场景 $N/M$/容量分布不同、最优基线本身处在不同量级；对比必须逐场景（per-scenario）呈现。

---

## 2. ILP 最优解 / ILP Optimal Solution (Dinkelbach 参数化)

原问题是分式规划（AR 本身是分式目标），采用 Dinkelbach 算法迭代求解等价的参数化子问题：
*The original AR-maximisation problem is a fractional program; it is solved via Dinkelbach's algorithm, iterating over an equivalent parametric subproblem:*

$$
\max_{x, y} \;\; F(\lambda) = \sum_{i} \sum_{j} x_{ij} \frac{n_i}{e_j} \;-\; \lambda \sum_j y_j
$$

**约束 / Constraints**：

$$
\sum_j x_{ij} = 1 \quad \forall i \qquad \text{（每个服务恰好分配一次 / each service assigned exactly once）}
$$

$$
\sum_i x_{ij} \, n_i \le e_j \, y_j \quad \forall j \qquad \text{（容量约束 / capacity constraint）}
$$

$$
\sum_{i \in C_k} x_{ij} \le 1 \quad \forall j, \, \forall \text{冲突集 } C_k \qquad \text{（冲突集约束 / conflict-set constraint：每个 ECU 每个冲突集最多容纳 1 个服务）}
$$

**迭代更新 / Iterative update**：

$$
\lambda \leftarrow \frac{F}{G}, \qquad G = \sum_j y_j \;\; \text{(当前解下被启用 ECU 数 / number of activated ECUs at current solution)}
$$

**收敛条件 / Convergence criterion**：$|\Delta\lambda| < 10^{-6}$

**中文**：$\lambda$ 相当于"每启用一个 ECU 的机会成本"，Dinkelbach 迭代不断用当前最优解重新估计这个成本，直到收敛，收敛时对应原分式目标 $AR$ 的全局最优。ILP 是**一次性求解**，不涉及时序决策。

---

## 3. RL 统一 MDP 框架 / Unified RL MDP Framework

所有 6 种 RL 方法（`ppo`、`ppo_mask`、`ppo_lagrangian`、`ppo_opt`、`dqn`、`ddqn`）共享同一套序贯决策 MDP 骨架，只在**约束处理机制**与对应奖励塑造上不同。
*All 6 RL methods share one sequential-decision MDP skeleton and differ only in how they handle constraints and shape reward.*

### 3.1 状态转移过程 / State-Transition Process

- 时间步 $t = 0, 1, \dots, M-1$，每步处理服务队列中的第 $t$ 个服务（按需求量降序排好）。
  *Timestep $t$ processes the $t$-th service in a pre-sorted (descending demand) queue.*
- 动作空间 / Action space：$a_t \in \{0, 1, \dots, N-1\}$，即为当前服务选择一个 ECU（$\text{Discrete}(N)$）。
- 终止条件 / Termination：$\text{done} = [\,t \ge M\,]$，即必须执行恰好 $M$ 步。

$$
\text{cap\_violated}_t = \mathbf{1}\!\left[\text{remaining\_vms}[a_t] < n_t\right], \qquad
\text{conflict\_violated}_t = \mathbf{1}\!\left[\exists\, C_k:\; t, a_t \in C_k \text{ 冲突}\right]
$$

$$
\text{ru}_t = \frac{n_t}{e_{a_t}} \quad \text{（本步资源利用贡献 / per-step utilisation contribution，违反时视算法而定是否计入）}
$$

### 3.2 统一分级终端奖励 / Unified Graded Terminal Reward

这是全部 6 个算法共享的**终端奖励**公式，在 `add_states`（v4.0.0冻结）和 `final_paper_experiments`（v4.1.0）两个分支上都不变。**非终端步奖励是否为 0，两个分支不同**：v4.0.0 上全部 6 算法非终端步恒为 0；v4.1.0 上 `ppo_mask`/`ppo_lagrangian`/`ppo_opt`/`dqn`/`ddqn` 五个算法的非终端步奖励已接入各自的逐步惩罚项（见第4节及 `v4.1.0_changelog.md`），只有 `ppo`(P3) 仍保持非终端步恒为 0。
*Shared by all 6 algorithms on both the `add_states` (v4.0.0, frozen) and `final_paper_experiments` (v4.1.0) branches. Whether the non-terminal step reward is 0 differs by branch: on v4.0.0 all 6 algorithms have it hardcoded to 0; on v4.1.0, `ppo_mask`/`ppo_lagrangian`/`ppo_opt`/`dqn`/`ddqn` now wire in their respective per-step penalty terms (Section 4, and the changelog) — only `ppo`(P3) still returns 0 for non-terminal steps.*

$$
R_{\text{terminal}} =
\begin{cases}
M \cdot (2\,AR - 1), & \text{若整个 episode 零违反 / if zero violations throughout the episode} \\[4pt]
-M \cdot \left(1 - \dfrac{\text{valid\_placed}}{M}\right), & \text{若存在违反 / if any violation occurred}
\end{cases}
$$

其中 $\text{valid\_placed}$ = episode 中未触发任何违反、被正常放置的服务数。
*where $\text{valid\_placed}$ is the count of services placed without triggering any violation.*

**中文解读**：
- 判定分支是**二元**的（成功 / 失败两支），但每支内部的奖励幅度是**连续分级**的，不是固定 $\pm1$：成功支随 AR 质量线性变化（值域 $(-M, M]$），失败支随"完成进度" $\text{valid\_placed}/M$ 线性变化（值域 $[-2M, -M)$，恒劣于任意成功支，二者只在不可达的 $AR \to 0$ 处取等）。
- 该公式最早在 `ppo_mask`（v2.6.0/v2.7.0）中提出，后被移植（"Ported from ppo_mask v2.6.0"）到其余 5 个算法。

**English**: The branch condition is binary (success vs. failure), but the reward *magnitude* within each branch is continuously graded — not a fixed ±1. This formula originated in `ppo_mask` (v2.6.0/v2.7.0) and was subsequently ported verbatim into the other five algorithms.

### 3.3 ppo_mask 退火权重变体 / ppo_mask Annealed-Weight Variant

`ppo_mask` 独有一个训练课程退火权重 $w \in [0,1]$（由训练回调按进度调度），成功支实际为：

$$
R_{\text{terminal}}^{\text{success}} = M \cdot \big[(1-w)\cdot 1.0 + w\cdot(2\,AR-1)\big]
$$

**中文**：训练初期 $w \to 0$，只奖励"是否完整放完"（不含 AR 质量噪声）；随训练推进 $w \to 1$，退化为 3.2 节的标准公式。这是一种课程学习（curriculum）设计，不影响其余 5 个算法。

---

## 4. 各算法约束处理机制 / Per-Algorithm Constraint-Handling Mechanisms

### 4.1 `ppo`（P3，无约束基线 / unconstrained baseline）

违反只记录不惩罚，非终端步奖励恒为 0：

$$
R_t = 0 \;\; (t < M), \qquad R_{\text{terminal}} \text{ 同第 3.2 节公式（zero shaping/penalty）}
$$

**中文**：这是刻意设计的"什么都不管"下限对照组，唯一奖励信号就是终端的分级公式本身，不含任何针对约束的塑形。

### 4.2 `ppo_mask`（P4，硬掩码 / hard action masking）

$$
\text{mask}[j] = \mathbf{1}\!\left[\text{remaining\_vms}[j] \ge n_t \;\wedge\; \neg\,\text{conflict}(j, t)\right]
$$

若 $\forall j: \text{mask}[j] = 0$（无合法 ECU），触发强制溢出兜底（fallback，选剩余容量最大的 ECU），处以 $-2.0$ 惩罚：

$$
R_t^{\text{violation-penalty}} = -2.0 \;\; \text{（仅当强制溢出触发时 / only on forced-overflow fallback）}
$$

外加基于势函数的塑形项（默认关闭）：

$$
F(s,a,s') = \gamma\,\Phi(s') - \Phi(s), \qquad \Phi(s) = -\beta \cdot \big(1 - \text{FFD\_feasibility}(s)\big)
$$

**中文**：$\beta$（`bottleneck_shaping_weight`）默认为 $0.0$，此时塑形项恒为 0（no-op），依据 Ng-Harada-Russell (1999) 的势函数塑形理论，该项在任意 $\beta \ge 0$ 下都不改变最优策略。

**版本差异**：v4.0.0 上 $R_t^{\text{violation-penalty}}$ 虽已计算但未组装进非终端步 `reward`（恒为 $0$，只有 shaping 项生效）；v4.1.0 上 `reward = violation_penalty`（lt/gt两场景；eq场景本身没有强制溢出兜底分支，不适用此项，未改动）。由于掩码已结构性防住了绝大多数情况，该修复预期影响很小。

### 4.3 `ppo_lagrangian`（P5，无动作掩码，容量固定惩罚 + 冲突拉格朗日软约束 / no action masking; fixed-penalty capacity + Lagrangian-soft conflict）

**⚠️ v4.0.0 阶段的表述订正 / correction to the v4.0.0-era description**：此前（包括本文档更早版本）曾把 P5 描述为"容量硬掩码 + 冲突拉格朗日"，这是**错误的**。核实 `run_all.py` 发现三场景（lt/eq/gt）训练用的都是普通 `PrunedPPO`（`stable_baselines3.PPO` 子类），从未用 `ActionMasker`/`MaskablePPO` 包装环境——`env.py` 里定义的 `action_masks()` 是**死代码，从未被调用**。这不是bug，是设计意图：硬掩码是 `ppo_mask`(P4) 专属机制，P5 的方法论意义就是"用惩罚/对偶变量代替掩码"，若P5也硬掩码就与P4没有区别。**准确描述是：P5 对容量和冲突都不做任何结构性阻止，两者都只是惩罚。**

（另外，`gt` 场景的 `env.py` 历史上有一段 `step()` 内部的容量重定向逻辑，会把违规动作事后纠正到可行ECU，与 lt/eq 不一致——v4.1.0 已删除该逻辑，三场景现已完全对齐。）

**逐步奖励公式 / per-step formula**（v4.1.0 已真正接入 `step()` 返回值；v4.0.0 阶段这几项虽已计算但从未组装进 reward，非终端步 `reward` 恒为 `0.0`，见下方"版本差异"）：

$$
r_t = \text{match\_gain}_t \;-\; \text{cap\_penalty}_t \;-\; (\lambda + \text{base\_penalty}) \cdot c_t
$$

$$
\text{cap\_penalty}_t = -2.0 \cdot \mathbf{1}[\text{cap\_violated}_t], \qquad c_t = \mathbf{1}[\text{conflict\_violated}_t], \qquad \text{base\_penalty} = 0.2
$$

**对偶上升 / dual ascent**（在训练回调中，每 `LAMBDA_UPDATE_WINDOW` 个 episode 更新一次，只针对冲突，不针对容量）：

$$
\lambda \leftarrow \text{clip}\big(\lambda + \eta \cdot \bar{v},\; 0,\; \lambda_{\max}\big), \qquad \eta = \text{LAMBDA\_LR} = 3\times10^{-4}
$$

$\bar{v}$ 为窗口内的平均冲突违反率，$\lambda_{\max}$ 为配置中的 `LAMBDA_MAX`。

**版本差异 / version history**：
- **v4.0.0（`add_states`分支，冻结）**：`match_gain` 累加进 `self._total_ru`（影响 AR），但 `cap_penalty`/`lagrange_penalty` 从未组装进 `reward`——非终端步 `reward` 恒为 `0.0`。唯一真正生效的信号是第 3.2 节的终端分级公式。
- **v4.1.0（`final_paper_experiments`分支）**：上述公式已真正接入 `reward`，见 `paper_contents/v4.1.0_changelog.md`。

**English**: Prior descriptions of P5 as "hard-masked capacity" were incorrect — verified that `run_all.py` trains a plain `PrunedPPO` (a `PPO` subclass) with no `ActionMasker`/`MaskablePPO` wrapping in any of the 3 scenarios; `action_masks()` is dead code by design (masking is P4-exclusive). P5 correctly has no structural constraint enforcement at all — both capacity and conflict are pure penalties. In v4.0.0 these penalty terms were computed but never assembled into the returned reward (non-terminal `reward` was hardcoded `0.0`); v4.1.0 wires them in (see changelog).

### 4.4 `ppo_opt`（P6，最佳适应修复启发式 / best-fit repair heuristic）

修复目标（当选中的 ECU 违反容量或冲突约束时触发）：

$$
j^* = \arg\max_{j \,\in\, \mathcal{V}(t)} \frac{n_t}{e_j}, \qquad \mathcal{V}(t) = \{\, j : \text{remaining\_vms}[j] \ge n_t \;\wedge\; \neg\text{conflict}(j,t) \,\}
$$

若 $\mathcal{V}(t) = \varnothing$（无可修复目标），episode 立即终止并处以：

$$
R_{\text{terminate-unrepairable}} = -M
$$

修复动作的 docstring 惩罚 / documented repair penalty：

$$
\text{repair\_penalty} = -0.1 \;\; \text{（每次触发修复 / per repair event）}
$$

**版本差异 / version history**：
- **v4.0.0（冻结）**：`repair_penalty`/`step_reward` 计算后未使用，非终端步 `reward` 恒为 $0$（仅"无法修复"分支立即返回 $-M$ 并终止）。docstring 里"Terminal bonus: $+AR\cdot(1-\text{repair\_rate})$"是过时描述，实际终端奖励一直是第 3.2 节统一分级公式，未按修复率加权。
- **v4.1.0**：非终端步 `reward = repair_penalty`，见 `v4.1.0_changelog.md`。

**English**: In v4.0.0, `repair_penalty`/`step_reward` were computed but unused (non-terminal reward hardcoded `0`, except the unrepairable-fallback branch returning `-M` immediately). The docstring's "$+AR\cdot(1-\text{repair\_rate})$" terminal formula was likewise stale. v4.1.0 wires `repair_penalty` into the non-terminal reward.

### 4.5 `dqn` / `ddqn`（Q-learning 族 / Q-learning family）

环境（`env.py`）与状态/动作/终端奖励定义同 4.1 节的无约束基线一致，仅训练算法（价值迭代 vs 策略梯度）不同。标准 Bellman 目标：

$$
y_t = r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a') \qquad \text{(DQN)}
$$

$$
y_t = r_t + \gamma \, Q_{\theta^-}\!\big(s_{t+1},\; \arg\max_{a'} Q_\theta(s_{t+1}, a')\big) \qquad \text{(Double DQN, 消除过高估计 / de-biases overestimation)}
$$

**中文**：DDQN 与 DQN 唯一区别在于目标 Q 值的动作选择与评估解耦（动作选择用在线网络 $\theta$，评估用目标网络 $\theta^-$）。环境侧：v4.0.0 上两者非终端步 `reward` 都恒为 $0$（`cap_penalty`/`conflict_penalty` 计算后未使用）；v4.1.0 已接入 `reward = cap_penalty + conflict_penalty`（gt场景无独立cap_penalty，容量违规是硬终止），见 `v4.1.0_changelog.md`。

**English**: The only DQN/DDQN difference is decoupling target-Q action-selection (online network) from evaluation (target network). Environment-wise: on v4.0.0 both had non-terminal reward hardcoded to `0`; v4.1.0 wires in `reward = cap_penalty + conflict_penalty` (gt has no separate cap_penalty — capacity violation there is an immediate hard termination).

---

## 5. 符号表 / Notation Table

| 符号 | 中文 | English |
|------|------|---------|
| $N$ | ECU 数量 | number of ECUs |
| $M$ | 服务数量 | number of services |
| $t$ | 当前时间步 / 当前服务下标 | current timestep / current service index |
| $e_j$ | ECU $j$ 的容量 | capacity of ECU $j$ |
| $n_i$, $n_t$ | 服务 $i$ / 当前服务 的资源需求 | requirement of service $i$ / current service |
| $x_{ij}$ | 服务-ECU 分配指示变量 | service-to-ECU assignment indicator |
| $y_j$ | ECU 启用指示变量 | ECU activation indicator |
| $AR$ | 平均资源利用率 | Allocation Ratio (mean utilisation) |
| $\mathcal{J}^*$ | 被启用 ECU 集合 | set of activated ECUs |
| $\lambda$（第2节） | Dinkelbach 分式规划参数 | Dinkelbach fractional-programming parameter |
| $\lambda$（第4.3节） | 拉格朗日对偶变量（冲突约束乘子） | Lagrangian dual variable (conflict multiplier) |
| $a_t$ | 第 $t$ 步动作（选择的 ECU） | action at step $t$ (chosen ECU) |
| $s_t$ | 第 $t$ 步状态（observation） | state/observation at step $t$ |
| $r_t$, $R_t$ | 第 $t$ 步即时奖励 | immediate reward at step $t$ |
| $\text{valid\_placed}$ | 未触发违反、成功放置的服务数 | count of services placed without violation |
| $C_k$ | 第 $k$ 个冲突集 | $k$-th conflict set |
| $\mathbf{1}[\cdot]$ | 指示函数 | indicator function |
| $\Phi(s)$ | 势函数（reward shaping） | potential function (for reward shaping) |
| $\gamma$ | 折扣因子 | discount factor |
| $\theta$, $\theta^-$ | 在线网络 / 目标网络参数 | online / target network parameters |

---

*本文档由代码逐行核对生成，涉及 `scenarios/{lt,eq,gt}/{ppo,ppo_mask,ppo_lagrangian,ppo_opt,dqn,ddqn}/env.py`，以 lt 场景为准（eq/gt 同构）。若代码后续更新，请重新核对本文档。*
