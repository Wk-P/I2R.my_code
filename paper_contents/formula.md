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

这是全部 6 个算法共享的**终端奖励**公式（非终端步奖励为 0，见各算法小节的例外情况）：
*This terminal-reward formula is shared by all 6 algorithms (non-terminal step reward is 0 unless noted otherwise below):*

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

### 4.3 `ppo_lagrangian`（P5，容量硬掩码 + 冲突拉格朗日软约束 / hard-cap + Lagrangian-soft-conflict）

**Docstring 中描述的逐步奖励公式 / per-step formula as documented**：

$$
r_t = \text{match\_gain} - (\lambda + \text{base\_penalty}) \cdot c_t - \text{forced\_overflow\_penalty}
$$

其中 $c_t = \mathbf{1}[\text{conflict\_violated}_t]$，$\text{base\_penalty} = 0.2$。

**对偶上升 / dual ascent**（在训练回调中，每 `LAMBDA_UPDATE_WINDOW` 个 episode 更新一次）：

$$
\lambda \leftarrow \text{clip}\big(\lambda + \eta \cdot \bar{v},\; 0,\; \lambda_{\max}\big), \qquad \eta = \text{LAMBDA\_LR} = 3\times10^{-4}
$$

$\bar{v}$ 为窗口内的平均冲突违反率，$\lambda_{\max}$ 为配置中的 `LAMBDA_MAX`。

**⚠️ 实现说明 / Implementation note**：核对 `scenarios/lt/ppo_lagrangian/env.py` 第 231-251 行发现，`match_gain`、`lagrange_penalty`、`forced_overflow_penalty`、`step_reward` 均已计算，但**实际返回的非终端步奖励恒为 `reward = 0.0`**，上述逐步公式并未真正接入 `step()` 返回值——只有 `match_gain` 被累加进 `self._total_ru`（用于计算 AR），冲突/容量惩罚在非终端步并不体现在返回给智能体的 reward 里。真正影响训练信号的，只有第 3.2 节的终端分级公式（其成功/失败分支由 `episode_violations` 决定，间接反映了整个 episode 的违反情况）。$\lambda$ 的对偶上升更新依据的是 episode 级违反率统计，与这段未接入的逐步惩罚公式无关。

**English**: Verified against the code — `match_gain`, `lagrange_penalty`, `forced_overflow_penalty`, and `step_reward` are all computed but **never assigned to the actual returned non-terminal `reward`, which is hardcoded to `0.0`**. Only `match_gain` feeds into the running AR total. The per-step formula in the docstring (and above) describes intent, not runtime behaviour; the only live training signal is the Section 3.2 terminal formula. The $\lambda$ dual-ascent update is driven by episode-level violation-rate statistics, independent of this dead per-step formula.

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

**⚠️ 实现说明**：核对 `scenarios/lt/ppo_opt/env.py` 第 243-267 行，`repair_penalty` 与 `step_reward` 同样是计算后未使用的变量——非终端步实际返回 `reward` 也恒为该分支未显式赋值时的 $0$（仅"无法修复"分支会立即返回 $-M$ 并终止）。docstring 里另一处描述的"Terminal bonus: $+AR \cdot (1-\text{repair\_rate})$"同样是过时文档，实际终端奖励用的是第 3.2 节的统一分级公式，未按修复率加权。

**English**: `repair_penalty` and `step_reward` are computed but unused for non-terminal steps (which return `0`, except the unrepairable-fallback branch which returns `-M` immediately and ends the episode). The docstring's alternative terminal formula "$+AR\cdot(1-\text{repair\_rate})$" is likewise stale; the actual terminal reward uses the unified Section 3.2 formula, not repair-rate weighting.

### 4.5 `dqn` / `ddqn`（Q-learning 族 / Q-learning family）

环境（`env.py`）与状态/动作/终端奖励定义同 4.1 节的无约束基线一致，仅训练算法（价值迭代 vs 策略梯度）不同。标准 Bellman 目标：

$$
y_t = r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a') \qquad \text{(DQN)}
$$

$$
y_t = r_t + \gamma \, Q_{\theta^-}\!\big(s_{t+1},\; \arg\max_{a'} Q_\theta(s_{t+1}, a')\big) \qquad \text{(Double DQN, 消除过高估计 / de-biases overestimation)}
$$

**中文**：DDQN 与 DQN 唯一区别在于目标 Q 值的动作选择与评估解耦（动作选择用在线网络 $\theta$，评估用目标网络 $\theta^-$），环境侧奖励结构完全相同，均沿用 4.1 节的 0/分级公式。

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
