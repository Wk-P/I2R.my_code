# 核心公式汇总 / Core Formula Reference

**说明 / Note**：本文档记录**最终定案**（`final_paper_experiments` 分支，tag `v4.1.0.2`）的公式与代码实现状态，不涉及论文的论证方式或叙事框架。所有公式均已与代码逐行核对。完整的排查/消融过程（谁改了、为什么改、数据支撑）见 `paper_contents/v4.1.0_changelog.md`，本文档只给最终结论。

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

三场景（lt/eq/gt）区别仅在于 $N$ 与 $M$ 的相对大小关系，核心公式结构不变：

| 场景 | $N$（ECU） | $M$（服务） | 难度定位 |
|---|---|---|---|
| lt | 10 | 15 | 资源紧缺（$N<M$），**论文核心困难场景** |
| eq | 10 | 10 | 供需均衡 |
| gt | 15 | 10 | 资源充裕（$N>M$） |

*The three scenarios (lt/eq/gt) differ only in the relative size of $N$ vs. $M$; the core formulas below are identical across all three. lt is deliberately the hardest (more services than ECUs), which is why most algorithms show the widest spread and lowest success_rate there — see the per-scenario numbers in Section 4.*

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

这是全部 6 个算法**始终共享**的终端奖励公式（唯一在所有历史版本、所有算法间都不变的部分）：

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
- 判定分支是**二元**的（成功 / 失败两支），但每支内部的奖励幅度是**连续分级**的，不是固定 $\pm1$：成功支随 AR 质量线性变化（值域 $(-M, M]$），失败支随"完成进度" $\text{valid\_placed}/M$ 线性变化（值域 $[-2M, -M)$，恒劣于任意成功支）。
- 该公式最早在 `ppo_mask`（v2.6.0/v2.7.0）中提出，后被移植到其余 5 个算法。

**English**: The branch condition is binary (success vs. failure), but the reward *magnitude* within each branch is continuously graded. Originated in `ppo_mask` (v2.6.0/v2.7.0), later ported verbatim to the other five algorithms.

### 3.3 ppo_mask 退火权重变体 / ppo_mask Annealed-Weight Variant

`ppo_mask` 独有一个训练课程退火权重 $w \in [0,1]$（由训练回调按进度调度），成功支实际为：

$$
R_{\text{terminal}}^{\text{success}} = M \cdot \big[(1-w)\cdot 1.0 + w\cdot(2\,AR-1)\big]
$$

**中文**：训练初期 $w \to 0$，只奖励"是否完整放完"；随训练推进 $w \to 1$，退化为 3.2 节的标准公式。课程学习设计，不影响其余 5 个算法。

### 3.4 非终端步奖励：最终定案一览 / Non-Terminal Reward: Final Status

| 算法 | 非终端步 $r_t$（最终，v4.1.0.2） | 是否曾修改后又回退 |
|---|---|---|
| `ppo`（P3） | $0$（设计如此，从未变过） | 否 |
| `ppo_mask`（P4） | $\text{violation\_penalty} + \text{shaping}$（详见4.2节） | 否（v4.1.0接入后保留） |
| `ppo_lagrangian`（P5） | $0$（**回退到v4.0.0行为**，详见4.3节） | **是**——v4.1.0接入过完整公式，v4.1.0.1试过去掉match_gain，两版在lt场景都明显更差，最终撤回 |
| `ppo_opt`（P6） | $\text{repair\_penalty}$（详见4.4节） | 否（v4.1.0接入后保留） |
| `dqn` | $\text{cap\_penalty} + \text{conflict\_penalty}$（详见4.5节） | 否（v4.1.0接入后保留） |
| `ddqn` | 同`dqn` | 否（v4.1.0接入后保留） |

---

## 4. 各算法约束处理机制 / Per-Algorithm Constraint-Handling Mechanisms

### 4.1 `ppo`（P3，无约束基线 / unconstrained baseline）

违反只记录不惩罚，非终端步奖励恒为 0：

$$
R_t = 0 \;\; (t < M), \qquad R_{\text{terminal}} \text{ 同第 3.2 节公式}
$$

**中文**：刻意设计的"什么都不管"下限对照组，唯一奖励信号就是终端的分级公式，不含任何针对约束的塑形。全程未改动，是6算法里唯一从v4.0.0到v4.1.0.2代码零变化的一个。

### 4.2 `ppo_mask`（P4，硬掩码 / hard action masking）—— **唯一真正使用动作掩码的算法**

用 `sb3_contrib.MaskablePPO` + `ActionMasker` 训练（已验证真实生效，非死代码接口）：

$$
\text{mask}[j] = \mathbf{1}\!\left[\text{remaining\_vms}[j] \ge n_t \;\wedge\; \neg\,\text{conflict}(j, t)\right]
$$

若 $\forall j: \text{mask}[j] = 0$（无合法 ECU，极罕见），触发强制溢出兜底（选剩余容量最大的 ECU）：

$$
r_t = \text{violation\_penalty} + F(s,a,s'), \qquad \text{violation\_penalty} = -2.0 \cdot \mathbf{1}[\text{forced-overflow triggered}]
$$

外加基于势函数的塑形项（默认关闭）：

$$
F(s,a,s') = \gamma\,\Phi(s') - \Phi(s), \qquad \Phi(s) = -\beta \cdot \big(1 - \text{FFD\_feasibility}(s)\big), \qquad \beta = 0 \text{（默认，no-op）}
$$

**中文**：依据 Ng-Harada-Russell (1999) 势函数塑形理论，$\beta \ge 0$ 时该项不改变最优策略。`violation_penalty` 在 v4.1.0 前是死代码（算了没接线，非终端步恒为0），v4.1.0接入后保留至今——因为掩码已结构性防住了99%以上的情况，这项修复本身影响很小，符合预期。**例外**：`eq` 场景没有强制溢出兜底分支（代码结构上不存在这种情况），非终端步恒为 $0$，无需任何改动。

### 4.3 `ppo_lagrangian`（P5，无动作掩码，纯惩罚约束 / no action masking, pure-penalty constraints）—— **最终回退到 v4.0.0 行为**

$$
r_t = 0 \quad (t < M), \qquad R_{\text{terminal}} \text{ 同第 3.2 节公式}
$$

**关键说明（务必准确表述）**：
1. **不使用任何动作掩码**——硬掩码是 `ppo_mask` 专属机制。`env.py` 里定义的 `action_masks()` 是死代码（`run_all.py` 训练的是普通 `PrunedPPO`，从未用 `ActionMasker`/`MaskablePPO` 包装），这是**设计意图**，不是bug：P5 的方法论意义就是"用惩罚/对偶变量代替掩码"，若也硬掩码就和P4没区别了。
2. **概念设计**上，容量违规该给固定惩罚、冲突违规该给拉格朗日自适应惩罚：
   $$
   r_t^{\text{concept}} = -\text{cap\_penalty}_t - (\lambda + \text{base\_penalty}) \cdot c_t, \qquad \text{cap\_penalty}_t = -2.0\cdot\mathbf{1}[\text{cap\_violated}_t], \quad c_t=\mathbf{1}[\text{conflict\_violated}_t],\ \text{base\_penalty}=0.2
   $$
   但**这个公式最终没有被接入训练**（`cap_penalty`/`lagrange_penalty` 在代码里仍然计算，但显式未使用，`reward` 硬编码为 `0.0`）——这是v4.1.0→v4.1.0.1→v4.1.0.2三轮消融实验后的**数据支撑的主动决策**，不是遗漏。
3. **对偶上升机制本身照常运行**，不受上述reward决策影响（$\lambda$ 更新依据episode级违反率统计，不依赖per-step reward）：
   $$
   \lambda \leftarrow \text{clip}\big(\lambda + \eta \cdot \bar{v},\; 0,\; \lambda_{\max}\big), \qquad \eta = 3\times10^{-4}
   $$

**为什么回退**（详见 `paper_contents/v4.1.0_changelog.md`，此处只摘结论）：
- v4.1.0接入完整公式（含正向 $\text{match\_gain}=n_t/e_{a_t}$ 项）：lt场景 success_rate $0.81\to0.52$，conflict_viol $0.18\to0.47$，明显变差。
- v4.1.0.1假设是match_gain重复计入AR信号，去掉它只留惩罚项重测：lt场景 $0.52\to0.54$，$0.47\to0.45$，**几乎没变化**，假设被推翻。
- 结论：不是公式细节问题，是"任何非零逐步奖励在这个环境结构下都会破坏PPO训练"——环境本身无结构性约束保护（不像`ppo_mask`有掩码、`ppo_opt`有修复），逐步惩罚量级（$-2.0$ 及自适应增长的 $\lambda$ 项）可能与终局奖励同量级，扰乱了GAE优势估计。eq/gt场景约束压力小，未观察到同等程度的负面影响。
- **gt场景额外说明**：v4.1.0期间同时删除了gt独有的一段"容量事后重定向"逻辑（历史遗留，与lt/eq不一致），这个删除**予以保留**（独立的正确性修正，与reward消融无关），经专项验证对gt结果无实质影响。

**English**: No action masking anywhere (masking is P4-exclusive by design). The conceptual penalty formula is computed but was ultimately **not** wired into training — confirmed via three rounds of ablation (v4.1.0 full formula, v4.1.0.1 penalty-only) that ANY non-zero per-step reward here regresses lt hard (success_rate 0.81→~0.52-0.54) without a clear cause tied to the match_gain term specifically. Reverted to v4.0.0's reward=0.0 behaviour as a data-backed final decision. The dual-ascent λ update is unaffected (driven by episode-level stats, not per-step reward).

### 4.4 `ppo_opt`（P6，最佳适应修复启发式 / best-fit repair heuristic）

策略在无约束动作空间采样；违规时环境**主动修复**（redirect），不是纯靠惩罚劝阻：

$$
j^* = \arg\max_{j \,\in\, \mathcal{V}(t)} \frac{n_t}{e_j}, \qquad \mathcal{V}(t) = \{\, j : \text{remaining\_vms}[j] \ge n_t \;\wedge\; \neg\text{conflict}(j,t) \,\}
$$

若 $\mathcal{V}(t) = \varnothing$（无可修复目标），episode 立即终止：$R_{\text{terminate-unrepairable}} = -M$。

**非终端步奖励（v4.1.0起接入，保留至今）**：

$$
r_t = \text{repair\_penalty}_t = -0.1 \cdot \mathbf{1}[\text{was\_repaired}_t]
$$

**中文**：v4.0.0阶段这行是死代码（算了没接线，非终端步恒为0，仅"无法修复"分支例外）。v4.1.0接入后，三场景**一致改善**（success_rate升、违规率降，lt场景尤其明显：conflict_viol $0.54\to0.15$），且实质缓解了此前发现的"策略靠环境修复钻空子（reward hacking）"问题——这是v4.1.0系列里效果最确定、最值得写进论文的一项修复。

### 4.5 `dqn` / `ddqn`（Q-learning 族 / Q-learning family）

环境结构与 4.1 节基本一致，无掩码无修复，唯一区别是引入了逐步惩罚。标准 Bellman 目标：

$$
y_t = r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a') \qquad \text{(DQN)}
$$

$$
y_t = r_t + \gamma \, Q_{\theta^-}\!\big(s_{t+1},\; \arg\max_{a'} Q_\theta(s_{t+1}, a')\big) \qquad \text{(Double DQN, 消除过高估计 / de-biases overestimation)}
$$

**非终端步奖励（v4.1.0起接入，保留至今）**：

$$
r_t = \text{cap\_penalty}_t + \text{conflict\_penalty}_t, \qquad \text{cap\_penalty}_t=\text{conflict\_penalty}_t=-2.0\cdot\mathbf{1}[\text{对应违规}]
$$

（`gt` 场景无独立 `cap_penalty`：容量违规在gt里是立即硬终止，故 $r_t = \text{conflict\_penalty}_t$）

**中文**：DDQN 与 DQN 唯一区别在于目标 Q 值的动作选择（在线网络 $\theta$）与评估（目标网络 $\theta^-$）解耦。v4.0.0阶段两者非终端步 `reward` 都恒为0（惩罚变量算了没接线）。v4.1.0接入后**三场景一致改善**（success_rate升、违规率降），是这套死代码bug里受益最直接的两个算法——因为它们既无掩码也无修复兜底，此前完全没有任何逐步反馈，密集惩罚信号补上后，Q-learning的Bellman bootstrap能相对有效地把这个信号传导回早期决策。

---

## 5. 六算法 v4.1.0 死代码修复：最终去留一览 / Final Disposition Table

| 算法 | v4.0.0非终端reward | v4.1.0是否接入修复 | 最终（v4.1.0.2）状态 | 效果 |
|---|---|---|---|---|
| ppo (P3) | 0（设计如此） | 不适用 | 0（未改动） | 不适用，对照基线 |
| ppo_mask (P4) | 0（死代码） | 是 | **保留** | 影响极小（掩码已结构性保证），无害 |
| ppo_lagrangian (P5) | 0（死代码） | 是→又撤回 | **回退为0** | 曾接入两版公式，lt场景均明显变差，数据支撑回退 |
| ppo_opt (P6) | 0（死代码） | 是 | **保留** | 三场景一致改善，lt尤其显著，缓解repair依赖问题 |
| dqn | 0（死代码） | 是 | **保留** | 三场景一致改善 |
| ddqn | 0（死代码） | 是 | **保留** | 三场景一致改善 |

---

## 6. 符号表 / Notation Table

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

*本文档反映 `final_paper_experiments` 分支 tag `v4.1.0.2` 的最终代码状态，涉及 `scenarios/{lt,eq,gt}/{ppo,ppo_mask,ppo_lagrangian,ppo_opt,dqn,ddqn}/env.py`。完整的修复/消融/回退过程见 `paper_contents/v4.1.0_changelog.md`。若代码后续更新，请重新核对本文档。*

