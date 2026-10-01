# 核心公式汇总（v4.2.0）/ Core Formula Reference (v4.2.0)

**范围 / Scope**：本文档描述 v4.2.0 最优模型合集中**每个模型实际训练时**所用的 MDP、奖励和学习算法，以及论文结果所用的评估与计时方法。v4.2.0 是混编集合，不同格子的模型由不同代码版本训练，下表给出每格对应的训练代码；所有公式都已与该版本代码逐行核对。

| 算法 | LT | EQ | GT |
|---|---|---|---|
| PPO | v4.0.0 | v4.0.0 | v4.0.0 |
| Mask-PPO | v4.0.0 | v4.0.0 | v4.0.0 |
| Lagrange-PPO | v4.0.0 | v4.1.0 | v4.1.0.1 |
| Repair-PPO | v4.1.0 | v4.1.0 | v4.1.0 |
| DQN | v4.1.0 | v4.1.0 | v4.1.0 |
| DDQN | v4.1.0 | v4.1.0 | v4.1.0 |

训练代码：v4.0.0 = tag `v4.0.0_final_1`（82513ac，`env.py` 与训练时的 ce1dab0 相同）；v4.1.0 = tag `v4.1.0`（d0899cc）；v4.1.0.1 = 4757d10。模型清单见 `manifest.json`。

符号见第 8 节。

---

## 1. 问题定义与 AR / Problem Definition and AR

| 符号 | 含义 | Meaning |
|---|---|---|
| $N$ | ECU 数量 | number of ECUs |
| $M$ | 服务数量 | number of services |
| $e_j$ | ECU $j$ 的容量 | capacity of ECU $j$ |
| $n_i$ | 服务 $i$ 的资源需求 | requirement of service $i$ |
| $C_k$ | 第 $k$ 个冲突集（同一 ECU 上同一冲突集至多一个服务） | $k$-th conflict set |
| $x_{ij}, y_j \in \{0,1\}$ | 服务 $i$ 放到 ECU $j$ / ECU $j$ 被启用 | assignment / activation indicators |

| 场景 | $N$ | $M$ | 特点 |
|---|---|---|---|
| LT | 10 | 15 | 资源紧缺（$N<M$） |
| EQ | 10 | 10 | 供需均衡 |
| GT | 15 | 10 | 资源充裕（$N>M$） |

**资源利用率 AR**（$\mathcal{J}^*$ 为被启用的 ECU 集合）：

$$
AR = \frac{1}{|\mathcal{J}^*|} \sum_{j \in \mathcal{J}^*} \sum_{i:\,x_{ij}=1} \frac{n_i}{e_j}
$$

AR 只在同一场景内比较，跨场景平均没有意义。

---

## 2. ILP 最优解 / ILP Optimum (Dinkelbach)

AR 是分式目标，用 Dinkelbach 迭代求解参数化子问题：

$$
\max_{x,y}\; F(\lambda_D) = \sum_i\sum_j x_{ij}\frac{n_i}{e_j} - \lambda_D\sum_j y_j
$$

$$
\text{s.t.}\quad \sum_j x_{ij}=1\;\forall i,\qquad \sum_i x_{ij}n_i \le e_j y_j\;\forall j,\qquad x_{ij}\le y_j,\qquad \sum_{i\in C_k}x_{ij}\le 1\;\forall j,k,\qquad x_{ij}=0 \text{ if } n_i>e_j
$$

每轮用 CBC 求解后更新 $\lambda_D \leftarrow \dfrac{\sum_{ij}x_{ij}n_i/e_j}{\sum_j y_j}$，直到 $|\Delta\lambda_D|<10^{-6}$。收敛时 $\lambda_D$ 即最优 AR。ILP 是离线一次性求解，作为 AR 上界与耗时对照。

---

## 3. RL 共用 MDP / Shared RL MDP

### 3.1 决策过程 / Decision process

- 每个 episode 是一个测试/训练实例（$N$ 个 ECU、$M$ 个服务、冲突集）。第 $t$ 步（$t=0,\dots,M-1$）为队列中第 $t$ 个服务选一个 ECU：$a_t\in\{0,\dots,N-1\}$。
- **服务顺序**：数据集中服务是乱序的；部分模型在 `reset()` 中按需求量降序重排：

  | 场景 | 降序重排 | 原始顺序 |
  |---|---|---|
  | LT | Mask-PPO, Lagrange-PPO, Repair-PPO, DQN, DDQN | PPO |
  | EQ | Mask-PPO, Lagrange-PPO, Repair-PPO | PPO, DQN, DDQN |
  | GT | — | 全部 |

- **违规指示**（$R_t[j]$ 为 ECU $j$ 当前剩余容量）：

$$
v^{\text{cap}}_t = \mathbf{1}\big[R_t[a_t] < n_t\big],\qquad
v^{\text{conf}}_t = \mathbf{1}\big[\exists k:\ t\in C_k,\ \text{ECU } a_t \text{ 上已有 } C_k \text{ 中的其他服务}\big]
$$

- **状态转移**：执行放置后 $R_{t+1}[a_t] = R_t[a_t]-n_t$（允许为负，即容量违规照样执行；Repair-PPO 例外，见 4.4）。
- **环境内 AR**：$AR_t = \dfrac{\sum_{\tau<t} u_\tau}{|\mathcal{J}_t|}$，$|\mathcal{J}_t|$ 为已承载服务的 ECU 数，$u_\tau = n_\tau/e_{a_\tau}$。违规的那次放置是否计入 $u_\tau$ 因算法而异：

  | 计入 $u_\tau$ | 不计入（$u_\tau=0$，但该 ECU 仍算启用） |
  |---|---|
  | PPO、Lagrange-PPO、Repair-PPO（其执行的放置都合法） | Mask-PPO、DQN、DDQN |

- **终止**：正常情况下恰好 $M$ 步。提前终止只有两种：Repair-PPO 修复失败（4.4）；GT 场景 DQN/DDQN 出现容量违规（4.5）。

### 3.2 观测 / Observation

所有模型共享一组基础特征，另按算法追加若干项：

| 特征 | 维度 | 含义 |
|---|---|---|
| 标量 | 6 | 当前服务需求、当前 AR、剩余可用容量和、剩余服务需求和、当前服务可用 ECU 比例、剩余服务数比例（均归一化） |
| `initial_cap` | $N$ | 各 ECU 初始容量 / 最大容量 |
| `remaining` | $N$ | 各 ECU 剩余容量（可为负）/ 最大容量 |
| `conflict_flag` | $N$ | 当前服务放到各 ECU 是否冲突 |
| `valid_flag` | $N$ | 各 ECU 是否可放：PPO/Lagrange-PPO/DQN/DDQN 只看容量；Mask-PPO = 动作掩码；Repair-PPO = 容量且无冲突 |
| `remaining_svcs` | $M$ | 未放置服务的需求 / 最大容量（已放置为 0） |

追加项：

| 追加特征 | 维度 | 出现在 |
|---|---|---|
| `ecu_allowed_frac`（各 ECU 还能接收的服务比例） | $N$ | LT 全部算法；EQ/GT 的 Mask-PPO |
| `svc_valid_ecus`（每个未放置服务当前可行 ECU 数 / $N$） | $M$ | 除 PPO 外全部 |
| `bottleneck_risk`（未放置服务 $\frac{1}{\#\text{可行ECU}+1}$ 的均值） | 1 | Mask-PPO |
| $\text{clip}(\lambda/\lambda_{\max},0,1)$ | 1 | Lagrange-PPO |

因此观测维度为：PPO $5N+6+M$（LT）/ $4N+6+M$（EQ、GT）；Mask-PPO $5N+7+2M$；Lagrange-PPO $5N+7+2M$（LT）/ $4N+7+2M$；Repair-PPO、DQN、DDQN $5N+6+2M$（LT）/ $4N+6+2M$。

### 3.3 终局奖励 / Terminal reward

基础形式（第 $M$ 步结束时；$\text{valid}$ 为合法放置数）：

$$
R_{\text{term}} =
\begin{cases}
M\,(2\,AR-1), & \text{本 episode 无任何违规} \\[2pt]
-M\left(1-\dfrac{\text{valid}}{M}\right), & \text{否则}
\end{cases}
$$

成功支取值 $(-M, M]$ 随 AR 线性变化；失败支随完成度变化，始终不高于成功支。各算法的差异：

| 算法 | 终局奖励 |
|---|---|
| PPO、Lagrange-PPO、DQN、DDQN | 基础形式 |
| Mask-PPO（LT） | 成功支改为 $M\,[(1-w)+w(2AR-1)]$，$w$ 在整个训练期间从 0 线性升到 1 |
| Mask-PPO（EQ、GT） | 基础形式 |
| Repair-PPO | 见 4.4 |

---

## 4. 各算法 / Per-algorithm

### 4.1 PPO（无约束基线）

动作空间不受限，违规只记录不惩罚：

$$
r_t = 0\quad(t<M-1),\qquad r_{M-1}=R_{\text{term}}
$$

### 4.2 Mask-PPO（硬掩码）

用 MaskablePPO 训练，采样前把不可行 ECU 的概率置零：

$$
m_t[j] = \mathbf{1}\big[R_t[j]\ge n_t\big]\cdot\mathbf{1}\big[\text{放到 } j \text{ 不冲突}\big]
$$

**LT 额外一步前瞻**：从 $m_t$ 中再去掉"放下后会使某个后续服务没有任何可行 ECU"的选项 $j$；若去掉后全为 0，则退回 $m_t$。EQ/GT 只用 $m_t$。

若 $m_t$ 全为 0（无合法 ECU），MaskablePPO 对所有动作的 logit 都置为 $-10^8$，相当于在 $N$ 个 ECU 中均匀随机选择，这一步记为违规。

$$
r_t = 0\quad(t<M-1),\qquad r_{M-1}=R_{\text{term}}\ \text{（LT 含课程权重 } w\text{）}
$$

（代码中另有势函数塑形项 $\gamma\Phi(s')-\Phi(s)$，$\Phi=-\beta(1-\text{FFD可行度})$，但训练时 $\beta=0$，该项恒为 0。）

### 4.3 Lagrange-PPO（无掩码，惩罚 + 对偶变量）

动作空间不受限。对偶变量 $\lambda$ 由训练回调按违规率做对偶上升，并作为一维特征进入观测。三个场景的模型训练时的逐步奖励不同：

| 场景 | 非终端步奖励 $r_t$ |
|---|---|
| LT | $0$ |
| EQ | $\dfrac{u_t}{|\mathcal{J}_{t+1}|} \;-\; 2\,v^{\text{cap}}_t \;-\; (\lambda+0.2)\,v^{\text{conf}}_t$ |
| GT | $-\,2\,v^{\text{cap}}_t \;-\; (\lambda+0.2)\,v^{\text{conf}}_t$ |

终局奖励为基础形式。也就是说，EQ/GT 中 $\lambda$ 是冲突惩罚的系数，直接进入优化目标；LT 中 $\lambda$ 只通过观测特征影响策略。

**对偶上升**：每个 episode 的违规率 $\nu = (\#\text{容量违规}+\#\text{冲突违规})/M$；预热 $E_w$ 个 episode 后，每 $W$ 个 episode 用最近 $W$ 个 episode 的平均违规率 $\bar\nu$ 更新一次：

$$
\lambda \leftarrow \text{clip}\big(\lambda + \eta\,(\bar\nu - \nu^{*}),\ 0,\ \lambda_{\max}\big),\qquad \nu^{*}=0
$$

| 场景 | $\lambda_0$ | $\eta$ | $\lambda_{\max}$ | $W$ | $E_w$ |
|---|---|---|---|---|---|
| LT | 0 | $3\times10^{-4}$ | 2.0 | 20 | 20000 |
| EQ | 0.5 | 0.01 | 5.0 | 200 | 0 |
| GT | 0.1 | 0.005 | 5.0 | 20 | 5000 |

评估时 $\lambda$ 固定为训练结束时的值。

### 4.4 Repair-PPO（最佳适应修复）

策略在不受限的动作空间采样；若所选 ECU 违规，环境把动作替换为

$$
a_t \leftarrow \arg\max_{j\in\mathcal{V}_t} \frac{n_t}{e_j},\qquad
\mathcal{V}_t=\{\,j:\ R_t[j]\ge n_t,\ \text{放到 } j \text{ 不冲突}\,\}
$$

再执行，因此**实际执行的放置都合法**。若 $\mathcal{V}_t=\varnothing$，episode 立即终止，奖励 $-M$。

$$
r_t = -0.1\cdot\mathbf{1}[\text{第 } t \text{ 步触发修复}]\quad(t<M-1)
$$

跑满 $M$ 步时的终局奖励：

| 场景 | $R_{\text{term}}$ |
|---|---|
| LT | 未触发过修复：$M(2AR-1)$；触发过修复：$0$（失败支 $-M(1-\text{valid}/M)$，而此时 $\text{valid}=M$） |
| EQ、GT | $M(2AR-1)$（修复不改变终局判定） |

### 4.5 DQN / DDQN

动作空间不受限，无掩码、无修复。两者环境完全相同，只在目标值上不同：

$$
y^{\text{DQN}} = r_t + \gamma\max_{a'}Q_{\theta^-}(s_{t+1},a'),\qquad
y^{\text{DDQN}} = r_t + \gamma\,Q_{\theta^-}\big(s_{t+1},\arg\max_{a'}Q_\theta(s_{t+1},a')\big)
$$

损失为 Huber（smooth L1）$\ \mathcal{L}=\text{Huber}\big(Q_\theta(s_t,a_t)-y\big)$，终止步 $y=r_t$；探索为 $\varepsilon$-greedy，$\varepsilon$ 在前 $f$ 比例的训练步内从 1 线性降到 0。

| 场景 | 非终端步奖励 $r_t$ | 容量违规 |
|---|---|---|
| LT、EQ | $-2\,v^{\text{cap}}_t - 2\,v^{\text{conf}}_t$ | 照常执行 |
| GT | $-2\,v^{\text{conf}}_t$ | episode 立即终止，奖励 $-M$ |

### 4.6 PPO 系列的优化目标

PPO、Mask-PPO、Lagrange-PPO、Repair-PPO 都用标准 PPO-Clip（Mask-PPO 的策略分布先经过掩码）：

$$
\mathcal{L}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t\hat A_t,\ \text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big] - c_v\,\mathbb{E}_t\big[(V_\theta(s_t)-\hat R_t)^2\big] + c_e\,\mathbb{E}_t\big[\mathcal{H}(\pi_\theta(\cdot|s_t))\big]
$$

$$
\rho_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)},\qquad
\hat A_t=\sum_{l\ge0}(\gamma\lambda_{\text{GAE}})^l\delta_{t+l},\qquad
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

$\hat A_t$ 在每个 minibatch 内标准化。$c_v=0.5$（SB3 默认）。熵系数 $c_e$ 见第 6 节（Mask-PPO LT 随训练线性退火）。

---

## 5. 非终端步奖励一览 / Non-terminal reward summary

| 算法 | LT | EQ | GT |
|---|---|---|---|
| PPO | 0 | 0 | 0 |
| Mask-PPO | 0 | 0 | 0 |
| Lagrange-PPO | 0 | $\frac{u_t}{|\mathcal{J}_{t+1}|}-2v^{\text{cap}}_t-(\lambda+0.2)v^{\text{conf}}_t$ | $-2v^{\text{cap}}_t-(\lambda+0.2)v^{\text{conf}}_t$ |
| Repair-PPO | $-0.1\cdot\mathbf{1}[\text{修复}]$ | 同左 | 同左 |
| DQN / DDQN | $-2v^{\text{cap}}_t-2v^{\text{conf}}_t$ | 同左 | $-2v^{\text{conf}}_t$（容量违规即终止） |

| 约束处理 | 算法 | 执行的放置能否违规 |
|---|---|---|
| 不处理（只记录） | PPO | 能 |
| 采样前掩码 | Mask-PPO | 只在无合法 ECU 时 |
| 惩罚 + 对偶变量 | Lagrange-PPO | 能 |
| 执行前修复 | Repair-PPO | 不能（修复失败则提前终止） |
| 惩罚 | DQN、DDQN | 能（GT 容量违规直接终止） |

---

## 6. 训练超参数 / Training hyperparameters

所有模型均训练 $5\times10^6$ 环境步；并行环境数 PPO 系列 40、DQN/DDQN 12。

**PPO 系列**

| | PPO | Mask-PPO | Lagrange-PPO | Repair-PPO |
|---|---|---|---|---|
| 学习率 | 3e-4 | 3e-4 | 3e-4 | 1e-4 |
| n_steps / batch / epochs | 512 / 256 / 10 | 512 / 256 / 10 | 512 / 256 / 10 | 1024 / 64 / 20 |
| $\gamma$ | 0.999 | 0.99 | 0.99 | 0.999（EQ 0.99） |
| $\lambda_{\text{GAE}}$ / clip $\epsilon$ | 0.95 / 0.2 | 0.95 / 0.2 | 0.95 / 0.2 | 0.95 / 0.3 |
| 熵系数 $c_e$ | 0 | LT 0.02→0.002 线性退火；EQ/GT 0.005 | 0.005 | 0.001 |
| 网络（策略 / 价值） | [256,256] / [256,256] | LT、GT [256,256]/[256,256]；EQ [256,256]/[512,512] | [256,256]/[512,512] | [256,256,256] / [256,256,256] |

**DQN / DDQN**（两者相同）

| 学习率 | buffer | learning_starts | batch | $\gamma$ | train_freq | 目标网络更新 | 探索比例 $f$ / 最终 $\varepsilon$ | 网络 |
|---|---|---|---|---|---|---|---|---|
| 1e-3 | 100,000 | 2000（EQ 64） | 64 | 0.99 | 4 | 每 500 步硬更新 | 0.5（EQ 0.1）/ 0 | [128,128] |

---

## 7. 评估与计时 / Evaluation and timing

- **数据划分**：每个场景 2000 个实例，按种子打乱后 80/20 划分，测试集 400 个。每个模型在它自己种子的测试集上评估。
- **评估方式**：每个测试实例只跑一次，策略取确定性动作（PPO 系列取概率最大的动作，DQN/DDQN 取 $\arg\max Q$，$\varepsilon=0$）。Lagrange-PPO 的 $\lambda$ 固定为训练结束值。Mask-PPO 遇到无合法 ECU 时该 episode 立即结束。
- **成功率**：

$$
\text{success} = \mathbf{1}\big[\text{valid}=M\big]
$$

其中 PPO、Lagrange-PPO、DQN、DDQN 同时要求无容量违规和冲突违规；Mask-PPO 和 Repair-PPO 执行的放置都合法，失败只来自提前结束（无合法 ECU / 修复失败）。

- **AR**：测试 episode 结束时的环境内 AR 的均值（含失败 episode）。
- **违规率**：实际执行的放置中出现 ≥1 次容量（或冲突）违规的测试 episode 比例。Repair-PPO 恒为 0；其"策略原始动作违规、被修复"的 episode 比例单独记为修复触发率。
- **统计**：表中为种子间均值 ± 样本标准差（ddof=1），PPO / Mask-PPO / LT 的 Lagrange-PPO 为 5 个种子，其余 3 个种子。
- **耗时**：单线程 CPU。ILP 为种子 1 测试集 400 个实例逐个求解的平均时间；RL 为一个 episode（$M$ 次决策，含环境计算与网络前向）的平均时间，先在每个种子的 400 个实例上平均，再在种子间平均，计时前预热 5 个 episode。

---

## 8. 符号表 / Notation

| 符号 | 含义 | Meaning |
|---|---|---|
| $N, M$ | ECU 数、服务数 | number of ECUs / services |
| $t$ | 时间步（当前服务下标） | timestep (current service index) |
| $e_j$, $n_t$ | ECU 容量、当前服务需求 | ECU capacity, current requirement |
| $R_t[j]$ | ECU $j$ 剩余容量 | remaining capacity of ECU $j$ |
| $C_k$ | 冲突集 | conflict set |
| $a_t$ | 动作（所选 ECU） | action (chosen ECU) |
| $v^{\text{cap}}_t, v^{\text{conf}}_t$ | 容量 / 冲突违规指示 | capacity / conflict violation indicator |
| $u_t$ | 本步利用率贡献 $n_t/e_{a_t}$ | per-step utilisation $n_t/e_{a_t}$ |
| $\mathcal{J}_t$, $\mathcal{J}^*$ | 已启用 ECU 集合 | set of activated ECUs |
| $AR$ | 平均资源利用率 | allocation ratio |
| valid | 合法放置的服务数 | number of validly placed services |
| $\lambda_D$ | Dinkelbach 参数 | Dinkelbach parameter |
| $\lambda$ | Lagrange-PPO 对偶变量 | dual variable of Lagrange-PPO |
| $\nu$ | episode 违规率 | episode violation rate |
| $m_t$ | 动作掩码 | action mask |
| $w$ | Mask-PPO（LT）终局奖励课程权重 | curriculum weight (Mask-PPO, LT) |
| $\gamma$, $\lambda_{\text{GAE}}$, $\epsilon$ | 折扣、GAE 参数、PPO 裁剪 | discount, GAE parameter, PPO clip |
| $\theta$, $\theta^-$ | 在线 / 目标网络参数 | online / target network parameters |
