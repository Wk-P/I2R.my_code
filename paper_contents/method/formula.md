# 公式汇总：ILP 与 RL 算法

本文给出论文结果所用的全部公式，每一条都已与代码逐行核对。

| 部分 | 代码版本 |
|---|---|
| 论文主表（v6.0.0-merge）的 Mask / Repair / Lagrangian / Unconstraint 模型 | v6.0.0，commit cbf08e9，`src/scripts/run_campaign.py` |
| 标准 Lagrangian（新定义，尚未训练） | 分支 `v6.0.1-lagrangian`，commit c2cc5e6 |
| ILP、实例生成、测试集 | `src/shared/ilp_utils.py`（`solve_ilp_max_ar`）、`src/paper_rl/data.py`、`src/scripts/diag_supervised.py`、`src/scripts/gen_testset_v6_0_4.py` |
| 评估指标 | `src/scripts/eval_testset_v6_0_4.py`（`summarize`） |

公式中下标从 1 开始，代码中从 0 开始。

---

## 1. 问题与记号

| 符号 | 含义 |
|---|---|
| N, M | ECU 数、服务数（LT：N = 10, M = 15；EQ：N = 10, M = 10；GT：N = 15, M = 10） |
| n ∈ {1, …, N} | ECU 下标 |
| m ∈ {1, …, M} | 服务下标 |
| c_n | ECU n 的容量（容器数），取值 {50, 55, …, 195} |
| e_m | 服务 m 的需求（容器数），取值 {10, 15, …, 95} |
| 𝒞_k, k = 1, …, K | 隐私冲突集（K = 10），同一冲突集的两个服务不能放在同一 ECU |
| x_mn ∈ {0, 1} | 服务 m 是否放在 ECU n |
| y_n ∈ {0, 1} | ECU n 是否开启 |

**AR（平均资源利用率）**：已开启 ECU 的利用率的平均值。

$$
\text{AR}(x, y) = \frac{F(x)}{G(y)},\qquad F(x) = \sum_{m=1}^{M}\sum_{n=1}^{N} x_{mn}\,\frac{e_m}{c_n},\qquad G(y) = \sum_{n=1}^{N} y_n
$$

**约束**：

$$
\begin{aligned}
&\textstyle\sum_{n} x_{mn} = 1 && \forall m &&\text{（每个服务恰好放一次）}\\
&\textstyle\sum_{m} x_{mn}\, e_m \le c_n\, y_n && \forall n &&\text{（容量；未开启的 ECU 不能放服务）}\\
&x_{mn} \le y_n && \forall m, n &&\text{（开启关联）}\\
&\textstyle\sum_{m \in \mathcal{C}_k} x_{mn} \le 1 && \forall k, \forall n &&\text{（隐私冲突）}\\
&x_{mn},\, y_n \in \{0, 1\}
\end{aligned}
$$

最优值记为 AR\*（ILP AR）：

$$
\text{AR}^* = \max_{(x, y)\ \text{可行}}\ \frac{F(x)}{G(y)}
$$

---

## 2. ILP：Dinkelbach 方法

目标 AR = F/G 是比值，不能直接交给 ILP 求解器。Dinkelbach 方法用一个变量 ρ 表示"目前找到的最好 AR"，每一轮求解一个线性目标的参数化 ILP（约束同第 1 节）：

$$
\varphi(\rho) = \max_{(x, y)\ \text{可行}}\ \big[\, F(x) - \rho\, G(y) \,\big]
$$

**每一轮在问什么**：因为 G(y) ≥ 1，

$$
F(x) - \rho\, G(y) > 0 \iff \frac{F(x)}{G(y)} > \rho \iff \text{方案 } (x, y) \text{ 的 AR} > \rho
$$

所以 φ(ρ) > 0 表示存在 AR 比 ρ 高的方案，φ(ρ) = 0 表示没有。

**迭代**：

1. ρ₀ = 0。
2. 用 CBC（PuLP）求解 φ(ρ_t)，状态必须为 `Optimal`，得到最优解 (x_t, y_t)。
3. ρ_{t+1} = AR(x_t, y_t) = F(x_t) / G(y_t)，其中 G 取实际放了服务的 ECU 数。
4. |ρ_{t+1} − ρ_t| < 10⁻⁶ 时停止，输出 AR\* = ρ_{t+1}；否则回到第 2 步（最多 30 轮）。

**收敛性**：

$$
\varphi(\rho) > 0 \iff \rho < \text{AR}^*,\qquad \varphi(\rho) = 0 \iff \rho = \text{AR}^*
$$

F(x_t) − ρ_{t+1} G(y_t) = 0 只对上一轮的那个解成立；φ 是对所有可行解取最大值，只要还有 AR 更高的方案，φ 就大于 0。只有当 ρ 已经是最大的 AR 时，φ 才等于 0。所以收敛时 φ = 0 是停止条件，输出的最优 AR 是 ρ，不是 φ。每轮的解都可行，ρ_t 单调不降；可行解只有有限个，迭代在有限步内停止。

**示意**（假设只有 3 个可行方案，AR 分别为 0.70、0.80、0.90）：

| 轮 | ρ | φ(ρ) | ILP 返回的方案（之一） | 新的 ρ |
|---|---|---|---|---|
| 1 | 0 | > 0 | AR 0.70 | 0.70 |
| 2 | 0.70 | > 0（0.80、0.90 都超过 0.70） | AR 0.90 | 0.90 |
| 3 | 0.90 | 0（没有方案超过 0.90） | — | 停止，AR\* = 0.90 |

**实例**（EQ 验证集第 1 个实例）：φ(0) = 5.3754（7 个 ECU，AR = 0.7679）；φ(0.7679) = 0.0000，停止，AR\* = 0.7679。

注：ρ 只用于 ILP 求解；第 6.4 节 Lagrangian 的 λ 是另一个量（违约惩罚的对偶变量）。

---

## 3. 数据

- **生成**：c_n、e_m 分别在上述取值中独立均匀抽取。K = 10 个冲突集，每个服务以概率 q 独立加入每个集合，成员少于 2 的集合丢弃。

$$
q = \sqrt{1 - (1 - p)^{1/K}},\qquad p = 1 - (1 - q^2)^K = 0.6
$$

  p 是任意两个服务共处同一冲突集（即互相冲突）的概率。

- **筛选**：每个实例都用第 2 节的方法求解，状态不是 `Optimal`（无可行解）就整体重抽。所以每个实例都可行，并且 AR\* 已知。
- **训练**：`results/sup_diag/<场景>.json`，每场景 5 万个实例，每局从中均匀随机抽一个。
- **测试**：`data/testset_v6.0.4/<场景>.json`，每场景 5000 个实例，所有种子和模型共用。与训练池生成器相同，种子不重叠，并且按实例内容去重。

---

## 4. MDP（四种机制共用）

### 4.1 回合

服务按需求降序排列（相同需求保持原顺序），重新编号使 e_1 ≥ e_2 ≥ … ≥ e_M。第 t 步（t = 1, …, M）放置服务 t。

full episode：除了 Mask 的 EXIT，不提前结束，每局固定 M 步。

### 4.2 动作

$$
a_t \in \{1, \dots, N\}\ \ (\text{Unconstraint / Repair / Lagrangian}),\qquad a_t \in \{1, \dots, N, \text{EXIT}\}\ \ (\text{Mask})
$$

### 4.3 违约与转移

设第 t 步实际执行的 ECU 为 n = ã_t（Repair 修复后的 ECU，其余机制即 a_t）。第 t 步之前已放在 ECU n 上的服务集合记为 H_n，剩余容量记为 r_n = c_n − Σ_{m∈H_n} e_m。H_n 包括违约的放置，所以 r_n 可以为负。

$$
v^{\text{cap}}_t = \mathbb{1}\big[r_n < e_t\big],\qquad
v^{\text{priv}}_t = \mathbb{1}\big[\exists\, m \in H_n,\ \exists\, k:\ \{t, m\} \subseteq \mathcal{C}_k\big]
$$

$$
c_t = v^{\text{cap}}_t + v^{\text{priv}}_t \in \{0, 1, 2\}
$$

注意：c_t 是第 t 步的违约数，与 ECU 容量 c_n 不是同一个量。

执行之后：H_n ← H_n ∪ {t}，r_n ← r_n − e_t。违约的放置也会照样执行。

服务 t 在 ECU n 上**可行**，当且仅当 r_n ≥ e_t 且没有隐私冲突。当前服务的可行集合记为

$$
\mathcal{A}_t = \{\, n : r_n \ge e_t,\ \text{服务 } t \text{ 与 } H_n \text{ 无冲突} \,\}
$$

### 4.4 AR 与合法放置数

只累计合法放置（c_t = 0 的步）：

$$
M_v = \sum_{t} \mathbb{1}[c_t = 0],\qquad
\mathcal{E}_{\text{legal}} = \{\, \tilde a_t : c_t = 0 \,\},\qquad
\text{AR} = \frac{\sum_{t:\, c_t = 0} e_t / c_{\tilde a_t}}{|\mathcal{E}_{\text{legal}}|}
$$

（𝓔_legal 为空时 AR = 0。）违约的放置既不计入分子，也不计入分母。EXIT 之后未放置的服务同样不计入。无违约完成（M_v = M）时，AR 就是该放置方案的 AR(x, y)，不会超过 AR\*。

### 4.5 状态

环境输出原始状态：c（N 维）、e（M 维，按放置顺序）、冲突图 A ∈ {0,1}^{M×M}（A_{mk} = 1 当且仅当 m ≠ k 且二者同属某个 𝒞_k）、每个服务所在 ECU（0 表示未放）、当前步 t。Lagrangian 另外在最后加一维 λ/(1 + λ)（λ 无上限时）。

网络内部再由原始状态构造特征（`graph_net.build`）。记 load_n = 已放在 ECU n 上的需求和（包括违约放置），free_n = c_n − load_n，c_max = max_n c_n：

| token | 特征 |
|---|---|
| 全局（1 个） | AR_net；(M − t)/M；未放服务需求和 / Σ_n c_n；Σ_n max(free_n, 0) / Σ_n c_n；［Lagrangian：λ/(1 + λ)］ |
| ECU n（N 个） | c_n / c_max；max(free_n, 0) / c_max；是否已开启；对当前服务是否可行；已放服务数 / M；load_n / c_n |
| 服务 k（M 个） | e_k / c_max；是否已放；是否为当前服务；对它可行的 ECU 数 / N；与其他未放服务的冲突数 / M（已放为 0） |

其中 AR_net = Σ_{开启的 n} (load_n / c_n) / 开启 ECU 数。它包括违约放置，只是网络的输入特征，不是第 4.4 节的 AR。

关系矩阵有 4 种，作为注意力的偏置：未放服务之间的冲突；服务放在 ECU 上；未放服务与 ECU 上已有服务冲突；未放服务的需求 ≤ ECU 的 free。四种机制的网络输入相同（Unconstraint 也能看到可行性特征）；Lagrangian 只多了 λ 这一维。

---

## 5. 奖励（ar_pen，γ = 1）

终局奖励：

$$
R_{\text{term}} =
\begin{cases}
\text{AR}, & M_v = M\ \ (\text{无违约完成；AR} \in (0, 1])\\[2pt]
-\left(1 - \dfrac{M_v}{M}\right), & \text{否则（含 EXIT；取值 } [-1, 0)\text{）}
\end{cases}
$$

每步奖励（T 为该局最后一步：T = M，EXIT 时为 EXIT 那一步）：

$$
r_t = \begin{cases} 0, & t < T \\ R_{\text{term}}, & t = T \end{cases}
\qquad\qquad
G = \sum_{t=1}^{T} r_t = R_{\text{term}}
$$

Lagrangian 在此基础上每步再减 λ c_t（第 6.4 节）。

---

## 6. 四种约束机制

### 6.1 Unconstraint

动作按原样执行，不做任何约束处理。奖励即第 5 节的 ar_pen：违约只通过终局奖励的失败分支间接受罚。

### 6.2 Mask（含 EXIT）

动作掩码：

$$
\text{mask}_t(n) = \mathbb{1}[n \in \mathcal{A}_t],\qquad \text{mask}_t(\text{EXIT}) = \mathbb{1}[\mathcal{A}_t = \varnothing]
$$

- 有可行 ECU 时，只能选可行 ECU，EXIT 被屏蔽。
- 没有可行 ECU 时，只能选 EXIT。回合立即结束，奖励为 −(1 − M_v/M)；此时 M_v = t − 1。
- 所以 Mask 从不执行违约放置：c_t ≡ 0，失败只来自 EXIT。

PPO：网络把不可行动作的 logit 置为 −10⁹，MaskablePPO 再按同一掩码采样。DQN / DDQN：探索、贪心选动作和 TD 目标都只在掩码内取（第 8.3 节）。

### 6.3 Repair

$$
\tilde a_t =
\begin{cases}
a_t, & a_t \in \mathcal{A}_t\\[2pt]
\arg\max_{n \in \mathcal{A}_t} \dfrac{e_t}{c_n}, & a_t \notin \mathcal{A}_t,\ \mathcal{A}_t \ne \varnothing \quad(\text{best-fit：容量最小的可行 ECU；并列取下标最小})\\[2pt]
a_t, & \mathcal{A}_t = \varnothing \quad(\text{无处可修，按原动作执行并记违约})
\end{cases}
$$

奖励为 ar_pen。学习器存储和训练用的是智能体选的 a_t，不是修复后的 ã_t。

### 6.4 Lagrangian

#### (a) v6.0.0：论文主表所用版本，实为「ar_pen + 自适应违约惩罚」

$$
r_t = -\lambda c_t\ \ (t < M),\qquad r_M = R_{\text{term}} - \lambda c_M,\qquad G = R_{\text{term}} - \lambda \sum_{t=1}^{M} c_t
$$

λ 在一局内固定，在每局开始（reset）时取最新值。对偶更新（Adam，lr = 0.035，λ₀ = 0.001，d = 0，λ 无上限）：

$$
\mathcal{L}(\lambda) = -\lambda\,(J_C - d),\qquad \lambda \leftarrow \text{Adam}\big(\nabla_\lambda \mathcal{L}\big),\qquad \lambda \leftarrow \max(\lambda, 0)
$$

$$
J_C = \frac{1}{|\mathcal{B}|} \sum_{\text{局} \in \mathcal{B}} \sum_{t} c_t\quad(\text{行为策略的每局平均违约数})
$$

- PPO：每个 rollout（40 × 512 = 20480 步）结束后更新一次，𝓑 = 该 rollout 内结束的局。
- DQN / DDQN：每 1000 局更新一次，𝓑 = 这 1000 局（不重叠）。

问题：R_term 已经因为违约罚过一次，λc_t 又罚一次；而且 d = 0、J_C ≥ 0，所以梯度恒 ≤ 0，λ 只升不降（201 条 λ 记录中下降 0 次）。所以它不是标准 Lagrangian，论文中称为「ar_pen + 自适应违约惩罚」。

#### (b) v6.0.1-lagrangian：标准 Lagrangian（新定义，尚未训练）

约束 MDP：

$$
\max_\pi\ J_R(\pi) = \mathbb{E}_\pi\big[\text{AR}_{\text{obj}}\big]\qquad \text{s.t.}\qquad J_C(\pi) = \mathbb{E}_\pi\Big[\sum_{t=1}^{M} c_t\Big] \le d,\quad d = 0.001
$$

无约束目标 AR_obj 包括违约放置，每个 ECU 的利用率截到 1（𝓔_exec 为放过服务的 ECU）：

$$
\text{AR}_{\text{obj}} = \frac{1}{|\mathcal{E}_{\text{exec}}|} \sum_{n \in \mathcal{E}_{\text{exec}}} \min\!\Big(\frac{\sum_{t:\, \tilde a_t = n} e_t}{c_n},\ 1\Big)
$$

$$
r_t = -\lambda c_t\ \ (t < M),\qquad r_M = \text{AR}_{\text{obj}} - \lambda c_M,\qquad G = \text{AR}_{\text{obj}} - \lambda \sum_{t} c_t
$$

G 正是 Lagrangian L(π, λ) = J_R − λ(J_C − d) 中与 π 有关的部分。对偶更新与 (a) 相同，只是 d = 0.001：J_C > d 时 λ 上升，J_C < d 时 λ 下降。启动参数：`--mechanism lagrange --reward ar_obj --lambda-target 0.001`。

---

## 7. 网络（结构感知注意力网络，`graph_net.GraphPolicyNet`）

L = 1 + N + M 个 token（全局、ECU、服务），宽度 d = 128，4 个头，3 层。

**输入**：h⁰ = Linear_τ(f) + E_τ。τ ∈ {全局, ECU, 服务} 是 token 类型，每类有自己的线性层 Linear_τ 和类型嵌入 E_τ（E 初始化为 0）。

**每层**（pre-LN）：

$$
\alpha^{(h)} = \text{softmax}\!\Big( \frac{Q^{(h)} K^{(h)\top}}{\sqrt{d/4}} + \sum_{r=1}^{4} w^{(\ell)}_{r,h}\, R_r \Big),\qquad
x \leftarrow x + W_o\,[\alpha^{(h)} V^{(h)}]_h,\qquad
x \leftarrow x + \text{FF}(\text{LN}(x))
$$

其中 Q、K、V 由 LN(x) 线性得到，R_r 是第 4.5 节的 4 种关系矩阵，w 是每层、每个头、每种关系的可学习标量（初始化为 0），FF 为 d → 2d → d（GELU）。最后再接一层 LN。

**输出头**：

$$
s_n = \text{MLP}_{\text{ecu}}(h_n)\ (d \to d \to 1,\ \text{所有 ECU 共享}),\qquad
s_{\text{EXIT}} = w^\top h_{\text{glob}} + b,\qquad
V(s) = \text{MLP}_v\big([\,h_{\text{glob}},\ \tfrac{1}{N+M}\textstyle\sum_{\text{非全局 token}} h\,]\big)
$$

- PPO：π(a | s) = softmax(s_a)。Mask 中不可行动作的 s_a 置为 −10⁹；其余机制只取 N 个 ECU 的分数，不做掩码。
- DQN / DDQN：Q(s, n) = s_n，Q(s, EXIT) = s_EXIT（只有 Mask 有 EXIT）；价值头不参与。
- 交换 ECU（或服务）的顺序，输出也随之交换，即网络对排列等变。

---

## 8. 学习算法

所有网络都用 Adam 训练，学习率在训练过程中保持不变；γ = 1；每个 run 训练 2 × 10⁷ 步。

### 8.1 PPO（Unconstraint / Repair / Lagrangian 用 SB3 PPO；Mask 用 sb3-contrib MaskablePPO）

优势用 GAE（γ = 1，λ_GAE = 0.95）：

$$
\delta_t = r_t + \gamma (1 - d_t)\, V(s_{t+1}) - V(s_t),\qquad
\hat A_t = \sum_{l \ge 0} (\gamma \lambda_{\text{GAE}})^l\, \delta_{t+l},\qquad
\hat R_t = \hat A_t + V(s_t)
$$

d_t 为终止标志；Â_t 在每个小批量内做标准化。损失：

$$
\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)},\qquad
L^{\text{clip}} = \mathbb{E}\big[\min\big(\rho_t \hat A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\, \hat A_t\big)\big]
$$

$$
L = -L^{\text{clip}} + c_v\, \mathbb{E}\big[(V_\theta(s_t) - \hat R_t)^2\big] - c_e\, \mathbb{E}\big[\mathcal{H}(\pi_\theta(\cdot \mid s_t))\big]
$$

| 参数 | 值 |
|---|---|
| 并行环境数 × 每环境步数 | 40 × 512 = 20480 步 / rollout |
| 小批量、轮数 | 256，10 轮（每个 rollout 800 次梯度更新） |
| ε（clip）、c_v、c_e | 0.1、0.5、0.005 |
| 优化器 | Adam，lr = 3 × 10⁻⁴，eps = 10⁻⁵（SB3 对 PPO 的设置），梯度范数裁剪 0.5 |

### 8.2 DQN / Double DQN（SB3 DQN 与 `shared/dqn_variants.py`）

TD 目标（Q̄ 为目标网络）：

$$
\text{DQN:}\quad y = r + \gamma (1 - d) \max_{a'} \bar Q(s', a')
$$

$$
\text{Double DQN:}\quad y = r + \gamma (1 - d)\, \bar Q\big(s',\ \arg\max_{a'} Q(s', a')\big)
$$

损失为 Huber（smooth L1，β = 1）：ℒ = 𝔼[Huber(Q(s, a) − y)]。

| 参数 | 值 |
|---|---|
| 并行环境数 | 12 |
| 回放池、批量、开始学习 | 10⁵，64，2000 步之后 |
| 更新频率 | 每 4 次环境调用（4 × 12 = 48 步）做 1 次梯度更新 |
| 目标网络 | 每 ⌊500/12⌋ = 41 次环境调用（≈ 492 步）完整复制一次（τ = 1） |
| ε-greedy | ε 在前 2 × 10⁶ 步内从 1 线性降到 0，之后为 0；同一次环境调用的 12 个环境共用一次随机判定 |
| 优化器 | Adam，lr = 10⁻³，PyTorch 默认参数（eps = 10⁻⁸），梯度范数裁剪 10 |

### 8.3 Mask-DQN / Mask-DDQN

掩码 𝒜(s) 与第 6.2 节相同（含 EXIT），由网络从原始状态算出。

- 探索与开始学习前：在 𝒜(s) 内均匀随机选动作。
- 贪心：a = argmax_{a ∈ 𝒜(s)} Q(s, a)。
- TD 目标：

$$
\text{Mask-DQN:}\quad y = r + \gamma (1 - d) \max_{a' \in \mathcal{A}(s')} \bar Q(s', a')
$$

$$
\text{Mask-DDQN:}\quad y = r + \gamma (1 - d)\, \bar Q\big(s',\ \arg\max_{a' \in \mathcal{A}(s')} Q(s', a')\big)
$$

### 8.4 Lagrangian 的对偶变量

见第 6.4 节。λ 用一个单独的 Adam（lr = 0.035），与网络的优化器无关。

---

## 9. 评估指标

每个测试实例运行一次确定性策略（PPO 取概率最大的动作，DQN 取 argmax Q；Mask 带掩码；Lagrangian 用训练结束时的 λ）。设测试集为 𝒯（|𝒯| = 5000），无违约完成的实例集合为 𝒯_ok = {i : M_v^{(i)} = M}：

$$
\text{gap} = 1 - \frac{\frac{1}{|\mathcal{T}|}\sum_{i \in \mathcal{T}} \text{AR}^{(i)}}{\frac{1}{|\mathcal{T}|}\sum_{i \in \mathcal{T}} \text{AR}^{*(i)}}
$$

$$
\text{无违约实例占比} = \frac{|\mathcal{T}_{\text{ok}}|}{|\mathcal{T}|},\qquad
\text{无违约实例相对 gap} = \frac{\overline{\text{AR}^*}_{\mathcal{T}_{\text{ok}}} - \overline{\text{AR}}_{\mathcal{T}_{\text{ok}}}}{\overline{\text{AR}^*}_{\mathcal{T}_{\text{ok}}}}
$$

$$
\text{容量违约率} = \frac{1}{|\mathcal{T}|}\sum_{i} \mathbb{1}\Big[\textstyle\sum_t v^{\text{cap}(i)}_t > 0\Big],\quad
\text{隐私违约率} = \frac{1}{|\mathcal{T}|}\sum_{i} \mathbb{1}\Big[\textstyle\sum_t v^{\text{priv}(i)}_t > 0\Big],\quad
\text{EXIT 率} = \frac{1}{|\mathcal{T}|}\sum_{i} \mathbb{1}[\text{EXIT}^{(i)}]
$$

- AR 只计合法放置（第 4.4 节）。gap 在全部实例上计算，失败实例也计入（取它的合法部分 AR）。
- 训练日志里的 "AR gap" 是 𝒯_ok 上的绝对差 ILP AR − AR，不是相对值。
- 报告中的表格对各种子的指标取均值 ± 标准差。

---

## 10. 实现细节（会影响解读）

1. Lagrangian 的第一局：训练环境用 λ = 0.1（`config.LAMBDA_INIT`）创建，训练开始时才设为 λ₀ = 0.001，并在下一次 reset 生效。所以每个并行环境的第一局用的是 λ = 0.1。
2. Lagrangian DQN / DDQN：回放池里存的是采样时的 r_t（已经减去当时的 λc_t），λ 变化后不会重算。
3. Repair：存入回放池（DQN）或用于计算 log π（PPO）的是智能体选的 a_t，不是修复后的 ã_t。
4. 网络特征里的 AR_net 和 load 都包括违约放置（第 4.5 节）。奖励和评估中的 AR 只计合法放置（第 4.4 节），两者不同。
