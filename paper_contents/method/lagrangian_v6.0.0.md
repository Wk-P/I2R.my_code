# v6.0.0 Lagrangian：AR 与奖励公式

> 本写法不是标准 Lagrangian，改称「ar_pen + 自适应违约惩罚」。标准定义见 `lagrangian_v6.0.1.md`。

代码：`src/paper_rl/env.py`（`step`、`_success_reward`、`_fail_reward`）、`src/paper_rl/train.py`（`_dual`、`_step_lambda`）；配置来自 tag v6.0.0（cbf08e9）的 `src/scripts/run_campaign.py`。Lagrangian 相关代码到当前 HEAD 未改动。

## 1. 记号

| 符号 | 含义 |
|---|---|
| N, M | ECU 数、服务数 |
| c_n | ECU n 的容量 |
| e_m | 服务 m 的需求 |
| a_m | 服务 m 被放到的 ECU |
| c_t | 第 t 步违约数 = 𝟙[容量违约] + 𝟙[隐私违约] |
| M_v | 合法放置的服务数 |
| λ | 对偶变量（episode 内固定，下次 reset 生效） |

## 2. AR 的累计（只计合法放置）

第 t 步把服务 m 放到 ECU n，若 c_t = 0：

$$
\text{legal\_ru} \leftarrow \text{legal\_ru} + \frac{e_m}{c_n},\qquad \mathcal{E}_{\text{legal}} \leftarrow \mathcal{E}_{\text{legal}} \cup \{n\}
$$

$$
\text{AR} = \frac{\sum_{\text{合法放置}} e_m / c_{a_m}}{|\mathcal{E}_{\text{legal}}|}
$$

即已用 ECU 的平均利用率；违约放置不进分子也不进分母。

## 3. 每步奖励（reward = ar_pen，γ = 1，full episode，固定 M 步）

$$
r_t = -\lambda\, c_t \quad (t < M)
$$

$$
r_M = R_{\text{term}} - \lambda\, c_M,\qquad
R_{\text{term}} =
\begin{cases}
\text{AR}, & M_v = M \\
-(1 - M_v / M), & M_v < M
\end{cases}
$$

## 4. 整局回报

$$
G = R_{\text{term}} - \lambda \sum_{t=1}^{M} c_t
$$

- 成功：G = AR
- 失败：G = −(1 − M_v/M) − λ·C_ep，C_ep 为该局违约总数

## 5. λ 的更新（对偶上升）

v6.0.0 配置：`lambda_update=adam`，lr = 0.035，λ₀ = 0.001，d = 0，λ 无上限，J_C 取行为策略。

- PPO：每个 rollout 结束更新一次，J_C = 该 rollout 内结束的 episode 的平均违约数
- DQN / DDQN：每 1000 个 episode 更新一次，J_C = 这 1000 局的平均违约数

$$
\mathcal{L}(\lambda) = -\lambda\,(J_C - d),\qquad \lambda \leftarrow \text{Adam}(\nabla_\lambda \mathcal{L}),\qquad \lambda \leftarrow \max(\lambda, 0)
$$

观测中附加 λ/(1+λ)（λ 无上限时）。

## 6. 评价

| 项 | 结论 |
|---|---|
| 与定稿口径（完整 ar_pen − λc_t）一致 | 是，实现无 bug |
| 是否标准 Lagrangian（L = J_R − λJ_C，J_R 为无约束目标） | 否：J_R 用的是 ar_pen，违约已被惩罚一次，再减 λc_t 是二次惩罚 |
| λ 能否回落 | 否：d = 0 且 J_C ≥ 0，梯度恒 ≤ 0；日志中 201 条 λ 记录下降 0 次 |
| 最终 λ | PPO 1.0–2.6，DQN/DDQN 4.8–5.5 |
| DQN/DDQN 回放池 | 存的是减过旧 λ·c_t 的奖励，λ 变化后未重算（推测影响 Lagrangian DQN 表现，未验证） |

标准形式的替代写法：r_M = AR_obj − λ·c_M（失败局也给 AR，`ar_obj` 见 v6.0.1，单 ECU 利用率截到 1），并设 d > 0。若保留现写法，论文中建议称为「ar_pen + 自适应违约惩罚」。
