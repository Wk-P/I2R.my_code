# v6.0.1-lagrangian：标准 Lagrangian 定义

分支 `v6.0.1-lagrangian`（commit c2cc5e6）。代码：`src/paper_rl/env.py`（`ar_obj`、`step` 中 `reward_mode == "ar_obj"` 分支）、`src/paper_rl/train.py`（`_dual`、`_step_lambda`）。启动参数：`--mechanism lagrange --reward ar_obj --lambda-target 0.001`。

v6.0.0 的写法（ar_pen − λc_t）改称「ar_pen + 自适应违约惩罚」，不再称为 Lagrangian，见 `lagrangian_v6.0.0.md`。

## 1. 记号

| 符号 | 含义 |
|---|---|
| N, M | ECU 数、服务数 |
| c_n | ECU n 的容量 |
| e_m | 服务 m 的需求 |
| a_m | 服务 m 被放到的 ECU |
| c_t | 第 t 步违约数 = 𝟙[容量违约] + 𝟙[隐私违约] |
| 𝓔_exec | 被放置过服务的 ECU 集合（含违约放置） |
| λ | 对偶变量（episode 内固定，下次 reset 生效） |
| d | 约束上限，d = 0.001 |

## 2. CMDP

$$
\max_\pi\ J_R(\pi) \quad \text{s.t.}\quad J_C(\pi) \le d
$$

$$
J_R(\pi) = \mathbb{E}_\pi\big[\text{AR}_{\text{obj}}\big],\qquad
J_C(\pi) = \mathbb{E}_\pi\Big[\sum_{t=1}^{M} c_t\Big]
$$

J_R 是无约束目标，不含任何违约惩罚；违约只出现在 J_C 中。

## 3. 目标 AR_obj（含违约放置）

$$
\text{AR}_{\text{obj}} = \frac{1}{|\mathcal{E}_{\text{exec}}|} \sum_{n \in \mathcal{E}_{\text{exec}}} \min\!\Big(\frac{\sum_{m:\,a_m = n} e_m}{c_n},\ 1\Big)
$$

每个 ECU 的利用率截到 1，超载不能让 AR_obj 超过 1。成功局（无违约）中 AR_obj = AR。

## 4. 每步奖励（γ = 1，full episode，固定 M 步）

$$
r_t = -\lambda\, c_t \quad (t < M),\qquad
r_M = \text{AR}_{\text{obj}} - \lambda\, c_M
$$

整局回报：

$$
G = \text{AR}_{\text{obj}} - \lambda \sum_{t=1}^{M} c_t
$$

即 Lagrangian L(π, λ) = J_R − λ(J_C − d) 中与 π 有关的部分（λd 与 π 无关）。

## 5. λ 的更新（对偶上升）

$$
\mathcal{L}(\lambda) = -\lambda\,(J_C - d),\qquad \lambda \leftarrow \text{Adam}(\nabla_\lambda \mathcal{L}),\qquad \lambda \leftarrow \max(\lambda, 0)
$$

- J_C > d：λ 上升；J_C < d：λ 下降（v6.0.0 中 d = 0，λ 只升不降）
- PPO：每个 rollout 更新一次，J_C = 该 rollout 内结束的 episode 的平均违约数
- DQN / DDQN：每 1000 个 episode 更新一次，J_C = 这 1000 局的平均违约数
- Adam lr = 0.035，λ₀ = 0.001，λ 无上限，J_C 取行为策略；观测中附加 λ/(1+λ)

## 6. 与 v6.0.0 的区别

| 项 | v6.0.0 | v6.0.1-lagrangian |
|---|---|---|
| 终局奖励 | ar_pen：成功 AR，失败 −(1 − M_v/M) | AR_obj（失败局也给） |
| 违约惩罚次数 | 两次（ar_pen + λc_t） | 一次（λc_t） |
| d | 0 | 0.001 |
| λ 能否下降 | 否 | 是 |
| 名称 | ar_pen + 自适应违约惩罚 | Lagrangian |

## 7. 未解决

DQN/DDQN 回放池存的是旧 λ 下的奖励，λ 变化后不重算（与 v6.0.0 相同，未处理）。
