# v5.1.0 lagrangian 20M version：沿用 v5.0.3.7 配置（λ 初值 0.001、Adam lr 0.035、λ ≥ 0 不设上限、J_C 取行为策略、d = 0；PPO 每个 rollout 更新一次，DQN / DDQN 每 1000 局、ε 1.0→0 固定 2M 步），扩展到 20M（种子 1、20M、5 万实例）

- 生成时间：2026-10-15 04:56:53；脚本 `src/scripts/run_v5.py --version 5.1.0`。
- 每个变体 1 个种子（1）× 3 场景（LT / EQ / GT）、各训练 20M 步，同一次训练在 5M / 10M / 15M 存 checkpoint；ar_pen 奖励、γ = 1、不提前结束、EXIT 只用于 Maskable、需求降序；PPO / DQN 超参取自 `paper_rl/config.py`；结构感知网络（Transformer）在 GPU 上训练。5 万实例 = 经 ILP 验证的实例池（`results/sup_diag/<场景>.json`）。
- 测试集 gap：各种子自己的 400 个测试实例；训练集 gap：1600 个训练实例（5 万实例池中按固定种子抽取，或 1600 实例划分本身）；均为确定性回放；均值 ± 样本标准差。
- manifest：`logs/v5.1.0/manifest.json`。

## λ 动态（训练过程，来自 training_curve.csv）

J_C = 每局违约次数（容量 + 隐私）；目标 d = 0。「到上限」= 第一次 λ ≥ λ_max 的局；「降幅」= 最高点到训练结束的下降。

| 场景 | 模型 | λ_max | 末 λ | 最高 λ | 到上限（局） | 上限占比 | 下降次数 | 降幅 | 前 1000 局 J_C | 末 1000 局 J_C |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · lagrangian 20M version | 无 | 2.971 | 2.971 | — | 0.0% | 0 | 0.000 | 1.981 | 0.0440 |
| LT | DQN · lagrangian 20M version | 无 | 5.597 | 5.597 | — | 0.0% | 0 | 0.000 | 1.983 | 0.0960 |
| LT | DDQN · lagrangian 20M version | 无 | 5.307 | 5.307 | — | 0.0% | 0 | 0.000 | 1.983 | 0.0890 |
| EQ | PPO · lagrangian 20M version | 无 | 1.554 | 1.554 | — | 0.0% | 0 | 0.000 | 1.820 | 0.0090 |
| EQ | DQN · lagrangian 20M version | 无 | 4.735 | 4.735 | — | 0.0% | 0 | 0.000 | 1.812 | 0.0000 |
| EQ | DDQN · lagrangian 20M version | 无 | 5.079 | 5.079 | — | 0.0% | 0 | 0.000 | 1.811 | 0.0000 |
| GT | PPO · lagrangian 20M version | 无 | 1.623 | 1.623 | — | 0.0% | 0 | 0.000 | 1.645 | 0.0000 |
| GT | DQN · lagrangian 20M version | 无 | 5.097 | 5.097 | — | 0.0% | 0 | 0.000 | 1.672 | 0.0000 |
| GT | DDQN · lagrangian 20M version | 无 | 4.923 | 4.923 | — | 0.0% | 0 | 0.000 | 1.672 | 0.0000 |

### λ 与 J_C（来自 lambda_updates.csv，各步数前最后一次更新）

每格：λ / 行为策略 J_C / 用于更新的 J_C（贪心时为 ε = 0 的局）。

| 场景 | 模型 | 5M | 10M | 15M | 20M |
|---|---|---|---|---|---|
| LT | PPO · lagrangian 20M version | 1.26 / 0.068 / 0.068 | 1.75 / 0.034 / 0.034 | 2.26 / 0.043 / 0.043 | 2.97 / 0.053 / 0.053 |
| EQ | PPO · lagrangian 20M version | 1.33 / 0.001 / 0.001 | 1.44 / 0.000 / 0.000 | 1.50 / 0.001 / 0.001 | 1.55 / 0.025 / 0.025 |
| GT | PPO · lagrangian 20M version | 1.02 / 0.003 / 0.003 | 1.28 / 0.000 / 0.000 | 1.34 / 0.000 / 0.000 | 1.62 / 0.001 / 0.001 |
| LT | DQN · lagrangian 20M version | 3.51 / 0.138 / 0.138 | 4.06 / 0.085 / 0.085 | 4.75 / 0.092 / 0.092 | 5.60 / 0.089 / 0.089 |
| EQ | DQN · lagrangian 20M version | 4.65 / 0.002 / 0.002 | 4.71 / 0.001 / 0.001 | 4.72 / 0.001 / 0.001 | 4.74 / 0.001 / 0.001 |
| GT | DQN · lagrangian 20M version | 5.01 / 0.000 / 0.000 | 5.04 / 0.000 / 0.000 | 5.05 / 0.000 / 0.000 | 5.10 / 0.000 / 0.000 |
| LT | DDQN · lagrangian 20M version | 3.46 / 0.109 / 0.109 | 3.89 / 0.098 / 0.098 | 4.53 / 0.096 / 0.096 | 5.31 / 0.081 / 0.081 |
| EQ | DDQN · lagrangian 20M version | 4.61 / 0.000 / 0.000 | 4.65 / 0.001 / 0.001 | 4.93 / 0.002 / 0.002 | 5.08 / 0.001 / 0.001 |
| GT | DDQN · lagrangian 20M version | 4.90 / 0.000 / 0.000 | 4.91 / 0.000 / 0.000 | 4.92 / 0.000 / 0.000 | 4.92 / 0.000 / 0.000 |

## 学习曲线：测试集相对 gap

### LT

| 步数 | PPO · lagrangian 20M version | DQN · lagrangian 20M version | DDQN · lagrangian 20M version |
|---|---|---|---|
| 5M | 9.3% | 31.1% | 20.4% |
| 10M | 7.8% | 24.9% | 30.5% |
| 15M | 7.0% | 26.9% | 26.8% |
| 20M | 6.7% | 25.9% | 26.7% |

### EQ

| 步数 | PPO · lagrangian 20M version | DQN · lagrangian 20M version | DDQN · lagrangian 20M version |
|---|---|---|---|
| 5M | 9.7% | 17.2% | 14.2% |
| 10M | 7.6% | 11.7% | 16.7% |
| 15M | 6.9% | 8.0% | 16.9% |
| 20M | 6.0% | 8.1% | 11.9% |

### GT

| 步数 | PPO · lagrangian 20M version | DQN · lagrangian 20M version | DDQN · lagrangian 20M version |
|---|---|---|---|
| 5M | 11.1% | 12.2% | 14.7% |
| 10M | 9.6% | 7.9% | 10.7% |
| 15M | 8.9% | 6.1% | 7.3% |
| 20M | 7.2% | 3.4% | 6.5% |

## 学习曲线：三场景平均（每个种子先对三场景取平均）

| 步数 | PPO · lagrangian 20M version | DQN · lagrangian 20M version | DDQN · lagrangian 20M version |
|---|---|---|---|
| 5M | 10.0% | 20.2% | 16.4% |
| 10M | 8.3% | 14.8% | 19.3% |
| 15M | 7.6% | 13.7% | 17.0% |
| 20M | 6.6% | 12.5% | 15.0% |

## 学习曲线：测试集未通过率（EXIT 或任一违约）

gap 只在未通过率之外的实例（无 EXIT、无违约）上计算；未通过率高时，gap 只反映容易的那部分实例。

### LT

| 步数 | PPO · lagrangian 20M version | DQN · lagrangian 20M version | DDQN · lagrangian 20M version |
|---|---|---|---|
| 5M | 2.2% | 7.5% | 6.0% |
| 10M | 2.7% | 7.8% | 6.5% |
| 15M | 2.7% | 6.0% | 7.3% |
| 20M | 2.7% | 6.8% | 6.0% |

### EQ

| 步数 | PPO · lagrangian 20M version | DQN · lagrangian 20M version | DDQN · lagrangian 20M version |
|---|---|---|---|
| 5M | 0.5% | 0.0% | 0.0% |
| 10M | 0.0% | 0.0% | 0.0% |
| 15M | 0.0% | 0.0% | 0.0% |
| 20M | 0.0% | 0.0% | 0.0% |

### GT

| 步数 | PPO · lagrangian 20M version | DQN · lagrangian 20M version | DDQN · lagrangian 20M version |
|---|---|---|---|
| 5M | 4.0% | 0.0% | 0.0% |
| 10M | 0.0% | 0.0% | 0.0% |
| 15M | 0.2% | 0.0% | 0.0% |
| 20M | 0.0% | 0.0% | 0.0% |


## 20M：测试集与训练集 gap、违约、前几步决策

| 场景 | 模型 | 测试集相对 gap | 训练集相对 gap | 测试未通过率（EXIT / 违约） | 训练集未通过率 | 容量违约率 | 隐私违约率 | 第 1–5 步 regret | 第 1–5 步选中最优 |
|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · lagrangian 20M version | 6.7% | 6.4% | 2.7% | 2.2% | 2.8% | 0.2% | — | — |
| LT | DQN · lagrangian 20M version | 25.9% | 25.3% | 6.8% | 5.8% | 2.0% | 5.2% | — | — |
| LT | DDQN · lagrangian 20M version | 26.7% | 26.5% | 6.0% | 6.4% | 3.8% | 3.8% | — | — |
| EQ | PPO · lagrangian 20M version | 6.0% | 6.1% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DQN · lagrangian 20M version | 8.1% | 8.5% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DDQN · lagrangian 20M version | 11.9% | 12.0% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| GT | PPO · lagrangian 20M version | 7.2% | 7.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DQN · lagrangian 20M version | 3.4% | 3.4% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DDQN · lagrangian 20M version | 6.5% | 6.2% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |

regret 按合法 ECU 之间的最优差距定义，只适用于 Maskable；其余机制可能执行违约动作，记为 —。


## 逐种子测试集 gap（20M）

| 场景 | 模型 | 种子 1 |
|---|---|---|
| LT | PPO · lagrangian 20M version | 6.7% |
| LT | DQN · lagrangian 20M version | 25.9% |
| LT | DDQN · lagrangian 20M version | 26.7% |
| EQ | PPO · lagrangian 20M version | 6.0% |
| EQ | DQN · lagrangian 20M version | 8.1% |
| EQ | DDQN · lagrangian 20M version | 11.9% |
| GT | PPO · lagrangian 20M version | 7.2% |
| GT | DQN · lagrangian 20M version | 3.4% |
| GT | DDQN · lagrangian 20M version | 6.5% |
