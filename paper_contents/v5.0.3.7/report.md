# v5.0.3.7 hyper-params lagrangian：常规 Safe RL 设置（OmniSafe PPOLag）——λ 初值 0.001、Adam（lr 0.035）更新 λ、λ ≥ 0 不设上限、J_C 取行为策略、d = 0；PPO 每个 rollout 更新一次，DQN / DDQN 每 1000 局更新一次（ε 1.0→0 固定 2M 步）（种子 1、5M、5 万实例）

- 生成时间：2026-10-14 21:00:08；脚本 `src/scripts/run_v5.py --version 5.0.3.7`。
- 每个变体 1 个种子（1）× 3 场景（LT / EQ / GT）、各训练 5M 步，同一次训练在 1M / 2M / 3M / 4M 存 checkpoint；ar_pen 奖励、γ = 1、不提前结束、EXIT 只用于 Maskable、需求降序；PPO / DQN 超参取自 `paper_rl/config.py`；结构感知网络（Transformer）在 GPU 上训练。5 万实例 = 经 ILP 验证的实例池（`results/sup_diag/<场景>.json`）。
- 测试集 gap：各种子自己的 400 个测试实例；训练集 gap：1600 个训练实例（5 万实例池中按固定种子抽取，或 1600 实例划分本身）；均为确定性回放；均值 ± 样本标准差。
- manifest：`logs/v5.0.3.7/manifest.json`。

## λ 动态（训练过程，来自 training_curve.csv）

J_C = 每局违约次数（容量 + 隐私）；目标 d = 0.001。「到上限」= 第一次 λ ≥ λ_max 的局；「降幅」= 最高点到训练结束的下降。

| 场景 | 模型 | λ_max | 末 λ | 最高 λ | 到上限（局） | 上限占比 | 下降次数 | 降幅 | 前 1000 局 J_C | 末 1000 局 J_C |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · lagrangian · hyper-params | 无 | 1.261 | 1.261 | — | 0.0% | 0 | 0.000 | 1.981 | 0.0540 |
| LT | DQN · lagrangian · hyper-params | 无 | 3.512 | 3.512 | — | 0.0% | 0 | 0.000 | 1.983 | 0.1110 |
| LT | DDQN · lagrangian · hyper-params | 无 | 3.461 | 3.461 | — | 0.0% | 0 | 0.000 | 1.983 | 0.0960 |
| EQ | PPO · lagrangian · hyper-params | 无 | 1.330 | 1.330 | — | 0.0% | 0 | 0.000 | 1.820 | 0.0040 |
| EQ | DQN · lagrangian · hyper-params | 无 | 4.646 | 4.646 | — | 0.0% | 0 | 0.000 | 1.812 | 0.0020 |
| EQ | DDQN · lagrangian · hyper-params | 无 | 4.606 | 4.606 | — | 0.0% | 0 | 0.000 | 1.811 | 0.0020 |
| GT | PPO · lagrangian · hyper-params | 无 | 1.025 | 1.025 | — | 0.0% | 0 | 0.000 | 1.645 | 0.1950 |
| GT | DQN · lagrangian · hyper-params | 无 | 5.012 | 5.012 | — | 0.0% | 0 | 0.000 | 1.672 | 0.0000 |
| GT | DDQN · lagrangian · hyper-params | 无 | 4.897 | 4.897 | — | 0.0% | 0 | 0.000 | 1.672 | 0.0000 |

### λ 与 J_C（来自 lambda_updates.csv，各步数前最后一次更新）

每格：λ / 行为策略 J_C / 用于更新的 J_C（贪心时为 ε = 0 的局）。

| 场景 | 模型 | 1M | 2M | 3M | 4M | 5M |
|---|---|---|---|---|---|---|
| LT | PPO · lagrangian · hyper-params | 0.85 / 0.210 / 0.210 | 0.99 / 0.105 / 0.105 | 1.09 / 0.084 / 0.084 | 1.18 / 0.058 / 0.058 | 1.26 / 0.068 / 0.068 |
| EQ | PPO · lagrangian · hyper-params | 1.05 / 0.012 / 0.012 | 1.28 / 0.015 / 0.015 | 1.30 / 0.001 / 0.001 | 1.32 / 0.005 / 0.005 | 1.33 / 0.001 / 0.001 |
| GT | PPO · lagrangian · hyper-params | 0.68 / 0.346 / 0.346 | 0.91 / 0.002 / 0.002 | 0.93 / 0.029 / 0.029 | 0.95 / 0.001 / 0.001 | 1.02 / 0.003 / 0.003 |
| LT | DQN · lagrangian · hyper-params | 2.11 / 5.224 / 5.224 | 3.22 / 0.223 / 0.223 | 3.35 / 0.124 / 0.124 | 3.44 / 0.100 / 0.100 | 3.51 / 0.138 / 0.138 |
| EQ | DQN · lagrangian · hyper-params | 3.07 / 2.179 / 2.179 | 4.58 / 0.041 / 0.041 | 4.61 / 0.000 / 0.000 | 4.64 / 0.001 / 0.001 | 4.65 / 0.002 / 0.002 |
| GT | DQN · lagrangian · hyper-params | 3.15 / 1.751 / 1.751 | 4.91 / 0.029 / 0.029 | 4.95 / 0.000 / 0.000 | 5.01 / 0.000 / 0.000 | 5.01 / 0.000 / 0.000 |
| LT | DDQN · lagrangian · hyper-params | 2.10 / 4.939 / 4.939 | 3.21 / 0.247 / 0.247 | 3.31 / 0.090 / 0.090 | 3.38 / 0.094 / 0.094 | 3.46 / 0.109 / 0.109 |
| EQ | DDQN · lagrangian · hyper-params | 3.06 / 2.099 / 2.099 | 4.56 / 0.041 / 0.041 | 4.60 / 0.000 / 0.000 | 4.60 / 0.001 / 0.001 | 4.61 / 0.000 / 0.000 |
| GT | DDQN · lagrangian · hyper-params | 3.11 / 1.740 / 1.740 | 4.83 / 0.029 / 0.029 | 4.89 / 0.000 / 0.000 | 4.89 / 0.000 / 0.000 | 4.90 / 0.000 / 0.000 |

## 学习曲线：测试集相对 gap

### LT

| 步数 | PPO · lagrangian · hyper-params | DQN · lagrangian · hyper-params | DDQN · lagrangian · hyper-params |
|---|---|---|---|
| 1M | 18.1% | 27.1% | 27.7% |
| 2M | 12.4% | 31.8% | 29.7% |
| 3M | 10.1% | 24.8% | 27.2% |
| 4M | 9.8% | 26.8% | 26.5% |
| 5M | 9.3% | 31.2% | 20.3% |

### EQ

| 步数 | PPO · lagrangian · hyper-params | DQN · lagrangian · hyper-params | DDQN · lagrangian · hyper-params |
|---|---|---|---|
| 1M | 17.0% | 46.3% | 44.1% |
| 2M | 12.1% | 36.5% | 19.9% |
| 3M | 10.5% | 17.5% | 16.9% |
| 4M | 9.7% | 16.7% | 17.6% |
| 5M | 9.7% | 17.0% | 14.1% |

### GT

| 步数 | PPO · lagrangian · hyper-params | DQN · lagrangian · hyper-params | DDQN · lagrangian · hyper-params |
|---|---|---|---|
| 1M | 12.4% | 30.7% | 45.4% |
| 2M | 10.3% | 16.0% | 22.4% |
| 3M | 10.7% | 15.8% | 16.2% |
| 4M | 10.6% | 16.1% | 14.8% |
| 5M | 10.9% | 12.2% | 14.6% |

## 学习曲线：三场景平均（每个种子先对三场景取平均）

| 步数 | PPO · lagrangian · hyper-params | DQN · lagrangian · hyper-params | DDQN · lagrangian · hyper-params |
|---|---|---|---|
| 1M | 15.8% | 34.7% | 39.1% |
| 2M | 11.6% | 28.1% | 24.0% |
| 3M | 10.4% | 19.3% | 20.1% |
| 4M | 10.0% | 19.8% | 19.6% |
| 5M | 10.0% | 20.2% | 16.3% |

## 学习曲线：测试集未通过率（EXIT 或任一违约）

gap 只在未通过率之外的实例（无 EXIT、无违约）上计算；未通过率高时，gap 只反映容易的那部分实例。

### LT

| 步数 | PPO · lagrangian · hyper-params | DQN · lagrangian · hyper-params | DDQN · lagrangian · hyper-params |
|---|---|---|---|
| 1M | 5.5% | 7.8% | 6.8% |
| 2M | 3.7% | 9.0% | 7.3% |
| 3M | 4.5% | 32.8% | 8.8% |
| 4M | 3.7% | 7.0% | 6.2% |
| 5M | 2.7% | 7.3% | 6.0% |

### EQ

| 步数 | PPO · lagrangian · hyper-params | DQN · lagrangian · hyper-params | DDQN · lagrangian · hyper-params |
|---|---|---|---|
| 1M | 16.8% | 0.2% | 0.0% |
| 2M | 0.2% | 0.2% | 0.5% |
| 3M | 0.2% | 0.0% | 0.2% |
| 4M | 0.0% | 0.0% | 0.0% |
| 5M | 0.2% | 0.0% | 0.0% |

### GT

| 步数 | PPO · lagrangian · hyper-params | DQN · lagrangian · hyper-params | DDQN · lagrangian · hyper-params |
|---|---|---|---|
| 1M | 5.0% | 0.0% | 0.0% |
| 2M | 0.2% | 0.0% | 0.0% |
| 3M | 1.0% | 0.0% | 0.0% |
| 4M | 0.2% | 0.0% | 0.0% |
| 5M | 0.7% | 0.0% | 0.0% |


## 5M：测试集与训练集 gap、违约、前几步决策

| 场景 | 模型 | 测试集相对 gap | 训练集相对 gap | 测试未通过率（EXIT / 违约） | 训练集未通过率 | 容量违约率 | 隐私违约率 | 第 1–5 步 regret | 第 1–5 步选中最优 |
|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · lagrangian · hyper-params | 9.3% | 9.2% | 2.7% | 2.7% | 2.5% | 1.2% | — | — |
| LT | DQN · lagrangian · hyper-params | 31.2% | 31.0% | 7.3% | 6.6% | 3.5% | 6.0% | — | — |
| LT | DDQN · lagrangian · hyper-params | 20.3% | 20.1% | 6.0% | 5.8% | 2.2% | 4.5% | — | — |
| EQ | PPO · lagrangian · hyper-params | 9.7% | 9.0% | 0.2% | 0.2% | 0.2% | 0.0% | — | — |
| EQ | DQN · lagrangian · hyper-params | 17.0% | 17.3% | 0.0% | 0.2% | 0.0% | 0.0% | — | — |
| EQ | DDQN · lagrangian · hyper-params | 14.1% | 14.2% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| GT | PPO · lagrangian · hyper-params | 10.9% | 10.7% | 0.7% | 0.8% | 0.2% | 0.5% | — | — |
| GT | DQN · lagrangian · hyper-params | 12.2% | 11.3% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DDQN · lagrangian · hyper-params | 14.6% | 14.6% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |

regret 按合法 ECU 之间的最优差距定义，只适用于 Maskable；其余机制可能执行违约动作，记为 —。


## 逐种子测试集 gap（5M）

| 场景 | 模型 | 种子 1 |
|---|---|---|
| LT | PPO · lagrangian · hyper-params | 9.3% |
| LT | DQN · lagrangian · hyper-params | 31.2% |
| LT | DDQN · lagrangian · hyper-params | 20.3% |
| EQ | PPO · lagrangian · hyper-params | 9.7% |
| EQ | DQN · lagrangian · hyper-params | 17.0% |
| EQ | DDQN · lagrangian · hyper-params | 14.1% |
| GT | PPO · lagrangian · hyper-params | 10.9% |
| GT | DQN · lagrangian · hyper-params | 12.2% |
| GT | DDQN · lagrangian · hyper-params | 14.6% |
