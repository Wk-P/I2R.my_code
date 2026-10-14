# v5.0.3.4 DQN / DDQN lagrangian λ research：λ_max = 5、归一化更新 η = 0.025，ε 从 1.0 固定 2M 步降到 0、J_C 取 ε = 0 贪心策略（每次更新 100 局）（种子 1、5M、5 万实例）

- 生成时间：2026-10-14 10:40:16；脚本 `src/scripts/run_v5.py --version 5.0.3.4`。
- 每个变体 1 个种子（1）× 3 场景（LT / EQ / GT）、各训练 5M 步，同一次训练在 1M / 2M / 3M / 4M 存 checkpoint；ar_pen 奖励、γ = 1、不提前结束、EXIT 只用于 Maskable、需求降序；PPO / DQN 超参取自 `paper_rl/config.py`；结构感知网络（Transformer）在 GPU 上训练。5 万实例 = 经 ILP 验证的实例池（`results/sup_diag/<场景>.json`）。
- 测试集 gap：各种子自己的 400 个测试实例；训练集 gap：1600 个训练实例（5 万实例池中按固定种子抽取，或 1600 实例划分本身）；均为确定性回放；均值 ± 样本标准差。
- manifest：`logs/v5.0.3.4/manifest.json`。

## λ 动态（训练过程，来自 training_curve.csv）

J_C = 每局违约次数（容量 + 隐私）；目标 d = 0.001。「到上限」= 第一次 λ ≥ λ_max 的局；「降幅」= 最高点到训练结束的下降。

| 场景 | 模型 | λ_max | 末 λ | 最高 λ | 到上限（局） | 上限占比 | 下降次数 | 降幅 | 前 1000 局 J_C | 末 1000 局 J_C |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | DQN · lagrangian · λ research | 5 | 5.000 | 5.000 | 198000 | 40.6% | 0 | 0.000 | 1.983 | 0.1780 |
| LT | DDQN · lagrangian · λ research | 5 | 5.000 | 5.000 | 198000 | 40.6% | 0 | 0.000 | 1.983 | 0.0950 |
| EQ | DQN · lagrangian · λ research | 5 | 0.101 | 0.208 | — | 0.0% | 168 | 0.107 | 1.810 | 0.0860 |
| EQ | DDQN · lagrangian · λ research | 5 | 0.119 | 0.192 | — | 0.0% | 188 | 0.073 | 1.812 | 0.0100 |
| GT | DQN · lagrangian · λ research | 5 | 0.149 | 0.215 | — | 0.0% | 141 | 0.067 | 1.669 | 0.0360 |
| GT | DDQN · lagrangian · λ research | 5 | 0.049 | 0.170 | — | 0.0% | 56 | 0.121 | 1.669 | 0.0620 |

### λ 与 J_C（来自 lambda_updates.csv，各步数前最后一次更新）

每格：λ / 行为策略 J_C / 用于更新的 J_C（贪心时为 ε = 0 的局）。

| 场景 | 模型 | 1M | 2M | 3M | 4M | 5M |
|---|---|---|---|---|---|---|
| LT | DQN · lagrangian · λ research | 1.74 / 5.075 / 0.240 | 3.39 / 0.189 / 0.120 | 5.00 / 0.257 / 0.250 | 5.00 / 0.148 / 0.180 | 5.00 / 0.408 / 0.100 |
| EQ | DQN · lagrangian · λ research | 0.02 / 2.541 / 0.020 | 0.00 / 0.046 / 0.000 | 0.00 / 0.001 / 0.000 | 0.07 / 0.108 / 0.110 | 0.10 / 0.015 / 0.070 |
| GT | DQN · lagrangian · λ research | 0.00 / 1.946 / 0.000 | 0.05 / 0.134 / 0.000 | 0.07 / 0.023 / 0.000 | 0.00 / 0.005 / 0.000 | 0.15 / 0.239 / 0.100 |
| LT | DDQN · lagrangian · λ research | 1.74 / 4.969 / 0.140 | 3.39 / 0.213 / 0.120 | 5.00 / 0.123 / 0.190 | 5.00 / 0.103 / 0.170 | 5.00 / 0.122 / 0.070 |
| EQ | DDQN · lagrangian · λ research | 0.02 / 2.467 / 0.020 | 0.00 / 0.048 / 0.000 | 0.06 / 0.003 / 0.000 | 0.12 / 0.038 / 0.020 | 0.12 / 0.003 / 0.000 |
| GT | DDQN · lagrangian · λ research | 0.00 / 1.926 / 0.000 | 0.00 / 0.056 / 0.000 | 0.00 / 0.000 / 0.000 | 0.07 / 0.094 / 0.210 | 0.05 / 0.090 / 0.300 |

## 学习曲线：测试集相对 gap

### LT

| 步数 | DQN · lagrangian · λ research | DDQN · lagrangian · λ research |
|---|---|---|
| 1M | 33.1% | 30.7% |
| 2M | 28.9% | 29.4% |
| 3M | 20.8% | 28.6% |
| 4M | 24.4% | 31.3% |
| 5M | 24.4% | 30.8% |

### EQ

| 步数 | DQN · lagrangian · λ research | DDQN · lagrangian · λ research |
|---|---|---|
| 1M | 29.2% | 42.1% |
| 2M | 31.8% | 32.0% |
| 3M | 14.7% | 16.7% |
| 4M | 12.6% | 14.4% |
| 5M | 13.6% | 12.0% |

### GT

| 步数 | DQN · lagrangian · λ research | DDQN · lagrangian · λ research |
|---|---|---|
| 1M | 27.1% | 28.3% |
| 2M | 26.2% | 20.5% |
| 3M | 20.2% | 14.3% |
| 4M | 15.2% | 15.7% |
| 5M | 16.1% | 16.0% |

## 学习曲线：三场景平均（每个种子先对三场景取平均）

| 步数 | DQN · lagrangian · λ research | DDQN · lagrangian · λ research |
|---|---|---|
| 1M | 29.8% | 33.7% |
| 2M | 29.0% | 27.3% |
| 3M | 18.6% | 19.9% |
| 4M | 17.4% | 20.5% |
| 5M | 18.0% | 19.6% |

## 学习曲线：测试集未通过率（EXIT 或任一违约）

gap 只在未通过率之外的实例（无 EXIT、无违约）上计算；未通过率高时，gap 只反映容易的那部分实例。

### LT

| 步数 | DQN · lagrangian · λ research | DDQN · lagrangian · λ research |
|---|---|---|
| 1M | 7.0% | 8.8% |
| 2M | 9.5% | 7.5% |
| 3M | 7.3% | 7.0% |
| 4M | 7.3% | 9.8% |
| 5M | 7.8% | 6.2% |

### EQ

| 步数 | DQN · lagrangian · λ research | DDQN · lagrangian · λ research |
|---|---|---|
| 1M | 0.5% | 0.2% |
| 2M | 0.5% | 0.5% |
| 3M | 0.2% | 0.5% |
| 4M | 4.5% | 0.2% |
| 5M | 24.0% | 0.5% |

### GT

| 步数 | DQN · lagrangian · λ research | DDQN · lagrangian · λ research |
|---|---|---|
| 1M | 0.0% | 0.0% |
| 2M | 0.0% | 0.0% |
| 3M | 0.5% | 0.0% |
| 4M | 0.5% | 4.5% |
| 5M | 3.0% | 3.0% |


## 5M：测试集与训练集 gap、违约、前几步决策

| 场景 | 模型 | 测试集相对 gap | 训练集相对 gap | 测试未通过率（EXIT / 违约） | 训练集未通过率 | 容量违约率 | 隐私违约率 | 第 1–5 步 regret | 第 1–5 步选中最优 |
|---|---|---|---|---|---|---|---|---|---|
| LT | DQN · lagrangian · λ research | 24.4% | 24.0% | 7.8% | 6.5% | 3.8% | 7.2% | — | — |
| LT | DDQN · lagrangian · λ research | 30.8% | 30.3% | 6.2% | 6.3% | 2.2% | 5.2% | — | — |
| EQ | DQN · lagrangian · λ research | 13.6% | 14.2% | 24.0% | 21.0% | 0.0% | 24.0% | — | — |
| EQ | DDQN · lagrangian · λ research | 12.0% | 12.4% | 0.5% | 0.6% | 0.2% | 0.5% | — | — |
| GT | DQN · lagrangian · λ research | 16.1% | 15.7% | 3.0% | 5.2% | 2.2% | 3.0% | — | — |
| GT | DDQN · lagrangian · λ research | 16.0% | 16.2% | 3.0% | 2.8% | 0.2% | 3.0% | — | — |

regret 按合法 ECU 之间的最优差距定义，只适用于 Maskable；其余机制可能执行违约动作，记为 —。


## 逐种子测试集 gap（5M）

| 场景 | 模型 | 种子 1 |
|---|---|---|
| LT | DQN · lagrangian · λ research | 24.4% |
| LT | DDQN · lagrangian · λ research | 30.8% |
| EQ | DQN · lagrangian · λ research | 13.6% |
| EQ | DDQN · lagrangian · λ research | 12.0% |
| GT | DQN · lagrangian · λ research | 16.1% |
| GT | DDQN · lagrangian · λ research | 16.0% |
