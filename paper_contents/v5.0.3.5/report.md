# v5.0.3.5 DQN series 20M：λ_max = 5、归一化更新 η = 0.025，ε 从 1.0 固定 2M 步降到 0、J_C 取 ε = 0 贪心策略（每次更新 100 局）（种子 1、20M、5 万实例）

- 生成时间：2026-10-14 13:56:59；脚本 `src/scripts/run_v5.py --version 5.0.3.5`。
- 每个变体 1 个种子（1）× 3 场景（LT / EQ / GT）、各训练 20M 步，同一次训练在 5M / 10M / 15M 存 checkpoint；ar_pen 奖励、γ = 1、不提前结束、EXIT 只用于 Maskable、需求降序；PPO / DQN 超参取自 `paper_rl/config.py`；结构感知网络（Transformer）在 GPU 上训练。5 万实例 = 经 ILP 验证的实例池（`results/sup_diag/<场景>.json`）。
- 测试集 gap：各种子自己的 400 个测试实例；训练集 gap：1600 个训练实例（5 万实例池中按固定种子抽取，或 1600 实例划分本身）；均为确定性回放；均值 ± 样本标准差。
- manifest：`logs/v5.0.3.5/manifest.json`。

## λ 动态（训练过程，来自 training_curve.csv）

J_C = 每局违约次数（容量 + 隐私）；目标 d = 0.001。「到上限」= 第一次 λ ≥ λ_max 的局；「降幅」= 最高点到训练结束的下降。

| 场景 | 模型 | λ_max | 末 λ | 最高 λ | 到上限（局） | 上限占比 | 下降次数 | 降幅 | 前 1000 局 J_C | 末 1000 局 J_C |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | DQN · lagrangian · DQN series 20M | 5 | 5.000 | 5.000 | 198000 | 85.1% | 0 | 0.000 | 1.983 | 0.1030 |
| LT | DDQN · lagrangian · DQN series 20M | 5 | 5.000 | 5.000 | 198000 | 85.1% | 0 | 0.000 | 1.983 | 0.0870 |
| EQ | DQN · lagrangian · DQN series 20M | 5 | 0.098 | 0.241 | — | 0.0% | 893 | 0.143 | 1.810 | 0.0610 |
| EQ | DDQN · lagrangian · DQN series 20M | 5 | 0.000 | 0.323 | — | 0.0% | 885 | 0.323 | 1.812 | 0.0000 |
| GT | DQN · lagrangian · DQN series 20M | 5 | 0.136 | 0.245 | — | 0.0% | 836 | 0.109 | 1.669 | 0.0020 |
| GT | DDQN · lagrangian · DQN series 20M | 5 | 0.212 | 0.264 | — | 0.0% | 583 | 0.052 | 1.669 | 0.0100 |

### λ 与 J_C（来自 lambda_updates.csv，各步数前最后一次更新）

每格：λ / 行为策略 J_C / 用于更新的 J_C（贪心时为 ε = 0 的局）。

| 场景 | 模型 | 5M | 10M | 15M | 20M |
|---|---|---|---|---|---|
| LT | DQN · lagrangian · DQN series 20M | 5.00 / 0.408 / 0.100 | 5.00 / 0.096 / 0.170 | 5.00 / 0.094 / 0.140 | 5.00 / 0.100 / 0.090 |
| EQ | DQN · lagrangian · DQN series 20M | 0.10 / 0.015 / 0.070 | 0.06 / 0.015 / 0.000 | 0.12 / 0.003 / 0.000 | 0.10 / 0.001 / 0.000 |
| GT | DQN · lagrangian · DQN series 20M | 0.15 / 0.239 / 0.100 | 0.07 / 0.003 / 0.000 | 0.18 / 0.000 / 0.000 | 0.14 / 0.001 / 0.000 |
| LT | DDQN · lagrangian · DQN series 20M | 5.00 / 0.122 / 0.070 | 5.00 / 0.099 / 0.150 | 5.00 / 0.098 / 0.220 | 5.00 / 0.076 / 0.080 |
| EQ | DDQN · lagrangian · DQN series 20M | 0.12 / 0.003 / 0.000 | 0.12 / 0.009 / 0.000 | 0.12 / 0.012 / 0.000 | 0.00 / 0.002 / 0.000 |
| GT | DDQN · lagrangian · DQN series 20M | 0.05 / 0.090 / 0.300 | 0.00 / 0.001 / 0.000 | 0.07 / 0.003 / 0.000 | 0.21 / 0.012 / 0.010 |

## 学习曲线：测试集相对 gap

### LT

| 步数 | DQN · lagrangian · DQN series 20M | DDQN · lagrangian · DQN series 20M |
|---|---|---|
| 5M | 24.3% | 30.6% |
| 10M | 22.8% | 24.4% |
| 15M | 22.8% | 19.9% |
| 20M | 31.4% | 18.7% |

### EQ

| 步数 | DQN · lagrangian · DQN series 20M | DDQN · lagrangian · DQN series 20M |
|---|---|---|
| 5M | 13.4% | 12.2% |
| 10M | 9.6% | 7.2% |
| 15M | 8.2% | 6.7% |
| 20M | 6.2% | 9.7% |

### GT

| 步数 | DQN · lagrangian · DQN series 20M | DDQN · lagrangian · DQN series 20M |
|---|---|---|
| 5M | 16.1% | 16.1% |
| 10M | 17.0% | 11.2% |
| 15M | 9.2% | 10.2% |
| 20M | 7.1% | 8.1% |

## 学习曲线：三场景平均（每个种子先对三场景取平均）

| 步数 | DQN · lagrangian · DQN series 20M | DDQN · lagrangian · DQN series 20M |
|---|---|---|
| 5M | 17.9% | 19.7% |
| 10M | 16.5% | 14.3% |
| 15M | 13.4% | 12.2% |
| 20M | 14.9% | 12.1% |

## 学习曲线：测试集未通过率（EXIT 或任一违约）

gap 只在未通过率之外的实例（无 EXIT、无违约）上计算；未通过率高时，gap 只反映容易的那部分实例。

### LT

| 步数 | DQN · lagrangian · DQN series 20M | DDQN · lagrangian · DQN series 20M |
|---|---|---|
| 5M | 7.5% | 6.2% |
| 10M | 6.8% | 6.2% |
| 15M | 6.0% | 7.0% |
| 20M | 6.8% | 6.5% |

### EQ

| 步数 | DQN · lagrangian · DQN series 20M | DDQN · lagrangian · DQN series 20M |
|---|---|---|
| 5M | 23.8% | 0.5% |
| 10M | 0.0% | 1.7% |
| 15M | 0.7% | 2.2% |
| 20M | 4.2% | 0.0% |

### GT

| 步数 | DQN · lagrangian · DQN series 20M | DDQN · lagrangian · DQN series 20M |
|---|---|---|
| 5M | 3.2% | 2.7% |
| 10M | 0.7% | 0.0% |
| 15M | 0.2% | 0.0% |
| 20M | 0.0% | 1.0% |


## 20M：测试集与训练集 gap、违约、前几步决策

| 场景 | 模型 | 测试集相对 gap | 训练集相对 gap | 测试未通过率（EXIT / 违约） | 训练集未通过率 | 容量违约率 | 隐私违约率 | 第 1–5 步 regret | 第 1–5 步选中最优 |
|---|---|---|---|---|---|---|---|---|---|
| LT | DQN · lagrangian · DQN series 20M | 31.4% | 31.1% | 6.8% | 7.0% | 5.0% | 2.8% | — | — |
| LT | DDQN · lagrangian · DQN series 20M | 18.7% | 18.1% | 6.5% | 6.1% | 5.0% | 4.8% | — | — |
| EQ | DQN · lagrangian · DQN series 20M | 6.2% | 6.2% | 4.2% | 3.3% | 0.0% | 4.2% | — | — |
| EQ | DDQN · lagrangian · DQN series 20M | 9.7% | 9.3% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| GT | DQN · lagrangian · DQN series 20M | 7.1% | 6.6% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DDQN · lagrangian · DQN series 20M | 8.1% | 8.0% | 1.0% | 0.9% | 0.0% | 1.0% | — | — |

regret 按合法 ECU 之间的最优差距定义，只适用于 Maskable；其余机制可能执行违约动作，记为 —。


## 逐种子测试集 gap（20M）

| 场景 | 模型 | 种子 1 |
|---|---|---|
| LT | DQN · lagrangian · DQN series 20M | 31.4% |
| LT | DDQN · lagrangian · DQN series 20M | 18.7% |
| EQ | DQN · lagrangian · DQN series 20M | 6.2% |
| EQ | DDQN · lagrangian · DQN series 20M | 9.7% |
| GT | DQN · lagrangian · DQN series 20M | 7.1% |
| GT | DDQN · lagrangian · DQN series 20M | 8.1% |
