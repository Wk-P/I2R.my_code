# v5.4.0 verication un/lag：objective 奖励（终局 M·AR，不管违约）下的 unconstraint 与 Lagrangian（objective − λc_t，λ 设置同 v5.1.0）；DQN / DDQN 的 ε 1.0→0 固定 2M 步（种子 1、20M、5 万实例）

- 生成时间：2026-10-16 19:32:16；脚本 `src/scripts/run_v5.py --version 5.4.0`。
- 每个变体 1 个种子（1）× 3 场景（LT / EQ / GT）、各训练 20M 步，同一次训练在 5M / 10M / 15M 存 checkpoint；ar_pen 奖励、γ = 1、不提前结束、EXIT 只用于 Maskable、需求降序；PPO / DQN 超参取自 `paper_rl/config.py`；结构感知网络（Transformer）在 GPU 上训练。5 万实例 = 经 ILP 验证的实例池（`results/sup_diag/<场景>.json`）。
- 测试集 gap：各种子自己的 400 个测试实例；训练集 gap：1600 个训练实例（5 万实例池中按固定种子抽取，或 1600 实例划分本身）；均为确定性回放；均值 ± 样本标准差。
- manifest：`logs/v5.4.0/manifest.json`。

## λ 动态（训练过程，来自 training_curve.csv）

J_C = 每局违约次数（容量 + 隐私）；目标 d = 0。「到上限」= 第一次 λ ≥ λ_max 的局；「降幅」= 最高点到训练结束的下降。

| 场景 | 模型 | λ_max | 末 λ | 最高 λ | 到上限（局） | 上限占比 | 下降次数 | 降幅 | 前 1000 局 J_C | 末 1000 局 J_C |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · lagrangian · objective | 无 | 14.929 | 14.929 | — | 0.0% | 0 | 0.000 | 1.981 | 0.0490 |
| LT | DQN · lagrangian · objective | 无 | 9.591 | 9.591 | — | 0.0% | 0 | 0.000 | 1.984 | 0.0250 |
| LT | DDQN · lagrangian · objective | 无 | 3.734 | 3.734 | — | 0.0% | 0 | 0.000 | 1.985 | 0.0150 |
| EQ | PPO · lagrangian · objective | 无 | 7.525 | 7.525 | — | 0.0% | 0 | 0.000 | 1.820 | 0.0010 |
| EQ | DQN · lagrangian · objective | 无 | 4.747 | 4.747 | — | 0.0% | 0 | 0.000 | 1.810 | 0.0000 |
| EQ | DDQN · lagrangian · objective | 无 | 4.686 | 4.686 | — | 0.0% | 0 | 0.000 | 1.812 | 0.0000 |
| GT | PPO · lagrangian · objective | 无 | 17.387 | 17.387 | — | 0.0% | 0 | 0.000 | 1.645 | 0.0000 |
| GT | DQN · lagrangian · objective | 无 | 4.725 | 4.725 | — | 0.0% | 0 | 0.000 | 1.672 | 0.0000 |
| GT | DDQN · lagrangian · objective | 无 | 4.768 | 4.768 | — | 0.0% | 0 | 0.000 | 1.672 | 0.0000 |

### λ 与 J_C（来自 lambda_updates.csv，各步数前最后一次更新）

每格：λ / 行为策略 J_C / 用于更新的 J_C（贪心时为 ε = 0 的局）。

| 场景 | 模型 | 5M | 10M | 15M | 20M |
|---|---|---|---|---|---|
| LT | PPO · lagrangian · objective | 8.97 / 27.707 / 27.707 | 14.85 / 0.090 / 0.090 | 14.89 / 0.059 / 0.059 | 14.93 / 0.093 / 0.093 |
| EQ | PPO · lagrangian · objective | 7.52 / 0.000 / 0.000 | 7.52 / 0.000 / 0.000 | 7.52 / 0.000 / 0.000 | 7.53 / 0.000 / 0.000 |
| GT | PPO · lagrangian · objective | 9.09 / 17.922 / 17.922 | 17.33 / 0.130 / 0.130 | 17.39 / 0.000 / 0.000 | 17.39 / 0.000 / 0.000 |
| LT | DQN · lagrangian · objective | 9.20 / 1.867 / 1.867 | 9.42 / 0.037 / 0.037 | 9.50 / 0.056 / 0.056 | 9.59 / 0.021 / 0.021 |
| EQ | DQN · lagrangian · objective | 4.65 / 0.000 / 0.000 | 4.67 / 0.001 / 0.001 | 4.72 / 0.000 / 0.000 | 4.75 / 0.000 / 0.000 |
| GT | DQN · lagrangian · objective | 4.71 / 0.000 / 0.000 | 4.71 / 0.000 / 0.000 | 4.72 / 0.000 / 0.000 | 4.72 / 0.000 / 0.000 |
| LT | DDQN · lagrangian · objective | 3.43 / 0.031 / 0.031 | 3.51 / 0.013 / 0.013 | 3.61 / 0.014 / 0.014 | 3.73 / 0.010 / 0.010 |
| EQ | DDQN · lagrangian · objective | 4.66 / 0.000 / 0.000 | 4.67 / 0.000 / 0.000 | 4.67 / 0.000 / 0.000 | 4.69 / 0.001 / 0.001 |
| GT | DDQN · lagrangian · objective | 4.71 / 0.006 / 0.006 | 4.73 / 0.000 / 0.000 | 4.74 / 0.000 / 0.000 | 4.77 / 0.000 / 0.000 |

## 学习曲线：测试集相对 gap

### LT

| 步数 | PPO · unconstraint · objective | DQN · unconstraint · objective | DDQN · unconstraint · objective | PPO · lagrangian · objective | DQN · lagrangian · objective | DDQN · lagrangian · objective |
|---|---|---|---|---|---|---|
| 5M | — | — | — | — | 15.6% | 8.0% |
| 10M | — | — | — | 14.6% | 11.2% | 7.3% |
| 15M | — | — | — | 12.4% | 9.8% | 6.1% |
| 20M | — | — | — | 11.8% | 9.1% | 5.3% |

### EQ

| 步数 | PPO · unconstraint · objective | DQN · unconstraint · objective | DDQN · unconstraint · objective | PPO · lagrangian · objective | DQN · lagrangian · objective | DDQN · lagrangian · objective |
|---|---|---|---|---|---|---|
| 5M | — | — | — | 11.2% | 5.5% | 5.3% |
| 10M | — | — | — | 5.1% | 5.4% | 4.4% |
| 15M | — | — | — | 4.3% | 4.2% | 3.3% |
| 20M | — | — | — | 3.9% | 3.5% | 4.0% |

### GT

| 步数 | PPO · unconstraint · objective | DQN · unconstraint · objective | DDQN · unconstraint · objective | PPO · lagrangian · objective | DQN · lagrangian · objective | DDQN · lagrangian · objective |
|---|---|---|---|---|---|---|
| 5M | — | — | — | — | 12.4% | 7.7% |
| 10M | — | — | — | 17.6% | 4.9% | 4.1% |
| 15M | — | — | — | 4.3% | 3.6% | 4.7% |
| 20M | — | — | — | 3.5% | 2.5% | 2.9% |

## 学习曲线：三场景平均（每个种子先对三场景取平均）

| 步数 | PPO · unconstraint · objective | DQN · unconstraint · objective | DDQN · unconstraint · objective | PPO · lagrangian · objective | DQN · lagrangian · objective | DDQN · lagrangian · objective |
|---|---|---|---|---|---|---|
| 5M | — | — | — | — | 11.2% | 7.0% |
| 10M | — | — | — | 12.5% | 7.2% | 5.3% |
| 15M | — | — | — | 7.0% | 5.9% | 4.7% |
| 20M | — | — | — | 6.4% | 5.0% | 4.1% |

## 学习曲线：测试集未通过率（EXIT 或任一违约）

gap 只在未通过率之外的实例（无 EXIT、无违约）上计算；未通过率高时，gap 只反映容易的那部分实例。

### LT

| 步数 | PPO · unconstraint · objective | DQN · unconstraint · objective | DDQN · unconstraint · objective | PPO · lagrangian · objective | DQN · lagrangian · objective | DDQN · lagrangian · objective |
|---|---|---|---|---|---|---|
| 5M | 100.0% | 100.0% | 100.0% | 100.0% | 9.3% | 2.5% |
| 10M | 100.0% | 100.0% | 100.0% | 5.5% | 2.5% | 1.7% |
| 15M | 100.0% | 100.0% | 100.0% | 4.5% | 3.7% | 1.7% |
| 20M | 100.0% | 100.0% | 100.0% | 5.2% | 2.7% | 1.5% |

### EQ

| 步数 | PPO · unconstraint · objective | DQN · unconstraint · objective | DDQN · unconstraint · objective | PPO · lagrangian · objective | DQN · lagrangian · objective | DDQN · lagrangian · objective |
|---|---|---|---|---|---|---|
| 5M | 100.0% | 100.0% | 100.0% | 0.2% | 0.0% | 0.0% |
| 10M | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| 15M | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| 20M | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |

### GT

| 步数 | PPO · unconstraint · objective | DQN · unconstraint · objective | DDQN · unconstraint · objective | PPO · lagrangian · objective | DQN · lagrangian · objective | DDQN · lagrangian · objective |
|---|---|---|---|---|---|---|
| 5M | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| 10M | 100.0% | 100.0% | 100.0% | 4.5% | 0.0% | 0.0% |
| 15M | 100.0% | 100.0% | 100.0% | 0.2% | 0.0% | 0.0% |
| 20M | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |


## 20M：测试集与训练集 gap、违约、前几步决策

| 场景 | 模型 | 测试集相对 gap | 训练集相对 gap | 测试未通过率（EXIT / 违约） | 训练集未通过率 | 容量违约率 | 隐私违约率 | 第 1–5 步 regret | 第 1–5 步选中最优 |
|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| LT | DQN · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| LT | DDQN · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| LT | PPO · lagrangian · objective | 11.8% | 11.1% | 5.2% | 4.2% | 3.2% | 4.5% | — | — |
| LT | DQN · lagrangian · objective | 9.1% | 9.0% | 2.7% | 1.9% | 1.5% | 1.5% | — | — |
| LT | DDQN · lagrangian · objective | 5.3% | 5.2% | 1.5% | 0.7% | 1.2% | 0.5% | — | — |
| EQ | PPO · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| EQ | DQN · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| EQ | DDQN · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| EQ | PPO · lagrangian · objective | 3.9% | 3.5% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DQN · lagrangian · objective | 3.5% | 3.3% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DDQN · lagrangian · objective | 4.0% | 4.2% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| GT | PPO · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| GT | DQN · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| GT | DDQN · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| GT | PPO · lagrangian · objective | 3.5% | 3.2% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DQN · lagrangian · objective | 2.5% | 2.6% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DDQN · lagrangian · objective | 2.9% | 3.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |

regret 按合法 ECU 之间的最优差距定义，只适用于 Maskable；其余机制可能执行违约动作，记为 —。


## 逐种子测试集 gap（20M）

| 场景 | 模型 | 种子 1 |
|---|---|---|
| LT | PPO · unconstraint · objective | — |
| LT | DQN · unconstraint · objective | — |
| LT | DDQN · unconstraint · objective | — |
| LT | PPO · lagrangian · objective | 11.8% |
| LT | DQN · lagrangian · objective | 9.1% |
| LT | DDQN · lagrangian · objective | 5.3% |
| EQ | PPO · unconstraint · objective | — |
| EQ | DQN · unconstraint · objective | — |
| EQ | DDQN · unconstraint · objective | — |
| EQ | PPO · lagrangian · objective | 3.9% |
| EQ | DQN · lagrangian · objective | 3.5% |
| EQ | DDQN · lagrangian · objective | 4.0% |
| GT | PPO · unconstraint · objective | — |
| GT | DQN · unconstraint · objective | — |
| GT | DDQN · unconstraint · objective | — |
| GT | PPO · lagrangian · objective | 3.5% |
| GT | DQN · lagrangian · objective | 2.5% |
| GT | DDQN · lagrangian · objective | 2.9% |
