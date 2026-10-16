# v5.3.2 unconstraint：PPO / DQN / DDQN，DQN / DDQN 的 ε 1.0→0 固定 2M 步（种子 1、20M、5 万实例）

- 生成时间：2026-10-16 01:32:52；脚本 `src/scripts/run_v5.py --version 5.3.2`。
- 每个变体 1 个种子（1）× 3 场景（LT / EQ / GT）、各训练 20M 步，同一次训练在 5M / 10M / 15M 存 checkpoint；ar_pen 奖励、γ = 1、不提前结束、EXIT 只用于 Maskable、需求降序；PPO / DQN 超参取自 `paper_rl/config.py`；结构感知网络（Transformer）在 GPU 上训练。5 万实例 = 经 ILP 验证的实例池（`results/sup_diag/<场景>.json`）。
- 测试集 gap：各种子自己的 400 个测试实例；训练集 gap：1600 个训练实例（5 万实例池中按固定种子抽取，或 1600 实例划分本身）；均为确定性回放；均值 ± 样本标准差。
- manifest：`logs/v5.3.2/manifest.json`。

## 学习曲线：测试集相对 gap

### LT

| 步数 | PPO · unconstraint | DQN · unconstraint | DDQN · unconstraint |
|---|---|---|---|
| 5M | 9.5% | 30.4% | 33.7% |
| 10M | 8.8% | 28.1% | 30.1% |
| 15M | 7.7% | 25.4% | 23.0% |
| 20M | 7.4% | 25.2% | 28.0% |

### EQ

| 步数 | PPO · unconstraint | DQN · unconstraint | DDQN · unconstraint |
|---|---|---|---|
| 5M | 10.3% | 24.0% | 25.1% |
| 10M | 7.7% | 27.7% | 19.4% |
| 15M | 6.8% | 36.8% | 23.3% |
| 20M | 5.6% | 31.9% | 29.3% |

### GT

| 步数 | PPO · unconstraint | DQN · unconstraint | DDQN · unconstraint |
|---|---|---|---|
| 5M | 9.1% | 16.3% | 24.0% |
| 10M | 7.8% | 16.0% | 22.0% |
| 15M | 6.7% | 16.9% | 15.6% |
| 20M | 6.3% | 18.7% | 15.8% |

## 学习曲线：三场景平均（每个种子先对三场景取平均）

| 步数 | PPO · unconstraint | DQN · unconstraint | DDQN · unconstraint |
|---|---|---|---|
| 5M | 9.6% | 23.6% | 27.6% |
| 10M | 8.1% | 23.9% | 23.8% |
| 15M | 7.1% | 26.4% | 20.6% |
| 20M | 6.4% | 25.3% | 24.4% |

## 学习曲线：测试集未通过率（EXIT 或任一违约）

gap 只在未通过率之外的实例（无 EXIT、无违约）上计算；未通过率高时，gap 只反映容易的那部分实例。

### LT

| 步数 | PPO · unconstraint | DQN · unconstraint | DDQN · unconstraint |
|---|---|---|---|
| 5M | 3.5% | 93.8% | 72.8% |
| 10M | 2.7% | 96.5% | 93.8% |
| 15M | 2.5% | 91.5% | 78.8% |
| 20M | 2.5% | 93.0% | 86.2% |

### EQ

| 步数 | PPO · unconstraint | DQN · unconstraint | DDQN · unconstraint |
|---|---|---|---|
| 5M | 0.5% | 42.5% | 12.5% |
| 10M | 0.5% | 19.0% | 49.8% |
| 15M | 0.0% | 0.5% | 7.3% |
| 20M | 0.0% | 2.0% | 1.2% |

### GT

| 步数 | PPO · unconstraint | DQN · unconstraint | DDQN · unconstraint |
|---|---|---|---|
| 5M | 0.0% | 0.0% | 0.0% |
| 10M | 0.2% | 0.0% | 0.2% |
| 15M | 0.2% | 17.5% | 0.0% |
| 20M | 0.0% | 13.5% | 35.2% |


## 20M：测试集与训练集 gap、违约、前几步决策

| 场景 | 模型 | 测试集相对 gap | 训练集相对 gap | 测试未通过率（EXIT / 违约） | 训练集未通过率 | 容量违约率 | 隐私违约率 | 第 1–5 步 regret | 第 1–5 步选中最优 |
|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · unconstraint | 7.4% | 6.9% | 2.5% | 1.4% | 1.5% | 2.5% | — | — |
| LT | DQN · unconstraint | 25.2% | 24.9% | 93.0% | 91.7% | 3.0% | 92.8% | — | — |
| LT | DDQN · unconstraint | 28.0% | 27.2% | 86.2% | 84.4% | 7.8% | 85.8% | — | — |
| EQ | PPO · unconstraint | 5.6% | 5.8% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DQN · unconstraint | 31.9% | 32.0% | 2.0% | 1.1% | 1.8% | 1.0% | — | — |
| EQ | DDQN · unconstraint | 29.3% | 28.7% | 1.2% | 0.5% | 0.0% | 1.2% | — | — |
| GT | PPO · unconstraint | 6.3% | 5.6% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DQN · unconstraint | 18.7% | 18.6% | 13.5% | 10.5% | 0.0% | 13.5% | — | — |
| GT | DDQN · unconstraint | 15.8% | 15.3% | 35.2% | 35.5% | 0.5% | 35.2% | — | — |

regret 按合法 ECU 之间的最优差距定义，只适用于 Maskable；其余机制可能执行违约动作，记为 —。


## 逐种子测试集 gap（20M）

| 场景 | 模型 | 种子 1 |
|---|---|---|
| LT | PPO · unconstraint | 7.4% |
| LT | DQN · unconstraint | 25.2% |
| LT | DDQN · unconstraint | 28.0% |
| EQ | PPO · unconstraint | 5.6% |
| EQ | DQN · unconstraint | 31.9% |
| EQ | DDQN · unconstraint | 29.3% |
| GT | PPO · unconstraint | 6.3% |
| GT | DQN · unconstraint | 18.7% |
| GT | DDQN · unconstraint | 15.8% |
