# v5.3.1 BFT Repair：PPO / DQN / DDQN，DQN / DDQN 的 ε 1.0→0 固定 2M 步（种子 1、20M、5 万实例）

- 生成时间：2026-10-15 20:22:59；脚本 `src/scripts/run_v5.py --version 5.3.1`。
- 每个变体 1 个种子（1）× 3 场景（LT / EQ / GT）、各训练 20M 步，同一次训练在 5M / 10M / 15M 存 checkpoint；ar_pen 奖励、γ = 1、不提前结束、EXIT 只用于 Maskable、需求降序；PPO / DQN 超参取自 `paper_rl/config.py`；结构感知网络（Transformer）在 GPU 上训练。5 万实例 = 经 ILP 验证的实例池（`results/sup_diag/<场景>.json`）。
- 测试集 gap：各种子自己的 400 个测试实例；训练集 gap：1600 个训练实例（5 万实例池中按固定种子抽取，或 1600 实例划分本身）；均为确定性回放；均值 ± 样本标准差。
- manifest：`logs/v5.3.1/manifest.json`。

## 学习曲线：测试集相对 gap

### LT

| 步数 | PPO · BFT Repair | DQN · BFT Repair | DDQN · BFT Repair |
|---|---|---|---|
| 5M | 10.0% | 16.7% | 16.9% |
| 10M | 9.1% | 14.8% | 13.7% |
| 15M | 8.2% | 14.2% | 14.7% |
| 20M | 7.8% | 11.6% | 13.7% |

### EQ

| 步数 | PPO · BFT Repair | DQN · BFT Repair | DDQN · BFT Repair |
|---|---|---|---|
| 5M | 8.2% | 5.0% | 5.1% |
| 10M | 7.7% | 4.2% | 4.8% |
| 15M | 7.5% | 5.3% | 9.3% |
| 20M | 7.2% | 3.7% | 3.8% |

### GT

| 步数 | PPO · BFT Repair | DQN · BFT Repair | DDQN · BFT Repair |
|---|---|---|---|
| 5M | 8.8% | 8.7% | 8.5% |
| 10M | 7.6% | 4.0% | 3.6% |
| 15M | 6.8% | 5.6% | 10.7% |
| 20M | 6.6% | 3.9% | 2.7% |

## 学习曲线：三场景平均（每个种子先对三场景取平均）

| 步数 | PPO · BFT Repair | DQN · BFT Repair | DDQN · BFT Repair |
|---|---|---|---|
| 5M | 9.0% | 10.1% | 10.2% |
| 10M | 8.1% | 7.6% | 7.4% |
| 15M | 7.5% | 8.4% | 11.6% |
| 20M | 7.2% | 6.4% | 6.7% |

## 学习曲线：测试集未通过率（EXIT 或任一违约）

gap 只在未通过率之外的实例（无 EXIT、无违约）上计算；未通过率高时，gap 只反映容易的那部分实例。

### LT

| 步数 | PPO · BFT Repair | DQN · BFT Repair | DDQN · BFT Repair |
|---|---|---|---|
| 5M | 2.7% | 9.8% | 8.5% |
| 10M | 3.0% | 5.5% | 3.2% |
| 15M | 1.7% | 9.5% | 3.2% |
| 20M | 2.5% | 3.7% | 5.5% |

### EQ

| 步数 | PPO · BFT Repair | DQN · BFT Repair | DDQN · BFT Repair |
|---|---|---|---|
| 5M | 0.0% | 0.0% | 0.0% |
| 10M | 0.0% | 0.0% | 0.0% |
| 15M | 0.0% | 0.0% | 0.0% |
| 20M | 0.0% | 0.0% | 0.0% |

### GT

| 步数 | PPO · BFT Repair | DQN · BFT Repair | DDQN · BFT Repair |
|---|---|---|---|
| 5M | 0.0% | 0.0% | 0.0% |
| 10M | 0.0% | 0.0% | 0.0% |
| 15M | 0.0% | 0.0% | 0.0% |
| 20M | 0.0% | 0.0% | 0.0% |


## 20M：测试集与训练集 gap、违约、前几步决策

| 场景 | 模型 | 测试集相对 gap | 训练集相对 gap | 测试未通过率（EXIT / 违约） | 训练集未通过率 | 容量违约率 | 隐私违约率 | 第 1–5 步 regret | 第 1–5 步选中最优 |
|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · BFT Repair | 7.8% | 7.9% | 2.5% | 1.6% | 1.8% | 2.0% | — | — |
| LT | DQN · BFT Repair | 11.6% | 11.3% | 3.7% | 2.8% | 1.2% | 3.5% | — | — |
| LT | DDQN · BFT Repair | 13.7% | 13.6% | 5.5% | 4.3% | 1.2% | 5.2% | — | — |
| EQ | PPO · BFT Repair | 7.2% | 6.5% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DQN · BFT Repair | 3.7% | 3.4% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DDQN · BFT Repair | 3.8% | 3.9% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| GT | PPO · BFT Repair | 6.6% | 6.1% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DQN · BFT Repair | 3.9% | 3.5% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DDQN · BFT Repair | 2.7% | 2.8% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |

regret 按合法 ECU 之间的最优差距定义，只适用于 Maskable；其余机制可能执行违约动作，记为 —。


## 逐种子测试集 gap（20M）

| 场景 | 模型 | 种子 1 |
|---|---|---|
| LT | PPO · BFT Repair | 7.8% |
| LT | DQN · BFT Repair | 11.6% |
| LT | DDQN · BFT Repair | 13.7% |
| EQ | PPO · BFT Repair | 7.2% |
| EQ | DQN · BFT Repair | 3.7% |
| EQ | DDQN · BFT Repair | 3.8% |
| GT | PPO · BFT Repair | 6.6% |
| GT | DQN · BFT Repair | 3.9% |
| GT | DDQN · BFT Repair | 2.7% |
