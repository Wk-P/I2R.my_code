# 1 Introduction

> 本章正文由作者撰写；以下为素材。原稿中的"Constraint-Handling Strategies""Reinforcement Learning Algorithms"两节按大纲移入第 4 章（见 Framework/4.4、4.5），PPO/DQN/DDQN 的英文简介和 bibtex 一并移过去。

## 动机（ILP 最优但 NP-hard、耗时）

- ILP（Dinkelbach + CBC）能给出最优 AR，测试集 3×5000 个实例全部求得 Optimal。
- 问题是强 NP-hard（见 3.5 命题 1）；实测单实例平均 1500 ms，LT 场景 3285 ms（中位数 1301 ms），长尾明显（见 5.7）。
- 隐私冲突是硬约束：违约不可接受，所以近似方法必须给出约束保证或报告违约率。

## 迁移思路

- 把 ILP 的目标（AR）直接作为 RL 的终局奖励，把 ILP 的约束（容量、隐私冲突）拆成逐步可判断的规则，用不同的约束处理策略放进 RL（见 4.1）。
- ILP 不参与训练，只用于实例可行性筛选和 AR gap 评估，因此方法是 model-free 的，不是监督学习。

## 贡献（3 条）

【空缺：贡献的措辞由作者撰写。可用的事实依据如下】
1. ILP→Safe RL 迁移框架：目标→奖励、约束→四种约束处理策略；在三个场景上系统比较 4 策略 × 3 算法共 12 个变体。
2. 关系感知 Transformer 策略网络（4 类关系偏置，对 ECU/服务排列等变，同一网络适用不同规模）。
3. 实验结论：PPO-M 在 0 违约下三场景平均相对 AR gap 4.38%，比 ILP 快 28×（LT 上约 48×）；强 NP-hard 与多项式推理的复杂度对比（命题 1–3）。

## 原稿（保留）

由于加入了硬约束机制，原本的RL模型需要升级成Safe RL模型。论文使用了4种策略来结合不同的RL算法，实现对Hard Constrainted问题下利用Safe RL模型的解决方案。4种策略包含：unconstrainted, action masking, Lagrangian multiplier, heuristic repair. 3种算法分别是：PPO, DQN, Double DQN.
