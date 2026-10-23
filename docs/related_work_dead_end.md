# 相关文献：屏蔽后的死局 / 可行性预测

背景：v6.0.2 诊断（`paper_contents/v6.0.2/exit_diag.md`）发现，LT 上 Mask 策略 EXIT 的实例平均在第 4.4 步就已无可行完成，第 12.0 步才 EXIT；一步必要条件平均只比 EXIT 早 0.4 步察觉。以下文献与"让 RL 在出错的那一步就学到"相关。

## 1. 组合优化中屏蔽之后仍走进死局

| 文献 | 要点 | 与本文关系 |
|---|---|---|
| Bi, Ma, Zhou, Song, Cao, Wu, Zhang. *Learning to Handle Complex Constraints for Vehicle Routing Problems.* NeurIPS 2024（PIP / PIP-D）。[论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/a9d2a5fd12d34250c21b5e4fa8d906b0-Paper-Conference.pdf) · [arXiv 2410.21066](https://arxiv.org/abs/2410.21066) | 当前步可行的屏蔽容易算，"会导致后面无解"的屏蔽难算（NP-hard）；用拉格朗日乘子加强约束意识 + 预防式屏蔽（preventative infeasibility masking）；PIP-D 用辅助解码器学习预测该屏蔽。TSPTW / TSPDL 上不可行率显著下降 | 最直接对照：同一类"屏蔽后仍死局"问题 |
| Zhao, She, Zhu, Yang, Xu. *Online 3D Bin Packing with Constrained Deep Reinforcement Learning.* AAAI 2021。[arXiv 2006.14978](https://arxiv.org/abs/2006.14978) | 训练可行性预测器输出放置屏蔽，调制 actor 的动作概率；可行性屏蔽损失 + 不可行惩罚，按 CMDP 建模 | 同属装箱类，学习型可行性屏蔽 |
| Solozabal, Ceberio, Takáč. *Constrained Combinatorial Optimization with Reinforcement Learning.* [arXiv 2006.11984](https://arxiv.org/abs/2006.11984) | 指出以往神经组合优化只处理能用屏蔽保证可行的问题；把约束写成奖励中的惩罚项，CMDP 建模 | 屏蔽 vs 惩罚的对照 |

## 2. 死局（dead-end）理论

| 文献 | 要点 | 与本文关系 |
|---|---|---|
| Fatemi, Sharma, van Seijen, Kahou. *Dead-ends and Secure Exploration in Reinforcement Learning.* ICML 2019。[PMLR](https://proceedings.mlr.press/v97/fatemi19a.html) | 定义 dead-end 状态：进入后无论如何行动都会到达不良终止；用辅助价值函数刻画"安全"条件并限制探索策略 | "已无解仍放置的 7.5 步"即处于 dead-end |
| Fatemi, Killian, Subramanian, Ghassemi. *Medical Dead-ends and Learning to Identify High-risk States and Treatments.* NeurIPS 2021。[arXiv 2110.04186](https://arxiv.org/abs/2110.04186) | 离线场景下学习识别会进入 dead-end 的高风险状态与动作 | 学习型 dead-end 识别 |

## 3. Safe RL：可行集 / 安全评论家

| 文献 | 要点 | 与本文关系 |
|---|---|---|
| Yu, Ma, Li, Chen. *Reachability Constrained Reinforcement Learning.* ICML 2022。[PMLR](https://proceedings.mlr.press/v162/yu22d.html) · [arXiv 2205.07536](https://arxiv.org/abs/2205.07536) | 用可达性分析刻画"能持续满足约束"的可行集，以安全价值函数作为约束 | 对应"前缀固定后是否仍有可行完成" |
| Thananjeyan, Balakrishna, Nair, Luo, Srinivasan, Hwang, Gonzalez, Ibarz, Finn, Goldberg. *Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones.* RA-L 2021。[arXiv 2010.15920](https://arxiv.org/abs/2010.15920) | 学习安全评论家估计状态-动作的违约风险，超过阈值切换到恢复策略 | 学习型风险估计 |

## 4. 与本文的差异

- 上述方法的可行性 / 风险信号都是**学习得到的近似**。本文实例全部经 ILP 验证可行，且问题规模小（N ≤ 15、M ≤ 15），训练时可以**精确**判断"当前前缀是否仍有可行完成"：ILP 每次约 44 ms，精确回溯搜索约 0.9 ms（200 个 LT 随机前缀上与 ILP 结论 200/200 一致）。
- 只做了少量检索，未见"在构造式 RL 训练中用精确可行性作为逐步信号 / 标签"的工作，但这不构成新颖性结论；如需作为贡献点，应以 feasibility oracle、preventative masking、dead-end、constructive neural solver 等关键词系统检索。
