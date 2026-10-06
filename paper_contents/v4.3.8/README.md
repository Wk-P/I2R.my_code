# v4.3.8 Maskable 加 EXIT 动作

在 v4.3.7（统一奖励、不提前结束）基础上，只改 Maskable：

- 动作空间为 [ECU1, …, ECUn, EXIT]。
- 只要还有合法 ECU，EXIT 被屏蔽，智能体只能在合法 ECU 中选，**永不违约**。
- 一个合法 ECU 都没有时，EXIT 是唯一合法动作；选后回合结束，得失败奖励 −(1 − valid/M) ∈ [−1, 0)。负奖励沿轨迹回传到前面"当时合法、但把后面逼进死路"的选择，模型要学会避开。
- mask 永远至少有一个合法动作，不再有全 False 的 mask，也不再强放。
- 测试：Maskable 的 capacity / privacy 违约率恒为 0，另报 EXIT 率（选了 EXIT 的测试实例比例）；AR / ILP AR / AR gap 在无违约且未 EXIT 的实例上算。
- 其他机制（无约束、Lagrangian、Repair）与 v4.3.7 完全相同。

运行：`scripts/run_v4.3.8.py --pilot`（只重训 Maskable PPO / DQN × 3 场景 × 种子 1 × 1M 步，其余 4 个模型取自 v4.3.7 试跑），报告 `pilot_report.md`。
