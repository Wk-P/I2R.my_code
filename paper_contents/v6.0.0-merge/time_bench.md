# 单实例耗时：ILP vs 12 个变种

- 生成时间 2026-10-29 15:58:56；脚本 `src/scripts/bench_time_v6.py`；耗时 1.4 分钟；只用 CPU，不训练。
- 实例：v6.0.4 固定测试集每场景前 200 个，所有方法相同；单位 ms/实例，CPU 单线程。
- RL：总图中每格的种子 1 模型，确定性完整 episode（reset + 掩码 + 前向 + step），先热身 5 个。ILP：`solve_ilp_max_ar`（Dinkelbach，CBC 单线程），含建模与全部迭代。加速比 = ILP 平均 / 变种平均。

| 方法 | LT | GT | EQ | 三场景平均 | 加速比（平均） |
|---|---|---|---|---|---|
| ILP (Dinkelbach, 3.06 次迭代) | 3192.5 | 807.7 | 543.3 | 1514.5 | 1× |
| Mask PPO | 74.6 | 49.9 | 46.8 | 57.1 | 24× |
| Mask DQN | 67.2 | 41.8 | 41.9 | 50.3 | 27× |
| Mask Double DQN | 64.9 | 43.5 | 41.9 | 50.1 | 27× |
| Repair PPO | 71.4 | 48.4 | 45.6 | 55.1 | 24× |
| Repair DQN | 64.7 | 43.7 | 50.6 | 53.0 | 26× |
| Repair Double DQN | 65.3 | 41.7 | 41.9 | 49.6 | 27× |
| Lagrangian PPO | 69.7 | 47.5 | 43.3 | 53.5 | 25× |
| Lagrangian DQN | 66.5 | 42.9 | 41.8 | 50.4 | 27× |
| Lagrangian Double DQN | 63.0 | 43.1 | 41.8 | 49.3 | 27× |
| Unconstraint PPO | 69.4 | 44.7 | 43.9 | 52.6 | 25× |
| Unconstraint DQN | 66.3 | 42.1 | 41.1 | 49.8 | 27× |
| Unconstraint Double DQN | 63.9 | 43.4 | 39.7 | 49.0 | 27× |

- 加速比为三场景各自加速比的平均。
