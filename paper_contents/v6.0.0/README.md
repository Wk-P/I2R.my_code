# v6.0.0 实验信息总览（总图 12 个变种）

由 `src/scripts/report_v6_0_0_info.py` 生成；所有数值从代码、run 的 results.json / run.json、已安装的库读取并用 assert 核对，没有手写数字。不训练、不评估。

## 文件

1. [01_algorithms](01_algorithms.md)：12 个变种（strategy × RL_algorithm）讲解、奖励、来源
2. [02_network](02_network.md)：Transformer（关系偏置注意力编码器）的结构、层数、维度、参数量
3. [03_training](03_training.md)：库与版本、PPO / DQN / DDQN 超参数、Lagrangian 对偶更新、自定义回调函数
4. [04_environment](04_environment.md)：环境变量、观测（= 状态）、动作空间，以及各变种的区别

## 范围与核对

- 范围：总图（/summary，`src/scripts/merge_2d_table.py`）使用的 108 个 run = 12 格 × 3 场景（LT / EQ / GT）× 3 种子（1, 2, 3）。
- 同一格内所有 run 的设置（奖励、网络、超参数、数据、步数等）逐项断言一致，只有代码提交与批次不同（见下表）。
- 代码一致性：这些 run 来自 5 个提交。用 `git diff <提交> HEAD` 核对了训练相关文件（env.py、graph_net.py、graph_policy.py、graph_dqn.py、dqn_variants.py、train.py、config.py、vec_env.py）：差异只有变量改名（cap/req → c/e，i/j → m/n）、新增但这些 run 未使用的 ar_obj 奖励与 feas_in 开关（这些 run 全部等价于 feas_in=True）、新增未使用的命令行参数。对这些 run 的设置而言，训练行为与当前代码相同，本文按当前代码描述。
- 记号：服务 m（共 M 个）、ECU n（共 N 个）、需求 e_m、容量 c_n、放置变量 x_mn、启用变量 y_n。DDQN = Double DQN。

### 每格的来源

| strategy | RL_algorithm | 批次 @ 提交：种子（只列不全的场景） |
|---|---|---|
| maskable | PPO | v6.0.5-PPO-hyparam @ `1a344ea`: 种子 1, 2, 3 |
| maskable | DQN | v6.0.0 @ `cbf08e9`: 种子 1, 2, 3 |
| maskable | DDQN (Double DQN) | v6.0.0 @ `cbf08e9`: 种子 1, 2, 3 |
| repair | PPO | v6.0.0 @ `cbf08e9`: 种子 1<br>v6.0.0 @ `4445cf9`: 种子 2, 3 |
| repair | DQN | v6.0.0 @ `cbf08e9`: 种子 1<br>v6.0.0 @ `4445cf9`: 种子 2, 3 |
| repair | DDQN (Double DQN) | v6.0.0 @ `cbf08e9`: 种子 1<br>v6.0.0 @ `4445cf9`: 种子 2, 3 |
| lagrangian | PPO | v6.0.1-lagrangian @ `7c165b5`: 种子 1, 2, 3 |
| lagrangian | DQN | v6.0.1-lagrangian @ `7c165b5`: 种子 1, 2, 3 |
| lagrangian | DDQN (Double DQN) | v6.0.1-lagrangian @ `7c165b5`: 种子 1, 2, 3 |
| unconstraint | PPO | v6.0.1-unconstraint @ `f56abd3`: 种子 1<br>v6.0.0 @ `4445cf9`: 种子 2, 3 |
| unconstraint | DQN | v6.0.1-unconstraint @ `1a344ea`: 种子 1 (LT)<br>v6.0.1-unconstraint @ `f56abd3`: 种子 1 (EQ, GT)<br>v6.0.0 @ `4445cf9`: 种子 2, 3 |
| unconstraint | DDQN (Double DQN) | v6.0.1-unconstraint @ `f56abd3`: 种子 1<br>v6.0.0 @ `4445cf9`: 种子 2, 3 |
