# v4.4.3 结构感知策略网络（纯 RL）

在 v4.3.8 基础上**只换策略 / 价值网络**，其余完全不动（数据 v4.3.1.4 p = 0.6、需求降序、只选 ECU、ar_pen 奖励、γ = 1、不提前结束、Maskable 带 EXIT、40 环境 × 512 步、batch 256、10 epochs、clip 0.1、熵系数 0.005、学习率 3e-4）。

- **网络**（`src/paper_rl/graph_net.py`）：全局 / 每台 ECU / 每个服务各一个 token；服务间冲突、服务在哪台 ECU、服务与 ECU 上的服务冲突、服务放得进 ECU 剩余容量 四种关系作为可学习偏置加到注意力上；3 层、d = 128、4 头，约 45 万参数。
- **actor / critic**（`src/paper_rl/graph_policy.py`）：共享编码器；每台 ECU 用同一个打分头（ECU 编号置换等变），EXIT 由全局 token 打分；V(s) 由全局 token 与所有 token 的平均得到（置换不变）。分布、mask、PPO 更新都是 MaskablePPO 原有的。
- **观测**（`obs = raw`，`src/paper_rl/env.py::_raw_obs`）：容量、需求（需求降序）、冲突图、每个服务所在 ECU、当前步。信息与原观测相同，特征在网络内部计算；已核对 2053 个状态的 mask / AR 与环境一致。
- **依据**：`arch_diag.md`——同一网络用 ILP 最优动作监督训练，相对 gap 3.5~6.1%，MLP 为 13~17%（大 MLP 也只到 13~15%）。
- **报告**：MLP Mask PPO（v4.3.8 试跑）、结构感知 Mask PPO（本版本）、ILP 监督参照（同一网络模仿 ILP 动作，只作表示能力参照，不是 RL 也不是上限）三行并排：相对 gap、EXIT、开启 ECU 数、被逼开启、第 1 步与全部步选中最优动作的比例（逐步 regret 诊断），以及按步号的最优率与 regret 占比。

运行：`src/scripts/run_v4.4.3.py --pilot`（3 场景 × 种子 1 × 1M 步），报告 `pilot_report.md`。训练选项 `src/paper_rl/train.py --net graph`（自动使用 `--obs raw`）。
