# v4.3.5 — 直接用 AR 作为奖励

新的次版本（tag `v4.3.5`）。

## 1. 动机

v4.3.1.6 起的 legacy 奖励与优化目标（硬约束下最大化 AR，即 ILP 的目标）不严格一致：

- 成功分支 M(2AR − 1)：AR < 0.5 时为负，可能低于"最后一步失败"的 −1；
- 失败分支 −M(1 − valid/M) 按"放进去几个服务"给部分分，ILP 目标里没有这一项；
- γ = 0.99：死局回合更短、折扣更少，回报不等于目标值。

## 2. 改动（唯一变量：奖励）

$$r_t = 0\ (t<T),\qquad r_T = AR\cdot\mathbb{1}\{M\text{ 个服务全部合法放置}\},\qquad \gamma = 1$$

回合回报恰为 $AR\cdot\mathbb{1}\{\text{可行}\}$，$J(\pi)=\mathbb{E}[AR\cdot\mathbb{1}\{\text{可行}\}]$。

- `paper_rl/env.py`：新增 `reward_mode="ar_raw"`。成功得 AR，失败（违规或死局，含 Repair 无可替换 ECU）得 0；不乘 M。Lagrangian 照旧在终局减 $\lambda\sum_t c_t$。
- 失败回合不能给原始 AR：死局前已放置部分的 AR 往往很高，会诱导提前走进死局（同 v4.3.1.4 的堆叠问题）。
- `paper_rl/train.py`：新增 `--gamma`（默认取 config 的 PPO_GAMMA / DQN_GAMMA，即旧行为），写入 `results.json` 的 `gamma` 字段；本版本所有学习器 γ = 1。
- 其余与 v4.3.1.6 完全相同：原观测（base）、服务按需求降序、12 个模型、其余超参、5M 步、种子 1–3、p = 0.6 数据与划分、评估方式（每个测试实例 1 次确定性输出）。
- `scripts/run_v4.3.5.py`：108 个任务；报告对比 legacy（v4.3.1.6，v4.3.1.7 重评）、AR（本版本）与贪心。

## 3. 历史依据与已知风险

- 中间步一律 0：v0.3.1、v4.1.0.0–.4 的逐步塑形全部无效或有害。
- 之前最接近的是 v4.3.1.3 的 `ar`（成功 M·AR、失败 −M(1−valid/M)，p = 0.3、单种子、1M 步），略好于 legacy，但未被采用。
- 不乘 M 后奖励在 [0, 1]：PPO 有 advantage 归一化，基本不受尺度影响；DQN 系列回归绝对回报、Huber 阈值 1，v4.3.1.8 中除以 M 使 Mask / Repair 的 DQN 变差（LT Repair 成功率 0.80 → 0.60），本版本可能重现，按原样报告。
- 失败统一为 0、没有部分分：无约束 / Lagrangian 的 DQN 系列可能更难学到可行性。
- 预期：以往所有奖励变体的 AR 都停在相近水平；本版本的主要意义是训练目标与论文目标在定义上一致。服务放置顺序（2026-10-06 贪心诊断：顺序可让 AR 变化 0.02–0.11）留到之后的版本。

## 4. 自检

- 3 场景 × 4 机制 × 150 个随机回合（共 1800，756 个成功）：中间步奖励全为 0，终局奖励与 $AR\cdot\mathbb{1}\{\text{可行}\}-\lambda\sum c_t$（仅 Lagrangian）0 处不一致。
- EQ 上 Mask-PPO / Lagrange-PPO / Mask-DQN / Repair-DDQN 各 3 万步跑通训练、评估、保存，`results.json` 含 `reward_mode=ar_raw`、`gamma=1.0`。
- ILP 基准复核（2026-10-06）：用 scipy milp（HiGHS）独立建模（按启用 ECU 数 k 枚举，AR* = max_k obj_k / k），不调用项目 ILP 代码，每场景随机 60 个实例，与存储的 AR* 差 ≤ 5e-7（存储值 6 位舍入），解经独立代码核对均可行。

## 5. 结果

见 `report.md`（完成后生成）。
