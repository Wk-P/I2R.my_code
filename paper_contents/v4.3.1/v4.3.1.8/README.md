# v4.3.1.8 — DQN / DDQN 奖励归一化（reward / M）

v4.3.1 的子版本（tag `v4.3.1.8`）。只重训 DQN 系列 8 个模型，PPO 系列不动。

## 1. 动机

v4.3.1.6 的奖励、回报、Q 值都没有归一化。PPO 有 advantage 归一化（SB3 默认 `normalize_advantage=True`），可以抵消奖励尺度；DQN / DDQN（含 Maskable 版，`shared/dqn_variants.py`）没有：

- Q 值直接回归原始回报，legacy 终止奖励的范围是 $[-M, M]$（LT 的 M = 15）；
- 损失是 Huber（smooth L1，阈值 1），|TD 误差| > 1 时梯度恒为 ±1。回报量级为十几时，大部分样本落在线性区，AR 带来的差异 $M\cdot\Delta AR$ 很难被区分出来。

## 2. 改动

- `paper_rl/train.py`：新增 `--reward-norm {none, m}`（环境变量 `REWARD_NORM`，默认 `none` = v4.3.1.6 行为）。`m` 时在 Monitor 外层套 `TransformReward`，学习器看到的是 $r/M$，因此终止奖励在 $[-1, 1]$；Lagrangian 的 $-\lambda\sum c_t$ 同比缩放（对偶更新只看违规率，不受影响）。Monitor / `training_curve.csv` 记录的仍是原始奖励，与旧曲线可直接比较。`results.json` 多一个 `reward_norm` 字段。
- `scripts/run_v4.3.1.8.py`：{DQN, DDQN} × {无约束, Lagrangian, Maskable, Repair} × LT/EQ/GT × 种子 1–3 × 5M，共 72 个任务。其余设定与 v4.3.1.6 完全相同。

对照组：v4.3.1.6 中相同的模型，按 v4.3.1.7 真正的 AR\* 重新评估的结果（`../v4.3.1.7/summary.json`）。

## 3. 结果

见 `report.md`（实验完成后生成）。
