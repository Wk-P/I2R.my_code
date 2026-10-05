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

完整表格见 `report.md`（2026-10-05 17:29 完成，72/72）。结论：**归一化不是普遍改进，只对 Lagrangian 有益，对 Maskable / Repair 有害或持平。**

| 组别 | 归一化的影响 |
|---|---|
| Lagrangian（DQN、DDQN，三场景） | 全面变好：LT success 0.21/0.10 → 0.46/0.47，相对最优 AR 提高 4~15 个百分点。原始版的训练曲线在后期**崩塌**（LT DQN 第 4→5 个 1M 段 success 0.25→0.12），归一化后单调上升、到 5M 仍未收敛 |
| Repair，LT | 明显变差：success 0.80/0.81 → 0.61/0.60（死局率翻倍），早期就学得更慢（第 2 段 0.53 vs 0.62），之后停在 ~0.6 |
| Maskable，LT | 略差（success −0.4~2.5pp） |
| Maskable / Repair，EQ/GT | success 不变（≈1），相对最优 AR 一致下降 0.7~2.3 个百分点 |
| 无约束对照 | 方向不一致，均在种子噪声内 |

**解释（推断，未单独验证）。** DQN 的 Huber 损失在 $|\delta|\le1$ 时是二次的。原始尺度下 TD 误差常大于 1，梯度被截成 ±1；除以 M（LT 为 15）后误差大多落入二次区，梯度 ∝ δ，相当于把有效学习率降了约一个数量级。
- Lagrangian 的原始回报含 $-\lambda\sum c_t$（λ 最大 50），尺度远大于 M，原始版的问题是**过大**、不稳定，缩小后反而稳定；
- Maskable / Repair 的回报本来就在 $[-M,M]$，缩小后更新太弱，5M 步内学不到原来的水平。

**对 v4.3.1.9 的影响。** 原计划"所有学习器除以 $M(\tilde B+1)$"在 $\tilde B=30$ 时会让 DQN 的 AR 信号再缩小约 31 倍，按本版本的结果，DQN 系列基本学不到 AR。DQN 系列不能只靠缩放奖励，需要同时调整损失或学习率（例如 Huber 阈值随尺度缩小 / 改 MSE + 提高 lr），或不把 $\tilde B$ 计入 DQN 的缩放。

## 4. 分析：PPO 的 advantage 归一化让成功奖励"自动按字典序"起作用

（理论分析，写于 2026-10-05，讨论 v4.3.1.9 奖励设计时得出，尚未单独做实验验证。）

**结论。** 训练早期，一个 batch 里有成功也有失败，方差主要来自成功与失败的差别，策略先学"不失败"；当 batch 里几乎全部成功，常数 B 被归一化消掉，AR 信号恢复到满强度，策略再优化 AR。

**机制。** 设终止奖励为成功 $B+\alpha AR$、失败 $-C-\beta f$（$f=1-M_v/M$），中间步为 0。PPO 的 advantage $A=\hat R-V(s)$ 先减去 critic 基线，再在每个 minibatch 内做 $(A-\mu)/\sigma$ 归一化（SB3 `normalize_advantage=True`）。在成功率为 $s$ 的 batch 里：

$$
\operatorname{Var}(A)\approx \underbrace{s(1-s)\,\Delta^2}_{\text{成功/失败}}+\underbrace{s\,\alpha^2\sigma_{AR}^2}_{\text{成功内部的 AR 差异}},\qquad \Delta\approx B+C+\alpha\overline{AR}+\beta\bar f .
$$

归一化把总方差缩放为 1，所以决定策略梯度方向的是两项的**相对**大小：
- $s(1-s)\Delta^2\gg\alpha^2\sigma_{AR}^2$ 时，梯度几乎全部用来提高成功率；
- 一个 batch 里没有失败（$s\to1$）时，第一项为 0，常数 $B$ 被基线和均值扣除，剩下的 $\alpha(AR-\overline{AR})$ 被放大到单位方差，AR 信号恢复到满强度，与 $B$ 取多大无关。

切换点约在失败率 $1-s\approx(\alpha\sigma_{AR}/\Delta)^2$（按奖励/M 计算，$\sigma_{AR}\approx0.09$、$\overline{AR}\approx0.70$、$s\approx0.85$）：

| 成功奖励 | $\Delta$ | AR 开始主导时的失败率 |
|---|---|---|
| legacy $M(2AR-1)$ | ≈ 0.47 | ≲ 15% |
| $M(1+2AR)$ | ≈ 2.5 | ≲ 0.5% |
| $M(\tilde B+AR)$，$\tilde B=30$ | ≈ 31 | 只有 batch 内完全无失败时 |

**含义。**
1. 对 PPO，加大成功溢价 $B$ 不会让它学不到 AR，只是把"优化 AR"推迟到失败基本消除之后，这正是"先不失败、再 AR"的字典序。eq/gt 上 Maskable 模型几乎总是成功，预期 $B$ 的大小对其 AR 基本没有影响；LT 长期存在失败，AR 会被压低，这是字典序目标该付的代价。
2. DQN / DDQN 没有这个机制：Q 值回归的是绝对回报，$B$ 直接进入 TD 目标，没有基线扣除，也没有方差归一化。$B$ 越大，Huber 损失的问题越严重，因此 DQN 系列必须做奖励归一化（即本版本的改动）。本版本的结果是采用大 $B$ 之前的前提检验。
3. 局限：归一化按 minibatch（256 个 transition，跨越十几个 episode）进行，"几乎全部成功"指的是 minibatch 层面；critic 拟合不准时，基线扣除也不完全；价值损失本身没有归一化，$B$ 很大时可能通过全局梯度裁剪（`max_grad_norm=0.5`）压低策略更新步长。这些需要在 v4.3.1.9 的 $\tilde B$ 试跑中观察。
