# v4.3.1.6 — 12 个模型统一回到 legacy 奖励、PPO clip ε 0.1

v4.3.1 的子版本（tag `v4.3.1.6`）。依据用户决定：

1. **12 个模型全部使用 legacy 奖励**（与 v4.2.0 相同形式）：

   $$R_{\text{term}}=\begin{cases}M(2AR-1), & M\text{ 个服务全部合法放置}\\ -M(1-\text{valid}/M), & \text{否则}\end{cases}\qquad r_t=0\ (t<M-1)$$

   - 无约束对照因此也受失败分支约束（与 v4.2.0 一致）：其含义是"只靠奖励、不加显式约束机制"。
   - v4.3.1.4 起的 objective 奖励不再作为默认（`REWARD_MODE=objective` 仍可用）。将 objective 推广到全部 12 个模型是 v4.3.1.4 中的自行推广，未经单独确认，本版本撤回。
   - legacy 的已知问题（AR < 0.5 − 1/(2M) 时成功回合得分低于"最后一步违规"）在 p = 0.6 数据上影响很小：ILP 最优解落入该区间的实例占 LT 0.1%、EQ 0.4%、GT 0%。
2. **PPO clip ε：0.2 → 0.1**（概率比限制在 [0.9, 1.1]），PPO 系列 4 个模型、三场景相同。
3. 沿用 v4.3.1.5：Lagrangian 终局一次扣 $\lambda\sum_t c_t$，$\lambda_{max}=50$；沿用 v4.3.1.4：p = 0.6 数据、Maskable / Repair 死局判失败并终止。

| 约束处理 | 中间步 | 终局 |
|---|---|---|
| 无约束 | 0 | legacy |
| Lagrangian | 0 | legacy $-\lambda\sum_t c_t$ |
| Maskable | 0 | legacy（死局：失败分支，终止） |
| Repair | 0 | legacy（死局：失败分支，终止） |

数据：每个场景 2000 个实例，按种子打乱 80/20 → 训练 1600、测试 400（无单独验证集）。

## 自检

LT / EQ 各 300 个随机 episode × 4 种机制：终局奖励与上式（Lagrangian 含 $-\lambda\sum c_t$）0 处不一致；Maskable / Repair 违规回合 0。

## 运行

- 12 个模型 × 3 场景 × legacy × 种子 1 × **5M 步** = 36 个任务；`scripts/run_v4.3.1.6.py`。
- 记录：`scripts/logs/v4.3.1.6_run/manifest.json`；报告 `report.md`（本目录）。

**已完成**（2026-10-02 15:30 – 17:32，36/36，无失败）。完整表格见 `report.md`。

### success_rate（单种子、5M 步；括号内为违规回合比例 = 容量或冲突至少一项违规）

| 模型 | 约束处理 | LT | EQ | GT |
|---|---|---|---|---|
| PPO | 无约束 | 0.848（~15%） | 0.995 | 0.985 |
| PPO | Lagrangian | **0.880** | 0.993 | 0.975 |
| PPO | Maskable | 0.855 | 1.000 | 1.000 |
| PPO | Repair | 0.825 | 0.995 | 1.000 |
| DQN | 无约束 | 0.172 | 0.593 | 0.600 |
| DQN | Lagrangian | 0.247 | 0.515 | 0.723 |
| DQN | Maskable | 0.858 | 1.000 | 1.000 |
| DQN | Repair | 0.823 | 1.000 | 1.000 |
| DDQN | 无约束 | 0.147 | 0.650 | 0.705 |
| DDQN | Lagrangian | 0.058 | 0.682 | 0.632 |
| DDQN | Maskable | 0.860 | 0.998 | 1.000 |
| DDQN | Repair | 0.780 | 0.998 | 1.000 |

AR/AR\*（成功 episode）：PPO 四种机制 LT 0.884～0.895、EQ 0.935～0.942、GT 0.951～0.956；DQN/DDQN 的 Maskable / Repair LT 0.875～0.887、EQ 0.928～0.941、GT 0.929～0.953；DQN/DDQN 的 Lagrangian 只有 0.60～0.81。

### 分析

1. **PPO：四种约束处理结果几乎相同**。无约束 PPO 只靠 legacy 奖励就学会了基本避开违规（LT success 0.848，约 15% 的回合仍有违规；EQ/GT 0.985～0.995）。Lagrangian-PPO 在 LT 上 success 最高（0.880，λ 收敛到 2.43；EQ/GT 为 1.16 / 1.00），说明 v4.3.1.4 的失败确由 objective 奖励 + λ 上限造成，换回 legacy 后正常。
2. **DQN / DDQN：只有 Maskable 和 Repair 有效**。无约束 success 只有 0.15～0.71、违规回合 30%～88%；Lagrangian 不但没改善，AR/AR\* 还明显下降（0.60～0.81），λ 一路升到 16～25。原因推测：λ 在训练中持续变化，经验池中的旧样本保留旧 λ 下的奖励，目标非平稳；λ 变大后 Q 值尺度被违规代价主导。
3. **学习算法的差异取决于约束机制**：
   - 没有硬性约束机制时（无约束 / Lagrangian），PPO 远好于 DQN / DDQN（LT 0.85～0.88 vs 0.06～0.25，EQ/GT 0.98～1.0 vs 0.52～0.72）；
   - 有硬性机制时（Maskable / Repair），三种学习算法几乎相同（LT Maskable：PPO 0.855、DQN 0.858、DDQN 0.860）。
   - 这支持"终局奖励下多步回报的 on-policy 方法能从奖励中学到约束，一步自举的 off-policy 方法学不到"的解释；可行性由机制保证时，学习算法的差别被抹平。
4. **Maskable vs Repair**：LT 上 Maskable 的 success_rate 高于 Repair（PPO 0.855 vs 0.825，DDQN 0.860 vs 0.780），Repair 的死局率更高（17.5%～22% vs 14%～14.5%）；EQ/GT 两者都接近 1.0，Repair 的 AR/AR\* 在 DQN/DDQN 上略高（GT 0.951 vs 0.929）。
5. **场景**：LT 是唯一能拉开差距的场景（所有方法 success ≤ 0.88，死局率 14%～22%）；EQ / GT 上 PPO 与带硬性机制的方法都接近 1.0。
6. 与 v4.3.1.4（objective 奖励、1M 步）不可直接比较：奖励、步数、PPO clip 都不同。

### 注意

- 单种子；正式结论需 3 种子。
- AR/AR\* 只在成功 episode 上计算，不同方法的成功实例集合不同。


## 正式实验（3 种子）

- 12 个模型 × 3 场景 × 种子 {1, 2, 3} × 5M 步 = 108 次运行。种子 1 复用上面的单种子运行（代码与设定完全相同，commit `ede2187`），本批次补跑种子 2、3，共 72 个任务。代码未改，仍属 v4.3.1.6。
- 脚本：`scripts/run_v4.3.1.6_campaign.py`；manifest：`scripts/logs/v4.3.1.6_campaign/manifest.json`（含种子 1 的 exp_id）；报告：`campaign_report.md`（本目录，3 种子均值 ± 样本标准差，新增"AR/AR\*×success"综合指标，失败计 0）。
- DQN / DDQN 的 Lagrangian 保持现有设定（未单独调整），其表现差作为结果如实报告。

（结果待补充）
