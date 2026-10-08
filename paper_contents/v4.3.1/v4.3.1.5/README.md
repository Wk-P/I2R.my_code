# v4.3.1.5 — Lagrangian 改为终局代价、λ_max 50

v4.3.1 的子版本（tag `v4.3.1.5`），只改 Lagrangian（PPO / DQN / DDQN 相同），其余与 v4.3.1.4 完全一致（p = 0.6 数据、objective 奖励、死局即失败）。

## 1. 改动

| | v4.3.1.4 | v4.3.1.5 |
|---|---|---|
| 约束代价 | 每步 $-\lambda c_t$ | 终局一次 $-\lambda\sum_t c_t$（$c_t$ = 第 $t$ 步违反的约束数） |
| $\lambda_{max}$ | 5 | 50 |

改后 12 个模型的奖励全部只出现在终局：

| 约束处理 | 中间步 | 终局（放完 M 个服务） | 死局 |
|---|---|---|---|
| 无约束 | 0 | $M\cdot AR_{exec}$ | — |
| Lagrangian | 0 | $M\cdot AR_{exec}-\lambda\sum_t c_t$ | — |
| Maskable | 0 | $M\cdot AR_{exec}$ | 失败，$-M(1-\text{valid}/M)$ |
| Repair | 0 | $M\cdot AR_{exec}$ | 失败，$-M(1-\text{valid}/M)$ |

原因：

1. **终局代价**：所有模型的奖励形式统一为终局奖励。不打折扣时与逐步扣除的回合总回报相同，$\gamma=0.99$ 时只差折扣。
2. **$\lambda_{max}$ 50**：v4.3.1.4 中 LT 的 Lagrangian-PPO 失败（success 0），$\lambda$ 卡在上限 5。违规的"全部堆到一个 ECU"在 LT 上终局奖励约 $15\times11.7\approx175$，违规总数至多 $2M=30$，$\lambda$ 需要超过约 6 才能让违规不划算。对偶上升规则与其余常数不变（初值 0.1、步长 0.005、窗口 20、预热 5000 episode）。

## 2. 关于 PPO 与 DQN 的比较（论文表述建议）

终局奖励下，PPO 用多步回报（GAE，$\lambda=0.95$）把终局信号分配到整条轨迹，DQN / DDQN 用一步自举，终局奖励需逐轮回传；PPO 为随机策略、on-policy，DQN 为 $\varepsilon$-greedy、off-policy（经验池中的旧样本保留旧 $\lambda$ 下的奖励）。因此有理由预期 PPO 更适合本设定，但这不是定理：差距主要来自回报估计方式而非"策略方法 vs 价值方法"本身（可用 n-step DQN 做消融检验），且需要多种子 5M 结果支持。v4.3.1.4 中 LT 的 Lagrangian-PPO 失败而 Lagrangian-DQN 成功，即为尚待解释的反例，本版本用于检验它是否由 $\lambda$ 尺度造成。

## 3. 自检

三个场景各 300 个随机 episode：Lagrangian 中间步奖励全为 0；终局奖励与 $M\cdot AR_{exec}-\lambda\sum c_t$ 完全一致（0 处不一致）。

## 4. 试跑

- 3 个 Lagrangian 模型 × 3 场景 × objective × 种子 1 × **5M 步** = 9 个任务；脚本 `src/scripts/run_v4.3.1.5_pilot.py`。其余 9 个模型不受本次改动影响。
- 注意：本轮 5M 步，v4.3.1.4 的其余模型为 1M 步，不能直接横向比较；本轮只用于验证 Lagrangian 能否学会。
- 记录：`logs/v4.3.1.5_pilot/manifest.json`；报告 `pilot_report.md`（本目录）。

**已作废**（2026-10-02 15:18 启动，约 16:30 停止，9 个任务均未完成）。停止原因：用户决定 12 个模型的奖励改回 legacy、PPO clip ε 0.2 → 0.1（见 v4.3.1.6），本轮设定不再代表最终方案。

停止前的观察：LT Lagrangian-PPO 训练到 1.1M 步时 λ 已升至 30.07（违规总数 ≤ 30，即代价可达约 900，远大于堆叠的终局奖励约 175），但最近 500 个 episode 的 success_rate 仍为 0。说明在 objective 奖励下，λ 放大后 PPO 仍未离开"全部堆到一个 ECU"的解，问题不只是 λ 的上限。
