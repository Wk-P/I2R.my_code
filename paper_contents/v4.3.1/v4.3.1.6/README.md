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

（结果待补充）
