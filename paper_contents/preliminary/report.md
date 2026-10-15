# 初步结果：Maskable 与 Lagrangian（5 万实例、20M、Transformer）

- 整理时间：2026-10-15。数据全部取自已完成批次的报告，未做新的计算。
- 状态：**初步**。Repair（v5.3.1，运行中）与 unconstraint（v5.3.2，排队中）尚无结果；Lagrangian 只有种子 1。

## 1. 实验设置

| 项目 | 设置 |
|---|---|
| 问题 | 带容量与隐私冲突约束的服务 → ECU 部署；场景 LT / EQ / GT |
| 网络 | 结构感知网络（Transformer），GPU 训练 |
| 训练数据 | 5 万个经 ILP 求出最优解的实例池（`results/sup_diag/<场景>.json`） |
| 训练步数 | 20M |
| 奖励 / 回合 | ar_pen 奖励、γ = 1、不提前结束（死局即违约）、服务按需求降序 |
| 评估 | 测试集：每个种子自己的 400 个实例；确定性回放；gap = 相对 ILP AR 的相对差距，只在未违约 / 未 EXIT 的实例上计算 |

各机制的数据来源：

| 机制 | 批次 | 种子 | 备注 |
|---|---|---|---|
| Maskable PPO | v4.4.15 → v4.4.16 → v4.4.17（在 v4.5.1 中汇总） | 1–3 | 5M → 10M → 20M 续训链 |
| Maskable DQN / DDQN | v4.5.2 | 1–3 | ε 1.0→0 用前 50% 步数（10M） |
| Lagrangian PPO / DQN / DDQN | v5.1.0 | 1 | λ 初值 0.001、Adam lr 0.035、λ ≥ 0 不设上限、J_C 取行为策略、d = 0；DQN / DDQN 的 ε 1.0→0 固定 2M 步 |

## 2. 主结果（20M，测试集）

### 2.1 相对 gap

Maskable 为 3 个种子的均值 ± 样本标准差；Lagrangian 为种子 1 单次结果。

| 机制 | Learner | LT | EQ | GT | 三场景平均 |
|---|---|---|---|---|---|
| Maskable | PPO | 7.0% ± 0.3% | 4.9% ± 0.3% | 4.6% ± 0.2% | **5.5% ± 0.0%** |
| Maskable | DQN | 9.9% ± 0.5% | 4.1% ± 0.1% | 3.1% ± 0.4% | **5.7% ± 0.3%** |
| Maskable | DDQN | 9.9% ± 0.5% | 4.6% ± 0.3% | 3.8% ± 0.8% | **6.1% ± 0.1%** |
| Lagrangian | PPO | 6.7% | 6.0% | 7.2% | **6.6%** |
| Lagrangian | DQN | 25.9% | 8.1% | 3.4% | **12.5%** |
| Lagrangian | DDQN | 26.7% | 11.9% | 6.5% | **15.0%** |

### 2.2 约束违约与未通过率

未通过 = EXIT 或任一约束违约。Maskable 用动作掩码保证不选不可行 ECU，只有在所有 ECU 都不可行（死局）时才会未通过。

| 机制 | Learner | LT 未通过率 | LT 容量违约率 | LT 隐私违约率 | EQ 未通过率 | GT 未通过率 |
|---|---|---|---|---|---|---|
| Maskable | PPO | 1.8% ± 0.3% | 0（掩码） | 0（掩码） | 0.1% ± 0.1% | 0.0% |
| Maskable | DQN | 2.6% ± 0.6% | 0（掩码） | 0（掩码） | 0.1% ± 0.1% | 0.0% |
| Maskable | DDQN | 2.5% ± 0.5% | 0（掩码） | 0（掩码） | 0.0% | 0.0% |
| Lagrangian | PPO | 2.7% | 2.8% | 0.2% | 0.0% | 0.0% |
| Lagrangian | DQN | 6.8% | 2.0% | 5.2% | 0.0% | 0.0% |
| Lagrangian | DDQN | 6.0% | 3.8% | 3.8% | 0.0% | 0.0% |

### 2.3 学习曲线（测试集三场景平均 gap）

| 步数 | Mask PPO | Mask DQN | Mask DDQN | Lag PPO | Lag DQN | Lag DDQN |
|---|---|---|---|---|---|---|
| 5M | 7.4% | 9.4% | 9.2% | 10.0% | 20.2% | 16.4% |
| 10M | 6.3% | 7.0% | 8.7% | 8.3% | 14.8% | 19.3% |
| 15M | 5.9% | 7.2% | 7.3% | 7.6% | 13.7% | 17.0% |
| 20M | 5.5% | 5.7% | 6.1% | 6.6% | 12.5% | 15.0% |

## 3. 初步结论

1. **Maskable 在三个 learner 上都是最好的机制。** 三场景平均 gap 为 5.5%（PPO）/ 5.7%（DQN）/ 6.1%（DDQN），种子间标准差 ≤ 0.3 个百分点，且约束违约为 0。
2. **Lagrangian PPO 接近 Maskable，但 LT 上仍有违约。** 平均 gap 6.6%，比 Maskable PPO 高约 1.1 个百分点；EQ / GT 违约率为 0，LT 容量违约 2.8%、隐私违约 0.2%。
3. **Lagrangian 下的 DQN / DDQN 在 LT 上失效。** LT gap 约 26%，违约率 2–5%；EQ / GT 表现正常，其中 DQN 在 GT 上 gap 3.4%，是该场景所有结果中最低。
4. **LT 是最难的场景。** 所有方法在 LT 上的 gap 和未通过率都最高；Lagrangian 的 λ 在 LT 上到 20M 仍在上升（PPO 2.97、DQN 5.60、DDQN 5.31），J_C 未降到 0，约束还没被完全满足。
5. **Maskable 的曲线到 20M 仍在下降**（15M → 20M 下降 0.4–1.5 个百分点），20M 还不是上限。

## 4. 局限与待补

- **Lagrangian 只有种子 1**，表中 Lagrangian 数值没有方差，不能作统计比较。
- **口径尚未完全统一**：Maskable DQN / DDQN 的 ε 为前 10M 步衰减，其余为固定 2M 步；Maskable PPO 为分段续训，其余为一次训练 20M。
- **Repair 与 unconstraint 尚无 20M 结果**（v5.3.1 运行中、v5.3.2 排队中，均为种子 1）。
- gap 只在通过的实例上计算；未通过率高的组合（Lagrangian DQN / DDQN 的 LT）gap 只反映较容易的那部分实例。
- 本报告只列相对 gap；绝对 AR 与 ILP AR 未在这些批次的报告中输出，正式版需补齐。
- 正式结论计划为：4 种机制 × 3 learner × 3 场景 × 3 种子，统一 5 万实例、20M、同一 ε 设置。

## 5. 数据出处

- Maskable PPO：`paper_contents/v4.5.1/report.md`（Transformer 列）
- Maskable DQN / DDQN：`paper_contents/v4.5.2/report.md`（Transformer 列）
- Lagrangian：`paper_contents/v5.1.0/report.md`、`paper_contents/v5.1.0/report_data.json`
