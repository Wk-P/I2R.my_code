# v4.3.1.3 — 统一实现：三场景只差 N / M

v4.3.1 的子版本（tag `v4.3.1.3`）。起因：v4.3.1.2 审计发现三个场景的代码、配置不一致，且 EQ/GT 的冲突数据几乎不允许共置（见 `../v4.3.1.2/README.md` 第 4 节），该版本试跑作废。

**原则**：LT / EQ / GT 在模型、环境、奖励、超参数上完全相同，唯一区别是 ECU 数 N 与服务数 M。12 个模型之间只差约束处理机制与学习算法。

## 1. 问题设定与约束

把 M 个服务放到 N 个 ECU 上，每步放一个服务。

| 约束 | 定义 |
|---|---|
| capacity | 放在同一 ECU 上的服务所需 container 数之和不超过该 ECU 的 container 上限 |
| privacy-conflict | 同一个冲突集中的两个服务不能放在同一 ECU；不共享任何冲突集的服务可以共置 |

ILP 约束与此相同（`shared/ilp_utils.solve_ilp`）。术语：容量与需求的单位为 container（原文中的 VM 一律改为 container）。

## 2. 场景数据（`paper_rl/data.py` → `data/v4.3.1.3/{lt,eq,gt}.yaml`）

| 场景 | N | M |
|---|---|---|
| LT | 10 | 15 |
| EQ | 10 | 10 |
| GT | 15 | 10 |

三个场景使用同一套分布：

- ECU 容量 ~ 均匀 {50, 55, …, 195} container；服务需求 ~ 均匀 {10, 15, …, 95} container（独立同分布，可重复）。
- 冲突集：K = 10 个；每个服务以概率 q 独立加入每个冲突集（不足 2 个成员的集合丢弃）。任意两个服务存在冲突的概率 $p = 1-(1-q^2)^K$，与 M 无关。取 **p = 0.3**。
- ILP 不可行的实例重抽；每个场景 2000 个实例，ILP 最优 AR\* 随实例存储。

冲突密度的选择（每个场景各 150 个实例）：

| p | 场景 | 可行率 | 实测冲突对比例 | 平均 AR\* | ILP 最优解每个启用 ECU 的服务数 |
|---|---|---|---|---|---|
| 0.1 | LT / EQ / GT | 97% / 100% / 100% | 10% / 11% / 10% | 0.922 / 0.880 / 0.884 | 1.91 / 1.63 / 1.46 |
| 0.2 | LT / EQ / GT | 97% / 100% / 100% | 20% / 21% / 19% | 0.909 / 0.860 / 0.876 | 1.88 / 1.59 / 1.45 |
| **0.3** | LT / EQ / GT | 97% / 100% / 100% | 31% / 32% / 30% | 0.896 / 0.838 / 0.860 | 1.86 / 1.54 / 1.42 |
| 0.5 | LT / EQ / GT | 95% / 100% / 100% | 51% / 52% / 49% | 0.838 / 0.784 / 0.816 | 1.74 / 1.45 / 1.35 |

正式数据（2000 个实例）的平均 AR\*：LT 0.9026，EQ 0.8485，GT 0.8547。对比旧数据：EQ/GT 服务两两冲突 99%、LT 74%。

注意：EQ / GT 的总容量约为总需求的 2 倍，Mask / Repair 即使随机选动作 success_rate 也接近 100%，这两个场景主要比较 AR。

## 3. 统一环境（`paper_rl/env.py::PlacementEnv`）

- 服务按需求降序依次放置（所有模型、场景相同）。
- 观测（维度 $6+5N+2M+1$）：当前服务需求、当前 AR、剩余可用容量和、剩余需求和、当前服务可行 ECU 比例、剩余服务比例；各 ECU 的初始容量、剩余容量、冲突标志、仍可接收（无冲突）的服务比例、可行标志（容量且无冲突）；各服务剩余需求、各服务当前可行 ECU 比例；$\lambda/\lambda_{\max}$（仅 Lagrange，其余为 0）。
- AR 只按合法执行的放置计算（违规放置不计入，也不使 ECU 记为启用）；无违规 episode 上即通常的 AR。
- success：M 个服务全部合法放置。

| 约束机制 | 做法 |
|---|---|
| 无约束 | 动作照常执行，只记录违规 |
| Mask | 动作掩码 = 同时满足两项约束的 ECU；无可行 ECU 时掩码全开，执行的放置记为违规 |
| Lagrange | 动作照常执行，每步奖励减 $\lambda c_t$（$c_t$ = 本步违反的约束数）；$\lambda$ 由对偶上升更新并进入观测 |
| Repair | 不可行动作替换为最佳适应 ECU（$\arg\max n_i/e_j$，可行 ECU 中）；无可替换 ECU 时 episode 终止 |

## 4. 奖励（`REWARD_MODE`，三场景、12 个模型相同）

| 模式 | 中间步 | 结束 |
|---|---|---|
| legacy | 0 | 成功 $M(2AR-1)$；否则 $-M(1-\text{valid}/M)$ |
| ar | 0 | 成功 $M\cdot AR$；否则 $-M(1-\text{valid}/M)$ |
| directional | $r^{obj}(\Delta AR_t)$ | 成功另加 $B=M$；死局（Mask：下一服务无可行 ECU；Repair：无可替换 ECU）$-C=-M$ 并终止 |

$r^{obj}(d)=1+\beta d\ (d>\epsilon)$，$\beta d\ (|d|\le\epsilon)$，$-\lambda_d+\beta d\ (d<-\epsilon)$；$\beta=10,\ \lambda_d=1,\ \epsilon=10^{-3}$。Lagrange 方法在三种模式下都另加 $-\lambda c_t$。Repair 在 legacy / ar 下无可替换 ECU 时以 $-M(1-\text{valid}/M)$ 终止。

去掉的旧特例：Mask-PPO 的一步前瞻掩码与 AR 课程权重 $w$（仅 LT）、Repair-PPO 的单步 −0.1 修复惩罚与"触发修复则终局 0"（仅 LT）、Lagrange-PPO 各场景不同的逐步奖励、`bottleneck_risk` 特征。

## 5. 超参数（`paper_rl/config.py`，三场景相同）

| | PPO 系列 | DQN / DDQN 系列 |
|---|---|---|
| 并行环境 | 40 | 12 |
| 学习率 | 3e-4 | 1e-3 |
| 其他 | n_steps 512，batch 256，epochs 10，γ 0.99，GAE λ 0.95，clip 0.2，熵系数 0.005（恒定），网络 pi/vf [256,256] | buffer 1e5，learning_starts 2000，batch 64，γ 0.99，train_freq 4，target 每 500 步硬更新，ε 在前 50% 步数从 1 降到 0，网络 [128,128] |

DQN 与 DDQN 只差目标值计算；Mask-DQN/DDQN 在探索、贪心动作和目标值中都只考虑可行 ECU（`shared/dqn_variants.py`）。

Lagrange（PPO / DQN / DDQN 相同）：$\lambda_0=0.1$，预热 5000 个 episode 后每 20 个 episode 更新 $\lambda\leftarrow\mathrm{clip}(\lambda+0.005\,\bar\nu,0,5)$，$\bar\nu$ 为平均（容量 + 冲突）违规数 / M；评估时 $\lambda$ 固定为最终值。

## 6. 代码与输出

| 文件 | 作用 |
|---|---|
| `paper_rl/data.py` | 数据生成（含 ILP） |
| `paper_rl/env.py` | 统一环境 |
| `paper_rl/config.py` | 统一超参数 |
| `paper_rl/train.py` | 训练 + 评估：`python -m paper_rl.train --scen lt --algo mask_ppo --reward ar --steps 1000000 --seed 1` |
| `scripts/run_v4.3.1.3_reward_pilot.py` | 奖励选型试跑 |

- 算法名：`ppo / mask_ppo / lagrange_ppo / repair_ppo / dqn / mask_dqn / … / repair_ddqn`。
- 结果写入 `results/unified/<scen>/<algo>/<exp_id>/`（`RESULTS_SPACE = "unified"`，与旧数据完全分开）：模型、`results.json`（含 version、commit、reward_mode、seed）、`summary.csv`、`training_curve.csv/png`。
- 评估：80/20 划分（按种子打乱），测试集 400 个实例，每个单次确定性评估。
- `scenarios/` 下的旧实现保留不动，仅用于复现 v4.3.1.2 及以前的结果；从本版本起论文实验只使用 `paper_rl/`。

## 7. 自检

- 36 种组合（3 场景 × 4 机制 × 3 奖励）各 150 个随机 episode：可行性判定与从零重算的参考定义 0 处不一致；观测维度正确且数值在 [−1, 1]；AR ≤ 1；共置普遍存在。
- 12 个模型在 EQ 上各训练 3 万步跑通训练、评估、保存全流程。

## 8. 奖励选型试跑

12 个模型 × 3 场景 × {legacy, ar, directional} × 种子 1 × 1M 步 = 108 个任务，三个场景交错排队。报告：`reward_pilot_report.md`（本目录）。选型标准同 v4.3.1.2：带约束机制方法的 success_rate 不下降、AR/AR\* 不明显下降、4 种约束机制之间的排序稳定。

**已完成**（2026-10-01 23:42 – 10-02 00:54，108/108，无失败）。完整表格见 `reward_pilot_report.md`。

结论（单种子、1M 步、p = 0.3，仅作选型依据）：

1. **directional 不采用**。
   - 带约束机制的 9 个模型上，directional 的 AR/AR\* 普遍最低，例如 GT Maskable-DQN 0.951（ar）→ 0.904，EQ Maskable-DDQN 0.938 → 0.889，GT Repair-DDQN 0.942 → 0.885。离散项 $+1/0/-\lambda_d$ 鼓励"多次小幅上升"，与最终 AR 不一致。
   - 无约束 DQN/DDQN 在 directional 下 success_rate 崩溃（LT 0.058 / 0.030，GT 0.107 / 0.325），因为违规步只是 $\Delta AR=0$、几乎没有代价。
2. **ar 与 legacy 接近，ar 略好，建议采用 ar**。
   - LT 上 PPO 的四种约束处理 success_rate 都提高 1.5～3 个百分点（无约束 0.970→0.985，Lagrangian 0.968→0.983，Maskable 0.958→0.973，Repair 0.912→0.940），AR/AR\* 基本不变。
   - ar 保证"任何成功 > 任何失败"，legacy 不保证。
   - 单种子波动：GT Lagrangian-DDQN 在 ar 下 success_rate 0.767（legacy 0.985），需多种子确认。
3. **约束处理机制之间的排序在 legacy / ar 下稳定**：PPO 在三个场景中 AR/AR\* 都是 Maskable 最高、Repair 最低；DQN/DDQN 上 Maskable 与 Repair 的 success_rate 稳定在 0.90～1.00，Lagrangian 次之，无约束最差。
4. **约束偏松**：无约束 PPO 的 success_rate 在 LT / EQ / GT 分别为 0.97～0.985 / 0.998～1.0 / 0.995，与有约束方法几乎相同（原因：奖励失败分支惩罚违规 + 观测含可行标志，PPO 学会了避开违规）。无约束 DQN/DDQN 学不会（0.19～0.81）。→ v4.3.1.4 提高冲突密度。
5. LT 上 Maskable 在 legacy / ar 下出现 2%～4% 的违规回合，来自死局时放开掩码的设计 → v4.3.1.4 改为死局即失败并终止。

## 下一步（已定，待本轮试跑结束后执行）

- **v4.3.1.4**：冲突密度 p 由 0.3 提高到 **0.6**，重新生成三场景数据后重跑试跑。原因：本轮早期结果中 LT 的无约束 PPO success_rate 已达 97%～98.5%，与 Maskable-PPO 接近，约束过松，约束处理机制之间拉不开差距。
- **v4.3.1.4**：死局一律判失败并终止。Maskable（下一服务无可行 ECU）与 Repair（无可修复 ECU）在任何奖励模式下都立即终止、success = 0、给失败惩罚（legacy / ar：$-M(1-\text{valid}/M)$；directional：$-C$），不执行违规放置。v4.3.1.3 中 legacy / ar 模式下 Maskable 在死局时放开掩码、执行一次违规放置并继续（LT 上 2%～4% 的回合因此出现违规），该行为取消。
- **v4.3.1.4 待定**：无约束对照的定义（保持"奖励惩罚违规 + 观测含可行信息"，或改为奖励只看 AR、不惩罚违规）。
