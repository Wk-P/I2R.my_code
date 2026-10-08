# v4.3.0 — DQN 系列改为 PPO 奖励 + 新增 Mask/Repair 变体

依据 `By-v4.2.0.md`。PPO 系列（PPO / Mask-PPO / Lagrange-PPO / Repair-PPO）不重训，沿用 v4.2.0 的模型与数据；本版本只重训 DQN 系列。

## 改动

### 1. DQN / DDQN：奖励与 PPO 相同

`scenarios/{lt,eq,gt}/{dqn,ddqn}/env.py` 去掉单步惩罚 $P_t = 2v^{\text{cap}}_t + 2v^{\text{conf}}_t$，改为与 PPO（4.1 节）一致的稀疏终局奖励：

$$
r_t = 0\quad(t<M-1),\qquad
r_{M-1} = R_{\text{term}} =
\begin{cases}
M\,(2\,AR-1), & \text{本 episode 无任何违规} \\[2pt]
-M\left(1-\dfrac{\text{valid}}{M}\right), & \text{否则}
\end{cases}
$$

GT 的 DQN/DDQN 原来在容量违规时直接终止 episode（奖励 $-M$），这和 v4.2.0 README 末尾"待办"里指出的问题是同一个。现在改为与 GT PPO 相同：违规的放置照常执行，只记录违规，episode 总是跑满 $M$ 步。违规那次放置的 $u_t$ 不计入 AR（与 LT/EQ 的 DQN 相同）。

观测、服务排序、超参数都不变。

改完后 DQN / DDQN 与 PPO（无约束基线）一一对应：无掩码、无修复、无逐步惩罚，违规只通过终局奖励的失败分支体现。论文中它们即作为**无约束 DQN / DDQN**，不另设新算法。

### 2. 新增四个变体

每个变体**直接复用对应 PPO 变体的环境**（观测、奖励、约束机制完全相同），所以 Mask-DQN 与 Mask-PPO、Repair-DQN 与 Repair-PPO 之间只差学习算法。超参数与同场景 DQN 完全相同（`config.py` 中 `DQN_*`）。

| 算法 | 环境 | 学习器 |
|---|---|---|
| Mask-DQN | `ppo_mask/env.py::P4Env` | `MaskableDQN` |
| Mask-DDQN | `ppo_mask/env.py::P4Env` | `MaskableDDQN` |
| Repair-DQN | `ppo_opt/env.py::P6Env` | SB3 `DQN` |
| Repair-DDQN | `ppo_opt/env.py::P6Env` | `DoubleDQN` |

**Mask-DQN / Mask-DDQN**（`src/shared/dqn_variants.py`）。掩码 $m_t$ 与 Mask-PPO 相同（LT 含一步前瞻）。P4Env 的观测里 `valid_flag` 段就是 `action_masks()`，所以掩码直接从观测读取，$s_{t+1}$ 的掩码随 replay buffer 里的 next_obs 一起得到。记 $\mathcal{A}(s)=\{j: m(s)[j]=1\}$，若为空则取全部 $N$ 个 ECU：

- 动作选择：$\varepsilon$-greedy 的随机动作与预热期动作都在 $\mathcal{A}(s_t)$ 内均匀采样；贪心动作 $a_t=\arg\max_{a\in\mathcal{A}(s_t)}Q_\theta(s_t,a)$。
- 目标值：

$$
y^{\text{Mask-DQN}} = r_t + \gamma\max_{a'\in\mathcal{A}(s_{t+1})}Q_{\theta^-}(s_{t+1},a'),\qquad
y^{\text{Mask-DDQN}} = r_t + \gamma\,Q_{\theta^-}\big(s_{t+1},\arg\max_{a'\in\mathcal{A}(s_{t+1})}Q_\theta(s_{t+1},a')\big)
$$

- 无合法 ECU 时退回全部动作，选中的放置由环境记为违规（对应 Mask-PPO 在全 0 掩码下的行为）。
- P4Env 的课程权重 $w$ 保持默认值 1，即终局奖励就是基础形式 $M(2AR-1)$；LT Mask-PPO 的 $w$ 退火是 PPO 训练回调做的，不属于环境。

**Repair-DQN / Repair-DDQN**。环境即 Repair-PPO 的 P6Env：所选 ECU 违规时替换为最佳适应 ECU 再执行，$\mathcal{V}_t=\varnothing$ 时终止并给 $-M$；单步 $-0.1$ 修复惩罚与终局奖励按场景同 Repair-PPO（v4.2.0 `formula.md` 4.4 节）。学习器为标准 DQN / Double DQN。

损失、探索、目标网络更新与 v4.2.0 `formula.md` 4.5 节相同。

## 指标口径

- success_rate：DQN/DDQN、Mask-* 为"$M$ 个服务全部合法放置且无违规"的测试实例比例；Repair-* 同 Repair-PPO，为"跑满 $M$ 步"的比例（执行的放置结构上不违规）。
- AR：average resource utilization，平均资源利用率，即优化目标。
- 违规率：测试 episode 中实际执行的放置出现 ≥1 次违规的比例。Repair-* 执行的放置结构上不违规，违规率恒为 0；修复触发率不记录、不统计。
- 评估：每个测试实例单次确定性评估（与 v4.2.0 相同）。

## 代码位置

- `scenarios/{lt,eq,gt}/{dqn,ddqn}/env.py`：奖励修改
- `src/shared/dqn_variants.py`：`DoubleDQN`、`MaskableDQN`、`MaskableDDQN`
- `src/shared/dqn_variant_runner.py`：四个变体共用的训练/评估/输出流程
- `scenarios/{lt,eq,gt}/{mask_dqn,mask_ddqn,repair_dqn,repair_ddqn}/`：`config.py` + `run_all.py`
- `src/shared/training_steps_config.py`：四个变体纳入统一 5M 步
- `src/shared/version_config.py`：`CURRENT_VERSION = "4.3.0"`
- `src/scripts/run_v4.3.0_campaign.py`：6 个 DQN 系算法 × 3 场景 × 3 种子（1/2/3）× 5M 步，共 54 个任务；结束后生成 `campaign_report.md`

## 验证

- 6 个 DQN/DDQN 环境各跑 200 个随机 episode：非终端奖励全为 0，episode 长度恒为 $M$。
- 18 个（场景, 算法）组合各训练 20k 步跑通全流程（训练、保存、评估、出图）。
- LT Mask-DDQN：100 个测试实例上 2772 次存在合法 ECU 的决策，确定性与随机动作都没有选到非法 ECU；模型保存后能用 `MaskableDDQN.load` 正确加载（`mask_start` 一并恢复）。

## 未做

- 论文叙述部分 "VM → containers" 的修改：正文由作者撰写，未改动。

## 已知问题（未修，所有算法共有）

终局奖励的失败分支上界是 $-1$（只在最后一步失败，$\text{valid}=M-1$），而成功分支 $M(2AR-1)$ 在 $AR < 0.5-\frac{1}{2M}$ 时低于 $-1$，所以"任何成功都优于任何失败"并不成立：

| 场景 | ILP 最优解的成功奖励 < −1 的测试实例 | Mask-PPO（v4.1.0）训练末 2 万个成功 episode 中奖励 < −1 |
|---|---|---|
| LT | 0% | 0.4% |
| EQ | 8.2% | 9.7% |
| GT | 1.2% | 1.6% |

可选修法：失败分支改为 $-M(2-\text{valid}/M)\in(-2M,-M]$，严格低于成功分支下界。该修改影响全部算法，需整体重训，v4.3.0 暂不改动。
