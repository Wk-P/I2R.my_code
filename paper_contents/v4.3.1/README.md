# v4.3.1 — 奖励设计探索（可切换奖励）+ 实验面板改版

在 v4.3.0（DQN 系列改用 PPO 奖励、新增 Mask/Repair-DQN/DDQN）之后的增量版本。**默认行为与 v4.3.0 完全相同**：新增的奖励写法全部通过环境变量 `REWARD_MODE` 切换，不设置时为 `legacy`。

## 1. 可切换的奖励 `REWARD_MODE`（`src/shared/reward_config.py`）

| 模式 | 无违规 episode 的终局奖励 | 有违规 episode 的终局奖励 | 中间步 |
|---|---|---|---|
| `legacy`（默认） | $M(2AR-1)\in(-M,M]$ | $-M(1-\text{valid}/M)\in[-M,0)$ | 0 |
| `ar` | $M\cdot AR\in(0,M]$ | 同上 | 0 |
| `ratio` | $M\cdot AR/AR^*\in(0,M]$ | 同上 | 0 |
| `directional` | 见第 2 节（逐步奖励，整体替换） | | |

- AR = average resource utilization（平均资源利用率，优化目标）；$AR^*$ 为该实例的 ILP 最优 AR。
- `legacy` 的问题：$AR<0.5-\frac{1}{2M}$ 时无违规 episode 的奖励低于"只在最后一步违规"的 $-1$（EQ 约 1/10 的实例受影响，见 `../v4.3.0/README.md`"已知问题"）。`ar` / `ratio` 保证任何无违规 episode（>0）严格优于任何违规 episode（<0）。
- $AR^*$ 由 `src/scripts/build_ar_star.py` 对每个场景 2000 个实例并行求 ILP 得到，存于 `results/<branch>/<scen>/ilp/ar_star.json`（按实例内容哈希索引），约 80 秒。
- 接入的环境：PPO、Mask-PPO、DQN、DDQN（Mask-DQN/DDQN 复用 Mask-PPO 环境）。Mask-PPO LT 的课程权重 $w$ 保留：成功分支为 $M[(1-w)+w\cdot q]$，$q$ 为上表中的质量项。

## 2. directional 奖励（依据 `../v4.3.0/Reward-Discussion.md`）

所有方法共用同一个 objective reward，方法之间只改约束处理机制。$\Delta AR_t=AR_{t+1}-AR_t$：

$$
r_t^{\text{obj}}=\begin{cases}1+\beta\Delta AR_t,&\Delta AR_t>\epsilon\\ \beta\Delta AR_t,&|\Delta AR_t|\le\epsilon\\ -\lambda_d+\beta\Delta AR_t,&\Delta AR_t<-\epsilon\end{cases}
$$

| 情形 | 奖励 | 适用 |
|---|---|---|
| 每一步 | $r_t^{\text{obj}}$ | 全部 |
| $M$ 个服务全部合法放置（最后一步） | $+B$ | 全部；Repair-* 执行的放置结构上合法，跑满即算 |
| 死局：还有服务未放，但下一状态已无合法 ECU | $-C$，episode 终止 | Mask-PPO / Mask-DQN / Mask-DDQN |
| 死局：修复找不到合法 ECU | $-C$，episode 终止 | Repair-PPO / Repair-DQN / Repair-DDQN |
| 约束代价 | $-\eta\,c_t$，$\eta$ = 环境中的对偶变量 $\lambda$，$c_t$ = 本步违规数 | Lagrange-PPO |
| 无约束处理 | 只有 $r_t^{\text{obj}}$（$+B$） | PPO / DQN / DDQN |

默认参数（环境变量可改）：`DIR_BETA` $\beta=10$，`DIR_LAMBDA` $\lambda_d=1$，`DIR_EPS` $\epsilon=10^{-3}$，`DIR_B` $B=M$，`DIR_C` $C=M$。

实现：`directional_step()` 装饰器包在 18 个环境（6 类 × 3 场景）的 `step()` 外，读取环境自身的 AR、违规计数、`action_masks()`、修复失败终止，整体替换奖励；`REWARD_MODE≠directional` 时装饰器直接返回原 `step`，不改变任何行为。Repair-PPO 原有的单步 −0.1 修复惩罚、Lagrange-PPO EQ/GT 原有的逐步惩罚在 directional 下都不再使用。

### 待验证的问题

1. **离散项与目标是否一致**：连续项 $\beta\sum_t\Delta AR_t=\beta\,AR_{\text{final}}$，与目标完全一致；离散项 $+1/0/-\lambda_d$ 统计"上升了几步、下降了几步"，不随最终 AR 单调，可能让策略偏向多次小幅改善而非最终 AR 最高。$\beta$ 越大越接近纯 AR 目标。
2. **无约束 PPO/DQN 的违规没有直接代价**：违规放置照样获得 $\Delta AR$ 奖励，唯一的代价是拿不到 $B$。随机策略下 EQ PPO 的平均回报约 8.8，success_rate 为 0。
3. **Lagrange-PPO 依赖 $\lambda$ 增长**：LT 的 $\lambda$ 预热 20000 个 episode、步长 $3\times10^{-4}$，1M 步内几乎为 0。
4. Reward-Discussion 中提到的 Lagrange-DQN / Lagrange-DDQN 当前还没有实现。

## 3. 1M 步奖励试跑（`legacy` / `ar` / `ratio`）

- 脚本：`src/scripts/run_v4.3.1_reward_pilot.py`；3 种奖励 × {PPO, Mask-PPO, DQN} × 3 场景 × 种子 1 = 27 个任务。
- 报告：`reward_pilot_report.md`（本目录）；manifest：`logs/v4.3.1_reward_pilot/manifest.json`。
- 调度权重按实测 CPU 占用（PPO 7 核、DQN 3 核），支持 `--resume` 接续。

结果（success_rate；单种子、1M 步，AR 为全部测试 episode 的均值）：

| 场景 | 算法 | legacy | ar | ratio |
|---|---|---|---|---|
| LT | Mask-PPO | 0.608 | 0.665 | 0.643 |
| LT | PPO | 0.355 | 0.345 | 0.353 |
| LT | DQN | 0.135 | 0.068 | 0.068 |
| EQ | Mask-PPO | 1.000 | 1.000 | 1.000 |
| EQ | PPO | 0.733 | 0.990 | 1.000 |
| EQ | DQN | 0.573 | 0.975 | 0.978 |
| GT | Mask-PPO | 1.000 | 1.000 | 1.000 |
| GT | PPO | 0.970 | 0.998 | 0.998 |
| GT | DQN | 0.740 | 0.875 | 0.918 |

- 与 legacy 漏洞的分析一致：受影响最大的 EQ 上，无约束 PPO / DQN 的 success_rate 从 0.73 / 0.57 升到 0.99–1.00 / 0.98，冲突违规率从 25% / 40% 降到 0–2%；GT 也有提升；LT（不受该漏洞影响）基本不变，DQN 反而下降（单种子，待复核）。
- 无约束算法在 ar / ratio 下 AR 略低（EQ PPO 0.506 → 0.489），但 legacy 的 AR 包含大量违规 episode，两者不完全可比；Mask-PPO 的 AR 三种奖励基本相同。
- ar 与 ratio 差别很小；ratio 在 GT DQN 上 success_rate 略高。

## 4. Repair-* 不再记录修复触发率

`scenarios/*/ppo_opt/run_all.py` 与 `src/shared/dqn_variant_runner.py`：违规列只统计实际执行的放置（Repair-* 结构上为 0），结果文件不再输出修复触发率。环境内部仍统计修复次数，因为 `legacy` 奖励要用。

## 5. git 分支整理与结果目录

- 原先 5 个分支（main → pretrain → paper-verfication → add_states → final_paper_experiments）是同一条直线上的研究阶段，均已包含在最新提交中。整理后只保留 **main** 作为主线（已快进到最新），版本用 tag 标记。
- 旧分支改为归档 tag 后删除（本地与 GitHub）：`archive/stage1-main`、`archive/stage2-pretrain`、`archive/stage3-paper-verification`（含原本地未推送的 2 个提交）、`archive/stage4-add-states`、`archive/stage5-final-paper-experiments`。查看旧阶段：`git checkout archive/stage4-add-states`。
- 结果写入位置与分支名解耦：`src/shared/version_config.RESULTS_SPACE = "final_paper_experiments"`（`$RESULTS_SPACE` 可覆盖），新训练继续写入 `results/final_paper_experiments/`；已有的 `results/<旧分支名>/` 目录和文档中的路径引用全部保持不变。面板中以"数据空间"展示，并标注阶段名与版本范围（`app/backend/result_spaces.json`）。

## 6. 实验面板（`app/`）改版

- 布局：左侧导航（总览 / 训练监控 / 批次管理 / 实验结果 / 版本记录 / 论文草稿），层级为 版本 → 批次 → 场景 → 算法 → 单次运行。
- 实验结果：版本→批次树 + 场景/算法/变体筛选 + 关键词；明细（排序、分页）与汇总对比（算法 × 场景透视、可选指标）；筛选条件写入 URL。
- 顶栏全局分支切换（只列出有结果的分支，记住选择）、全局搜索；手机端抽屉导航与响应式布局。
- 术语：success_rate、AR（average resource utilization）、AR gap = ILP AR − AR。
- 后端修复：版本列表遇到非纯数字 tag 时 500；manifest 批次识别；旧批次结果分支推断；并发任务进度按各自日志读取。
