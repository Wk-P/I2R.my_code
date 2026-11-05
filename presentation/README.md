# 讲解目录（v6.0.0）

按顺序读，每一篇只讲一件事，先讲直观，再讲细节。

| 顺序 | 文件 | 讲什么 | 一句话 |
|---|---|---|---|
| 1 | [01_problem.md](01_problem.md) | 问题 | 把服务塞进 ECU，塞得越满越好，但不能超容量、冲突的服务不能放一起 |
| 2 | [02_ilp.md](02_ilp.md) | 标准答案 ILP | 用求解器算出最优解，当作"满分" |
| 3 | [03_environment.md](03_environment.md) | 把问题变成游戏 | 一步放一个服务，智能体每步选一个 ECU |
| 4 | [04_strategies.md](04_strategies.md) | 4 种约束处理 | 智能体选错了怎么办：挡住 / 帮它改 / 罚它 / 不管 |
| 5 | [05_algorithms.md](05_algorithms.md) | 3 种 RL 算法 | PPO 学"概率"，DQN/DDQN 学"打分" |
| 6 | [06_network.md](06_network.md) | 网络（Transformer） | 每个 ECU、每个服务当成一个词，互相看，再给每个 ECU 打分 |
| 7 | [07_training.md](07_training.md) | 怎么训练 | 2000 万步、3 个种子、回调负责记录和更新 λ |
| 8 | [08_results.md](08_results.md) | 结果 | Mask PPO 最好，平均差 ILP 4.38%，快 20 多倍 |

4 种约束处理 × 3 种算法 = **12 个变种**，它们共用同一个环境（第 3 篇）和同一个网络（第 6 篇）。

更正式、更完整的版本见 `paper_contents/v6.0.0/`；本目录只负责"讲明白"，数字都取自那里和 `paper_contents/v6.0.0-merge/`。

## 表格与图片（`figs/`，PDF + EPS）

与前端"总览"页（/summary）一一对应，由 `src/scripts/presentation_figs.py` 生成，重跑即更新；风格与论文图一致（衬线字体、Okabe-Ito 配色）。

| 文件 | 对应总览页的卡片 | 对应篇 |
|---|---|---|
| fig1_gap_heatmap | AR gap 热力表（加粗 = 最低，虚线框 = 低于 5% 目标） | 8 |
| fig2_ar_maskable / _repair / _lagrangian / _unconstraint | AR 对比：ILP / PPO / DQN / DDQN（页面上的切换，一种约束处理一张） | 8 |
| fig3_reward_curves | 训练 reward 曲线（4 种约束处理，Lagrangian 是 ar_obj − λc，量纲不同） | 7 |
| fig4_time、fig4_time_table | 耗时对比：ILP vs 12 个变种（图 + 表） | 2、8 |
| fig_gap_bars | 论文图 1：AR gap 分组柱状图（与论文同一段代码） | 8 |
| fig_gap_viol | 论文图 2：gap–违约率（与论文同一段代码） | 4、8 |
| table_main（+ .tex） | 论文表 1：主结果表（均值 ± 标准差，加粗最好、下划线第二） | 8 |

"Dinkelbach 更新步骤"卡片是文字，见第 2 篇。
