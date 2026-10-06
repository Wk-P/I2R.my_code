# v4.4.1 随机冲突密度数据

在 v4.3.8（统一奖励、不提前结束、Maskable 带 EXIT、原观测、需求降序、只选 ECU）基础上，**只换数据**。v4.4.0 的（服务, ECU）动作不再使用。

- **冲突密度不再固定**：每个实例单独抽 p ~ U[0,1)，再按原生成器（K = 10 个冲突集，每个服务以概率 q 加入每个集合，p = 1 − (1 − q²)^K）生成。每个实例记录自己的 `p` 和实测冲突服务对比例 `conflict_ratio`（ρ）；分档统计一律用 ρ。
- **只保留 ILP 可行实例**：ILP 无可行解时，连同 p 一起重抽。LT 在 ρ ≥ 0.8 时多数不可行，所以 LT 的 p 偏低（均值 0.420），EQ / GT 几乎都可行（均值约 0.50）。可行率随 ρ 的曲线见 `density_report.md`（另抽 1 万个实例，只判断可行性）。
- **ILP AR**：生成时直接用 Dinkelbach（`solve_ilp_max_ar`）求真正的 AR 最优；`paper_rl/data.py` 原来调用的旧 `solve_ilp` 已改掉。用 p = 0.6 重生成前 40 个实例核对，与 `data/v4.3.1.4`（已重算）完全一致。
- 每个场景 2000 个实例，`data/v4.4.1/{lt,eq,gt}.yaml`；划分（80% 训练）与种子同前。数据版本由环境变量 `DATA_VERSION` 选择，默认仍为 v4.3.1.4。
- **评估**：每个实例都 ILP 可行，所以 EXIT 率 = 策略把一个本可完成的实例做成死局的比例。AR、ILP AR 在未 EXIT 的同一批实例上平均；**绝对 gap = ILP AR − AR，相对 gap = (ILP AR − AR) / ILP AR，两者都报**。总体和按 ρ 分 10 档各报一次。

运行：`scripts/run_v4.4.1.py --pilot`（Maskable PPO × 3 场景 × 种子 1 × 1M 步），报告 `pilot_report.md`；`scripts/density_v4.4.1.py` 生成 `density_report.md`。
