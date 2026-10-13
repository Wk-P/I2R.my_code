# 实验数据规范（schema 3）

> 2026-10-13 起（v5.0.4 及以后的批次）。字段名由用户规定：**RL_algorithm、strategies、seeds**，其余一律放进 **params**。
> v5.0.3（schema 2，字段名 learner / mechanism）由新看板后端换算成本规范显示；更早的批次只在旧看板（/classic/）。

## 1. 字段

| 字段 | 取值 | 说明 |
|---|---|---|
| `RL_algorithm` | `PPO`、`DQN`、`DDQN` | 强化学习算法。批次里是列表，run 里是单个值 |
| `strategies` / `strategy` | `unconstraint`、`lagrangian`、`maskable`、`repair` | 约束处理策略。批次里是列表 `strategies`，run 里是单个值 `strategy` |
| `seeds` / `seed` | 整数 | 批次里是列表 `seeds`，run 里是单个值 `seed` |
| `scenarios` / `scenario` | `lt`、`eq`、`gt` | 场景 |
| `params` | 对象 | 其余所有参数。批次的 `params` 是全部 run 共用的；run 的 `params` 只放这个 run 与批次不同的参数 |

结果目录名（如 `lagrange_dqn`）只是存储路径：`results/<results_space>/<scenario>/<dir>/<exp_id>/`，`dir` 由 RL_algorithm 与 strategy 决定：

| strategy | dir |
|---|---|
| unconstraint | `ppo` / `dqn` / `ddqn` |
| lagrangian | `lagrange_<小写算法>` |
| maskable | `mask_<小写算法>` |
| repair | `repair_<小写算法>` |

## 2. 批次 manifest（`logs/<批次名>/manifest.json`）

驱动脚本**开跑前**写出完整计划，启动一个 run 时只填入它的 `exp_id`。

```json
{
  "schema": 3,
  "batch": "v5.0.4",
  "version": "5.0.4",
  "purpose": "main",
  "commit": "a3b2109",
  "results_space": "unified",
  "driver_pid": 12345,
  "created_at": "2026-10-14T09:00:00",
  "title": {"zh": "...", "en": "..."},

  "RL_algorithm": ["PPO", "DQN", "DDQN"],
  "strategies": ["unconstraint", "lagrangian", "maskable", "repair"],
  "seeds": [1],
  "scenarios": ["lt", "eq", "gt"],
  "params": {
    "steps": 20000000,
    "checkpoints": [1000000, 3000000, 5000000, 7500000, 10000000, 15000000],
    "net": "graph",
    "reward": "ar_pen",
    "gamma": 1.0,
    "full_episode": true,
    "train_pool": "results/sup_diag/{scen}.json",
    "data": "v4.3.1.4",
    "lambda_max": {"PPO": 2, "DQN": 1, "DDQN": 5},
    "lambda_max_from": "v5.0.3"
  },

  "runs": [
    {"key": "dqn_lagrangian", "role": "main", "group": null, "depends_on": [],
     "RL_algorithm": "DQN", "strategy": "lagrangian", "seed": 1, "scenario": "lt",
     "params": {"lambda_max": 1}, "dir": "lagrange_dqn", "exp_id": "3cf63e77"}
  ],
  "references": []
}
```

λ_max 扫描批次（v5.0.3 这类）：`strategies` 只有 `["lagrangian"]`，扫描设置放在 `params.lambda_max_sweep`：

```json
"params": {"steps": 3000000, "checkpoints": [1000000],
           "lambda_max_sweep": {"values": [1, 2, 5], "select": "三场景平均有效 gap 最小"},
           "lambda_max": {"PPO": 2, "DQN": 1, "DDQN": 5}}
```

扫描 run 的 `role` 为 `sweep`，`group` 为 `lambda_max_sweep_<RL_algorithm>`，`params.lambda_max` 为该 run 的取值；扫描结束后驱动把选定值写进批次 `params.lambda_max`（键为 RL_algorithm）。

规则：

1. 计划总数 = `len(runs)`。
2. run 的 `params` 中值为 `null` 表示尚未确定（例如等待同批次扫描的 `lambda_max`），确定后驱动回填。
3. `depends_on` 列出要等待的组名；组内 run 全部结束之前，依赖它的 run 显示为「等待」。
4. `driver_pid`：驱动还活着、run 尚未全部完成时，批次显示为「运行中」。
5. smoke 批次：`purpose: "smoke"`，批次名以 `_smoke` 结尾；数据写 `results/smoke/`，报告写 `logs/<批次名>/report/`。
6. 报告目录（正式批次 `paper_contents/v<版本>/`）除 `report.md` / `report.en.md` 外，还有给看板画图的 `report_data.json`（学习曲线、计时）和 `lam_sweep_<算法>.json`（λ_max 扫描）。

## 3. run.json（每个 run 一份，训练开始时写入结果目录）

```json
{
  "schema": 3,
  "exp_id": "3cf63e77",
  "batch": "v5.0.4",
  "key": "dqn_lagrangian",
  "role": "main",
  "group": null,
  "version": "5.0.4",
  "commit": "a3b2109",
  "RL_algorithm": "DQN",
  "strategy": "lagrangian",
  "seed": 1,
  "scenario": "lt",
  "params": {"steps": 20000000, "net": "graph", "lambda_max": 1},
  "dir": "lagrange_dqn",
  "started_at": "2026-10-14T09:00:00",
  "argv": ["--scen", "lt", "..."]
}
```

`batch` / `key` / `role` / `group` / `params` 由驱动通过环境变量 `RUN_BATCH`、`RUN_KEY`、`RUN_ROLE`、`RUN_GROUP`、`RUN_PARAMS` 传入。训练结束后的指标仍写在 `results.json`。

## 4. 看板

- 新看板（`/`，`app/web` + `app/backend/v2/`）只显示 v5.0.3 起的批次（schema 2 / 3）；旧批次只在旧看板 `/classic/`。两边代码互不调用。
- 页面上只用本规范的名字：RL_algorithm（PPO / DQN / DDQN）、strategy（unconstraint / lagrangian / maskable / repair）、seed、params。
- 批次详情的结果矩阵：每个场景一张，行 = RL_algorithm，列 = strategy；格内为相对 gap 与容量 / 隐私违约率，或运行状态。
- run 状态：`done` 已完成 / `running` 运行中 / `waiting` 等待 / `queued` 排队中 / `stopped` 已中断 / `skipped` 未运行。

## 5. schema 2 的换算（只有 v5.0.3 及其 smoke）

| schema 2 | schema 3 |
|---|---|
| `learner`: ppo / dqn / ddqn | `RL_algorithm`: PPO / DQN / DDQN |
| `mechanism`: none / lagrange / mask / repair | `strategy`: unconstraint / lagrangian / maskable / repair |
| `algo` | `dir` |
| `steps`（run 级） | `params.steps` |
| 顶层 `lam_sweep`、`lambda_max`（键为小写算法） | `params.lambda_max_sweep`、`params.lambda_max`（键为 RL_algorithm） |
| 组 `lam_sweep_<learner>` | `lambda_max_sweep_<RL_algorithm>` |
