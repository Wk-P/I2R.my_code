# 实验数据规范（manifest v2 / run.json）

> 自 v5.0.3 起，所有训练批次都按本规范写数据；看板只负责显示，不再推断。旧批次由后端的转换函数映射到同样的结构。

## 1. 名词

| 名词 | 含义 |
|---|---|
| 批次（batch） | 一次驱动脚本的运行，目录 `logs/<批次名>/`，例如 `v5.0.3`、`v5.0.3_smoke` |
| run | 一次训练（一个模型），由 `exp_id` 标识，结果目录 `results/<数据空间>/<场景>/<algo>/<exp_id>/` |
| 用途（purpose） | 批次级：`main` 正式实验 / `smoke` 冒烟测试 / `test` 其他测试 |
| 角色（role） | run 级：`main` 正式结果 / `sweep` 参数扫描 / `reference` 跨批次参照（不在本批次训练） |
| 组（group） | 同一角色下一组相关的 run，例如 `lam_sweep`；其他 run 可以用 `depends_on` 等待某个组结束 |
| 参数（params） | 区分同一 learner × mechanism 下不同 run 的超参，例如 `{"lambda_max": 2, "steps": 3000000}` |

## 2. `run.json`（每个 run 一份，训练开始时由 `paper_rl/train.py` 写入）

位置：`results/<数据空间>/<场景>/<algo>/<exp_id>/run.json`。

```json
{
  "schema": 1,
  "exp_id": "3cf63e77",
  "batch": "v5.0.3",
  "key": "lam_sweep_1",
  "role": "sweep",
  "group": "lam_sweep",
  "version": "5.0.3",
  "commit": "410a38c",
  "scenario": "lt",
  "seed": 1,
  "algo": "lagrange_dqn",
  "learner": "dqn",
  "mechanism": "lagrange",
  "net": "graph",
  "steps": 3000000,
  "params": {"lambda_max": 1.0},
  "started_at": "2026-10-13T14:26:29",
  "argv": ["--scen", "lt", "..."]
}
```

- `batch` / `key` / `role` / `group` 由驱动脚本通过环境变量 `RUN_BATCH`、`RUN_KEY`、`RUN_ROLE`、`RUN_GROUP` 传入；手动启动的 run 没有这些变量时，分别为 `null` / `null` / `"main"` / `null`。
- `params` 只放**偏离默认值、且用来区分 run 的参数**（例如 `lambda_max`）。其余超参照旧写在 `results.json`。
- 训练结束后的指标仍写在 `results.json`，格式不变。

## 3. manifest v2（每个批次一份，`logs/<批次名>/manifest.json`）

驱动脚本**开跑前**就写出完整的计划清单；启动一个 run 时只填入它的 `exp_id`。

```json
{
  "schema": 2,
  "batch": "v5.0.3",
  "version": "5.0.3",
  "purpose": "main",
  "commit": "410a38c",
  "results_space": "unified",
  "title": {"zh": "...", "en": "..."},
  "created_at": "2026-10-13T14:26:29",
  "runs": [
    {"key": "lam_sweep_1", "role": "sweep", "group": "lam_sweep", "depends_on": [],
     "scenario": "lt", "seed": 1, "algo": "lagrange_dqn", "learner": "dqn", "mechanism": "lagrange", "net": "graph",
     "steps": 3000000, "params": {"lambda_max": 1.0}, "exp_id": "3cf63e77"},
    {"key": "ddqn_lagrange", "role": "main", "group": "lagrange", "depends_on": ["lam_sweep"],
     "scenario": "lt", "seed": 1, "algo": "lagrange_ddqn", "learner": "ddqn", "mechanism": "lagrange", "net": "graph",
     "steps": 20000000, "params": {"lambda_max": null}, "exp_id": null}
  ],
  "references": [
    {"label": {"zh": "无约束 DQN（v5.0.2）", "en": "Unconstrained DQN (v5.0.2)"},
     "results_space": "unified",
     "scenario": "lt", "seed": 1, "algo": "dqn", "learner": "dqn", "mechanism": "none", "net": "graph",
     "steps": 20000000, "params": {}, "exp_id": "2d3a6181"}
  ],
  "exp_ids": {"1": {"lt": {"lam_sweep_1": "3cf63e77"}}}
}
```

规则：

1. **计划总数 = `len(runs)`**，`references` 不计入。
2. `params` 中值为 `null` 表示"由依赖的组决定、尚未确定"（例如扫描结束前的 `lambda_max`）；确定后驱动脚本回填。
3. `depends_on` 列出要等待的组名；组内 run 全部结束之前，依赖它的 run 显示为"等待 <组名>"。
4. `exp_ids`（旧的嵌套格式 `{种子: {场景: {key: exp_id}}}`）由 `runs` 派生，只为 `diag_regret.py` 等旧脚本保留。
5. smoke 批次：`purpose: "smoke"`，批次名以 `_smoke` 结尾，数据写到单独的数据空间（`RESULTS_SPACE`）。

## 4. 看板显示

- 批次详情表的列固定为：**场景 / 模型（learner）/ 约束处理（mechanism）/ 参数 / 种子 / 指标**。参数列把 `params` 显示成徽章：`λ_max=2`、`步数=3M`（`steps` 与批次默认步数不同时才显示）。
- 角色标签页：**正式 / 扫描 / 参照**。参照行标出来源批次，指标从来源 run 的 `results.json` 读。
- 批次列表默认隐藏 `purpose` 为 `smoke` / `test` 的批次，勾选"显示测试批次"后才显示。
- run 状态：`done` 已完成 / `running` 运行中 / `waiting` 等待依赖 / `queued` 排队中 / `stopped` 已中断 / `skipped` 未运行。

## 5. 旧批次（schema 缺省）的转换

后端 `_legacy_runs` 把旧 manifest 转成第 3 节的 `runs`：

- 场景 / 种子 / algo 按旧规则从嵌套的 `exp_ids` 中识别；`learner`、`mechanism` 由 algo 名拆出；
- 旧 `keys` 的标签（奖励模式、网络、`· 1600` 等）放进 `params.variant`；与网络名相同时丢弃（网络已有自己的列）；
- `role` 一律为 `main`；批次名含 `smoke` 时 `purpose` 为 `smoke`；
- 计划总数：有 `keys` 时为 场景 × 种子 × keys（× modes），否则为 场景 × algos × 种子（× modes）。
