# Experiment data spec (schema 3)

> From 2026-10-13 (v5.0.4 and later batches). Field names set by the user: **RL_algorithm, strategies, seeds**; everything else goes into **params**.
> v5.0.3 (schema 2, fields learner / mechanism) is converted to this spec by the new panel's backend; older batches are only in the classic panel (/classic/).

## 1. Fields

| Field | Values | Meaning |
|---|---|---|
| `RL_algorithm` | `PPO`, `DQN`, `DDQN` | RL algorithm. A list in a batch, one value in a run |
| `strategies` / `strategy` | `unconstraint`, `lagrangian`, `maskable`, `repair` | Constraint-handling strategy. A list `strategies` in a batch, one value `strategy` in a run |
| `seeds` / `seed` | integers | A list `seeds` in a batch, one value `seed` in a run |
| `scenarios` / `scenario` | `lt`, `eq`, `gt` | Scenario |
| `params` | object | Every other parameter. A batch's `params` apply to all its runs; a run's `params` hold only what differs from the batch |

The results directory name (e.g. `lagrange_dqn`) is storage only: `results/<results_space>/<scenario>/<dir>/<exp_id>/`, with `dir` given by RL_algorithm and strategy:

| strategy | dir |
|---|---|
| unconstraint | `ppo` / `dqn` / `ddqn` |
| lagrangian | `lagrange_<algorithm, lower case>` |
| maskable | `mask_<algorithm, lower case>` |
| repair | `repair_<algorithm, lower case>` |

## 2. Batch manifest (`logs/<batch>/manifest.json`)

The driver writes the complete plan **before anything starts**; launching a run only fills in its `exp_id`.

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

A λ_max sweep batch (like v5.0.3): `strategies` is `["lagrangian"]` only, and the sweep settings go into `params.lambda_max_sweep`:

```json
"params": {"steps": 3000000, "checkpoints": [1000000],
           "lambda_max_sweep": {"values": [1, 2, 5], "select": "lowest mean effective gap over the scenarios"},
           "lambda_max": {"PPO": 2, "DQN": 1, "DDQN": 5}}
```

Sweep runs have `role` `sweep`, `group` `lambda_max_sweep_<RL_algorithm>` and their own value in `params.lambda_max`; once a sweep is done the driver writes the chosen value into the batch's `params.lambda_max` (keyed by RL_algorithm).

Rules:

1. Planned total = `len(runs)`.
2. A `null` in a run's `params` means "not known yet" (e.g. a `lambda_max` waiting for a sweep in the same batch); the driver fills it in.
3. `depends_on` lists the groups to wait for; until all their runs have finished, a dependent run shows as waiting.
4. `driver_pid`: while the driver is alive and not every run has finished, the batch shows as running.
5. Smoke batches: `purpose: "smoke"`, the name ends in `_smoke`; data in `results/smoke/`, the report in `logs/<batch>/report/`.
6. The report directory (`paper_contents/v<version>/` for a main batch) holds, besides `report.md` / `report.en.md`, `report_data.json` (learning curves, timing) and `lam_sweep_<algorithm>.json` (λ_max sweep) for the panel's charts.

## 3. run.json (one per run, written into its results directory when training starts)

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

`batch` / `key` / `role` / `group` / `params` come from the driver through `RUN_BATCH`, `RUN_KEY`, `RUN_ROLE`, `RUN_GROUP`, `RUN_PARAMS`. Metrics after training are still written to `results.json`.

## 4. Panel

- The new panel (`/`, `app/web` + `app/backend/v2/`) shows only batches from v5.0.3 on (schema 2 / 3); older batches are only in the classic panel `/classic/`. Neither side calls the other's code.
- Pages use only this spec's names: RL_algorithm (PPO / DQN / DDQN), strategy (unconstraint / lagrangian / maskable / repair), seed, params.
- A batch's result matrix: one per scenario, rows = RL_algorithm, columns = strategy; a cell shows the relative gap and the capacity / privacy violation rates, or the run's status.
- Run status: `done` / `running` / `waiting` / `queued` / `stopped` (interrupted) / `skipped` (never ran).
- **EXIT belongs to maskable only**: the result matrix, the variant table, All runs and the run page show EXIT only when strategy = maskable (also when it is 0); never for another strategy.
- **λ belongs to lagrangian only**: the λ curve, current / final λ and checkpoint λ are shown only when strategy = lagrangian; a batch without lagrangian hides the λ column altogether.

## 5. Converting schema 2 (v5.0.3 and its smoke only)

| schema 2 | schema 3 |
|---|---|
| `learner`: ppo / dqn / ddqn | `RL_algorithm`: PPO / DQN / DDQN |
| `mechanism`: none / lagrange / mask / repair | `strategy`: unconstraint / lagrangian / maskable / repair |
| `algo` | `dir` |
| `steps` (per run) | `params.steps` |
| top-level `lam_sweep`, `lambda_max` (keyed by lower-case algorithm) | `params.lambda_max_sweep`, `params.lambda_max` (keyed by RL_algorithm) |
| group `lam_sweep_<learner>` | `lambda_max_sweep_<RL_algorithm>` |

## 6. Batches that train nothing (evaluation / diagnosis / data)

A script that launches no training run and only writes a report or data (e.g. v6.0.2 EXIT attribution, v6.0.3 checkpoint evaluation, v6.0.3.1 PPO update diagnosis, v6.0.4-testset test-set build) must also write `logs/<batch>/manifest.json`, or the panel cannot see it. Wrap `main()` in `nontrain_batch(...)` from `src/shared/batch_manifest.py`: it writes `status: running` and `driver_pid` at the start, `finished` at the end (`failed` plus `error` on an exception).

| Field | Meaning |
|---|---|
| `schema` | 3 |
| `purpose` | `eval` (evaluates existing models) / `diag` (diagnosis) / `data` (builds data) |
| `script` | script path, e.g. `src/scripts/eval_ckpt_v6.py`; with `driver_pid` the panel tells whether it still runs |
| `depends_on` | input batch names (linked on the panel), e.g. `["v6.0.0", "v6.0.1"]` |
| `inputs` / `outputs` | non-batch input / output files (`{scen}` placeholder allowed) |
| `report_dir` | report directory, e.g. `paper_contents/v6.0.3`; `null` without a report |
| `logs` | optional; script log paths (under `logs/`). Default: `logs/<batch>/*.log` |
| `status` / `created_at` / `finished_at` / `minutes` | run status and times |
| `commit` / `script_uncommitted` | HEAD when it ran, and whether the script had uncommitted changes |
| `runs` | empty list |

Panel: the batch list shows these by default (tagged Evaluation / Diagnosis / Data; the progress columns give the script and the input batches); the detail page has only the Report / log and Config tabs, logs shown verbatim.
