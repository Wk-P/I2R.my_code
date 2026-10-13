# Experiment data spec (manifest v2 / run.json)

> From v5.0.3 on, every training batch writes its data by this spec; the panel only displays it and no longer infers anything. Older batches are mapped to the same shape by a converter in the backend.

## 1. Terms

| Term | Meaning |
|---|---|
| batch | one driver-script run, directory `logs/<batch>/`, e.g. `v5.0.3`, `v5.0.3_smoke` |
| run | one training (one model), identified by `exp_id`, results in `results/<data space>/<scenario>/<algo>/<exp_id>/` |
| purpose | per batch: `main` formal experiment / `smoke` smoke test / `test` other tests |
| role | per run: `main` formal result / `sweep` parameter sweep / `reference` cross-batch reference (not trained in this batch) |
| group | related runs of one role, e.g. `lam_sweep`; other runs can wait for a group with `depends_on` |
| params | the hyper-parameters that tell runs of the same learner × mechanism apart, e.g. `{"lambda_max": 2, "steps": 3000000}` |

## 2. `run.json` (one per run, written by `paper_rl/train.py` when training starts)

Location: `results/<data space>/<scenario>/<algo>/<exp_id>/run.json`.

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

- `batch` / `key` / `role` / `group` come from the driver through the environment variables `RUN_BATCH`, `RUN_KEY`, `RUN_ROLE`, `RUN_GROUP`; a manually started run without them gets `null` / `null` / `"main"` / `null`.
- `params` holds only the **parameters that differ from the defaults and tell runs apart** (e.g. `lambda_max`). All other hyper-parameters stay in `results.json` as before.
- Metrics after training are still written to `results.json`, unchanged.

## 3. manifest v2 (one per batch, `logs/<batch>/manifest.json`)

The driver writes the complete plan **before anything starts**; launching a run only fills in its `exp_id`.

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

Rules:

1. **Planned total = `len(runs)`**; `references` do not count.
2. A `null` value in `params` means "decided by a group this run depends on, not known yet" (e.g. `lambda_max` before the sweep ends); the driver fills it in once known.
3. `depends_on` lists the groups to wait for; until every run of such a group has finished, a dependent run shows as "waiting for <group>".
4. `exp_ids` (the old nested form `{seed: {scenario: {key: exp_id}}}`) is derived from `runs` and kept only for older scripts such as `diag_regret.py`.
5. Smoke batches: `purpose: "smoke"`, the batch name ends in `_smoke`, data go to a separate data space (`RESULTS_SPACE`).

## 4. Panel display

- The batch detail table has fixed columns: **scenario / model (learner) / constraint handling (mechanism) / params / seeds / metrics**. The params column shows `params` as badges: `λ_max=2`, `steps=3M` (`steps` only when it differs from the batch's default).
- Role tabs: **main / sweep / reference**. Reference rows show their source; their metrics come from the source run's `results.json`.
- The batch list hides batches whose `purpose` is `smoke` / `test` by default; tick "show test batches" to see them.
- Run status: `done` / `running` / `waiting` (for a dependency) / `queued` / `stopped` (interrupted) / `skipped` (never ran).

## 5. Converting old batches (no schema)

The backend's `_legacy_runs` turns an old manifest into the `runs` of §3:

- scenario / seed / algo are recognised in the nested `exp_ids` by the old rules; `learner` and `mechanism` come from the algo name;
- the old `keys` label (reward mode, network, `· 1600`, ...) goes into `params.variant`; it is dropped when it equals the network name (the network has its own column);
- `role` is always `main`; a batch name containing `smoke` gives `purpose` `smoke`;
- planned total: scenarios × seeds × keys (× modes) when there are `keys`, otherwise scenarios × algos × seeds (× modes).
