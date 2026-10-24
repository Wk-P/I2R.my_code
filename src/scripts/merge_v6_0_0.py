"""v6.0.0-merge: one panel batch that collects, without retraining, the v6 runs that follow the final definition.

  maskable / repair / lagrangian  v6.0.0 runs              (role main)
  unconstraint                    v6.0.1 runs              (role main; v6.0.0's Penalty runs are the old definition)
  maskable PPO, n_epochs=4        v6.0.0.2 runs            (role reference; PPO update-size fix)
Every run is checked against the final reward / parameter definition before the manifest is written; any mismatch
aborts. Report: paper_contents/v6.0.0-merge/report.md (+ .en.md) from report_v6_merged.py.
    .venv/bin/python src/scripts/merge_v6_0_0.py
"""
import glob
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shared.batch_manifest import _commit, _now, write  # noqa: E402

BATCH = "v6.0.0-merge"
REPORT = ROOT / "paper_contents" / BATCH
STRAT = {"mask": "maskable", "repair": "repair", "lagrange": "lagrangian", "none": "unconstraint"}
ALGO = {"ppo": "PPO", "dqn": "DQN", "ddqn": "DDQN"}
SOURCES = [("v6.0.0", ("mask", "repair", "lagrange"), "main"),
           ("v6.0.1", ("none",), "main"),
           ("v6.0.0.2", ("mask",), "reference")]
# final definition (memory: reward-formula / full-episode-violation); value or callable(results) -> bool
COMMON = {"reward_mode": "ar_pen", "reward_norm": "none", "full_episode": True, "gamma": 1.0, "net": "graph",
          "obs": "raw", "order": "desc", "action_mode": "ecu", "bc": None, "train_pool": "results/sup_diag/{scen}.json",
          "data": {"conflict_pair_prob": 0.6, "k_sets": 10}, "test_count": 400}
CODE = {"v6.0.0": "4445cf9", "v6.0.1": "14cce61", "v6.0.0.2": "c8f195b"}   # 14cce61 adds only the unconstraint branch


def check(r, version, role):
    bad = [f"{k}={r.get(k)!r}" for k, v in COMMON.items() if r.get(k) != v]
    if r["training"]["total_steps"] != 20_000_000:
        bad.append(f"steps={r['training']['total_steps']}")
    if r["exit_action"] != (r["mechanism"] == "mask"):
        bad.append(f"exit_action={r['exit_action']}")
    if r["version"] != version:
        bad.append(f"version={r['version']}")
    if r["learner"] == "ppo":
        want = (4, None) if version == "v6.0.0.2" else (10, None)
        got = (r.get("n_epochs", 10) or 10, r.get("target_kl"))
        if got != want:
            bad.append(f"ppo n_epochs/target_kl={got}")
    return bad


def main():
    runs, problems = [], []
    for version, mechs, role in SOURCES:
        for run in json.loads((ROOT / f"logs/{version}/manifest.json").read_text())["runs"]:
            d = Path(glob.glob(str(ROOT / f"results/unified/*/*/{run['exp_id']}"))[0])
            r = json.loads((d / "results.json").read_text())
            if r["mechanism"] not in mechs:
                continue
            for b in check(r, version, role):
                problems.append(f"{version} {run['exp_id']} {r['mechanism']}/{r['learner']}/{r['scenario']}/s{r['seed']}: {b}")
            runs.append({"key": f"{r['learner']}_{STRAT[r['mechanism']]}", "role": role,
                         "group": f"{version}{'' if role == 'main' else ' n_epochs=4'}", "depends_on": [],
                         "RL_algorithm": ALGO[r["learner"]], "strategy": STRAT[r["mechanism"]], "seed": r["seed"],
                         "scenario": r["scenario"], "params": {"n_epochs": 4} if version == "v6.0.0.2" else {},
                         "dir": d.parent.name, "exp_id": run["exp_id"], "attempts": [run["exp_id"]], "status": "ok",
                         "source_batch": version, "source_commit": CODE[version]})
    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} definition mismatches -- manifest not written")
    print(f"{len(runs)} runs checked: all follow the final definition")

    REPORT.mkdir(parents=True, exist_ok=True)
    for f in ("merged.md", "merged.en.md"):
        src = ROOT / "logs/v6.0.1/report" / f
        if src.is_file():
            shutil.move(src, REPORT / f.replace("merged", "report"))
    head, dirty = _commit(__file__)
    write(BATCH, schema=3, batch=BATCH, version=BATCH, purpose="main", label="Results Merge",
          description={"zh": "合并v6.0.0及补充数据。", "en": "v6.0.0 merged with its supplementary data."}, commit=head, script="src/scripts/merge_v6_0_0.py", script_uncommitted=dirty,
          results_space="unified", driver_pid=None, created_at=_now(),
          title={"zh": "v6.0.0-merge：合并 v6.0.0 及补充数据（Mask / Repair / Lagrangian 取 v6.0.0，Unconstraint 取 v6.0.1；"
                       "v6.0.0.2 的 Mask PPO n_epochs=4 作参照），不重新训练",
                 "en": "v6.0.0-merge: v6.0.0 merged with its follow-ups (Mask / Repair / Lagrangian from v6.0.0, Unconstraint "
                       "from v6.0.1; v6.0.0.2 Mask PPO n_epochs=4 as reference), no retraining"},
          RL_algorithm=["PPO", "DQN", "DDQN"], strategies=["unconstraint", "lagrangian", "maskable", "repair"],
          seeds=[1, 2, 3], scenarios=["eq", "gt", "lt"],
          params={"steps": 20_000_000, "checkpoints": [5_000_000, 10_000_000, 15_000_000], "net": "graph",
                  "reward": "ar_pen", "gamma": 1.0, "full_episode": True, "exit_action": "maskable only",
                  "unconstraint": "v6.0.1: no penalty, terminal AR_obj, no constraint info in the state",
                  "train_pool": "results/sup_diag/{scen}.json", "data": "v4.3.1.4",
                  "lagrangian": {"lambda_max": "none", "lambda_update": "adam", "lambda_lr": 0.035, "lambda_init": 0.001,
                                 "lambda_target": 0.0, "lambda_cost": "behavior"},
                  "seeds_note": "lagrangian / unconstraint: seeds 2, 3 only"},
          depends_on=["v6.0.0", "v6.0.1", "v6.0.0.2", "v6.0.4-test-50k"], report_dir=f"paper_contents/{BATCH}",
          runs=runs, references=[], status="finished", finished_at=_now())
    print("->", ROOT / "logs" / BATCH / "manifest.json")


if __name__ == "__main__":
    main()
