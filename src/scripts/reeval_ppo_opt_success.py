#!/usr/bin/env python3
"""Re-evaluate already-trained ppo_opt models with the corrected success
formula (v2.8.1: success == full completion only, not also zero repairs --
see scenarios/{lt,eq,gt}/ppo_opt/run_all.py's comment on this change), WITHOUT
retraining. Loads each saved model_*.zip under
results/<branch>/<scenario>/ppo_opt/<exp_id>/, reruns run_episodes() (now
using the corrected formula since it imports the live run_all.py module),
and rewrites that run's results.json + summary.csv success_rate fields in
place. Everything else in those files (ar_mean, training section, etc.) is
left untouched.

Usage:
    .venv/bin/python src/scripts/reeval_ppo_opt_success.py <scenario> [exp_id ...]

    <scenario>: lt, eq, or gt
    [exp_id ...]: optional — restrict to these run dirs; default is all
                  run dirs under that scenario's ppo_opt/ that have a
                  results.json.
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_run_all_module(scenario: str):
    """Import scenarios/<scenario>/ppo_opt/run_all.py as a module the same
    way it imports itself when run directly (its own sys.path.insert calls
    at the top handle `import config as C`, `from ppo_opt.env import
    P6Env`, etc. — this just needs cwd-independent sys.path setup mirroring
    what run_all.py's HERE-relative inserts already do)."""
    algo_dir = PROJECT_ROOT / "scenarios" / scenario / "ppo_opt"
    sys.path.insert(0, str(algo_dir))
    sys.path.insert(0, str(algo_dir.parent))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(f"run_all_{scenario}", algo_dir / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reeval_one(mod, run_dir: Path, device: str = "cpu"):
    results_path = run_dir / "results.json"
    summary_path = run_dir / "summary.csv"
    if not results_path.is_file():
        print(f"  skip {run_dir.name}: no results.json")
        return None

    # SB3's model.save() is called here without a .zip suffix on the target
    # path (see run_all.py's `model_path = run_dir / f"model_{exp_id}_v..."`),
    # and on this SB3 version that leaves the saved file's name WITHOUT a
    # .zip extension too (confirmed via `file`: it's zip data internally,
    # just an extensionless filename) -- glob for "model_*" broadly, not
    # "model_*.zip", and exclude the .png/.json/.csv siblings explicitly.
    model_candidates = [
        p for p in run_dir.glob("model_*")
        if p.suffix not in (".png", ".json", ".csv")
    ]
    if not model_candidates:
        print(f"  skip {run_dir.name}: no saved model file")
        return None
    model_path = model_candidates[0]

    model = mod.PPO.load(str(model_path), device=device)

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    ppo_res = mod.run_episodes(None, None, ppo_policy)
    new_success_rate = float(__import__("numpy").mean(ppo_res["success"]))

    payload = json.loads(results_path.read_text())
    method_key = next((k for k, v in payload.items() if isinstance(v, dict) and "success_rate" in v), None)
    old_success_rate = payload[method_key]["success_rate"] if method_key else None
    if method_key:
        payload[method_key]["success_rate"] = round(new_success_rate, 6)
        payload[method_key]["success_rate_formula"] = "v2.8.1: valid_placed==M only (no zero-repair requirement)"
    results_path.write_text(json.dumps(payload, indent=2))

    if summary_path.is_file():
        rows = list(csv.reader(summary_path.open()))
        header = rows[0]
        if "success_rate" in header:
            sr_idx = header.index("success_rate")
            for row in rows[1:]:
                if row[0] != "ILP (Optimal)":
                    row[sr_idx] = f"{new_success_rate:.4f}"
        with summary_path.open("w", newline="") as f:
            csv.writer(f).writerows(rows)

    print(f"  {run_dir.name}: success_rate {old_success_rate} -> {new_success_rate:.4f}")
    return new_success_rate


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    scenario = sys.argv[1]
    exp_ids = sys.argv[2:]

    mod = load_run_all_module(scenario)

    algo_results_dir = PROJECT_ROOT / "results" / "add_states" / scenario / "ppo_opt"
    if exp_ids:
        run_dirs = [algo_results_dir / e for e in exp_ids]
    else:
        run_dirs = sorted(d for d in algo_results_dir.iterdir() if d.is_dir())

    print(f"=== re-evaluating {len(run_dirs)} ppo_opt run(s) in scenario={scenario} ===")
    for run_dir in run_dirs:
        reeval_one(mod, run_dir)


if __name__ == "__main__":
    main()
