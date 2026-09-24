#!/usr/bin/env python3
"""v4.1.0.1 targeted ablation: lt/eq/gt x ppo_lagrangian ONLY x 3-seed at 5M steps.

Purpose: the v4.1.0 full campaign (5M steps, matching v4.0.0's budget) showed
ppo_lagrangian regressing on lt specifically (success_rate 0.81->0.52,
conflict_viol 0.18->0.47) after wiring in its full docstring reward formula,
which included a dense per-step match_gain (utilisation) term. v4.1.0.1
dropped match_gain from ppo_lagrangian's env.py (penalty terms only remain)
-- see paper_contents/v4.1.0_changelog.md. This script reruns ONLY
ppo_lagrangian (the other 5 algorithms are unchanged since v4.1.0's 5M run,
no need to rerun them) to check whether dropping match_gain recovers lt's
performance back toward the v4.0.0 reference (success_rate ~0.81, conflict_viol
~0.18).

Same step budget (5M) and seed count (3) as the v4.1.0 5M campaign, so this
result is directly comparable to that run's ppo_lagrangian rows -- isolating
match_gain as the only variable that changed.

Results land in results/final_paper_experiments/<scenario>/<algo>/<exp_id>/
(branch-namespaced per shared/paths.py), completely separate from
results/add_states/. A manifest of every (scenario, algo, seed) -> exp_id is
written to scripts/logs/v4.1.0.1_ppo_lagrangian_5M/manifest.json so the report
generator (scripts/generate_v4.1.0_report.py) knows exactly which exp_ids
belong to this run. On completion this script automatically invokes the
report generator (per project requirement: every full experiment on
final_paper_experiments must auto-produce a full report).

Usage:
    nohup .venv/bin/python scripts/run_v4.1.0.1_ppo_lagrangian_5M.py > scripts/logs/run_v4.1.0.1_ppo_lagrangian_5M_driver.log 2>&1 &
    disown
"""
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "scripts" / "logs" / "v4.1.0.1_ppo_lagrangian_5M"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = LOG_DIR / "manifest.json"

TOTAL_CORES = 56
CAPACITY = int(TOTAL_CORES * 0.93)

THREAD_WEIGHT = {
    "ppo": 10, "ppo_mask": 10, "ppo_lagrangian": 10, "ppo_opt": 10,
    "dqn": 5, "ddqn": 5,
}

SCENARIOS = ["lt", "eq", "gt"]
ALGOS = ["ppo_lagrangian"]
SEEDS = [1, 2, 3]
STEPS = 5_000_000


def new_exp_id() -> str:
    out = subprocess.run(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-c",
         "from shared.paths import new_exp_id; print(new_exp_id())"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def build_queue():
    queue = []
    for seed in SEEDS:
        for scenario in SCENARIOS:
            for algo in ALGOS:
                queue.append((scenario, algo, seed, THREAD_WEIGHT[algo]))
    return queue


def launch(scenario: str, algo: str, seed: int, manifest: dict) -> tuple[subprocess.Popen, "TextIOWrapper", str]:
    exp_id = new_exp_id()
    log_path = LOG_DIR / f"{scenario}_{algo}_{STEPS}_seed{seed}_{exp_id}.log"
    log_f = open(log_path, "w")
    env = {"PYTHONUNBUFFERED": "1", "TRAIN_SEED": str(seed), "EXP_ID": exp_id}
    import os
    full_env = {**os.environ, **env}
    proc = subprocess.Popen(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-u",
         f"scenarios/{scenario}/{algo}/run_all.py", "--total-timesteps", str(STEPS)],
        cwd=PROJECT_ROOT, env=full_env, stdout=log_f, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    manifest.setdefault(scenario, {}).setdefault(algo, {})[str(seed)] = exp_id
    _write_manifest(manifest)
    print(f"[{time.strftime('%H:%M:%S')}] launched {scenario}/{algo} seed={seed} "
          f"exp_id={exp_id} pid={proc.pid} weight={THREAD_WEIGHT[algo]}", flush=True)
    return proc, log_f, exp_id


def _write_manifest(manifest: dict) -> None:
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({
            "version": "4.1.0",
            "branch": "final_paper_experiments",
            "steps": STEPS,
            "scenarios": SCENARIOS,
            "algos": ALGOS,
            "seeds": SEEDS,
            "exp_ids": manifest,
        }, f, indent=2)
    tmp.replace(MANIFEST_PATH)


def main():
    queue = build_queue()
    manifest: dict = {}
    print(f"=== v4.1.0 campaign start {time.strftime('%Y-%m-%d %H:%M:%S')} — "
          f"{len(queue)} jobs, capacity={CAPACITY}/{TOTAL_CORES} cores, steps={STEPS} ===", flush=True)

    active = []
    completed = 0
    failed = []

    while queue or active:
        current_load = sum(w for _, _, w, _ in active)
        i = 0
        while i < len(queue) and current_load + queue[i][3] <= CAPACITY:
            scenario, algo, seed, weight = queue.pop(i)
            proc, log_f, exp_id = launch(scenario, algo, seed, manifest)
            active.append((proc, log_f, weight, f"{scenario}/{algo}/seed{seed}"))
            current_load += weight

        still_active = []
        for proc, log_f, weight, label in active:
            ret = proc.poll()
            if ret is None:
                still_active.append((proc, log_f, weight, label))
            else:
                log_f.close()
                completed += 1
                status = "OK" if ret == 0 else f"EXIT={ret}"
                print(f"[{time.strftime('%H:%M:%S')}] finished {label} ({status}) "
                      f"— {completed} done, {len(queue)} queued, {len(still_active)} running", flush=True)
                if ret != 0:
                    failed.append(label)
        active = still_active

        if queue or active:
            time.sleep(15)

    print(f"=== v4.1.0 campaign done {time.strftime('%Y-%m-%d %H:%M:%S')} — "
          f"{completed} completed, {len(failed)} failed ===", flush=True)
    if failed:
        print("FAILED:", failed, flush=True)

    print("=== auto-generating report ===", flush=True)
    report_proc = subprocess.run(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"),
         "scripts/generate_v4.1.0_report.py", "--manifest", str(MANIFEST_PATH)],
        cwd=PROJECT_ROOT,
    )
    if report_proc.returncode != 0:
        print("!!! report generation FAILED, see output above", flush=True)
    else:
        print("=== report generated ===", flush=True)


if __name__ == "__main__":
    main()
