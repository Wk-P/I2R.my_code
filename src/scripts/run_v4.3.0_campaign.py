#!/usr/bin/env python3
"""v4.3.0 campaign: DQN family on lt/eq/gt x 3 seeds x 5M steps.

v4.3.0 changes (see paper_contents/v4.3.0/README.md):
  - dqn / ddqn now use the same sparse reward as PPO (no per-step violation
    penalty; gt no longer terminates on capacity violation);
  - four new variants: mask_dqn / mask_ddqn (Mask-PPO env + hard action
    mask) and repair_dqn / repair_ddqn (Repair-PPO env + best-fit repair).

The PPO family is not retrained. Results land in
results/final_paper_experiments/<scenario>/<algo>/<exp_id>/; the
(scenario, algo, seed) -> exp_id manifest is written to
logs/v4.3.0_campaign/manifest.json, and a summary report to
paper_contents/v4.3.0/campaign_report.md when every job has finished.

Usage:
    nohup .venv/bin/python src/scripts/run_v4.3.0_campaign.py > logs/run_v4.3.0_campaign_driver.log 2>&1 &
    disown
"""
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
PY = str(PROJECT_ROOT / ".venv" / "bin" / "python")
LOG_DIR = PROJECT_ROOT / "logs" / "v4.3.0_campaign"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = LOG_DIR / "manifest.json"
REPORT_PATH = PROJECT_ROOT / "paper_contents" / "v4.3.0" / "campaign_report.md"
RESULTS_ROOT = PROJECT_ROOT / "results" / "final_paper_experiments"

TOTAL_CORES = 56
CAPACITY = int(TOTAL_CORES * 0.93)
THREAD_WEIGHT = 5  # TORCH_NUM_THREADS in every DQN-family config.py

SCENARIOS = ["lt", "eq", "gt"]
ALGOS = ["dqn", "ddqn", "mask_dqn", "mask_ddqn", "repair_dqn", "repair_ddqn"]
LABELS = {"dqn": "DQN", "ddqn": "DDQN", "mask_dqn": "Mask-DQN", "mask_ddqn": "Mask-DDQN",
          "repair_dqn": "Repair-DQN", "repair_ddqn": "Repair-DDQN"}
SEEDS = [1, 2, 3]
STEPS = 5_000_000


def new_exp_id() -> str:
    out = subprocess.run([PY, "-c", "from shared.paths import new_exp_id; print(new_exp_id())"],
                         cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def write_manifest(exp_ids: dict) -> None:
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "version": "4.3.0", "branch": "final_paper_experiments", "steps": STEPS,
        "scenarios": SCENARIOS, "algos": ALGOS, "seeds": SEEDS, "exp_ids": exp_ids,
    }, indent=2))
    tmp.replace(MANIFEST_PATH)


def launch(scenario, algo, seed, exp_ids):
    exp_id = new_exp_id()
    log_f = open(LOG_DIR / f"{scenario}_{algo}_seed{seed}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TRAIN_SEED": str(seed), "EXP_ID": exp_id}
    proc = subprocess.Popen(
        [PY, "-u", f"scenarios/{scenario}/{algo}/run_all.py", "--total-timesteps", str(STEPS)],
        cwd=PROJECT_ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    exp_ids.setdefault(scenario, {}).setdefault(algo, {})[str(seed)] = exp_id
    write_manifest(exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched {scenario}/{algo} seed={seed} "
          f"exp_id={exp_id} pid={proc.pid}", flush=True)
    return proc, log_f


def read_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[1] if len(rows) > 1 else None


def write_report(exp_ids: dict) -> None:
    def ms(xs):
        if not xs:
            return "—"
        sd = np.std(xs, ddof=1) if len(xs) > 1 else 0.0
        return f"{np.mean(xs):.4f}±{sd:.4f}"

    lines = ["# v4.3.0 campaign 报告", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"- {STEPS:,} 步 × 种子 {SEEDS}；每个种子在自己的测试集上单次确定性评估（训练脚本内置评估）。",
             "- 违规率 = 测试 episode 中出现 ≥1 次违规的比例；Repair-* 两列为修复触发率（执行的放置不违规）。",
             "- manifest：`logs/v4.3.0_campaign/manifest.json`", ""]
    for scen in SCENARIOS:
        lines += [f"## {scen.upper()}", "",
                  "| 算法 | success_rate | AR | 容量违规率 | 冲突违规率 | 种子数 | ILP AR |",
                  "|---|---|---|---|---|---|---|"]
        for algo in ALGOS:
            rows, ilp = [], []
            for seed, eid in exp_ids.get(scen, {}).get(algo, {}).items():
                d = RESULTS_ROOT / scen / algo / eid
                r = read_row(d / "summary.csv")
                if r:
                    rows.append(r)
                    ilp.append(json.loads((d / "results.json").read_text())["ilp"]["ar"])
            f = lambda k: [float(r[k]) for r in rows]
            lines.append(f"| {LABELS[algo]} | {ms(f('success_rate'))} | {ms(f('ar_mean'))} | "
                         f"{ms(f('cap_viol_rate'))} | {ms(f('conflict_viol_rate'))} | {len(rows)} | "
                         f"{ms(ilp)} |")
        lines.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"report -> {REPORT_PATH}", flush=True)


def main():
    queue = [(s, a, seed) for seed in SEEDS for s in SCENARIOS for a in ALGOS]
    exp_ids: dict = {}
    print(f"=== v4.3.0 campaign start {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(queue)} jobs, "
          f"capacity={CAPACITY}/{TOTAL_CORES} cores, steps={STEPS:,} ===", flush=True)
    active, failed, done = [], [], 0
    while queue or active:
        while queue and (len(active) + 1) * THREAD_WEIGHT <= CAPACITY:
            s, a, seed = queue.pop(0)
            proc, log_f = launch(s, a, seed, exp_ids)
            active.append((proc, log_f, f"{s}/{a}/seed{seed}"))
        still = []
        for proc, log_f, label in active:
            ret = proc.poll()
            if ret is None:
                still.append((proc, log_f, label))
                continue
            log_f.close()
            done += 1
            print(f"[{time.strftime('%H:%M:%S')}] finished {label} ({'OK' if ret == 0 else f'EXIT={ret}'}) "
                  f"— {done} done, {len(queue)} queued, {len(still)} running", flush=True)
            if ret != 0:
                failed.append(label)
        active = still
        if queue or active:
            time.sleep(15)
    print(f"=== v4.3.0 campaign done {time.strftime('%Y-%m-%d %H:%M:%S')} — "
          f"{done} completed, {len(failed)} failed ===", flush=True)
    if failed:
        print("FAILED:", failed, flush=True)
    write_report(exp_ids)


if __name__ == "__main__":
    main()
