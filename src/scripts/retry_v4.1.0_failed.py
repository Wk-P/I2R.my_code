#!/usr/bin/env python3
"""One-off retry driver for the 12 jobs that failed in the first v4.1.0
2M-step campaign run (all 12 failures were the same ILP-cache race condition
in shared/ilp_utils.py, fixed in this branch's follow-up commit -- not a
reward-fix issue). Relaunches exactly these 12 (scenario, algo, seed) combos,
updates the campaign manifest with their new exp_ids, and regenerates the
report on completion.

Usage:
    nohup .venv/bin/python scripts/retry_v4.1.0_failed.py > scripts/logs/retry_v4.1.0_failed_driver.log 2>&1 &
    disown
"""
import json
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "scripts" / "logs" / "v4.1.0_2M_campaign"
MANIFEST_PATH = LOG_DIR / "manifest.json"

THREAD_WEIGHT = {
    "ppo": 10, "ppo_mask": 10, "ppo_lagrangian": 10, "ppo_opt": 10,
    "dqn": 5, "ddqn": 5,
}
STEPS = 2_000_000
CAPACITY = int(56 * 0.93)

FAILED = [
    ("lt", "ppo_lagrangian", 1), ("lt", "ppo_opt", 1), ("lt", "dqn", 1),
    ("lt", "ddqn", 1), ("lt", "ppo_mask", 1), ("eq", "ppo_mask", 1),
    ("eq", "ppo_lagrangian", 1), ("eq", "ppo", 1), ("gt", "ppo_mask", 1),
    ("lt", "ppo_mask", 2), ("lt", "ppo", 2), ("lt", "ppo_lagrangian", 3),
]


def new_exp_id() -> str:
    out = subprocess.run(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-c",
         "from shared.paths import new_exp_id; print(new_exp_id())"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def launch(scenario, algo, seed, manifest):
    exp_id = new_exp_id()
    log_path = LOG_DIR / f"RETRY_{scenario}_{algo}_{STEPS}_seed{seed}_{exp_id}.log"
    log_f = open(log_path, "w")
    import os
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TRAIN_SEED": str(seed), "EXP_ID": exp_id}
    proc = subprocess.Popen(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-u",
         f"scenarios/{scenario}/{algo}/run_all.py", "--total-timesteps", str(STEPS)],
        cwd=PROJECT_ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    manifest.setdefault(scenario, {}).setdefault(algo, {})[str(seed)] = exp_id
    print(f"[{time.strftime('%H:%M:%S')}] retry-launched {scenario}/{algo} seed={seed} "
          f"exp_id={exp_id} pid={proc.pid}", flush=True)
    return proc, log_f, f"{scenario}/{algo}/seed{seed}", THREAD_WEIGHT[algo]


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    exp_ids = manifest["exp_ids"]

    queue = list(FAILED)
    active = []
    completed = 0
    failed = []

    print(f"=== retry start {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(queue)} jobs ===", flush=True)

    while queue or active:
        current_load = sum(w for _, _, w, _ in active)
        i = 0
        while i < len(queue):
            weight = THREAD_WEIGHT[queue[i][1]]
            if current_load + weight > CAPACITY:
                i += 1
                continue
            scenario, algo, seed = queue.pop(i)
            proc, log_f, label, w = launch(scenario, algo, seed, exp_ids)
            active.append((proc, log_f, w, label))
            current_load += w

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
                      f"— {completed} done, {len(queue)} queued", flush=True)
                if ret != 0:
                    failed.append(label)
        active = still_active

        # persist manifest progress after every poll cycle
        tmp = MANIFEST_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(MANIFEST_PATH)

        if queue or active:
            time.sleep(10)

    print(f"=== retry done {time.strftime('%Y-%m-%d %H:%M:%S')} — {completed} completed, {len(failed)} failed ===", flush=True)
    if failed:
        print("STILL FAILED:", failed, flush=True)

    print("=== auto-generating report ===", flush=True)
    r = subprocess.run(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"),
         "scripts/generate_v4.1.0_report.py", "--manifest", str(MANIFEST_PATH)],
        cwd=PROJECT_ROOT,
    )
    print("=== report generated ===" if r.returncode == 0 else "!!! report generation FAILED", flush=True)


if __name__ == "__main__":
    main()
