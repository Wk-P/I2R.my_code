#!/usr/bin/env python3
"""Full lt/eq/gt x 6-algorithm x 5-seed campaign at a uniform 5M steps,
launched with WEIGHTED bin-packing concurrency control instead of a flat
"N processes at a time" throttle.

Why weighted: each algorithm's config.py sets its own TORCH_NUM_THREADS
(ppo/ppo_mask/ppo_lagrangian/ppo_opt = 10, dqn/ddqn = 5) -- a naive
"5 processes at once" throttle mixes cheap and expensive jobs arbitrarily
and either oversubscribes (5x PPO-family = 50 threads, fine; but the
gt_full_5M_rerun incident launched 25-30 processes at once = 200+ threads
of demand on a 56-core box, a 4x oversubscription that collapsed
throughput to ~1-2% of single-process speed) or leaves cores idle. This
scheduler instead tracks live thread-demand and only starts a new job when
it fits under CAPACITY, so utilization stays in the 90-95% band the whole
run without ever blowing past it.

Usage:
    nohup .venv/bin/python src/scripts/run_full_5M_campaign.py > logs/run_full_5M_campaign_driver.log 2>&1 &
    disown
"""
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "lt_eq_gt_5M_full"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_CORES = 56
CAPACITY = int(TOTAL_CORES * 0.93)  # ~90-95% target, leaves ~7% (~4 cores) headroom

# TORCH_NUM_THREADS per algorithm (must match each scenario's config.py --
# verified identical across lt/eq/gt in the pre-flight check).
THREAD_WEIGHT = {
    "ppo": 10, "ppo_mask": 10, "ppo_lagrangian": 10, "ppo_opt": 10,
    "dqn": 5, "ddqn": 5,
}

SCENARIOS = ["lt", "eq", "gt"]
ALGOS = ["ppo_mask", "ppo_lagrangian", "ppo", "ppo_opt", "dqn", "ddqn"]
SEEDS = [1, 2, 3, 4, 5]
STEPS = 5_000_000


def new_exp_id() -> str:
    out = subprocess.run(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-c",
         "from shared.paths import new_exp_id; print(new_exp_id())"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def build_queue():
    """seed-major ordering: fills a full (scenario, seed) 6-algo bundle
    (weight 50) before moving to the next seed, so bundles tend to finish
    close together rather than being scattered across the whole campaign."""
    queue = []
    for seed in SEEDS:
        for scenario in SCENARIOS:
            for algo in ALGOS:
                queue.append((scenario, algo, seed, THREAD_WEIGHT[algo]))
    return queue


def launch(scenario: str, algo: str, seed: int) -> tuple[subprocess.Popen, int]:
    exp_id = new_exp_id()
    log_path = LOG_DIR / f"{scenario}_{algo}_{STEPS}_seed{seed}_{exp_id}.log"
    log_f = open(log_path, "w")
    env = {
        "PYTHONUNBUFFERED": "1",
        "TRAIN_SEED": str(seed),
        "EXP_ID": exp_id,
    }
    import os
    full_env = {**os.environ, **env}
    proc = subprocess.Popen(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-u",
         f"scenarios/{scenario}/{algo}/run_all.py", "--total-timesteps", str(STEPS)],
        cwd=PROJECT_ROOT, env=full_env, stdout=log_f, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    print(f"[{time.strftime('%H:%M:%S')}] launched {scenario}/{algo} seed={seed} "
          f"exp_id={exp_id} pid={proc.pid} weight={THREAD_WEIGHT[algo]}", flush=True)
    return proc, log_f


def main():
    queue = build_queue()
    print(f"=== campaign start {time.strftime('%Y-%m-%d %H:%M:%S')} — "
          f"{len(queue)} jobs, capacity={CAPACITY}/{TOTAL_CORES} cores ===", flush=True)

    active = []  # list of (proc, log_f, weight, label)
    completed = 0
    failed = []

    while queue or active:
        # fill available capacity greedily from the front of the queue
        current_load = sum(w for _, _, w, _ in active)
        i = 0
        while i < len(queue) and current_load + queue[i][3] <= CAPACITY:
            scenario, algo, seed, weight = queue.pop(i)
            proc, log_f = launch(scenario, algo, seed)
            active.append((proc, log_f, weight, f"{scenario}/{algo}/seed{seed}"))
            current_load += weight
            # i stays 0: popping index 0 shifts the next item into position 0,
            # so re-checking queue[0] against the updated current_load is
            # exactly "keep taking from the front while it still fits".

        # poll active jobs
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

    print(f"=== campaign done {time.strftime('%Y-%m-%d %H:%M:%S')} — "
          f"{completed} completed, {len(failed)} failed ===", flush=True)
    if failed:
        print("FAILED:", failed, flush=True)


if __name__ == "__main__":
    main()
