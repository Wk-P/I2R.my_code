#!/usr/bin/env python3
"""Extension campaign: full lt/eq/gt x 6-algorithm x 5-seed at 10M steps.

Follow-up to src/scripts/run_full_5M_campaign.py after tail-trend analysis of
that campaign's training curves showed dqn/ddqn (all 3 scenarios) and
ppo/ppo_opt (lt scenario) still had a non-trivial upward success_rate trend
in the last 30% of training -- i.e. not clearly converged at 5M. ppo_mask
and ppo_lagrangian had already plateaued. Rather than extend only the
non-converged subset (which would leave those algorithms' cross-scenario
training budget inconsistent with ppo_mask/ppo_lagrangian's 5M budget), this
campaign uniformly reruns ALL 6 algorithms x 3 scenarios x 5 seeds at 10M so
every cell in the final comparison table again comes from the same budget,
matching the project's established "same steps count for every algorithm in
a cross-algorithm comparison" principle (see src/shared/training_steps_config.py
comments). The existing 5M results under results/add_states/ are left
untouched (this campaign writes fresh exp_ids), so both budgets remain
available for comparison.

Estimated wall-clock (linear x2 extrapolation from the 5M campaign's
observed per-job durations and concurrency efficiency): ~42.7 hours. This
is within the user's 48-hour window but with only ~5 hours of buffer --
ppo_opt (~197min/5M-job, ~394min/10M-job) is the dominant cost.

Uses the same WEIGHTED bin-packing concurrency control as
src/scripts/run_full_5M_campaign.py (see that file's docstring for why weighted
scheduling matters); copied and re-parameterized rather than imported so the
two campaigns stay independently auditable.

Usage:
    nohup .venv/bin/python src/scripts/run_extended_10M_campaign.py > logs/run_extended_10M_campaign_driver.log 2>&1 &
    disown
"""
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "lt_eq_gt_10M_extension"
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
STEPS = 10_000_000


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
