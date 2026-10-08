"""eval_find_hard_scenarios.py — push best-of-N toward success_rate=100% and
isolate exactly which test scenarios never succeed, instead of re-running
every one of the 40 test scenarios at a huge N (wasteful once most of them
already succeed at a much smaller N).

Step 1: one pass at `--base-n` across all TEST_SCENARIOS, record per-scenario
        success (index order matches C.TEST_SCENARIOS).
Step 2: for scenarios that failed, retry ONLY those with `--push-n` (much
        larger) to see whether they ever succeed given enough tries, or are
        genuinely stuck (structural infeasibility for this policy) — the
        real question this session is chasing: is the residual gap to 100%
        closeable with more resampling, or a hard floor?

Usage:
    TRAIN_SEED=42 python src/scripts/eval_find_hard_scenarios.py \
        results/paper-verfication/lt/ppo_mask/0e1a36bb_bc/model_0e1a36bb_bc_v1.2.4 \
        --base-n 1024 --push-n 8192
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scenarios" / "lt" / "ppo_mask"))

import numpy as np
from sb3_contrib import MaskablePPO

import config as C
from shared.ilp_utils import load_scenario
import run_all as RA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--base-n", type=int, default=1024)
    ap.add_argument("--push-n", type=int, default=8192)
    args = ap.parse_args()

    ecus, services, _, _ = load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
    print(f"TRAIN_SEED={C.SEED}  test_scenarios={len(C.TEST_SCENARIOS)}  model={args.model_path}")

    model = MaskablePPO.load(args.model_path)

    def ppo_policy(obs, mask):
        action, _ = model.predict(obs, deterministic=False, action_masks=mask)
        return int(action)

    all_scenarios = C.TEST_SCENARIOS
    print(f"\n[1/2] base pass  N={args.base_n}  over all {len(all_scenarios)} scenarios ...")
    res = RA.run_episodes(ecus, services, ppo_policy, n_samples=args.base_n)
    success = res["success"]
    print(f"  success_rate = {float(np.mean(success)):.4f}  ({int(success.sum())}/{len(success)})")

    failed_idx = [i for i, ok in enumerate(success) if not ok]
    print(f"  failed scenario indices: {failed_idx}")
    if not failed_idx:
        print("  all scenarios already succeed at base-n -- nothing to push further.")
        return

    print(f"\n[2/2] pushing N={args.push_n} on the {len(failed_idx)} failed scenario(s) only ...")
    C.TEST_SCENARIOS = [all_scenarios[i] for i in failed_idx]
    res2 = RA.run_episodes(ecus, services, ppo_policy, n_samples=args.push_n)
    for local_i, orig_i in enumerate(failed_idx):
        ok = bool(res2["success"][local_i])
        vp = int(res2["valid_placed"][local_i])
        attempts = int(res2["attempts"][local_i])
        caps, reqs, cs = all_scenarios[orig_i]
        M_sc = len(reqs)
        util = sum(reqs) / sum(caps)
        print(f"  scenario idx={orig_i:>3}  success={ok}  valid_placed={vp}/{M_sc}  "
              f"attempts_used={attempts}/{args.push_n}  utilization={util:.3f}")

    n_recovered = int(np.sum(res2["success"]))
    overall_success = (int(success.sum()) + n_recovered) / len(all_scenarios)
    print(f"\n  recovered {n_recovered}/{len(failed_idx)} at N={args.push_n}")
    print(f"  overall success_rate now = {overall_success:.4f}")
    print(f"  still-failing after N={args.push_n}: "
          f"{[failed_idx[i] for i in range(len(failed_idx)) if not res2['success'][i]]}")


if __name__ == "__main__":
    main()
