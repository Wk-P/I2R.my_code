"""test_new_lt_scenarios.py — generate fresh lt-feasible scenarios (not in
config_ecu_lt_svc.yaml's 200), solve each with ILP, and evaluate the current
best-trained lt/ppo_mask model against them.

Purpose: sanity-check whether the model's measured success_rate/AR on the
existing 40 held-out test scenarios generalises to entirely new scenarios
drawn from the same generator, or whether it was overfit to that fixed pool.

Usage:
    python src/scripts/test_new_lt_scenarios.py [--groups 4] [--size 100] \
        [--model results/paper-verfication/lt/ppo_mask/e251a0fb_bc/model_e251a0fb_bc_v2.1.0] \
        [--best-of-n 8]

Output: test/<YYYYMMDD_HHMMSS>/compare/
    group{i}_scenarios.json   -- generated scenarios (ECUs/SVCs/conflict_sets)
    group{i}_results.json     -- per-scenario ILP + RL comparison
    group{i}_summary.csv
    summary_all.json          -- aggregated stats across all 4 groups
"""
import argparse
import csv
import datetime
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scenarios" / "lt"))  # for `ilp` package
sys.path.insert(0, str(ROOT / "scenarios" / "lt" / "ppo_mask"))

import numpy as np
from sb3_contrib import MaskablePPO

from ilp.objects import ECU, SVC
from shared.ilp_utils import solve_ilp
from ppo_mask.env import P4Env

N_ECUS = 10
N_SVCS = 15
K_SETS = 10
DEFAULT_MODEL = ROOT / "results/paper-verfication/lt/ppo_mask/e251a0fb_bc/model_e251a0fb_bc_v2.1.0"


def _is_conflict_feasible(conflict_sets, M, N):
    adj = [set() for _ in range(M)]
    for cs in conflict_sets:
        valid = [i for i in cs if i < M]
        for i in valid:
            for j in valid:
                if i != j:
                    adj[i].add(j)
    colors = [-1] * M
    for node in range(M):
        used = {colors[nb] for nb in adj[node] if colors[nb] >= 0}
        for c in range(N):
            if c not in used:
                colors[node] = c
                break
        if colors[node] == -1:
            return False
    return True


def generate_feasible_scenario(rng: random.Random):
    """Same generation rule as scenarios/lt/ilp/config/generate_config.py,
    but driven by an explicit Random instance (not the global module) so
    groups don't share/perturb each other's stream."""
    while True:
        ecu_capacity = rng.sample(range(50, 200, 5), N_ECUS)
        svc_requirement = rng.sample(range(10, 100, 5), N_SVCS)
        conflict_sets = [
            sorted(rng.sample(range(N_SVCS), rng.randint(2, N_ECUS)))
            for _ in range(K_SETS)
        ]
        if not _is_conflict_feasible(conflict_sets, N_SVCS, N_ECUS):
            continue
        ecus = [ECU(f"ECU{i}", c) for i, c in enumerate(ecu_capacity)]
        svcs = [SVC(f"SVC{i}", r) for i, r in enumerate(svc_requirement)]
        res = solve_ilp(ecus, svcs, conflict_sets)
        if res["status"] == "Optimal":
            return ecu_capacity, svc_requirement, conflict_sets, res


def run_rl(ecus, services, conflict_sets, model, n_samples: int):
    """Best-of-n_samples eval of one scenario, mirrors run_all.run_episodes."""
    best = None
    used_attempts = 0
    for attempt in range(max(1, n_samples)):
        env = P4Env(ecus, services, scenarios=[(
            [e.capacity for e in ecus],
            [s.requirement for s in services],
            conflict_sets,
        )])
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            mask = env.action_masks()
            if not np.any(mask):
                break
            action, _ = model.predict(obs, deterministic=False, action_masks=mask)
            obs, _, done, _, info = env.step(int(action))
        M_sc = len(services)
        placed = info.get("services_placed", 0)
        valid_placed = int(info.get("valid_placed", placed))
        ar = info.get("ar", 0.0)
        success = bool(valid_placed == M_sc)
        used_attempts = attempt + 1
        candidate = (success, valid_placed, ar, placed)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
        if success:
            break
    success, valid_placed, ar, placed = best
    return dict(
        success=bool(success),
        valid_placed=int(valid_placed),
        ar=float(ar),
        placed=int(placed),
        attempts=int(used_attempts),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--best-of-n", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=9000,
                     help="Base RNG seed; kept far from the SEED=42 used to "
                          "generate the original 200-scenario pool, and each "
                          "group gets base_seed + group_idx*100000 so groups "
                          "don't share a stream.")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ROOT / "test" / ts / "compare"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {outdir}")

    print(f"Loading model: {args.model}")
    model = MaskablePPO.load(args.model)

    all_group_stats = []
    for g in range(args.groups):
        seed = args.base_seed + g * 100_000
        rng = random.Random(seed)
        print(f"\n=== Group {g+1}/{args.groups}  (seed={seed}, size={args.size}) ===")

        scenarios_out = []
        rows = []
        ilp_ars, rl_ars, rl_success = [], [], []
        attempts_all = []

        for i in range(args.size):
            ecu_cap, svc_req, conflict_sets, ilp_res = generate_feasible_scenario(rng)
            ecus = [ECU(f"ECU{k}", c) for k, c in enumerate(ecu_cap)]
            svcs = [SVC(f"SVC{k}", r) for k, r in enumerate(svc_req)]

            rl_res = run_rl(ecus, svcs, conflict_sets, model, args.best_of_n)

            scenarios_out.append({
                "idx": i,
                "ECUs": ecu_cap,
                "SVCs": svc_req,
                "conflict_sets": conflict_sets,
            })
            row = {
                "idx": i,
                "demand_cap_ratio": round(sum(svc_req) / sum(ecu_cap), 4),
                "ilp_ar": round(ilp_res["avg_utilization"], 6),
                "ilp_active_ecus": ilp_res["active_ecus"],
                "rl_success": rl_res["success"],
                "rl_ar": round(rl_res["ar"], 6),
                "rl_valid_placed": rl_res["valid_placed"],
                "rl_attempts": rl_res["attempts"],
            }
            rows.append(row)
            ilp_ars.append(ilp_res["avg_utilization"])
            attempts_all.append(rl_res["attempts"])
            rl_success.append(rl_res["success"])
            if rl_res["success"]:
                rl_ars.append(rl_res["ar"])

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{args.size}] running success_rate so far = {np.mean(rl_success):.3f}")

        with open(outdir / f"group{g+1}_scenarios.json", "w") as f:
            json.dump(scenarios_out, f, indent=2)

        stats = {
            "group": g + 1,
            "seed": seed,
            "size": args.size,
            "ilp_ar_mean": round(float(np.mean(ilp_ars)), 6),
            "ilp_ar_std": round(float(np.std(ilp_ars)), 6),
            "rl_success_rate": round(float(np.mean(rl_success)), 6),
            "rl_ar_mean": round(float(np.mean(rl_ars)), 6) if rl_ars else None,
            "rl_ar_std": round(float(np.std(rl_ars)), 6) if rl_ars else None,
            "rl_attempts_mean": round(float(np.mean(attempts_all)), 3),
            "best_of_n": args.best_of_n,
        }
        all_group_stats.append(stats)

        with open(outdir / f"group{g+1}_results.json", "w") as f:
            json.dump({"stats": stats, "scenarios": rows}, f, indent=2)

        with open(outdir / f"group{g+1}_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"  Group {g+1} done: ILP_AR={stats['ilp_ar_mean']:.4f}  "
              f"RL success_rate={stats['rl_success_rate']:.4f}  "
              f"RL AR={stats['rl_ar_mean']}")

    overall = {
        "model": args.model,
        "best_of_n": args.best_of_n,
        "groups": all_group_stats,
        "overall_rl_success_rate": round(float(np.mean([s["rl_success_rate"] for s in all_group_stats])), 6),
        "overall_ilp_ar_mean": round(float(np.mean([s["ilp_ar_mean"] for s in all_group_stats])), 6),
        "overall_rl_ar_mean": round(float(np.mean([s["rl_ar_mean"] for s in all_group_stats if s["rl_ar_mean"] is not None])), 6),
    }
    with open(outdir / "summary_all.json", "w") as f:
        json.dump(overall, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All groups done. Overall RL success_rate = {overall['overall_rl_success_rate']:.4f}")
    print(f"Overall ILP AR = {overall['overall_ilp_ar_mean']:.4f}  Overall RL AR = {overall['overall_rl_ar_mean']:.4f}")
    print(f"Output: {outdir}")


if __name__ == "__main__":
    main()
