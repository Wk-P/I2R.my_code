"""Greedy baseline (no learning), evaluated exactly like the RL models.

Same episode as the RL agents (services in descending demand, one per step) and
the same feasibility test as Maskable: an ECU is a candidate only if it has the
capacity and hosts no conflicting service, so the greedy policy never violates a
constraint; if no ECU is feasible the episode is a dead end (failure).
Rule among the feasible ECUs:
  1. prefer ECUs that already host a service (open as few ECUs as possible);
  2. then the largest demand / capacity (fill the ECU as much as possible).
This is Repair's best-fit rule plus "prefer an already used ECU".

    .venv/bin/python -m paper_rl.greedy            # all scenarios, seeds 1-3
"""
import numpy as np

from paper_rl.env import PlacementEnv
from paper_rl.train import split_instances


def greedy_action(env: PlacementEnv) -> int:
    cand = np.flatnonzero(env._feasible(env.t))
    if len(cand) == 0:
        return 0                     # unreachable: the env ends the episode before a dead end
    used = np.array([bool(env.hosted[j]) for j in cand])
    return int(cand[np.argmax(used * 10.0 + env.req[env.t] / env.cap[cand])])


def evaluate(scen: str, seed: int) -> dict:
    _, _, test = split_instances(scen, seed)
    env = PlacementEnv(test, "mask", "legacy")
    succ, ratio = [], []
    for k in range(len(test)):
        env.use_instance(k)
        env.reset()
        done = False
        while not done:
            _, _, done, _, info = env.step(greedy_action(env))
        ok = info["valid_placed"] == env.M
        succ.append(ok)
        ratio.append(info["ar"] / test[k]["ar_star"] if ok else 0.0)
    succ, ratio = np.array(succ), np.array(ratio)
    return {"success": float(succ.mean()), "ratio_success": float(ratio[succ].mean()) if succ.any() else None,
            "ratio_all": float(ratio.mean()), "dead_end": float(1 - succ.mean())}


if __name__ == "__main__":
    for s in ["lt", "eq", "gt"]:
        for sd in [1, 2, 3]:
            r = evaluate(s, sd)
            print(f"{s} seed{sd}: 成功率 {r['success']:.3f} | 相对最优 AR（成功回合）{100 * r['ratio_success']:.1f}% "
                  f"| 失败计 0 {100 * r['ratio_all']:.1f}%", flush=True)
