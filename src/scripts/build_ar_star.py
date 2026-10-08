#!/usr/bin/env python3
"""Solve the ILP for every instance of each scenario pool and write
results/<branch>/<scenario>/ilp/ar_star.json = {scenario_key: AR*}.

Needed by REWARD_MODE=ratio (src/shared/reward_config.py). Infeasible instances
(no Optimal status) are left out -- a success, the only place AR* is read,
is impossible on them.

Usage:
    .venv/bin/python src/scripts/build_ar_star.py [lt eq gt]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def _solve(sc):
    from ilp.objects import ECU, SVC
    from shared.ilp_utils import solve_ilp
    caps, reqs, cs = sc
    res = solve_ilp([ECU(f"ECU{i}", c) for i, c in enumerate(caps)],
                    [SVC(f"SVC{i}", r) for i, r in enumerate(reqs)], cs)
    return res["status"], res["avg_utilization"]


def _init(scen):
    sys.path[:0] = [str(ROOT / "scenarios" / scen), str(ROOT / "src")]


def build(scen: str) -> None:
    sys.path[:0] = [str(ROOT / "scenarios" / scen / "ppo"), str(ROOT / "scenarios" / scen), str(ROOT / "src")]
    import config as C
    from shared.reward_config import ar_star_path, scenario_key
    pool_sc = C.SCENARIOS
    with Pool(40, initializer=_init, initargs=(scen,)) as pool:
        out = pool.map(_solve, pool_sc, chunksize=8)
    ar_star = {scenario_key(*sc): float(ar) for sc, (st, ar) in zip(pool_sc, out) if st == "Optimal"}
    path = ar_star_path(scen)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ar_star))
    vals = list(ar_star.values())
    print(f"{scen}: {len(ar_star)}/{len(pool_sc)} optimal, mean AR*={sum(vals) / len(vals):.4f} -> {path}",
          flush=True)
    for m in ("config",):
        sys.modules.pop(m, None)
    del sys.path[:3]


if __name__ == "__main__":
    for scen in sys.argv[1:] or ["lt", "eq", "gt"]:
        build(scen)
