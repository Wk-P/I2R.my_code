"""Scenario data for the unified implementation (v4.3.1.3).

Problem: place M services on N ECUs.
  - capacity constraint: the containers required by the services placed on an
    ECU must not exceed that ECU's container capacity;
  - privacy-conflict constraint: two services of the same conflict set must
    not share an ECU; services in no common set may share one.

Every scenario uses the same distributions; only N and M differ:
  - ECU capacity  ~ uniform{50, 55, ..., 195} containers (i.i.d.)
  - SVC demand    ~ uniform{10, 15, ..., 95}  containers (i.i.d.)
  - K = 10 conflict sets; each service joins each set independently with
    probability q, sets with < 2 members are dropped. Two services then
    conflict with probability p = 1 - (1 - q^2)^K, independent of M.
Instances that the ILP cannot solve are redrawn. The ILP optimum (AR*) is
stored with every instance.

    python -m paper_rl.data --p 0.6 --version v4.3.1.4 [--n 2000]   # writes data/<version>/<scen>.yaml
(v4.3.1.3 used p = 0.3, v4.3.1.4 uses p = 0.6; paper_rl/config.DATA_VERSION selects the set.)
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scenarios" / "lt"))   # ilp.objects (ECU, SVC)

SCENARIOS = {"lt": (10, 15), "eq": (10, 10), "gt": (15, 10)}   # name -> (N, M)
K_SETS = 10
CAP_CHOICES = list(range(50, 200, 5))
REQ_CHOICES = list(range(10, 100, 5))
DATA_ROOT = ROOT / "data"


def data_dir(version: str | None = None) -> Path:
    from paper_rl import config as C
    return DATA_ROOT / (version or C.DATA_VERSION)


def q_for(p: float, k: int = K_SETS) -> float:
    return math.sqrt(1 - (1 - p) ** (1 / k))


def draw(n: int, m: int, p: float, rng: random.Random):
    q = q_for(p)
    caps = [rng.choice(CAP_CHOICES) for _ in range(n)]
    reqs = [rng.choice(REQ_CHOICES) for _ in range(m)]
    sets = []
    for _ in range(K_SETS):
        s = [i for i in range(m) if rng.random() < q]
        if len(s) >= 2:
            sets.append(s)
    return caps, reqs, sets


def solve(caps, reqs, sets) -> dict:
    from ilp.objects import ECU, SVC
    from shared.ilp_utils import solve_ilp
    return solve_ilp([ECU(f"ECU{i}", c) for i, c in enumerate(caps)],
                     [SVC(f"SVC{i}", r) for i, r in enumerate(reqs)], sets)


def _gen_one(args):
    n, m, p, seed = args
    rng = random.Random(seed)
    tries = 0
    while True:
        tries += 1
        caps, reqs, sets = draw(n, m, p, rng)
        res = solve(caps, reqs, sets)
        if res["status"] == "Optimal":
            return {"ECUs": caps, "SVCs": reqs, "conflict_sets": sets,
                    "ar_star": round(float(res["avg_utilization"]), 6),
                    "ilp_active_ecus": res["active_ecus"], "draws": tries}


def generate(scen: str, p: float, n_inst: int, seed: int = 42, workers: int = 32) -> dict:
    from multiprocessing import Pool
    n, m = SCENARIOS[scen]
    base = {"lt": 1, "eq": 2, "gt": 3}[scen] * 10_000_000 + seed * 100_000
    with Pool(workers) as pool:
        inst = pool.map(_gen_one, [(n, m, p, base + i) for i in range(n_inst)], chunksize=8)
    return {"scenario": scen, "N": n, "M": m, "conflict_pair_prob": p, "k_sets": K_SETS,
            "cap_choices": [CAP_CHOICES[0], CAP_CHOICES[-1], 5],
            "req_choices": [REQ_CHOICES[0], REQ_CHOICES[-1], 5],
            "seed": seed, "instances": inst}


def load(scen: str, version: str | None = None) -> dict:
    import yaml
    return yaml.safe_load(open(data_dir(version) / f"{scen}.yaml"))


if __name__ == "__main__":
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--version", required=True)
    a = ap.parse_args()
    out = data_dir(a.version)
    out.mkdir(parents=True, exist_ok=True)
    for scen in SCENARIOS:
        d = generate(scen, a.p, a.n)
        yaml.safe_dump(d, open(out / f"{scen}.yaml", "w"), sort_keys=False)
        ars = [x["ar_star"] for x in d["instances"]]
        print(f"{scen}: {len(ars)} instances, mean AR* {sum(ars)/len(ars):.4f}")
