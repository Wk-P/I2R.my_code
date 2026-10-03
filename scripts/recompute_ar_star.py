#!/usr/bin/env python3
"""v4.3.1.7: replace `ar_star` in data/<version>/<scen>.yaml by the true AR optimum.

Until v4.3.1.6, ar_star came from shared.ilp_utils.solve_ilp, which maximises
the total utilisation and is not the AR optimum. This recomputes it with
solve_ilp_max_ar (Dinkelbach). Instances (ECUs, SVCs, conflict sets) are not
changed; the old value is kept as `ar_star_total_util`.

    python scripts/recompute_ar_star.py v4.3.1.4
"""
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from paper_rl.data import data_dir
from shared.ilp_utils import solve_ilp_max_ar


def one(x):
    r = solve_ilp_max_ar(x["ECUs"], x["SVCs"], x["conflict_sets"])
    assert r["status"] == "Optimal", r["status"]
    return round(float(r["avg_utilization"]), 6), r["active_ecus"]


if __name__ == "__main__":
    version = sys.argv[1]
    for scen in ("lt", "eq", "gt"):
        path = data_dir(version) / f"{scen}.yaml"
        d = yaml.safe_load(open(path))
        if d.get("ar_star_solver") == "dinkelbach_max_ar":
            print(f"{scen}: already recomputed"); continue
        with Pool(48) as pool:
            res = pool.map(one, d["instances"], chunksize=4)
        for x, (ar, act) in zip(d["instances"], res):
            x["ar_star_total_util"] = x.pop("ar_star")
            x["ilp_active_ecus_total_util"] = x.pop("ilp_active_ecus")
            x["ar_star"], x["ilp_active_ecus"] = ar, act
        d["ar_star_solver"] = "dinkelbach_max_ar"
        yaml.safe_dump(d, open(path, "w"), sort_keys=False)
        old = sum(x["ar_star_total_util"] for x in d["instances"]) / len(d["instances"])
        new = sum(x["ar_star"] for x in d["instances"]) / len(d["instances"])
        print(f"{scen}: mean AR* {old:.4f} (total-util ILP) -> {new:.4f} (max-AR ILP)")
