#!/usr/bin/env python3
"""v6.0.4: fixed 5000-instance test set per scenario (the same for every seed and every run).

Same generator as the 50k training pool (results/sup_diag/<scen>.json, diag_supervised.gen): p = 0.6, K = 10,
every instance solved by solve_ilp_max_ar and kept only if the status is Optimal (AR* and the ILP allocation are
stored). Instance seeds 800_000_000 + scenario * 10_000_000 + i, disjoint from the pool (700_000_000 + ...) and
from v4.3.1.4 (scenario * 10_000_000 + 42 * 100_000 + i); in addition every instance is checked by content against
the pool and the 2000 v4.3.1.4 instances, and duplicates inside the set are dropped (redrawn with the next seed).

    .venv/bin/python src/scripts/gen_testset_v6_0_4.py            # -> data/testset_v6.0.4/<scen>.json
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))
from diag_supervised import _gen                      # noqa: E402  (the pool's own generator)

N_TEST = 5000
SCENARIOS = ("lt", "eq", "gt")                         # the order of diag_supervised (seed offsets)
OUT = ROOT / "data" / "testset_v6.0.4"
POOL = ROOT / "results" / "sup_diag" / "{scen}.json"


def key(x):
    return json.dumps([x["ECUs"], x["SVCs"], sorted(map(sorted, x["conflict_sets"]))])


def main():
    from paper_rl.data import load
    OUT.mkdir(parents=True, exist_ok=True)
    for k, scen in enumerate(SCENARIOS):
        t0 = time.time()
        seen = {key(x) for x in json.loads(Path(str(POOL).format(scen=scen)).read_text())}
        seen |= {key(x) for x in load(scen, "v4.3.1.4")["instances"]}
        base, nxt, out = 800_000_000 + k * 10_000_000, 0, []
        with Pool(48) as pool:
            while len(out) < N_TEST:
                need = N_TEST - len(out)
                for x in pool.map(_gen, [(scen, base + nxt + i) for i in range(need)], chunksize=16):
                    if key(x) not in seen and x["ar_star"] > 0:
                        seen.add(key(x))
                        out.append(x)
                nxt += need
        assert len(out) == N_TEST
        (OUT / f"{scen}.json").write_text(json.dumps(out))
        print(f"{scen}: {len(out)} instances ({nxt} seeds), mean AR* {sum(x['ar_star'] for x in out) / len(out):.4f}, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    from shared.batch_manifest import nontrain_batch      # docs/data_spec.md §6: listed by the v2 panel
    with nontrain_batch("v6.0.4-testset", "data", __file__,
                        {"zh": "v6.0.4-testset：固定测试集，每场景 5000 个实例（与 5 万训练池同一生成器，去重训练池与 v4.3.1.4，均经 ILP 求出最优）",
                         "en": "v6.0.4-testset: fixed test set, 5000 instances per scenario (the 50k pool's generator, deduplicated against the pool and v4.3.1.4, every instance solved to optimality by the ILP)"},
                        depends_on=[], report_dir=None, version="v6.0.4", inputs=["results/sup_diag/{scen}.json", "v4.3.1.4"], outputs=["data/testset_v6.0.4/{scen}.json"], logs=["logs/run_v6.0.4-testset_driver.log"]):
        main()
