#!/usr/bin/env python3
"""v6.1.0: validation set for hyper-parameter tuning, 500 instances per scenario (never used for the reported results).

Same generator and checks as the v6.0.4 test set (gen_testset_v6_0_4.py): p = 0.6, K = 10, every instance solved by
solve_ilp_max_ar and kept only if Optimal. Instance seeds 900_000_000 + scenario * 10_000_000 + i, disjoint from the
pool (700_000_000 + ...) and the test set (800_000_000 + ...); in addition every instance is checked by content
against the pool, the test set and the 2000 v4.3.1.4 instances.

    .venv/bin/python src/scripts/gen_valset_v6_1_0.py            # -> data/valset_v6.1.0/<scen>.json
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
from gen_testset_v6_0_4 import POOL, SCENARIOS, key   # noqa: E402

N_VAL = 500
OUT = ROOT / "data" / "valset_v6.1.0"
TEST = ROOT / "data" / "testset_v6.0.4" / "{scen}.json"


def main():
    from paper_rl.data import load
    OUT.mkdir(parents=True, exist_ok=True)
    for k, scen in enumerate(SCENARIOS):
        t0 = time.time()
        seen = {key(e) for f in (POOL, TEST) for e in json.loads(Path(str(f).format(scen=scen)).read_text())}
        seen |= {key(e) for e in load(scen, "v4.3.1.4")["instances"]}
        base, nxt, out = 900_000_000 + k * 10_000_000, 0, []
        with Pool(48) as pool:
            while len(out) < N_VAL:
                need = N_VAL - len(out)
                for e in pool.map(_gen, [(scen, base + nxt + m) for m in range(need)], chunksize=4):
                    if key(e) not in seen and e["ar_star"] > 0:
                        seen.add(key(e))
                        out.append(e)
                nxt += need
        assert len(out) == N_VAL
        (OUT / f"{scen}.json").write_text(json.dumps(out))
        print(f"{scen}: {len(out)} instances ({nxt} seeds), mean AR* {sum(e['ar_star'] for e in out) / len(out):.4f}, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    from shared.batch_manifest import nontrain_batch      # docs/data_spec.md §6: listed by the v2 panel
    with nontrain_batch("v6.1.0-valset", "data", __file__,
                        {"zh": "v6.1.0-valset：调参用验证集，每场景 500 个实例（与 5 万训练池同一生成器，去重训练池、v6.0.4 测试集与 v4.3.1.4，均经 ILP 求出最优）",
                         "en": "v6.1.0-valset: tuning validation set, 500 instances per scenario (the 50k pool's generator, deduplicated against the pool, the v6.0.4 test set and v4.3.1.4, every instance solved to optimality by the ILP)"},
                        depends_on=[], report_dir=None, version="v6.1.0",
                        inputs=["results/sup_diag/{scen}.json", "data/testset_v6.0.4/{scen}.json", "v4.3.1.4"],
                        outputs=["data/valset_v6.1.0/{scen}.json"], logs=["logs/run_v6.1.0-valset_driver.log"]):
        main()
