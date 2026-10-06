#!/usr/bin/env python3
"""v4.4.1: ILP feasibility and ILP AR against the measured conflict density.

The v4.4.1 data keeps only ILP-feasible instances (infeasible draws are redrawn
together with p), so the feasibility curve needs its own sample: per scenario,
--n fresh draws with p ~ U[0, 1), each checked for ILP feasibility only (no
objective). Both tables are binned by conflict_ratio = measured fraction of
service pairs sharing a conflict set (p only drives the generator).

    .venv/bin/python scripts/density_v4.4.1.py [--n 10000]
    -> paper_contents/v4.4.1/density_report.md
"""
import argparse
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pulp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paper_rl.data import SCENARIOS, conflict_ratio, draw, load  # noqa: E402

VERSION = "v4.4.1"
REPORT = ROOT / "paper_contents" / VERSION / "density_report.md"
BINS = [i / 10 for i in range(11)]


def feasible(caps, reqs, sets) -> bool:
    n, m = len(caps), len(reqs)
    prob = pulp.LpProblem("feas", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (range(m), range(n)), cat="Binary")
    prob += 0
    for i in range(m):
        prob += pulp.lpSum(x[i][j] for j in range(n)) == 1
    for j in range(n):
        prob += pulp.lpSum(x[i][j] * reqs[i] for i in range(m)) <= caps[j]
        for s in sets:
            prob += pulp.lpSum(x[i][j] for i in s) <= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.LpStatus[prob.status] == "Optimal"


def one(args):
    scen, seed = args
    n, m = SCENARIOS[scen]
    rng = random.Random(seed)
    p = rng.random()
    caps, reqs, sets = draw(n, m, p, rng)
    return conflict_ratio(m, sets), feasible(caps, reqs, sets)


def bin_of(r):
    return min(int(r * 10), 9)


def label(b):
    return f"[{BINS[b]:.1f}, {BINS[b + 1]:.1f}" + ("]" if b == 9 else ")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    lines = [f"# {VERSION} 冲突密度与 ILP 可行率 / ILP AR", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 冲突密度 ρ = 实测冲突服务对比例（`conflict_ratio`），按 ρ 分 10 档；生成参数 p 只用于驱动生成器。",
             f"- **ILP 可行率**：每个场景另抽 {a.n} 个实例（p ~ U[0,1)，与正式数据同一生成器、不同随机种子），只判断 ILP 是否有可行解（不求最优）。",
             "- **正式数据**：`data/v4.4.1/`，每场景 2000 个实例，全部 ILP 可行（不可行的已连同 p 重抽）；ILP AR 为 Dinkelbach 最优值。",
             "- 每格写 均值（实例数）。", ""]
    for scen in SCENARIOS:
        with Pool(a.workers) as pool:
            res = pool.map(one, [(scen, 900_000_000 + i) for i in range(a.n)], chunksize=8)
        data = load(scen, VERSION)
        inst = data["instances"]
        ps = np.array([x["p"] for x in inst])
        lines += [f"## {scen.upper()}（N = {data['N']}，M = {data['M']}）", "",
                  f"正式数据：p 均值 {ps.mean():.3f}，中位数 {np.median(ps):.3f}；共抽 {sum(x['draws'] for x in inst)} 次得到 {len(inst)} 个可行实例。", "",
                  "| ρ 档 | ILP 可行率（抽样） | 正式数据实例数 | ILP AR | ILP 启用 ECU 数 | p 均值 |",
                  "|---|---|---|---|---|---|"]
        for b in range(10):
            fs = [f for r, f in res if bin_of(r) == b]
            xs = [x for x in inst if bin_of(x["conflict_ratio"]) == b]
            feas = f"{100 * np.mean(fs):.1f}%（{len(fs)}）" if fs else "—（0）"
            if xs:
                ar = f"{np.mean([x['ar_star'] for x in xs]):.4f}"
                act = f"{np.mean([x['ilp_active_ecus'] for x in xs]):.2f}"
                pm = f"{np.mean([x['p'] for x in xs]):.3f}"
            else:
                ar = act = pm = "—"
            lines.append(f"| {label(b)} | {feas} | {len(xs)} | {ar} | {act} | {pm} |")
        lines.append("")
        print(f"{scen} done", flush=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    main()
