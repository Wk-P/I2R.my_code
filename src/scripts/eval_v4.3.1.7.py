#!/usr/bin/env python3
"""v4.3.1.7: re-evaluate the 108 v4.3.1.6 models against the true ILP optimum, and time ILP vs RL.

No retraining. For every (scenario, model, seed) of scripts/logs/v4.3.1.6_campaign/manifest.json:
  - load the saved model, run one deterministic episode per test instance of
    that seed (400), record per-instance AR, success, violations, dead end and
    wall time per episode (env + inference, 1 torch thread);
  - compare with the instance's AR* from solve_ilp_max_ar (data/v4.3.1.4,
    recomputed by scripts/recompute_ar_star.py).
ILP timing: solve_ilp_max_ar on the seed-1 test set of every scenario (CBC, 1 thread).
Both run as WORKERS parallel single-thread processes, so they see the same load.

Outputs (paper_contents/v4.3.1/v4.3.1.7/): eval_raw.jsonl (per run), ilp_timing.json,
summary.json, report.md.

    .venv/bin/python scripts/eval_v4.3.1.7.py
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

MANIFEST = ROOT / "scripts" / "logs" / "v4.3.1.6_campaign" / "manifest.json"
OUT = ROOT / "paper_contents" / "v4.3.1" / "v4.3.1.7"
RESULTS = ROOT / "results" / "unified"
WORKERS = 16
WARMUP = 5
SCENARIOS = ["lt", "eq", "gt"]
LEARNERS = ["ppo", "dqn", "ddqn"]
MECHS = ["none", "lagrange", "mask", "repair"]
MECH_LABEL = {"none": "无约束（对照）", "lagrange": "Lagrangian", "mask": "Maskable", "repair": "Repair"}


def eval_run(job):
    import torch
    torch.set_num_threads(1)
    from paper_rl.env import PlacementEnv
    from paper_rl.train import model_class, split_algo, split_instances
    scen, algo, seed, exp_id = job
    mech, learner = split_algo(algo)
    run_dir = RESULTS / scen / algo / exp_id
    model = model_class(learner, mech).load(str(next(run_dir.glob("model_*"))), device="cpu")
    lam = json.loads((run_dir / "results.json").read_text())["training"].get("final_lambda", 0.0)
    _, _, test = split_instances(scen, seed)
    env = PlacementEnv(test, mech, "legacy", lam=lam)

    def episode(k):
        env.use_instance(k)
        obs, _ = env.reset()
        done = False
        while not done:
            if learner == "ppo" and mech == "mask":
                a, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            else:
                a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(int(a))
        return info

    for k in range(WARMUP):
        episode(k)
    rows = []
    for k in range(len(test)):
        t0 = time.perf_counter()
        info = episode(k)
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"ar": info["ar"], "ar_star": test[k]["ar_star"], "success": info["valid_placed"] == env.M,
                     "viol": info["capacity_violations"] + info["conflict_violations"] > 0,
                     "dead_end": info["dead_end"], "ms": ms})
    return {"scen": scen, "algo": algo, "seed": seed, "exp_id": exp_id, "rows": rows}


def ilp_one(x):
    from shared.ilp_utils import solve_ilp_max_ar
    t0 = time.perf_counter()
    r = solve_ilp_max_ar(x["ECUs"], x["SVCs"], x["conflict_sets"])
    return (time.perf_counter() - t0) * 1000, r["avg_utilization"], r["iterations"]


def stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def summarize(runs, ilp):
    out = {"scenarios": {}}
    for s in SCENARIOS:
        S = {"ilp": ilp[s], "models": {}}
        for l in LEARNERS:
            for m in MECHS:
                algo = l if m == "none" else f"{m}_{l}"
                per_seed = []
                for r in (x for x in runs if x["scen"] == s and x["algo"] == algo):
                    rows = r["rows"]
                    succ = [x for x in rows if x["success"]]
                    per_seed.append({
                        "success": np.mean([x["success"] for x in rows]),
                        "ar_success": np.mean([x["ar"] for x in succ]) if succ else None,
                        "ratio_success": np.mean([x["ar"] / x["ar_star"] for x in succ]) if succ else None,
                        "ratio_all": np.mean([x["ar"] / x["ar_star"] if x["success"] else 0.0 for x in rows]),
                        "gap_success": np.mean([x["ar_star"] - x["ar"] for x in succ]) if succ else None,
                        "viol": np.mean([x["viol"] for x in rows]),
                        "dead_end": np.mean([x["dead_end"] for x in rows]),
                        "ms": np.mean([x["ms"] for x in rows]),
                    })
                S["models"][algo] = {k: stats([p[k] for p in per_seed]) for k in per_seed[0]} | {"n_seeds": len(per_seed)}
        out["scenarios"][s] = S
    return out


def report(summary):
    f = lambda v, d=4: "—" if v is None else f"{v:.{d}f}"
    pm = lambda t, d=4: "—" if t[0] is None else f"{t[0]:.{d}f} ± {t[1]:.{d}f}"
    L = ["# v4.3.1.7 与 ILP 的对比（v4.3.1.6 的 108 个模型重新评估，AR* 为真正最优）", "",
         f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
         "- ILP：`shared.ilp_utils.solve_ilp_max_ar`（Dinkelbach，直接最大化 AR，CBC 单线程）。",
         "- RL：每个种子的模型在自己的测试集（400 个实例）上单次确定性评估；表中为 3 个种子均值 ± 样本标准差。",
         "- AR/AR*（成功）= 成功 episode 上 AR 与该实例 ILP 最优 AR 之比；综合 = 失败计 0；AR gap = AR* − AR（成功 episode）。",
         f"- 计时：{WORKERS} 个单线程进程并行，ILP 与 RL 条件相同；RL 计时含环境 reset/step/掩码与网络推理，预热 {WARMUP} 个 episode。", ""]
    for s in SCENARIOS:
        S = summary["scenarios"][s]
        il = S["ilp"]
        L += [f"## {s.upper()}", "",
              f"ILP 最优：平均 AR\\* = **{il['ar_mean']:.4f}**；求解耗时 {il['ms_mean']:.1f} ± {il['ms_std']:.1f} ms/实例"
              f"（中位数 {il['ms_median']:.1f}，最大 {il['ms_max']:.0f}；Dinkelbach 平均 {il['iter_mean']:.2f} 次迭代；种子 1 测试集 {il['n']} 个实例）", "",
              "| 模型 | 约束处理 | success_rate | AR（成功） | AR/AR*（成功） | 综合 AR/AR*×success | AR gap | 违规率 | 耗时 ms/实例 | 加速比 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for l in LEARNERS:
            for m in MECHS:
                algo = l if m == "none" else f"{m}_{l}"
                r = S["models"][algo]
                L.append("| " + " | ".join([l.upper(), MECH_LABEL[m], pm(r["success"], 3), pm(r["ar_success"]),
                                            pm(r["ratio_success"]), pm(r["ratio_all"]), pm(r["gap_success"]),
                                            pm(r["viol"], 3), f(r["ms"][0], 2), f"{il['ms_mean'] / r['ms'][0]:.1f}×"]) + " |")
        L.append("")
    (OUT / "report.md").write_text("\n".join(L))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = json.loads(MANIFEST.read_text())["exp_ids"]
    jobs = [(s, a, int(sd), e) for sd, d in ids.items() for s, d2 in d.items() for a, e in d2.items()]
    print(f"[1/2] evaluating {len(jobs)} models with {WORKERS} workers ...", flush=True)
    with Pool(WORKERS) as pool:
        runs = pool.map(eval_run, jobs, chunksize=1)
    with open(OUT / "eval_raw.jsonl", "w") as fh:
        for r in runs:
            fh.write(json.dumps(r) + "\n")
    print("[2/2] ILP timing ...", flush=True)
    from paper_rl.train import split_instances
    ilp = {}
    with Pool(WORKERS) as pool:
        for s in SCENARIOS:
            _, _, test = split_instances(s, 1)
            res = pool.map(ilp_one, test, chunksize=1)
            ms = [r[0] for r in res]
            ilp[s] = {"n": len(test), "ar_mean": float(np.mean([r[1] for r in res])),
                      "ms_mean": float(np.mean(ms)), "ms_std": float(np.std(ms, ddof=1)),
                      "ms_median": float(np.median(ms)), "ms_max": float(np.max(ms)),
                      "iter_mean": float(np.mean([r[2] for r in res]))}
            print(f"  {s}: AR* {ilp[s]['ar_mean']:.4f}, {ilp[s]['ms_mean']:.1f} ms/instance", flush=True)
    (OUT / "ilp_timing.json").write_text(json.dumps(ilp, indent=2))
    summary = summarize(runs, ilp)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    report(summary)
    print("done ->", OUT / "report.md", flush=True)


if __name__ == "__main__":
    main()
