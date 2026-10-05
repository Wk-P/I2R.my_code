#!/usr/bin/env python3
"""v4.3.3: best-of-K sampling at inference, PPO models of v4.3.1.6 vs a randomised greedy.

No retraining. Question: does the PPO policy's distribution contain better solutions
than its deterministic one (and than greedy)?
For each test instance (400 per seed) we draw K_MAX solutions:
  - RL: sample 0 is the deterministic policy, samples 1.. are drawn from the policy
    distribution (Maskable-PPO with its action mask);
  - greedy: sample 0 is paper_rl/greedy.py, samples 1.. take a uniformly random
    feasible ECU with probability GREEDY_EPS at each step, the greedy one otherwise.
Best-of-K uses the first K samples and picks, without looking at AR*, a successful
solution first, then the highest AR. AR* is only used afterwards for scoring.
Time per instance = K x the mean episode time (1 torch thread).

Outputs (paper_contents/v4.3.3/): raw.jsonl (per run), report.md.

    nohup .venv/bin/python scripts/eval_v4.3.3.py > scripts/logs/eval_v4.3.3.log 2>&1 &
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
OUT = ROOT / "paper_contents" / "v4.3.3"
RESULTS = ROOT / "results" / "unified"
WORKERS = 4
K_MAX = 64
KS = [1, 4, 16, 64]
GREEDY_EPS = 0.2
SCENARIOS = ["lt", "eq", "gt"]
ALGOS = ["ppo", "lagrange_ppo", "mask_ppo", "repair_ppo"]
LABEL = {"ppo": "PPO 无约束（对照）", "lagrange_ppo": "Lagrangian-PPO", "mask_ppo": "Maskable-PPO",
         "repair_ppo": "Repair-PPO", "greedy": "随机贪心"}


def best_of(samples, k):
    """samples: [(success, ar)] for one instance; best of the first k, AR* not used."""
    return max(samples[:k], key=lambda x: (x[0], x[1]))


def run(job):
    import torch
    torch.set_num_threads(1)
    from paper_rl.env import PlacementEnv
    from paper_rl.greedy import greedy_action
    from paper_rl.train import model_class, split_algo, split_instances
    scen, algo, seed, exp_id = job
    _, _, test = split_instances(scen, seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if algo == "greedy":
        env = PlacementEnv(test, "mask", "legacy")

        def act(obs, k_sample):
            if k_sample > 0 and rng.random() < GREEDY_EPS:
                return int(rng.choice(np.flatnonzero(env._feasible(env.t))))
            return greedy_action(env)
    else:
        mech, learner = split_algo(algo)
        run_dir = RESULTS / scen / algo / exp_id
        model = model_class(learner, mech).load(str(next(run_dir.glob("model_*"))), device="cpu")
        lam = json.loads((run_dir / "results.json").read_text())["training"].get("final_lambda", 0.0)
        env = PlacementEnv(test, mech, "legacy", lam=lam)

        def act(obs, k_sample):
            kw = {"action_masks": env.action_masks()} if mech == "mask" else {}
            a, _ = model.predict(obs, deterministic=(k_sample == 0), **kw)
            return int(a)

    rows, t0 = [], time.perf_counter()
    for k in range(len(test)):
        samples = []
        for j in range(K_MAX):
            env.use_instance(k)
            obs, _ = env.reset()
            done = False
            while not done:
                obs, _, done, _, info = env.step(act(obs, j))
            samples.append((bool(info["valid_placed"] == env.M), float(info["ar"])))
        rows.append({"ar_star": test[k]["ar_star"], "samples": samples})
    ms = (time.perf_counter() - t0) * 1000 / (len(test) * K_MAX)
    print(f"[{time.strftime('%H:%M:%S')}] done {scen} {algo} seed{seed} ({ms:.1f} ms/episode)", flush=True)
    return {"scen": scen, "algo": algo, "seed": seed, "exp_id": exp_id, "ms_episode": ms, "rows": rows}


def summarise(r, k):
    best = [best_of(x["samples"], k) + (x["ar_star"],) for x in r["rows"]]
    succ = np.array([b[0] for b in best])
    ratio = np.array([b[1] / b[2] if b[0] else 0.0 for b in best])
    return succ.mean(), (ratio[succ].mean() if succ.any() else np.nan), ratio.mean()


def write_report(runs):
    def ms(v, f):
        v = [x for x in v if not np.isnan(x)]
        return f"{f(np.mean(v))} ± {f(np.std(v, ddof=1) if len(v) > 1 else 0.0)}" if v else "—"

    f3 = lambda v: f"{v:.3f}"
    pc = lambda v: f"{100 * v:.1f}%"
    L = ["# v4.3.3 推理时采样 K 个解取最好（best-of-K）", "",
         f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；不重新训练，模型为 v4.3.1.6 的 PPO 系列（5M，种子 1–3）。",
         f"- 每个测试实例采样 {K_MAX} 个解；第 1 个为确定性输出（RL）或纯贪心，其余按策略概率采样（RL）"
         f"或每步以 {GREEDY_EPS} 的概率随机选一个可行 ECU（随机贪心）。K 个解取前 K 个。",
         "- 挑选规则只用解本身：先要成功，再取 AR 最高；最优 AR（ILP）只用于最后计分。",
         "- 相对最优 AR = 模型 AR ÷ 同一实例的最优 AR。「成功回合」只在成功实例上平均；「失败计 0」把失败实例记为 0。"
         "均值 ± 样本标准差（3 种子的测试集）。耗时 = K × 单次 episode 平均耗时（单线程）。", ""]
    for s in SCENARIOS:
        L += [f"## {s.upper()}", "", "| 方法 | K | 成功率 | 相对最优 AR（成功回合） | 相对最优 AR（失败计 0） | 耗时 ms/实例 |",
              "|---|---|---|---|---|---|"]
        for a in ["greedy"] + ALGOS:
            rs = [r for r in runs if r["scen"] == s and r["algo"] == a]
            for k in KS:
                st = [summarise(r, k) for r in rs]
                L.append("| " + " | ".join([LABEL[a] if k == KS[0] else "", str(k),
                                            ms([x[0] for x in st], f3), ms([x[1] for x in st], pc),
                                            ms([x[2] for x in st], pc),
                                            f"{k * np.mean([r['ms_episode'] for r in rs]):.1f}"]) + " |")
        L.append("")
    (OUT / "report.md").write_text("\n".join(L))
    print(f"report -> {OUT / 'report.md'}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.loads(MANIFEST.read_text())["exp_ids"]
    jobs = [(s, a, int(sd), m[sd][s][a]) for sd in m for s in SCENARIOS for a in ALGOS]
    jobs += [(s, "greedy", sd, "") for s in SCENARIOS for sd in [1, 2, 3]]
    print(f"=== v4.3.3 start {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(jobs)} runs ===", flush=True)
    with Pool(WORKERS) as pool:
        runs = pool.map(run, jobs, chunksize=1)
    with open(OUT / "raw.jsonl", "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    write_report(runs)


if __name__ == "__main__":
    main()
