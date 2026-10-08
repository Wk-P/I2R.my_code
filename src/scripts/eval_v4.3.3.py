#!/usr/bin/env python3
"""v4.3.3: best-of-K sampling at inference, PPO models of v4.3.1.6 vs a randomised greedy.

No retraining. Question: does the PPO policy's distribution contain better solutions
than its deterministic one (and than greedy)?
For each test instance (400 per seed) we draw K_MAX solutions:
  - RL: sample 0 is the deterministic policy, samples 1.. are drawn from the policy
    distribution (Maskable-PPO with its action mask);
  - greedy: sample 0 is src/paper_rl/greedy.py, samples 1.. take a uniformly random
    feasible ECU with probability GREEDY_EPS at each step, the greedy one otherwise.
Best-of-K uses the first K samples and picks, without looking at AR*, a successful
solution first, then the highest AR. AR* is only used afterwards for scoring.
Time per instance = K x the mean episode time (1 torch thread).

Outputs (paper_contents/v4.3.3/): raw.jsonl (per run), report.md.

    nohup .venv/bin/python src/scripts/eval_v4.3.3.py > logs/eval_v4.3.3.log 2>&1 &
    nohup .venv/bin/python src/scripts/eval_v4.3.3.py --resume >> logs/eval_v4.3.3.log 2>&1 &   # skip runs in raw.jsonl
    .venv/bin/python src/scripts/eval_v4.3.3.py --report      # regenerate report.md from raw.jsonl
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

MANIFEST = ROOT / "logs" / "v4.3.1.6_campaign" / "manifest.json"
OUT = ROOT / "paper_contents" / "v4.3.3"
RESULTS = ROOT / "results" / "unified"
WORKERS = 4
K_MAX = 64
KS = [1, 4, 16, 64]
GREEDY_EPS = 0.2
SCENARIOS = ["lt", "eq", "gt"]
ALGOS = ["ppo", "lagrange_ppo", "mask_ppo", "repair_ppo"]
LABEL = {"ppo": "PPO 无约束（对照）", "lagrange_ppo": "Lagrangian-PPO", "mask_ppo": "Maskable-PPO",
         "repair_ppo": "Repair-PPO", "greedy": "贪心（K>1 时为随机贪心）"}


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
    """success rate; AR and ILP AR over the same successful instances (best of the first k samples)."""
    best = [tuple(best_of(x["samples"], k)) + (x["ar_star"],) for x in r["rows"]]
    ok = [b for b in best if b[0]]
    ar = np.mean([b[1] for b in ok]) if ok else np.nan
    ilp = np.mean([b[2] for b in ok]) if ok else np.nan
    return len(ok) / len(best), ar, ilp, ilp - ar


def write_report(runs):
    def ms(v, f):
        v = [x for x in v if not np.isnan(x)]
        return f"{f(np.mean(v))} ± {f(np.std(v, ddof=1) if len(v) > 1 else 0.0)}" if v else "—"

    f4 = lambda v: f"{v:.4f}"
    pc = lambda v: f"{100 * v:.1f}%"
    ilp = json.loads((ROOT / "paper_contents" / "v4.3.1" / "v4.3.1.7" / "summary.json").read_text())["scenarios"]
    L = ["# v4.3.3 推理时采样 K 个解取最好（best-of-K）", "",
         f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；不重新训练，模型为 v4.3.1.6 的 PPO 系列（5M，种子 1–3）。",
         f"- 每个测试实例采样 {K_MAX} 个解；第 1 个为确定性输出（RL）或纯贪心，其余按策略概率采样（RL）"
         f"或每步以 {GREEDY_EPS} 的概率随机选一个可行 ECU（随机贪心）。K 个解取前 K 个。",
         "- 挑选规则只用解本身：先要成功，再取 AR 最高；ILP AR 只用于最后计分。",
         "- AR 与 ILP AR 都只在该方法成功的测试实例上平均（同一批实例）；AR gap = ILP AR − AR。"
         "均值 ± 样本标准差（3 种子的测试集）。",
         "- 耗时 = K × 单次 episode 平均耗时（单线程，逐个运行）。本次与 v4.3.1.10 训练同时运行，机器满载，"
         "耗时偏高；ILP 耗时取自 v4.3.1.7，两者不是同一负载下测得，只作参考。", ""]
    for s in SCENARIOS:
        L += [f"## {s.upper()}", "",
              f"ILP（v4.3.1.7）：全部测试实例平均 AR {ilp[s]['ilp']['ar_mean']:.4f}，"
              f"平均耗时 {ilp[s]['ilp']['ms_mean']:.1f} ms/实例（中位数 {ilp[s]['ilp']['ms_median']:.1f}）。", "",
              "| 方法 | K | 成功率 | AR | ILP AR | AR gap | 耗时 ms/实例 | 比 ILP 快 |",
              "|---|---|---|---|---|---|---|---|"]
        for a in ["greedy"] + ALGOS:
            rs = [r for r in runs if r["scen"] == s and r["algo"] == a]
            for k in KS:
                st = [summarise(r, k) for r in rs]
                t = k * np.mean([r["ms_episode"] for r in rs])
                L.append("| " + " | ".join([LABEL[a] if k == KS[0] else "", str(k),
                                            ms([x[0] for x in st], pc), ms([x[1] for x in st], f4),
                                            ms([x[2] for x in st], f4), ms([x[3] for x in st], f4),
                                            f"{t:.1f}", f"{ilp[s]['ilp']['ms_mean'] / t:.1f}×"]) + " |")
        L.append("")
    (OUT / "report.md").write_text("\n".join(L))
    print(f"report -> {OUT / 'report.md'}", flush=True)


def write_panel_manifest(jobs):
    """Evaluation-batch manifest for the monitor panel (app/backend/main.py, _eval_batch_state)."""
    ilp = json.loads((ROOT / "paper_contents" / "v4.3.1" / "v4.3.1.7" / "summary.json").read_text())["scenarios"]
    d = ROOT / "logs" / "eval_v4.3.3"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({
        "kind": "eval", "version": "4.3.3", "script": "src/scripts/eval_v4.3.3.py", "log": "logs/eval_v4.3.3.log",
        "workers": WORKERS, "started_at": time.time(), "raw": "paper_contents/v4.3.3/raw.jsonl",
        "raw_format": "best_of_k", "ks": KS, "report": "paper_contents/v4.3.3/report.md",
        "ilp": {s: {k: ilp[s]["ilp"][k] for k in ("ar_mean", "ms_mean", "ms_median")}
                | {"source": "v4.3.1.7（CBC，单线程，16 进程并行）"} for s in SCENARIOS},
        "jobs": [{"scenario": s, "algo": a, "seed": int(sd)} for s, a, sd, _ in jobs],
        "description": "推理时采样 K 个解取最好（best-of-K）：v4.3.1.6 的 PPO 系列（5M）vs 随机贪心，不重新训练",
    }, ensure_ascii=False, indent=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--report" in sys.argv:                  # regenerate report.md from raw.jsonl
        write_report([json.loads(l) for l in open(OUT / "raw.jsonl")])
        return
    m = json.loads(MANIFEST.read_text())["exp_ids"]
    jobs = [(s, a, int(sd), m[sd][s][a]) for sd in m for s in SCENARIOS for a in ALGOS]
    jobs += [(s, "greedy", sd, "") for s in SCENARIOS for sd in [1, 2, 3]]
    write_panel_manifest(jobs)
    print(f"=== v4.3.3 start {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(jobs)} runs ===", flush=True)
    # Append each run as soon as it finishes, so the panel shows finished jobs'
    # metrics while the rest are running; --resume skips runs already in raw.jsonl.
    raw = OUT / "raw.jsonl"
    runs = [json.loads(l) for l in open(raw)] if "--resume" in sys.argv and raw.exists() else []
    have = {(r["scen"], r["algo"], int(r["seed"])) for r in runs}
    todo = [j for j in jobs if (j[0], j[1], int(j[2])) not in have]
    with open(raw, "a" if runs else "w") as f, Pool(WORKERS) as pool:
        for r in pool.imap_unordered(run, todo, chunksize=1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            runs.append(r)
    write_report(runs)


if __name__ == "__main__":
    main()
