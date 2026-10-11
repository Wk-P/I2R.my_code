#!/usr/bin/env python3
"""Per-instance wall time and gap: trained Mask PPO models vs the ILP, every seed on its own test split.

Same convention as bench_time.py (v4.3.1.7 timing): one CPU thread each.
  - policy: full deterministic episode per instance (env reset, action mask, forward pass, step), and the
    forward pass alone; torch.set_num_threads(1); 5 warm-up episodes.
  - ILP   : shared.ilp_utils.solve_ilp_max_ar (Dinkelbach, CBC single thread), model building included.
Both on the 400 test instances of each training seed, so the gap and the time refer to the same instances.

    .venv/bin/python src/scripts/bench_infer.py --manifest logs/v4.4.16/manifest.json --key mask_ppo_pool10m \\
        --label "v4.4.16 结构感知 Mask PPO（5 万实例，10M）" --label-en "v4.4.16 structure-aware Mask PPO (50k instances, 10M)" \\
        --out paper_contents/v4.4.16/time_bench.md
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results" / "unified"
SCENARIOS = ["lt", "eq", "gt"]
WARMUP, ILP_WORKERS = 5, 12


def test_split(scen, seed):
    from paper_rl.train import split_instances
    return split_instances(scen, seed)[2]


def policy_times(job):
    import torch
    torch.set_num_threads(1)
    from paper_rl.env import PlacementEnv
    from paper_rl.policy_io import load_policy
    scen, seed, exp_id = job
    test = test_split(scen, seed)
    predict, obs_mode = load_policy(scen, "mask_ppo", exp_id, "model_*", len(test[0]["ECUs"]), len(test[0]["SVCs"]))
    env = PlacementEnv(test, "mask", "ar_pen", full_episode=True, exit_action=True, obs_mode=obs_mode)

    def episode(k):
        env.use_instance(k)
        obs, _ = env.reset()
        fwd, done = 0.0, False
        while not done:
            mask = env.action_masks()
            t = time.perf_counter()
            a = predict(obs, mask)
            fwd += time.perf_counter() - t
            obs, _, done, _, info = env.step(a)
        return fwd, info

    for k in range(WARMUP):
        episode(k)
    rows = []
    for k in range(len(test)):
        t0 = time.perf_counter()
        fwd, info = episode(k)
        rows.append({"ms": (time.perf_counter() - t0) * 1000, "fwd_ms": fwd * 1000, "exited": bool(info["exited"]),
                     "ar": float(info["ar"]), "ilp": float(test[k]["ar_star"])})
    return scen, seed, rows


def ilp_one(args):
    from shared.ilp_utils import solve_ilp_max_ar
    scen, seed, k = args
    x = test_split(scen, seed)[k]
    t0 = time.perf_counter()
    r = solve_ilp_max_ar(x["ECUs"], x["SVCs"], x["conflict_sets"])
    return scen, seed, k, (time.perf_counter() - t0) * 1000, r["iterations"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--label", required=True)
    ap.add_argument("--label-en", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    ids = json.loads((ROOT / a.manifest).read_text())["exp_ids"]
    t_start = time.time()
    jobs = [(s, sd, ids[str(sd)][s][a.key]) for sd in seeds for s in SCENARIOS]
    n = {(s, sd): len(test_split(s, sd)) for s in SCENARIOS for sd in seeds}
    with Pool(len(jobs)) as pp, Pool(ILP_WORKERS) as ip:
        pol = pp.map_async(policy_times, jobs)
        ilp = {}
        tasks = [(s, sd, k) for (s, sd), m in n.items() for k in range(m)]
        for i, (s, sd, k, ms, it) in enumerate(ip.imap_unordered(ilp_one, tasks, chunksize=2), 1):
            ilp.setdefault(s, []).append((ms, it))
            if i % max(1, len(tasks) // 20) == 0:
                print(f"[bench] ILP {i}/{len(tasks)} ({100 * i / len(tasks):.0f}%)", flush=True)
        pol = {(s, sd): rows for s, sd, rows in pol.get()}
    secs = time.time() - t_start
    print(f"done in {secs / 60:.1f} min", flush=True)

    st = lambda v: (np.mean(v), np.median(v), np.percentile(v, 95), np.max(v))
    f = lambda v: f"{v:.1f}" if v >= 10 else f"{v:.2f}"
    out = {}
    for lang in ("zh", "en"):
        zh = lang == "zh"
        T = lambda z, e: z if zh else e
        L = [T("# 测试实例耗时与 gap：结构感知网络 vs ILP", "# Per-instance time and gap: structure-aware network vs ILP"), "",
             T(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/bench_infer.py`；总耗时 {secs / 60:.1f} 分钟。",
               f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; script `src/scripts/bench_infer.py`; took {secs / 60:.1f} min."),
             T(f"- 模型：{a.label}（`{a.manifest}`，key `{a.key}`），种子 {', '.join(map(str, seeds))}；每个模型在它自己训练种子的 400 个测试实例上计时，ILP 在同一批实例上计时，因此耗时与 gap 一一对应。",
               f"- Models: {a.label_en} (`{a.manifest}`, key `{a.key}`), seeds {', '.join(map(str, seeds))}; each model is timed on the 400 test "
               "instances of its own training seed and the ILP on the same instances, so time and gap refer to the same instances."),
             T(f"- 网络：CPU 单线程（`torch.set_num_threads(1)`），确定性策略；「整条 episode」= reset + 每步动作掩码 + 前向 + 环境 step；「仅前向」= 其中网络前向（含 SB3 predict 开销）的累计；先热身 {WARMUP} 个 episode。",
               f"- Network: one CPU thread (`torch.set_num_threads(1)`), deterministic policy; \"full episode\" = reset + action mask + forward pass "
               f"+ environment step at every step; \"forward only\" = the forward passes (SB3 predict overhead included); {WARMUP} warm-up episodes."),
             T(f"- ILP：`solve_ilp_max_ar`（Dinkelbach，PuLP + CBC 默认单线程），计时包含建模与全部迭代；{ILP_WORKERS} 个实例并行求解，每个实例单独计时。",
               f"- ILP: `solve_ilp_max_ar` (Dinkelbach, PuLP + CBC single thread), model building and all iterations included; {ILP_WORKERS} "
               "instances solved in parallel, each timed on its own."),
             T("- 计时时机器上同时有 GPU 训练进程（各占约 1 个 CPU 核）；56 核未占满，但不是完全空载。加速比 = ILP 耗时 / 网络耗时（平均 / 中位数）。",
               "- GPU training processes were running at the same time (about one CPU core each); the 56 cores were not saturated but the machine "
               "was not idle. Speed-up = ILP time / network time (mean / median)."), "",
             T("| 场景 | 方法 | 相对 gap | 平均 ms | 中位数 ms | P95 ms | 最大 ms | 加速比（平均） | 加速比（中位数） |",
               "| Scenario | Method | Relative gap | Mean ms | Median ms | P95 ms | Max ms | Speed-up (mean) | Speed-up (median) |"),
             "|---|---|---|---|---|---|---|---|---|"]
        for s in SCENARIOS:
            im = [x[0] for x in ilp[s]]
            ia = st(im)
            L.append(f"| {s.upper()} | {T('ILP（Dinkelbach，平均', 'ILP (Dinkelbach, mean')} {np.mean([x[1] for x in ilp[s]]):.2f} {T('次迭代）', 'iterations)')} | 0 | "
                     + " | ".join(f(v) for v in ia) + " | 1× | 1× |")
            rows = [r for sd in seeds for r in pol[(s, sd)]]
            gaps = []
            for sd in seeds:
                ok = [r for r in pol[(s, sd)] if not r["exited"]]
                ar, il = np.mean([r["ar"] for r in ok]), np.mean([r["ilp"] for r in ok])
                gaps.append((il - ar) / il)
            gtxt = f"{100 * np.mean(gaps):.1f}% ± {100 * np.std(gaps, ddof=1):.1f}%" if len(gaps) > 1 else f"{100 * gaps[0]:.1f}%"
            for part, key in ((T("整条 episode", "full episode"), "ms"), (T("仅前向", "forward only"), "fwd_ms")):
                b = st([r[key] for r in rows])
                L.append(f"| {s.upper()} | {T('网络：', 'network: ')}{part} | {gtxt} | " + " | ".join(f(x) for x in b)
                         + f" | {ia[0] / b[0]:.0f}× | {ia[1] / b[1]:.0f}× |")
        L.append("")
        out[lang] = L
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out["zh"]))
    p.with_suffix(".en.md").write_text("\n".join(out["en"]))
    p.with_suffix(".json").write_text(json.dumps({"ilp": {s: [x[0] for x in ilp[s]] for s in SCENARIOS},
                                                  "policy": {f"{s}|{sd}": pol[(s, sd)] for s in SCENARIOS for sd in seeds}}))
    print(f"report -> {p} (+ .en.md)", flush=True)


if __name__ == "__main__":
    main()
