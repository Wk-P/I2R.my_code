#!/usr/bin/env python3
"""Per-instance wall time: structure-aware Mask PPO vs ILP (v4.3.1.7 timing convention), no training.

Same test instances (seed-1 split, 400 per scenario), one CPU thread each:
  - policy: full deterministic episode (env reset, action mask, forward pass, step) per instance,
    torch.set_num_threads(1); the forward pass alone is timed too. 5 warm-up episodes.
  - ILP   : shared.ilp_utils.solve_ilp_max_ar (Dinkelbach, CBC single thread) per instance, timed
    around the whole call (model building included).
Models: v4.4.3 5M seed 1 (structure-aware, logs/v4.4.3/manifest.json) and its MLP 5M control.

    .venv/bin/python src/scripts/bench_time.py      -> paper_contents/v4.4.3/time_bench.md
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
MANIFEST = ROOT / "logs" / "v4.4.3" / "manifest.json"
REPORT = ROOT / "paper_contents" / "v4.4.3" / "time_bench.md"
SCENARIOS = ["lt", "eq", "gt"]
MODELS = [("结构感知 Mask PPO（5M）", "mask_ppo"), ("MLP Mask PPO（5M，对照）", "mask_ppo_mlp")]
SEED, WARMUP, ILP_WORKERS = 1, 5, 12


def test_split(scen):
    from paper_rl.train import split_instances
    return split_instances(scen, SEED)[2]


def policy_times(job):
    import torch
    torch.set_num_threads(1)
    from paper_rl.env import PlacementEnv
    from paper_rl.policy_io import load_policy
    label, scen, exp_id = job
    test = test_split(scen)
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
        rows.append({"ms": (time.perf_counter() - t0) * 1000, "fwd_ms": fwd * 1000, "exited": bool(info["exited"])})
    return label, scen, rows


def ilp_one(args):
    from shared.ilp_utils import solve_ilp_max_ar
    scen, k = args
    x = test_split_cache(scen)[k]
    t0 = time.perf_counter()
    r = solve_ilp_max_ar(x["ECUs"], x["SVCs"], x["conflict_sets"])
    return scen, k, (time.perf_counter() - t0) * 1000, r["iterations"]


_CACHE = {}


def test_split_cache(scen):
    if scen not in _CACHE:
        _CACHE[scen] = test_split(scen)
    return _CACHE[scen]


def main():
    t_start = time.time()
    ids = json.loads(MANIFEST.read_text())["exp_ids"][str(SEED)]
    jobs = [(lb, s, ids[s][key]) for lb, key in MODELS for s in SCENARIOS]
    n = {s: len(test_split(s)) for s in SCENARIOS}
    with Pool(len(jobs)) as pp, Pool(ILP_WORKERS) as ip:
        pol = pp.map_async(policy_times, jobs)
        ilp = {}
        for s, k, ms, it in ip.imap_unordered(ilp_one, [(s, k) for s in SCENARIOS for k in range(n[s])], chunksize=2):
            ilp.setdefault(s, []).append((ms, it))
        pol = {(lb, s): rows for lb, s, rows in pol.get()}
    print(f"done in {(time.time() - t_start) / 60:.1f} min", flush=True)

    st = lambda v: (np.mean(v), np.median(v), np.percentile(v, 95), np.max(v))
    f = lambda v: f"{v:.1f}" if v >= 10 else f"{v:.2f}"
    lines = ["# 测试实例耗时：结构感知网络 vs ILP", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/bench_time.py`；总耗时 {(time.time() - t_start) / 60:.1f} 分钟。",
             f"- 实例：种子 1 测试集（每场景 {n['lt']} 个，与训练报告同一划分），逐个实例计时，单位 ms/实例。",
             "- 网络：v4.4.3 5M 种子 1 模型（`logs/v4.4.3/manifest.json`），CPU 单线程（`torch.set_num_threads(1)`），确定性策略；"
             f"「整条 episode」= reset + 每步动作掩码 + 前向 + 环境 step，直到放完或 EXIT；「仅前向」= 其中网络前向（含 SB3 predict 开销）的累计。先热身 {WARMUP} 个 episode。",
             "- ILP：`solve_ilp_max_ar`（Dinkelbach，PuLP + CBC 默认单线程），计时包含建模与全部迭代，与 v4.3.1.7 的口径相同。"
             f"{ILP_WORKERS} 个实例并行求解，每个实例单独计时。",
             "- 计时时机器上同时有 9 个 GPU 训练进程（各占约 1 个 CPU 核）；56 核未占满，单个实例的计时基本不受影响，但不是完全空载。",
             "- 加速比 = ILP 平均耗时 / 网络平均耗时；中位数加速比 = ILP 中位数 / 网络中位数。", ""]
    lines += ["| 场景 | 方法 | 平均 | 中位数 | P95 | 最大 | 加速比（平均） | 加速比（中位数） |", "|---|---|---|---|---|---|---|---|"]
    for s in SCENARIOS:
        im = [x[0] for x in ilp[s]]
        a = st(im)
        lines.append(f"| {s.upper()} | ILP（Dinkelbach，平均 {np.mean([x[1] for x in ilp[s]]):.2f} 次迭代） | "
                     + " | ".join(f(v) for v in a) + " | 1× | 1× |")
        for lb, _ in MODELS:
            for part, key in (("整条 episode", "ms"), ("仅前向", "fwd_ms")):
                v = [r[key] for r in pol[(lb, s)]]
                b = st(v)
                lines.append(f"| {s.upper()} | {lb}：{part} | " + " | ".join(f(x) for x in b)
                             + f" | {a[0] / b[0]:.0f}× | {a[1] / b[1]:.0f}× |")
    lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    (REPORT.with_suffix(".json")).write_text(json.dumps({"ilp": {s: [x[0] for x in ilp[s]] for s in SCENARIOS},
                                                         "policy": {f"{lb}|{s}": pol[(lb, s)] for lb, _ in MODELS for s in SCENARIOS}}))
    print(f"report -> {REPORT}", flush=True)


if __name__ == "__main__":
    main()
