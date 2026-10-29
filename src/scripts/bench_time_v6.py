#!/usr/bin/env python3
"""Per-instance wall time on the summary page: ILP vs the 12 strategy x RL_algorithm variants (no training).

Same instances for every method: the first N_INST of the fixed v6.0.4 test set per scenario. One CPU thread each:
  - RL : the seed-1 model of every summary cell (src/scripts/merge_2d_table.py:load, the same models as the table),
         paper_rl.train.evaluate (deterministic full episode: env reset, mask for Mask PPO, forward pass, step) on all
         N_INST instances after WARMUP warm-up episodes; ms/instance = total / N_INST. torch.set_num_threads(1).
  - ILP: shared.ilp_utils.solve_ilp_max_ar (Dinkelbach, PuLP + CBC single thread), timed per instance around the whole
         call (model building and every iteration included); the iteration counts are kept too.
Speed-up = ILP mean / variant mean, per scenario.

    .venv/bin/python src/scripts/bench_time_v6.py [--workers 24]
        -> paper_contents/v6.0.0-merge/time_bench.{json,md,en.md}; merge_2d_table.py puts it on the summary page
        log: stdout only
"""
import argparse
import json
import statistics as st
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))
BATCH = "v6.0.0-merge"
OUT = ROOT / "paper_contents" / BATCH
TEST = ROOT / "data/testset_v6.0.4/{scen}.json"
N_INST, WARMUP, SEED = 200, 5, 1


def instances(scen):
    return json.loads(Path(str(TEST).format(scen=scen)).read_text())[:N_INST]


def rl_time(row):
    import torch
    torch.set_num_threads(1)
    from paper_rl.train import evaluate, model_class
    d = Path(row["dir"])
    res = json.loads((d / "results.json").read_text())
    mech, learner = res["mechanism"], res["learner"]
    tr = res.get("training") or {}
    lam = tr.get("final_lambda", 0.0) if mech == "lagrange" else 0.0
    lm = tr.get("lambda_max", 50.0)
    lm = float("inf") if lm == "none" else float(lm)
    model = model_class(learner, mech).load(str(next(d.glob("model_*"))), device="cpu")
    ev = lambda xs: evaluate(model, xs, mech, res["reward_mode"], lam, learner, res["obs"], res["full_episode"],
                             res["exit_action"], res["action_mode"], res["order"], res["order_seed"], lam_max=lm)
    test = instances(row["scen"])
    ev(test[:WARMUP])
    t0 = time.perf_counter()
    ev(test)
    return row["mech"], row["learner"], row["scen"], (time.perf_counter() - t0) * 1000 / len(test)


def ilp_time(args):
    from shared.ilp_utils import solve_ilp_max_ar
    scen, k = args
    x = instances(scen)[k]
    t0 = time.perf_counter()
    r = solve_ilp_max_ar(x["ECUs"], x["SVCs"], x["conflict_sets"])
    return scen, (time.perf_counter() - t0) * 1000, r["iterations"]


def build(d, lang):
    from merge_2d_table import MECHS, LEARNERS, SCENS, NAME
    T = (lambda z, e: z if lang == "zh" else e)
    f = lambda v: f"{v:.1f}" if v >= 10 else f"{v:.2f}"
    L = [T("# 单实例耗时：ILP vs 12 个变种", "# Per-instance time: ILP vs the 12 variants"), "",
         T(f"- 生成时间 {d['generated_at']}；脚本 `src/scripts/bench_time_v6.py`；耗时 {d['minutes']:.1f} 分钟；只用 CPU，不训练。",
           f"- Generated {d['generated_at']}; script `src/scripts/bench_time_v6.py`; took {d['minutes']:.1f} min; CPU only, no training."),
         T(f"- 实例：v6.0.4 固定测试集每场景前 {N_INST} 个，所有方法相同；单位 ms/实例，CPU 单线程。",
           f"- Instances: the first {N_INST} of the fixed v6.0.4 test set per scenario, the same for every method; ms per instance, one CPU thread."),
         T(f"- RL：总图中每格的种子 {SEED} 模型，确定性完整 episode（reset + 掩码 + 前向 + step），先热身 {WARMUP} 个。"
           "ILP：`solve_ilp_max_ar`（Dinkelbach，CBC 单线程），含建模与全部迭代。加速比 = ILP 平均 / 变种平均。",
           f"- RL: the seed-{SEED} model of every summary cell, deterministic full episode (reset + mask + forward + step), {WARMUP} warm-up episodes. "
           "ILP: `solve_ilp_max_ar` (Dinkelbach, CBC single thread), model building and every iteration included. Speed-up = ILP mean / variant mean."), "",
         "| " + " | ".join([T("方法", "Method")] + [s.upper() for s in SCENS] + [T("三场景平均", "Mean"), T("加速比（平均）", "Speed-up (mean)")]) + " |",
         "|---" * (len(SCENS) + 3) + "|"]
    ilp = d["ilp"]
    it = st.mean(ilp[s]["iters"] for s in SCENS)
    L.append(f"| ILP (Dinkelbach, {it:.2f} {T('次迭代', 'iterations')}) | " + " | ".join(f(ilp[s]["mean_ms"]) for s in SCENS)
             + f" | {f(st.mean(ilp[s]['mean_ms'] for s in SCENS))} | 1× |")
    for m in MECHS:
        for l in LEARNERS:
            v = d["rl"][f"{m}|{l}"]
            L.append(f"| {NAME[m]} {NAME[l]} | " + " | ".join(f(v[s]) for s in SCENS) + f" | {f(st.mean(v.values()))} | "
                     f"{st.mean(ilp[s]['mean_ms'] / v[s] for s in SCENS):.0f}× |")
    L.append("")
    L.append(T("- 加速比为三场景各自加速比的平均。", "- Speed-up is the mean of the per-scenario speed-ups."))
    L.append("")
    return "\n".join(L)


def main():
    from merge_2d_table import load, SCENS
    t0 = time.time()
    rows = [r for r in load() if r["seed"] == SEED]
    assert len(rows) == 36, len(rows)
    with Pool(ARGS.workers) as pool:
        rl = pool.map_async(rl_time, rows, chunksize=1)
        ilp = {}
        for s, ms, it in pool.imap_unordered(ilp_time, [(s, k) for s in SCENS for k in range(N_INST)], chunksize=4):
            ilp.setdefault(s, []).append((ms, it))
        rl = rl.get()
    out = dict(generated_at=time.strftime("%Y-%m-%d %H:%M:%S"), script="src/scripts/bench_time_v6.py", n_inst=N_INST, seed=SEED,
               minutes=(time.time() - t0) / 60,
               ilp={s: dict(mean_ms=st.mean(x[0] for x in v), median_ms=st.median(x[0] for x in v),
                            iters=st.mean(x[1] for x in v)) for s, v in ilp.items()}, rl={})
    for m, l, s, ms in rl:
        out["rl"].setdefault(f"{m}|{l}", {})[s] = ms
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "time_bench.json").write_text(json.dumps(out, indent=1))
    (OUT / "time_bench.md").write_text(build(out, "zh"))
    (OUT / "time_bench.en.md").write_text(build(out, "en"))
    print(OUT / "time_bench.md", f"({out['minutes']:.1f} min)", flush=True)
    if not ARGS.no_mail:
        from shared.report_mail import send_report
        send_report(BATCH, OUT / "time_bench.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--no-mail", action="store_true")
    ARGS = ap.parse_args()
    # no manifest of its own: the output belongs to v6.0.0-merge (merge_v6_0_0.py owns that manifest) and is shown on /summary
    main()
