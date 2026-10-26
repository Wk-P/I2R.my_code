#!/usr/bin/env python3
"""v6.0.2 diagnostic: why do v6.0.0 Mask policies EXIT? (replay + ILP, no training)

Every v6.0.0 Mask run with EXITs is replayed deterministically on its own 400 test instances (same env as
paper_rl.train.evaluate). For every EXIT instance, with the first t placements of the trajectory pinned:
  t_inf   = smallest t for which the ILP has no feasible completion (the step that made the instance unsolvable)
  t_cheap = smallest t at which some unplaced service has no legal ECU left (one-step necessary condition)
  t_exit  = placements made before EXIT
so t_inf <= t_cheap <= t_exit. At the EXIT itself each ECU is blocked by capacity only, privacy only or both.

    .venv/bin/python src/scripts/diag_exit_v6.py      -> paper_contents/v6.0.2/exit_diag.md (+ .en.md, .json), mailed
"""
import argparse
import glob
import json
import statistics as st
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "paper_contents/v6.0.2"
MANIFEST = ROOT / "logs/v6.0.0/manifest.json"
SCENS = ["eq", "gt", "lt"]
LEARNERS = ["ppo", "dqn", "ddqn"]


def mask_runs():
    runs = []
    for r in json.loads(MANIFEST.read_text())["runs"]:
        if r["strategy"] != "maskable":
            continue
        d = Path(glob.glob(str(ROOT / f"results/unified/{r['scenario']}/{r['dir']}/{r['exp_id']}/results.json"))[0]).parent
        res = json.loads((d / "results.json").read_text())
        runs.append(dict(scen=r["scenario"], learner=res["learner"], seed=r["seed"], dir=str(d), res=res))
    return runs


def replay(run):
    """Deterministic replay; returns the EXIT trajectories (original service index and ECU per step)."""
    import torch
    torch.set_num_threads(1)
    from paper_rl.env import PlacementEnv
    from paper_rl.train import model_class, split_instances
    res = run["res"]
    _, _, test = split_instances(run["scen"], run["seed"])
    env = PlacementEnv(test, "mask", res["reward_mode"], lam=0.0, obs_mode=res["obs"], full_episode=res["full_episode"],
                       order=res["order"], order_seed=res["order_seed"], exit_action=res["exit_action"],
                       action_mode=res["action_mode"])
    model = model_class(run["learner"], "mask").load(str(next(Path(run["dir"]).glob("model_*"))), device="cpu")
    out = []
    for k in range(len(test)):
        env.use_instance(k)
        obs, _ = env.reset()
        steps, done, info = [], False, {}
        while not done:
            if run["learner"] == "ppo":
                a, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            else:
                a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            if a < env.N:
                steps.append((int(env.order[env.t]), a))
            obs, _, done, _, info = env.step(a)
        if info["exited"]:
            out.append({"k": k, "steps": steps})
    return run, out, test


def analyse(args):
    """t_inf / t_cheap / t_exit and the blocking cause at the EXIT for one EXIT trajectory."""
    from shared.ilp_utils import solve_ilp_max_ar
    x, steps = args
    caps, reqs, sets = x["ECUs"], x["SVCs"], x["conflict_sets"]
    n, m = len(caps), len(reqs)
    partners = [set() for _ in range(m)]
    for s in sets:
        for i in s:
            partners[i] |= set(s) - {i}

    def feasible_ilp(t):
        r = solve_ilp_max_ar(caps, reqs, sets, max_iter=1, fixed=dict(steps[:t]))
        return r["status"] == "Optimal"

    def blocked(t):
        """per unplaced service: per ECU (cap_blocked, conf_blocked) after the first t placements"""
        rem = list(caps)
        hosted = [set() for _ in range(n)]
        for i, j in steps[:t]:
            rem[j] -= reqs[i]
            hosted[j].add(i)
        placed = {i for i, _ in steps[:t]}
        return {i: [(rem[j] < reqs[i], bool(partners[i] & hosted[j])) for j in range(n)]
                for i in range(m) if i not in placed}

    t_exit = len(steps)
    assert feasible_ilp(0), "instance must be ILP-feasible"
    assert not feasible_ilp(t_exit), "an EXIT state must have no feasible completion"
    t_inf = next(t for t in range(1, t_exit + 1) if not feasible_ilp(t))
    t_cheap = next(t for t in range(t_exit + 1) if any(not any(not c and not f for c, f in b) for b in blocked(t).values()))
    nxt = blocked(t_exit)
    svc = next(i for i in nxt if not any(not c and not f for c, f in nxt[i]))
    cause = nxt[svc]
    return dict(t_inf=t_inf, t_cheap=t_cheap, t_exit=t_exit, M=m,
                cap_only=sum(c and not f for c, f in cause), conf_only=sum(f and not c for c, f in cause),
                both=sum(c and f for c, f in cause), N=n)


def pct(v):
    return f"{100 * v:.3f}%"


def build(rows, meta, lang, minutes):
    zh = lang == "zh"
    T = (lambda z, e: z if zh else e)
    L = ["# v6.0.2 " + T("Mask EXIT 归因诊断", "Mask EXIT attribution diagnostic"), ""]
    L.append(T(f"- 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/diag_exit_v6.py`；耗时 {minutes:.1f} 分钟；不训练。",
               f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; script `src/scripts/diag_exit_v6.py`; took {minutes:.1f} min; no training."))
    L.append(T("- 模型：v6.0.0 的 Maskable 最终模型（PPO / DQN / DDQN，种子 1–3），在各自种子的 400 个测试实例上确定性回放；"
               "回放得到的 EXIT 实例数与 results.json 逐 run 核对一致。",
               "- Models: the v6.0.0 Maskable final models (PPO / DQN / DDQN, seeds 1–3), replayed deterministically on each seed's 400 "
               "test instances; the replayed EXIT counts match results.json run by run."))
    L.append(T("- 对每个 EXIT 实例，固定轨迹的前 t 步放置：", "- For every EXIT instance, with the first t placements of the trajectory fixed:"))
    L.append(T("  - **t_inf**：ILP 判定剩余部分已无可行解的最小 t，即让实例变成无解的那一步；",
               "  - **t_inf**: the smallest t for which the ILP finds no feasible completion, i.e. the step that made the instance unsolvable;"))
    L.append(T("  - **t_cheap**：某个尚未放置的服务已没有任何合法 ECU 的最小 t（只看当前状态的一步必要条件）；",
               "  - **t_cheap**: the smallest t at which some unplaced service has no legal ECU left (a one-step necessary condition on the current state);"))
    L.append(T("  - **t_exit**：EXIT 前已放置的服务数。三者满足 t_inf ≤ t_cheap ≤ t_exit。",
               "  - **t_exit**: services placed before the EXIT. Always t_inf ≤ t_cheap ≤ t_exit."))
    L.append(T("- **潜伏步数** = t_exit − t_inf：实例已经无解、但策略还在继续放置的步数。**可见步数** = t_exit − t_cheap。",
               "- **Latent steps** = t_exit − t_inf: steps placed after the instance was already unsolvable. **Visible steps** = t_exit − t_cheap."))
    L.append("")

    L.append("## " + T("1. EXIT 实例数", "1. EXIT instances"))
    L.append("")
    L.append("| " + T("场景 | 算法 | EXIT 实例 / 总实例 | 比例", "Scen | Algo | EXIT instances / total | Rate") + " |")
    L.append("|---|---|---|---|")
    for s in SCENS:
        for l in LEARNERS:
            e, tot = meta[(s, l)]
            if e:
                L.append(f"| {s.upper()} | {l.upper()} | {e} / {tot} | {pct(e / tot)} |")
    L.append("")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["scen"], r["learner"])].append(r)
        groups[(r["scen"], "all")].append(r)
    keys = [(s, l) for s in SCENS for l in LEARNERS + ["all"] if groups.get((s, l))]
    name = lambda s, l: f"{s.upper()} {T('合计', 'all') if l == 'all' else l.upper()}"

    L.append("## " + T("2. 无解从哪一步开始", "2. When does the instance become unsolvable"))
    L.append("")
    L.append("| " + T("组 | 实例 | t_inf 均值 | t_cheap 均值 | t_exit 均值 | 潜伏步数均值 | 潜伏 = 0 | 潜伏 ≥ 3 | t_cheap = t_inf",
                       "Group | Instances | mean t_inf | mean t_cheap | mean t_exit | mean latent | latent = 0 | latent ≥ 3 | t_cheap = t_inf") + " |")
    L.append("|---" * 9 + "|")
    for s, l in keys:
        g = groups[(s, l)]
        lat = [r["t_exit"] - r["t_inf"] for r in g]
        L.append(f"| {name(s, l)} | {len(g)} | {st.mean(r['t_inf'] for r in g):.3f} | {st.mean(r['t_cheap'] for r in g):.3f} | "
                 f"{st.mean(r['t_exit'] for r in g):.3f} | {st.mean(lat):.3f} | {sum(v == 0 for v in lat)} / {len(g)} | "
                 f"{sum(v >= 3 for v in lat)} / {len(g)} | {sum(r['t_cheap'] == r['t_inf'] for r in g)} / {len(g)} |")
    L.append("")

    L.append("### " + T("潜伏步数分布（t_exit − t_inf）", "Latent-step distribution (t_exit − t_inf)"))
    L.append("")
    mx = max(r["t_exit"] - r["t_inf"] for r in rows)
    L.append("| " + T("组", "Group") + " | " + " | ".join(str(d) for d in range(mx + 1)) + " |")
    L.append("|---" * (mx + 2) + "|")
    for s, l in keys:
        g = groups[(s, l)]
        L.append(f"| {name(s, l)} | " + " | ".join(str(sum(r["t_exit"] - r["t_inf"] == d for r in g)) for d in range(mx + 1)) + " |")
    L.append("")

    L.append("## " + T("3. EXIT 时被屏蔽的原因", "3. Why every ECU is masked at the EXIT"))
    L.append("")
    L.append(T("对 EXIT 时无处可放的那个服务，逐个 ECU 判断被屏蔽的原因；实例按全部 ECU 的原因归类。",
               "For the service that cannot be placed at the EXIT, each ECU's blocking reason; instances are classified over all ECUs."))
    L.append("")
    L.append("| " + T("组 | 实例 | 仅容量的 ECU 占比 | 仅隐私的 ECU 占比 | 两者都有的 ECU 占比 | 实例：全由容量 | 实例：全由隐私 | 实例：混合",
                       "Group | Instances | ECUs capacity only | ECUs privacy only | ECUs both | Instances: all capacity | "
                       "Instances: all privacy | Instances: mixed") + " |")
    L.append("|---" * 8 + "|")
    for s, l in keys:
        g = groups[(s, l)]
        ne = sum(r["N"] for r in g)
        allc = sum(r["conf_only"] == 0 and r["both"] == 0 for r in g)
        allf = sum(r["cap_only"] == 0 and r["both"] == 0 for r in g)
        L.append(f"| {name(s, l)} | {len(g)} | {pct(sum(r['cap_only'] for r in g) / ne)} | {pct(sum(r['conf_only'] for r in g) / ne)} | "
                 f"{pct(sum(r['both'] for r in g) / ne)} | {allc} / {len(g)} | {allf} / {len(g)} | {len(g) - allc - allf} / {len(g)} |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-mail", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    runs = mask_runs()
    todo = [r for r in runs if r["res"][r["res"]["algo"]].get("exit_rate", 0) > 0]
    meta = defaultdict(lambda: [0, 0])
    for r in runs:
        meta[(r["scen"], r["learner"])][1] += r["res"]["test_count"]
    print(f"{len(todo)} runs with EXITs to replay", flush=True)
    with Pool(a.workers) as pool:
        tasks, info = [], []
        for run, out, test in pool.imap_unordered(replay, todo):
            want = round(run["res"][run["res"]["algo"]]["exit_rate"] * run["res"]["test_count"])
            assert len(out) == want, f"{run['scen']} {run['learner']} seed {run['seed']}: replay {len(out)} EXITs, results.json {want}"
            meta[(run["scen"], run["learner"])][0] += len(out)
            print(f"  {run['scen']} {run['learner']} seed {run['seed']}: {len(out)} EXITs (matches results.json)", flush=True)
            for e in out:
                tasks.append((test[e["k"]], e["steps"]))
                info.append(dict(scen=run["scen"], learner=run["learner"], seed=run["seed"], k=e["k"]))
        print(f"{len(tasks)} EXIT instances to analyse", flush=True)
        rows = [{**i, **r} for i, r in zip(info, pool.map(analyse, tasks, chunksize=1))]
    minutes = (time.time() - t0) / 60
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exit_diag.json").write_text(json.dumps(rows, indent=1))
    meta = {k: tuple(v) for k, v in meta.items()}
    (OUT / "exit_diag.md").write_text(build(rows, meta, "zh", minutes))
    (OUT / "exit_diag.en.md").write_text(build(rows, meta, "en", minutes))
    print(OUT / "exit_diag.md", f"({minutes:.1f} min)", flush=True)
    if not a.no_mail:
        from shared.report_mail import send_report
        send_report("v6.0.2", OUT / "exit_diag.md")


if __name__ == "__main__":
    from shared.batch_manifest import nontrain_batch      # docs/data_spec.md §6: listed by the v2 panel
    with nontrain_batch("v6.0.2", "diag", __file__,
                        {"zh": "v6.0.2：Mask EXIT 归因诊断（v6.0.0 Maskable 最终模型，测试集确定性回放；ILP 判定实例从哪一步开始无解）",
                         "en": "v6.0.2: Mask EXIT attribution (v6.0.0 Maskable final models, deterministic test replay; the step at which the ILP finds the instance infeasible)"},
                        depends_on=['v6.0.0'], report_dir="paper_contents/v6.0.2"):
        main()
