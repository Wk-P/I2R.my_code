#!/usr/bin/env python3
"""Diagnostic: how does a trained Mask policy open ECUs? (no training, no greedy)

Replays saved models on their seed-1 test split (deterministic, as in the reports) and
classifies every step that puts a service on a not-yet-active ECU:
  first     -- t = 0, no ECU active yet;
  voluntary -- some active ECU could legally take the service, the policy opened a new one;
  forced    -- no active ECU could take it legally.
For every open: u = d_i / c_j of the chosen ECU and u* = max d_i / c_k over the legal inactive
ECUs (best capacity match available), du = u* - u >= 0.
Per instance (completed without EXIT): active ECUs vs ILP, absolute / relative AR gap,
number of voluntary opens, sum of du; Spearman correlations with the relative gap.

    .venv/bin/python scripts/diag_open_ecu.py   -> paper_contents/v4.4.1/open_diag.md
"""
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results" / "unified"
REPORT = ROOT / "paper_contents" / "v4.4.1" / "open_diag.md"
SCENARIOS = ["lt", "eq", "gt"]
# (label, data version, manifest, algo)
MODELS = [("v4.3.8 Mask PPO（p = 0.6）", "v4.3.1.4", "scripts/logs/v4.3.8_pilot/manifest.json", "mask_ppo"),
          ("v4.3.8 Mask DQN（p = 0.6）", "v4.3.1.4", "scripts/logs/v4.3.8_pilot/manifest.json", "mask_dqn"),
          ("v4.4.1 Mask PPO（随机密度）", "v4.4.1", "scripts/logs/v4.4.1_pilot/manifest.json", "mask_ppo")]
SEED = 1


def test_split(scen, version, seed):
    from paper_rl.data import load
    inst = load(scen, version)["instances"]
    idx = list(range(len(inst)))
    random.Random(seed).shuffle(idx)                   # same as paper_rl.train.split_instances
    return [inst[i] for i in idx[int(0.8 * len(inst)):]]


def replay(job):
    import torch
    torch.set_num_threads(1)
    from paper_rl.env import PlacementEnv
    from paper_rl.train import model_class, split_algo
    label, version, scen, algo, exp_id = job
    run_dir = RESULTS / scen / algo / exp_id
    mech, learner = split_algo(algo)
    model = model_class(learner, mech).load(str(next(run_dir.glob("model_*"))), device="cpu")
    test = test_split(scen, version, SEED)
    env = PlacementEnv(test, mech, "ar_pen", full_episode=True, exit_action=True)
    opens, insts = [], []
    for k in range(len(test)):
        env.use_instance(k)
        obs, _ = env.reset()
        done, n_vol, sum_du = False, 0, 0.0
        while not done:
            mask = env.action_masks()
            if learner == "ppo":
                a, _ = model.predict(obs, deterministic=True, action_masks=mask)
            else:
                a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            if a < env.N:
                i = env.t
                active = [j for j in range(env.N) if env.hosted[j]]
                if a not in active:
                    feas = env._feasible(i)
                    inactive_ok = [j for j in range(env.N) if feas[j] and j not in active]
                    u = float(env.req[i] / env.cap[a])
                    u_best = max(float(env.req[i] / env.cap[j]) for j in inactive_ok)
                    kind = "first" if not active else ("voluntary" if any(feas[j] for j in active) else "forced")
                    # forced: did some active ECU have room but a privacy conflict?
                    room = any(env.remaining[j] >= env.req[i] for j in active)
                    opens.append({"kind": kind, "u": u, "u_best": u_best, "du": u_best - u, "room": room})
                    n_vol += kind == "voluntary"
                    sum_du += u_best - u
            obs, _, done, _, info = env.step(a)
        if not info["exited"]:
            ilp = test[k]["ar_star"]
            insts.append({"ar": info["ar"], "ilp": ilp, "gap": ilp - info["ar"], "rel": (ilp - info["ar"]) / ilp,
                          "act": len([j for j in range(env.N) if env.hosted[j]]), "ilp_act": test[k]["ilp_active_ecus"],
                          "n_vol": n_vol, "sum_du": sum_du})
    return job, opens, insts, len(test)


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(set(x)) < 2 or len(set(y)) < 2:
        return "—"
    r, p = spearmanr(x, y)
    return f"{r:+.2f}（p={p:.1g}）"


def main():
    jobs = []
    for label, version, man, algo in MODELS:
        ids = json.loads((ROOT / man).read_text())["exp_ids"][str(SEED)]
        jobs += [(label, version, s, algo, ids[s][algo]) for s in SCENARIOS if algo in ids.get(s, {})]
    with Pool(len(jobs)) as pool:
        out = pool.map(replay, jobs)
    f4 = lambda v: f"{v:.4f}"
    pc = lambda v: f"{100 * v:.1f}%"
    lines = ["# 开启 ECU 诊断（回放已有模型，不训练）", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `scripts/diag_open_ecu.py`。",
             "- 模型：v4.3.8 试跑的 Mask PPO / Mask DQN（p = 0.6 数据）、v4.4.1 试跑的 Mask PPO（随机密度数据）；均为种子 1、1M 步，"
             "在各自种子 1 的测试集（400 个实例）上确定性回放。",
             "- 「开启」= 把服务放到一个还没有服务的 ECU 上。**首次**：t = 0，还没有任何 ECU 开启；**主动**：已开启的 ECU 中至少有一个能合法放下该服务，"
             "策略却开了新的；**被逼**：已开启的 ECU 都放不下（容量或冲突）。",
             "- 被逼开启再分两类：**有空间但冲突** = 至少一个已开启 ECU 剩余容量够，但与其上服务有 privacy 冲突；**全都容量不足** = 所有已开启 ECU 剩余容量都不够。",
             "- u = d_i / c_j（所开 ECU 的利用率贡献）；u* = 当时所有合法、未开启 ECU 中 d_i / c_k 的最大值（容量最匹配的选择）；Δu = u* − u ≥ 0。",
             "- 实例级相对 gap 为逐实例 (ILP AR − AR) / ILP AR 的平均，与试跑报告（均值之比）略有差别。",
             "- 实例级指标只在未 EXIT 的实例上算；开启 ECU 数多于 ILP = 最终开启数 − ILP 最优解开启数。相关性为 Spearman 秩相关。", ""]
    for label, version, man, algo in MODELS:
        lines += [f"## {label}", "",
                  "| 场景 | 开启总次数（不含首次） | 主动 | 被逼 | 被逼中：有空间但冲突 | 被逼中：全都容量不足 | 被逼时 u | 被逼时 u* | 被逼时 Δu | 首次 Δu | 主动时 Δu |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        rows2 = []
        for (lb, _, scen, _, _), opens, insts, n in out:
            if lb != label:
                continue
            rest = [o for o in opens if o["kind"] != "first"]
            vol = [o for o in rest if o["kind"] == "voluntary"]
            frc = [o for o in rest if o["kind"] == "forced"]
            fst = [o for o in opens if o["kind"] == "first"]
            m = lambda xs, key: f4(np.mean([o[key] for o in xs])) if xs else "—"
            lines.append(f"| {scen.upper()} | {len(rest)} | {len(vol)}（{pc(len(vol) / max(len(rest), 1))}） | "
                         f"{len(frc)}（{pc(len(frc) / max(len(rest), 1))}） | {pc(np.mean([o['room'] for o in frc]))} | "
                         f"{pc(1 - np.mean([o['room'] for o in frc]))} | {m(frc, 'u')} | {m(frc, 'u_best')} | {m(frc, 'du')} | "
                         f"{m(fst, 'du')} | {m(vol, 'du')} |")
            if insts:
                g = lambda key: [x[key] for x in insts]
                extra = np.array(g("act")) - np.array(g("ilp_act"))
                rows2.append(f"| {scen.upper()} | {len(insts)}/{n} | {np.mean(g('act')):.2f} | {np.mean(g('ilp_act')):.2f} | "
                             f"{np.mean(extra):+.2f} | {pc(np.mean(np.array(g('n_vol')) > 0))} | {f4(np.mean(g('gap')))} | "
                             f"{pc(np.mean(g('rel')))} | {spearman(g('rel'), g('n_vol'))} | {spearman(g('rel'), g('sum_du'))} | "
                             f"{spearman(g('rel'), list(extra))} |")
        lines += ["", "| 场景 | 未 EXIT 实例 | 开启 ECU 数 | ILP 开启数 | 多于 ILP | 有主动开启的实例 | 绝对 gap | 相对 gap | "
                      "相对 gap ~ 主动开启次数 | 相对 gap ~ ΣΔu | 相对 gap ~ 多开数 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"] + rows2 + [""]
    REPORT.write_text("\n".join(lines))
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    main()
