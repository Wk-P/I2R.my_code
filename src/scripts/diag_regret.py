#!/usr/bin/env python3
"""Diagnostic: per-step regret of a trained Mask policy, measured with the ILP (no training).

V*(s_t) = best final AR reachable from the prefix s_t (placements made so far pinned, the rest
solved optimally by solve_ilp_max_ar(fixed=...)); None if no feasible completion exists.
Regret(s_t, a) = V*(s_t) - V*(T(s_t, a)) for every legal ECU a (V*(s_t) = max_a V*(T(s_t, a)),
since the next placement of any feasible completion is legal). Along the policy's own
trajectory the regrets telescope: sum_t Regret(s_t, a_t) = ILP AR - AR on completed instances.
Empty ECUs of equal capacity are interchangeable, so they are solved once. EXIT instances are
reported separately: t_first_infeasible = first step whose action leaves no feasible completion.

    .venv/bin/python src/scripts/diag_regret.py [--n 400]   -> paper_contents/v4.4.2/regret_diag.md
"""
import argparse
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results" / "unified"
REPORT = ROOT / "paper_contents" / "v4.4.2" / "regret_diag.md"
SCENARIOS = ["lt", "eq", "gt"]
DATA, MANIFEST, ALGO, SEED = "v4.3.1.4", "logs/v4.3.8_pilot/manifest.json", "mask_ppo", 1
EPS = 1e-4


def test_split(scen):
    from paper_rl.data import load
    inst = load(scen, DATA)["instances"]
    idx = list(range(len(inst)))
    random.Random(SEED).shuffle(idx)                   # same as paper_rl.train.split_instances
    return [inst[i] for i in idx[int(0.8 * len(inst)):]]


def rollout(args):
    """Replay the policy on test instances; per step: action, legal ECUs, empty flags, open kind."""
    import torch
    torch.set_num_threads(1)
    from paper_rl.env import PlacementEnv
    from paper_rl.policy_io import env_order, load_policy
    scen, exp_id, ks = args
    test = test_split(scen)
    predict, obs_mode = load_policy(scen, ALGO, exp_id, "model_*", len(test[0]["ECUs"]), len(test[0]["SVCs"]))
    env = PlacementEnv(test, "mask", "ar_pen", full_episode=True, exit_action=True, obs_mode=obs_mode,
                       **env_order(scen, ALGO, exp_id))
    out = []
    for k in ks:
        env.use_instance(k)
        obs, _ = env.reset()
        order = env.order                                      # v4.4.7: the run's own service order
        steps, done = [], False
        while not done:
            mask = env.action_masks()
            a = predict(obs, mask)
            if a == env.N:                                     # EXIT
                obs, _, done, _, info = env.step(a)
                break
            active = [j for j in range(env.N) if env.hosted[j]]
            feas = env._feasible(env.t)
            kind = None
            if a not in active and active:
                kind = "voluntary" if any(feas[j] for j in active) else "forced"
            steps.append({"svc": order[env.t], "a": a, "legal": [int(j) for j in np.flatnonzero(mask[:env.N])],
                          "empty": [not env.hosted[j] for j in range(env.N)], "open": kind})
            obs, _, done, _, info = env.step(a)
        out.append({"k": k, "steps": steps, "exited": bool(info["exited"]), "ar": float(info["ar"]),
                    "ilp": float(test[k]["ar_star"]), "M": env.M})
    return scen, out


def values(args):
    """V*(T(s_t, a)) for every legal a at one step (equal-capacity empty ECUs solved once)."""
    from shared.ilp_utils import solve_ilp_max_ar
    scen, k, t, prefix, step = args
    x = test_split_cache(scen)[k]
    caps = x["ECUs"]
    vals, seen = {}, {}
    for a in step["legal"]:
        key = ("empty", caps[a]) if step["empty"][a] else ("ecu", a)
        if key not in seen:
            r = solve_ilp_max_ar(caps, x["SVCs"], x["conflict_sets"], fixed={**prefix, step["svc"]: a})
            seen[key] = float(r["avg_utilization"]) if r["status"] == "Optimal" else None
        vals[a] = seen[key]
    return scen, k, t, vals


_CACHE = {}


def test_split_cache(scen):
    if scen not in _CACHE:
        _CACHE[scen] = test_split(scen)
    return _CACHE[scen]


def main():
    global SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=48)
    # v4.4.3: any Mask PPO run (manifest) or the supervised reference (--ref <dir with attn_<scen>_50000.pt>)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--key", default=ALGO, help="manifest key of the run (v4.4.3: mask_ppo_mlp = MLP control)")
    ap.add_argument("--label", default="v4.3.8 试跑的从零训练 Mask PPO（MLP，种子 1、1M 步，p = 0.6 数据）")
    ap.add_argument("--out", default=str(REPORT))
    ap.add_argument("--label-en", default=None, help="English model label for the .en.md report (default: built from --key / --manifest)")
    ap.add_argument("--seed", type=int, default=SEED, help="v4.4.5: training seed of the run (selects its test split)")
    a = ap.parse_args()
    SEED = a.seed                                      # set before the Pool forks, so the workers see it
    t0 = time.time()
    if a.ref:
        ids = {s: {ALGO: f"ref:{Path(a.ref) / f'attn_{s}_50000.pt'}"} for s in SCENARIOS}
    else:
        ids = {s: {ALGO: d[a.key]} for s, d in json.loads((ROOT / a.manifest).read_text())["exp_ids"][str(SEED)].items()}
    chunks = [(s, ids[s][ALGO], list(range(c, min(c + 25, a.n)))) for s in SCENARIOS for c in range(0, a.n, 25)]
    with Pool(a.workers) as pool:
        traj = {}
        for s, out in pool.map(rollout, chunks):
            for e in out:
                traj[(s, e["k"])] = e
        tasks = []
        for (s, k), e in traj.items():
            prefix = {}
            for t, st in enumerate(e["steps"]):
                tasks.append((s, k, t, dict(prefix), st))
                prefix[st["svc"]] = st["a"]
        print(f"{len(traj)} trajectories, {len(tasks)} steps to solve", flush=True)
        for s, k, t, vals in pool.imap_unordered(values, tasks, chunksize=4):
            traj[(s, k)]["steps"][t]["vals"] = vals
    print(f"solved in {(time.time() - t0) / 60:.1f} min", flush=True)

    pc = lambda v: f"{100 * v:.1f}%"
    f4 = lambda v: f"{v:.4f}"
    zh, en = [], []                                     # v4.4.11: every report in Chinese and English

    def L(z, e=None):                                   # one line in both languages (e = z if identical)
        zh.append(z)
        en.append(z if e is None else e)

    label_en = a.label_en or f"model '{a.key}' of {a.manifest if not a.ref else a.ref}, seed {SEED}"
    L("# 逐步 regret 分解（用 ILP 测量，不训练）", "# Per-step regret decomposition (measured with the ILP, no training)")
    L("")
    L(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/diag_regret.py`；耗时 {(time.time() - t0) / 60:.1f} 分钟。",
      f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; script `src/scripts/diag_regret.py`; took {(time.time() - t0) / 60:.1f} min.")
    L(f"- 模型：{a.label}，在种子 {SEED} 测试集的前 {a.n} 个实例上确定性回放。",
      f"- Model: {label_en}, replayed deterministically on the first {a.n} instances of the seed-{SEED} test split.")
    L("- V*(s_t) = 固定已做的放置、其余由 ILP（Dinkelbach）最优完成时能达到的最终 AR；Regret(s_t, a) = V*(s_t) − V*(执行 a 之后)。"
      "沿策略自己的轨迹逐步相加，恰好等于该实例的绝对 gap（ILP AR − AR）。",
      "- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP "
      "(Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's "
      "absolute gap (ILP AR − AR).")
    L(f"- 最优动作 = Regret ≤ {EPS:g} 的合法动作（容差防 ILP 数值误差）。同容量的空 ECU 互换等价，只求一次但各算一个动作。",
      f"- Optimal action = a legal action with Regret ≤ {EPS:g} (tolerance against ILP numerical error). Empty ECUs of equal "
      "capacity are interchangeable: solved once, but each counts as an action.")
    L("- EXIT 实例单独统计：第一次把状态推进到「ILP 已无可行完成」的步号。",
      "- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.")
    L("")
    summary = {}
    for s in SCENARIOS:
        ok, ex = [], []
        for (sc, k), e in sorted(traj.items()):
            if sc != s:
                continue
            M = e["M"]
            regs, nopt, nleg, picked, inf_t = [], [], [], [], None
            v_prev = max(v for v in e["steps"][0]["vals"].values() if v is not None)
            for t, st in enumerate(e["steps"]):
                vals = st["vals"]
                feas_v = [v for v in vals.values() if v is not None]
                vs = max(feas_v) if feas_v else None
                va = vals[st["a"]]
                if vs is None or va is None:
                    inf_t = t if inf_t is None else inf_t
                    break
                r = vs - va
                regs.append(r)
                nleg.append(len(vals))
                nopt.append(sum(v is not None and vs - v <= EPS for v in vals.values()))
                picked.append(r <= EPS)
            rec = {"regs": regs, "nopt": nopt, "nleg": nleg, "picked": picked, "M": M, "inf_t": inf_t,
                   "gap": e["ilp"] - e["ar"], "v0": v_prev,
                   "forced": [t for t, st in enumerate(e["steps"]) if st["open"] == "forced"]}
            (ex if e["exited"] or inf_t is not None else ok).append(rec)
        M = ok[0]["M"] if ok else ex[0]["M"]
        tele = max(abs(sum(r["regs"]) - r["gap"]) for r in ok) if ok else float("nan")
        all_regs = [x for r in ok for x in r["regs"]]
        tot = sum(sum(r["regs"]) for r in ok)
        summary[s] = {"opt_rate": float(np.mean([p for r in ok for p in r["picked"]])),
                      "opt_rate_t": [float(np.mean([r["picked"][t] for r in ok if len(r["picked"]) > t])) for t in range(M)],
                      # v4.4.4: mean absolute regret per step (sums to the mean gap over completed instances)
                      "regret_t": [float(np.mean([r["regs"][t] for r in ok if len(r["regs"]) > t])) for t in range(M)],
                      "regret_share_t": [sum(r["regs"][t] for r in ok if len(r["regs"]) > t) / tot if tot else 0.0 for t in range(M)],
                      "n_ok": len(ok), "n_ex": len(ex)}
        L(f"## {s.upper()}（M = {M}）", f"## {s.upper()} (M = {M})")
        L("")
        L(f"完成的实例 {len(ok)} 个，EXIT / 中途无可行完成 {len(ex)} 个。逐步 regret 之和与绝对 gap 的最大偏差 {tele:.1e}（核对 telescoping）。",
          f"{len(ok)} completed instances, {len(ex)} with EXIT / no feasible completion on the way. Max deviation between the sum of "
          f"per-step regrets and the absolute gap: {tele:.1e} (telescoping check).")
        L("")
        L("| 指标 | 值 |", "| Metric | Value |")
        L("|---|---|")
        L(f"| 平均绝对 gap（= 平均 Σ regret） | {f4(np.mean([r['gap'] for r in ok]))} |",
          f"| Mean absolute gap (= mean Σ regret) | {f4(np.mean([r['gap'] for r in ok]))} |")
        L(f"| PPO 每步 regret：均值 / 中位数 | {f4(np.mean(all_regs))} / {f4(np.median(all_regs))} |",
          f"| PPO regret per step: mean / median | {f4(np.mean(all_regs))} / {f4(np.median(all_regs))} |")
        L(f"| PPO 选中最优动作的比例 | {pc(np.mean([p for r in ok for p in r['picked']]))} |",
          f"| Share of steps where PPO picks an optimal action | {pc(np.mean([p for r in ok for p in r['picked']]))} |")
        nl, no = np.mean([x for r in ok for x in r['nleg']]), np.mean([x for r in ok for x in r['nopt']])
        L(f"| 每步合法动作数 / 其中最优动作数（均值） | {nl:.2f} / {no:.2f} |",
          f"| Legal actions / optimal actions per step (mean) | {nl:.2f} / {no:.2f} |")
        L(f"| 最优动作唯一的步所占比例 | {pc(np.mean([x == 1 for r in ok for x in r['nopt']]))} |",
          f"| Share of steps with a unique optimal action | {pc(np.mean([x == 1 for r in ok for x in r['nopt']]))} |")
        nr = np.mean([sum(x > EPS for x in r['regs']) for r in ok])
        L(f"| 每个实例有 regret 的步数（均值） | {nr:.2f} |", f"| Steps with regret per instance (mean) | {nr:.2f} |")
        thirds = [(0, M // 3), (M // 3, 2 * M // 3), (2 * M // 3, M)]
        for lo, hi in thirds:
            part = sum(sum(r["regs"][lo:hi]) for r in ok)
            v = pc(part / tot) if tot else "—"
            L(f"| 第 {lo + 1}–{hi} 步贡献的 gap 占比 | {v} |", f"| Share of the gap from steps {lo + 1}–{hi} | {v} |")
        # regret steps vs later forced opens
        before = [x for r in ok for t, x in enumerate(r["regs"]) if x > EPS and any(f > t for f in r["forced"])]
        at_f = [x for r in ok for t, x in enumerate(r["regs"]) if x > EPS and t in r["forced"]]
        lag = [min(f for f in r["forced"] if f > t) - t for r in ok for t, x in enumerate(r["regs"])
               if x > EPS and any(f > t for f in r["forced"])]
        rsum = sum(x for r in ok for x in r["regs"] if x > EPS)
        vb, va_ = (pc(sum(before) / rsum), pc(sum(at_f) / rsum)) if rsum else ("—", "—")
        L(f"| 有 regret 的步之后还会出现被逼开启：regret 占比 | {vb} |",
          f"| Regret at steps followed by a forced opening: share of regret | {vb} |")
        L(f"| 有 regret 的步本身就是被逼开启：regret 占比 | {va_} |",
          f"| Regret at steps that are themselves forced openings: share of regret | {va_} |")
        vl = f"{np.mean(lag):.2f}" if lag else "—"
        L(f"| 有 regret 的步到下一次被逼开启的步数（均值） | {vl} |",
          f"| Steps from a regret step to the next forced opening (mean) | {vl} |")
        if ex:
            it = [r["inf_t"] for r in ex if r["inf_t"] is not None]
            if it:
                L(f"| EXIT 实例：第一次无可行完成的步号（1 起，均值 / 中位数） | {np.mean(it) + 1:.2f} / {np.median(it) + 1:.0f}（{len(it)} 个） |",
                  f"| EXIT instances: first step with no feasible completion (1-based, mean / median) | "
                  f"{np.mean(it) + 1:.2f} / {np.median(it) + 1:.0f} ({len(it)} instances) |")
            else:
                L("| EXIT 实例 | — |", "| EXIT instances | — |")
        L("")
        L("按步号：", "By step:")
        L("")
        L("| 步 | 平均 regret | gap 占比 | 有 regret 的实例比例 | 合法动作数 | 最优动作数 | PPO 选中最优 |",
          "| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |")
        L("|---|---|---|---|---|---|---|")
        for t in range(M):
            rs = [r["regs"][t] for r in ok if len(r["regs"]) > t]
            if not rs:
                continue
            L(f"| {t + 1} | {f4(np.mean(rs))} | {pc(sum(rs) / tot) if tot else '—'} | {pc(np.mean([x > EPS for x in rs]))} | "
              f"{np.mean([r['nleg'][t] for r in ok]):.2f} | {np.mean([r['nopt'][t] for r in ok]):.2f} | "
              f"{pc(np.mean([r['picked'][t] for r in ok]))} |")
        L("")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(zh))
    out.with_suffix(".en.md").write_text("\n".join(en))
    out.with_suffix(".json").write_text(json.dumps(summary))
    print(f"report -> {out} (+ .en.md)")


if __name__ == "__main__":
    main()
