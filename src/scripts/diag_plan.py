#!/usr/bin/env python3
"""Diagnostic (v4.4.12): what the global plan P does along the plan-conditioned policy's own trajectories.

For a model trained with --plan, the ECU logits are P[i_t, j] + dl_j(s_t). Along the deterministic
trajectory, every step records: std over the legal ECUs of P[i_t, .] and of dl (per-step score alone),
the action taken (argmax of P + dl) and the action dl alone would take, and -- with the ILP, as in
diag_regret.py -- V*(T(s_t, a)) for every legal a, hence the regret of both actions at the same state.
Reported by step and by phase: per-step regret, std ratio, share of steps where P changes the argmax, and
on those steps the regret of P's choice vs dl's choice (does P help or hurt, early vs late?).

    .venv/bin/python src/scripts/diag_plan.py --scen lt --seeds 1,2,3 --out paper_contents/v4.4.12/plan_diag.md
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
sys.path.insert(0, str(ROOT / "src" / "scripts"))
import diag_regret as DR                                      # noqa: E402

MANIFEST = ROOT / "logs" / "v4.4.12" / "manifest.json"
KEY, EPS = "mask_ppo_plan", 1e-4


def rollout(args):
    import torch
    torch.set_num_threads(1)
    from paper_rl.env import PlacementEnv
    from paper_rl.graph_net import decode_raw
    from paper_rl.train import model_class
    scen, exp_id, ks = args
    test = DR.test_split(scen)
    model = model_class("ppo", "mask").load(str(next((DR.RESULTS / scen / "mask_ppo" / exp_id).glob("model_*"))), device="cpu")
    net = model.policy.gnet.eval()
    env = PlacementEnv(test, "mask", "ar_pen", full_episode=True, exit_action=True, obs_mode="raw")
    out = []
    for k in ks:
        env.use_instance(k)
        obs, _ = env.reset()
        steps, done = [], False
        while not done:
            mask = env.action_masks()
            legal = [int(j) for j in np.flatnonzero(mask[:env.N])]
            x = decode_raw(torch.as_tensor(obs, dtype=torch.float32)[None], env.N, env.M)
            with torch.no_grad():
                logits = net(*x)[0][0].numpy()
                P = net.plan_enc(x[0], x[1], x[2])[0, min(env.t, env.M - 1)].numpy()
            if not legal:                                       # only EXIT left
                obs, _, done, _, _ = env.step(env.N)
                break
            dl = logits[:env.N] - P
            a_full = legal[int(np.argmax(logits[legal]))]
            a_dyn = legal[int(np.argmax(dl[legal]))]
            steps.append({"svc": env.order[env.t], "a": a_full, "a_dyn": a_dyn, "legal": legal,
                          "empty": [not env.hosted[j] for j in range(env.N)],
                          "std_p": float(np.std(P[legal])), "std_dl": float(np.std(dl[legal]))})
            obs, _, done, _, info = env.step(a_full)
        out.append({"k": k, "steps": steps, "exited": bool(env.exited)})
    return scen, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", default="lt")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=str(ROOT / "paper_contents" / "v4.4.12" / "plan_diag.md"))
    a = ap.parse_args()
    ids = json.loads(MANIFEST.read_text())["exp_ids"]
    t0, per_seed = time.time(), {}
    for sd in [int(x) for x in a.seeds.split(",")]:
        DR.SEED = sd                                            # test split of this seed (set before forking)
        DR._CACHE.clear()
        eid = ids[str(sd)][a.scen][KEY]
        with Pool(a.workers) as pool:
            traj = {}
            for s, out in pool.map(rollout, [(a.scen, eid, list(range(c, min(c + 25, a.n)))) for c in range(0, a.n, 25)]):
                for e in out:
                    traj[e["k"]] = e
            tasks = []
            for k, e in traj.items():
                prefix = {}
                for t, st in enumerate(e["steps"]):
                    tasks.append((a.scen, k, t, dict(prefix), st))
                    prefix[st["svc"]] = st["a"]
            for i, (s, k, t, vals) in enumerate(pool.imap_unordered(DR.values, tasks, chunksize=4), 1):
                traj[k]["steps"][t]["vals"] = vals
                if i % max(1, len(tasks) // 20) == 0:
                    print(f"[plan-diag] seed {sd}: solved {i}/{len(tasks)} ({100 * i / len(tasks):.0f}%)", flush=True)
        rows = []                                               # one row per step with a feasible completion
        for k, e in traj.items():
            for t, st in enumerate(e["steps"]):
                feas = [v for v in st["vals"].values() if v is not None]
                if not feas or st["vals"][st["a"]] is None:
                    break
                vs = max(feas)
                rd = st["vals"][st["a_dyn"]]
                rows.append({"t": t, "reg": vs - st["vals"][st["a"]], "reg_dyn": None if rd is None else vs - rd,
                             "changed": st["a"] != st["a_dyn"], "std_p": st["std_p"], "std_dl": st["std_dl"]})
        per_seed[sd] = {"eid": eid, "rows": rows, "n_inst": len(traj), "n_exit": sum(e["exited"] for e in traj.values())}
        print(f"seed {sd} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    write(a, per_seed, time.time() - t0)


def write(a, per_seed, secs):
    M = max(r["t"] for d in per_seed.values() for r in d["rows"]) + 1
    phases = [(0, 5), (5, 10), (10, M)] if M > 10 else [(0, 3), (3, 6), (6, M)]
    pc = lambda v: f"{100 * v:.1f}%"
    f4 = lambda v: f"{v:.4f}"

    def stats(rows):
        ch = [r for r in rows if r["changed"] and r["reg_dyn"] is not None]
        ratio = [r["std_p"] / r["std_dl"] for r in rows if r["std_dl"] > 1e-9]
        better = sum(r["reg"] < r["reg_dyn"] - EPS for r in ch)
        worse = sum(r["reg"] > r["reg_dyn"] + EPS for r in ch)
        return {"n": len(rows), "reg": np.mean([r["reg"] for r in rows]) if rows else np.nan,
                "ratio": np.median(ratio) if ratio else np.nan, "chg": len(ch) / len(rows) if rows else np.nan,
                "n_ch": len(ch), "reg_p": np.mean([r["reg"] for r in ch]) if ch else np.nan,
                "reg_d": np.mean([r["reg_dyn"] for r in ch]) if ch else np.nan,
                "better": better / len(ch) if ch else np.nan, "worse": worse / len(ch) if ch else np.nan,
                "net": sum(r["reg_dyn"] - r["reg"] for r in ch) / len(rows) if rows else np.nan}

    out = {}
    for lang in ("zh", "en"):
        zh = lang == "zh"
        T = lambda z, e: z if zh else e
        L = [T(f"# 全局计划 P 的作用诊断（{a.scen.upper()}，v4.4.12 模型）", f"# What the global plan P does ({a.scen.upper()}, v4.4.12 models)"), "",
             T(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/diag_plan.py`；耗时 {secs / 60:.1f} 分钟。",
               f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; script `src/scripts/diag_plan.py`; took {secs / 60:.1f} min."),
             T("- 沿计划策略自己的确定性轨迹（各种子测试集前 400 个实例），每一步 ECU logit = P[i_t, j] + Δℓ_j(s_t)。"
               "记录合法 ECU 上 P 行与 Δℓ 的标准差；实际动作 = argmax(P + Δℓ)，对照动作 = argmax(Δℓ)（只用逐步打分时会选的动作）。",
               "- Along the plan-conditioned policy's own deterministic trajectories (first 400 instances of each seed's test split); at every "
               "step the ECU logit is P[i_t, j] + Δℓ_j(s_t). Recorded: std of the P row and of Δℓ over the legal ECUs; action taken = "
               "argmax(P + Δℓ), counterfactual = argmax(Δℓ) (what the per-step score alone would pick)."),
             T("- 两个动作的 regret 在同一个状态上用 ILP 计算（固定已做的放置，其余最优完成，同 `diag_regret.py`），因此是同一状态下的反事实对比。",
               "- Both actions' regrets are computed with the ILP at the same state (placements so far fixed, the rest completed optimally, as in "
               "`diag_regret.py`), i.e. a counterfactual comparison at the same state."),
             T("- 「P 改变动作」= 两个动作不同；其中「P 更好 / 更差」按 regret 差超过 1e−4 判定；「净收益」= Σ(对照 regret − 实际 regret) / 该段步数，"
               "正值表示 P 平均每步减少了多少 regret。", "- \"P changes the action\" = the two actions differ; \"P better / worse\" = regret differs "
               "by more than 1e−4; \"net gain\" = Σ(counterfactual regret − actual regret) / steps in the phase, positive = regret P removes per step."), ""]
        for sd, d in sorted(per_seed.items()):
            L += [T(f"## 种子 {sd}（exp_id {d['eid']}，完成实例 {d['n_inst'] - d['n_exit']} 个，EXIT {d['n_exit']} 个）",
                    f"## Seed {sd} (exp_id {d['eid']}, {d['n_inst'] - d['n_exit']} completed instances, {d['n_exit']} EXIT)"), "",
                  T("| 步 | 步数 | 实际 regret | std(P) / std(Δℓ) 中位数 | P 改变动作 | 改变时：P 的 regret | 改变时：对照 regret | P 更好 | P 更差 | 净收益 / 步（×1e−4） |",
                    "| Steps | n | Actual regret | Median std(P) / std(Δℓ) | P changes action | When changed: P's regret | When changed: counterfactual regret | P better | P worse | Net gain / step (×1e−4) |"),
                  "|---|---|---|---|---|---|---|---|---|---|"]
            groups = [(T(f"第 {lo + 1}–{hi} 步", f"{lo + 1}–{hi}"), lo, hi) for lo, hi in phases] + [(T("全部", "all"), 0, M)]
            for name, lo, hi in groups:
                s = stats([r for r in d["rows"] if lo <= r["t"] < hi])
                L.append(f"| {name} | {s['n']} | {f4(s['reg'])} | {s['ratio']:.2f} | {pc(s['chg'])} | {f4(s['reg_p'])} | {f4(s['reg_d'])} | "
                         f"{pc(s['better'])} | {pc(s['worse'])} | {1e4 * s["net"]:+.2f} |")
            L += ["", T("按步号：", "By step:"), "",
                  T("| 步 | 实际 regret | std(P)/std(Δℓ) | P 改变动作 | P 更好 | P 更差 | 净收益 / 步（×1e−4） |",
                    "| Step | Actual regret | std(P)/std(Δℓ) | P changes action | P better | P worse | Net gain / step (×1e−4) |"), "|---|---|---|---|---|---|---|"]
            for t in range(M):
                s = stats([r for r in d["rows"] if r["t"] == t])
                if s["n"]:
                    L.append(f"| {t + 1} | {f4(s['reg'])} | {s['ratio']:.2f} | {pc(s['chg'])} | {pc(s['better'])} | {pc(s['worse'])} | {1e4 * s["net"]:+.2f} |")
            L.append("")
        out[lang] = L
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out["zh"]))
    p.with_suffix(".en.md").write_text("\n".join(out["en"]))
    print(f"report -> {p} (+ .en.md)", flush=True)


if __name__ == "__main__":
    main()
