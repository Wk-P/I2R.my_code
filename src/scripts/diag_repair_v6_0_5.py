#!/usr/bin/env python3
"""v6.0.5-check-DQN/DDQN: why Repair helps DQN / DDQN (and not PPO) -- evaluation only, CPU, no training.

Repair (env.py) replaces an infeasible action by the best-fit feasible ECU (argmax e_m / c_n). For a value-based
learner every infeasible action is then a "let best-fit place it" macro action. Per step of the deterministic
evaluation on the fixed v6.0.4 test set (5000 ILP-optimal instances per scenario) this records:
  repair rate      share of steps whose chosen action is infeasible and gets repaired (as a share of all steps and
                   of the steps with a feasible ECU), split by the episode third the step falls in;
  best-fit share   share of steps whose EXECUTED ECU equals the best-fit ECU (repaired steps count as best-fit);
  Q margin         DQN / DDQN: max Q over all ECUs - max Q over the feasible ECUs on repaired steps (absolute);
                   PPO: probability mass on infeasible ECUs on the same steps;
  masked gap       the same model with the choice restricted to the feasible ECUs (argmax of the masked Q /
                   logits), so Repair never fires: gap with vs without the best-fit fallback.
Runs: Repair (v6.0.0, 3 seeds); Mask DQN / DDQN (v6.0.0) and Mask PPO (v6.0.0.2) for the best-fit share only.
gap = 1 - mean(AR) / mean(ILP AR), AR over legal placements.

    .venv/bin/python src/scripts/diag_repair_v6_0_5.py [--workers 32]
        -> paper_contents/v6.0.5-check-DQN-DDQN/diag.{json,md,en.md}, mailed
"""
import argparse
import json
import statistics as st
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
VERSION = "v6.0.5-check-DQN/DDQN"
BATCH = "v6.0.5-check-DQN-DDQN"                     # VERSION with "/" -> "-" for paths
OUT = ROOT / "paper_contents" / BATCH
TEST = ROOT / "data/testset_v6.0.4/{scen}.json"
CHUNK = 500
SCENS = ["lt", "gt", "eq"]
LEARNERS = ["ppo", "dqn", "ddqn"]
NAME = {"ppo": "PPO", "dqn": "DQN", "ddqn": "Double DQN", "repair": "Repair", "mask": "Mask"}


def runs():
    out = []
    for src, keep in (("v6.0.0", lambda m, l: m == "repair" or (m == "mask" and l != "ppo")),
                      ("v6.0.0.2", lambda m, l: m == "mask" and l == "ppo")):
        man = json.loads((ROOT / f"logs/{src}/manifest.json").read_text())
        seen = set()
        for r in man["runs"]:
            if not r.get("exp_id") or r["exp_id"] in seen:
                continue
            seen.add(r["exp_id"])
            d = ROOT / "results" / man["results_space"] / r["scenario"] / r["dir"] / r["exp_id"]
            res = json.loads((d / "results.json").read_text())
            if keep(res["mechanism"], res["learner"]):
                out.append(dict(dir=str(d), res=res, src=src))
    return out


def scores(model, learner, obs):
    """Q-values (DQN / DDQN) or logits (PPO) over the actions, as numpy."""
    import torch
    with torch.no_grad():
        t = model.policy.obs_to_tensor(obs)[0]
        if learner == "ppo":
            return model.policy.get_distribution(t).distribution.logits[0].numpy()
        return model.q_net(t)[0].numpy()


def eval_chunk(args):
    import torch
    torch.set_num_threads(1)
    from paper_rl.env import PlacementEnv
    from paper_rl.train import model_class
    run, lo, masked = args
    res, d = run["res"], Path(run["dir"])
    mech, learner = res["mechanism"], res["learner"]
    test = json.loads(Path(str(TEST).format(scen=res["scenario"])).read_text())[lo:lo + CHUNK]
    model = model_class(learner, mech).load(str(next(d.glob("model_*"))), device="cpu")
    env = PlacementEnv(test, mech, res["reward_mode"], lam=0.0, obs_mode=res["obs"], full_episode=res["full_episode"],
                       order=res["order"], order_seed=res["order_seed"], exit_action=res["exit_action"],
                       action_mode=res["action_mode"])
    N = env.N
    out = []
    for k in range(len(test)):
        env.use_instance(k)
        obs, _ = env.reset()
        done, steps = False, []
        while not done:
            m = env.t
            feas = env._feasible(m) if m < env.M else np.zeros(N, bool)
            s = scores(model, learner, obs)
            if mech == "mask" or masked:
                ss = s[:N].copy()
                if feas.any():
                    ss[~feas] = -np.inf
                    a = int(np.argmax(ss))
                else:
                    a = N if res["exit_action"] and mech == "mask" else int(np.argmax(s[:N]))
            else:
                a = int(np.argmax(s))
            bf = int(np.flatnonzero(feas)[np.argmax(env.e[m] / env.c[feas])]) if feas.any() else -1
            rep = bool(a < N and feas.any() and not feas[a])
            st_ = dict(t=m / env.M, feas=bool(feas.any()), rep=rep, bf=bool(feas.any() and (rep or a == bf)))
            if rep:
                if learner == "ppo":
                    p = np.exp(s[:N] - s[:N].max())
                    st_["margin"] = float(p[~feas].sum() / p.sum())
                else:
                    st_["margin"] = float(s[:N].max() - s[:N][feas].max())
            steps.append(st_)
            obs, _, done, _, info = env.step(a)
        out.append(dict(ar=info["ar"], ar_star=info["ar_star"], success=info["valid_placed"] == env.M, steps=steps))
    return res["exp_id"], masked, lo, out


def summarize(ev):
    ar, star = np.array([e["ar"] for e in ev]), np.array([e["ar_star"] for e in ev])
    S = [s for e in ev for s in e["steps"]]
    F = [s for s in S if s["feas"]]
    R = [s for s in S if s["rep"]]
    third = lambda lo, hi: [s for s in F if lo <= s["t"] < hi]
    rate = lambda xs: float(np.mean([s["rep"] for s in xs])) if xs else float("nan")
    return dict(gap=float(1 - ar.mean() / star.mean()), vf=float(np.mean([e["success"] for e in ev])),
                rep_all=rate(S), rep_feas=rate(F), rep_t1=rate(third(0, 1 / 3)), rep_t2=rate(third(1 / 3, 2 / 3)),
                rep_t3=rate(third(2 / 3, 1.01)), bf=float(np.mean([s["bf"] for s in F])) if F else float("nan"),
                margin=float(np.median([s["margin"] for s in R])) if R else float("nan"))


def p(v):
    return "—" if v is None or v != v else f"{100 * v:.2f}%"


def build(rows, lang, minutes):
    zh = lang == "zh"
    T = lambda z, e: z if zh else e
    g = lambda m, l, s, k, key="std": [r[key][k] for r in rows if (r["mech"], r["learner"], r["scen"]) == (m, l, s) and r.get(key)]
    mean = lambda xs: st.mean(xs) if xs else float("nan")
    L = [T(f"# {VERSION}：Repair 为什么帮 DQN / DDQN", f"# {VERSION}: why Repair helps DQN / DDQN"), ""]
    L.append(T(f"- 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}，耗时 {minutes:.1f} 分钟；脚本 `src/scripts/diag_repair_v6_0_5.py`；只评估，不训练。",
               f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}, {minutes:.1f} min; script `src/scripts/diag_repair_v6_0_5.py`; evaluation only, no training."))
    L.append(T("- 测试集：v6.0.4 固定测试集，每场景 5000 个 ILP 最优实例；确定性推理；表内为种子均值。",
               "- Test set: the fixed v6.0.4 test set, 5000 ILP-optimal instances per scenario; deterministic; seed means."))
    L.append(T("- 模型：Repair = v6.0.0（种子 1–3）；Mask DQN/DDQN = v6.0.0；Mask PPO = v6.0.0.2（n_epochs=4）。",
               "- Models: Repair = v6.0.0 (seeds 1–3); Mask DQN/DDQN = v6.0.0; Mask PPO = v6.0.0.2 (n_epochs=4)."))
    L.append(T("- Repair 规则：选到不可行 ECU 时换成可行 ECU 中 argmax e_m / c_n（best-fit）。",
               "- Repair rule: an infeasible choice is replaced by the feasible ECU with argmax e_m / c_n (best-fit)."))
    L.append("")
    L.append(T("## 1. Repair：修正率与关闭修正的反事实", "## 1. Repair: repair rate and the no-repair counterfactual"))
    L.append("")
    L.append(T("- **修正率**：有可行 ECU 的步中，所选动作不可行而被修正的比例；前/中/后 = episode 的三等分。",
               "- **Repair rate**: share of steps with a feasible ECU whose choice is infeasible and repaired; early / mid / late = thirds of the episode."))
    L.append(T("- **best-fit 比例**：有可行 ECU 的步中，实际执行的 ECU 等于 best-fit ECU 的比例（被修正的步计入）。",
               "- **Best-fit share**: share of steps with a feasible ECU whose executed ECU is the best-fit ECU (repaired steps included)."))
    L.append(T("- **Q 差 / 不可行概率**：被修正步上的中位数。DQN/DDQN = 全体 ECU 最大 Q − 可行 ECU 最大 Q（绝对值）；PPO = 不可行 ECU 上的概率总和。",
               "- **Q margin / infeasible mass**: median over repaired steps. DQN/DDQN = max Q over all ECUs − max Q over feasible ECUs (absolute); PPO = probability mass on infeasible ECUs."))
    L.append(T("- **gap（只在可行中选）**：同一模型只在可行 ECU 中取 argmax，Repair 永不触发。",
               "- **gap (feasible only)**: the same model, argmax over the feasible ECUs only, so Repair never fires."))
    L.append("")
    hd = [T("场景", "Scen"), T("算法", "Algo"), "gap (Repair)", T("gap（只在可行中选）", "gap (feasible only)"), T("差", "Δ"),
          T("修正率", "Repair rate"), T("前", "early"), T("中", "mid"), T("后", "late"), T("best-fit 比例", "best-fit share"),
          T("Q 差 / 不可行概率", "Q margin / infeasible mass")]
    L.append("| " + " | ".join(hd) + " |")
    L.append("|---" * len(hd) + "|")
    for s in SCENS:
        for l in LEARNERS:
            a, b = mean(g("repair", l, s, "gap")), mean(g("repair", l, s, "gap", "masked"))
            mg = mean(g("repair", l, s, "margin"))
            mg_s = p(mg) if l == "ppo" else ("—" if mg != mg else f"{mg:.4f}")
            L.append(f"| {s.upper()} | {NAME[l]} | {p(a)} | {p(b)} | {'—' if b != b else f'{100 * (b - a):+.2f}pp'} | "
                     f"{p(mean(g('repair', l, s, 'rep_feas')))} | {p(mean(g('repair', l, s, 'rep_t1')))} | {p(mean(g('repair', l, s, 'rep_t2')))} | "
                     f"{p(mean(g('repair', l, s, 'rep_t3')))} | {p(mean(g('repair', l, s, 'bf')))} | {mg_s} |")
    L.append("")
    L.append(T("## 2. 各机制执行 best-fit 的比例", "## 2. Best-fit share by mechanism"))
    L.append("")
    L.append("| " + T("场景", "Scen") + " | " + " | ".join(f"{NAME[m]} {NAME[l]}" for m in ("mask", "repair") for l in LEARNERS) + " |")
    L.append("|---" * 7 + "|")
    for s in SCENS:
        L.append(f"| {s.upper()} | " + " | ".join(f"{p(mean(g(m, l, s, 'bf')))} (gap {p(mean(g(m, l, s, 'gap')))})"
                                                for m in ("mask", "repair") for l in LEARNERS) + " |")
    L.append("")
    L.append(T("## 3. 逐 run 明细", "## 3. Per run"))
    L.append("")
    L.append("| exp_id | " + " | ".join([T("机制", "Mech"), T("算法", "Algo"), T("场景", "Scen"), T("种子", "Seed"), "gap",
                                         T("gap（只在可行中选）", "gap (feasible only)"), T("修正率", "Repair rate"), "best-fit"]) + " |")
    L.append("|---" * 9 + "|")
    for r in sorted(rows, key=lambda r: (r["mech"], r["scen"], r["learner"], r["seed"])):
        L.append(f"| {r['exp_id']} | {NAME[r['mech']]} | {NAME[r['learner']]} | {r['scen'].upper()} | {r['seed']} | {p(r['std']['gap'])} | "
                 f"{p(r['masked']['gap']) if r.get('masked') else '—'} | {p(r['std']['rep_feas']) if r['mech'] == 'repair' else '—'} | {p(r['std']['bf'])} |")
    L.append("")
    return "\n".join(L)


def main():
    a = ARGS
    t0 = time.time()
    rs = runs()
    tasks = [(r, lo, mk) for r in rs for mk in ((False, True) if r["res"]["mechanism"] == "repair" else (False,))
             for lo in range(0, 5000, CHUNK)]
    print(f"{len(rs)} runs, {len(tasks)} chunks, {a.workers} workers", flush=True)
    parts = {}
    with Pool(a.workers) as pool:
        for i, (eid, mk, lo, ev) in enumerate(pool.imap_unordered(eval_chunk, tasks), 1):
            parts.setdefault((eid, mk), {})[lo] = ev
            if i % 20 == 0 or i == len(tasks):
                print(f"[chunk] {i}/{len(tasks)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    rows = []
    for r in rs:
        res = r["res"]
        cat = lambda mk: [e for lo in sorted(parts[res["exp_id"], mk]) for e in parts[res["exp_id"], mk][lo]]
        rows.append(dict(exp_id=res["exp_id"], src=r["src"], mech=res["mechanism"], learner=res["learner"], scen=res["scenario"],
                         seed=res["seed"], std=summarize(cat(False)),
                         masked=summarize(cat(True)) if (res["exp_id"], True) in parts else None))
    minutes = (time.time() - t0) / 60
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diag.json").write_text(json.dumps(rows, indent=1))
    (OUT / "diag.md").write_text(build(rows, "zh", minutes))
    (OUT / "diag.en.md").write_text(build(rows, "en", minutes))
    print(OUT / "diag.md", f"({minutes:.1f} min)", flush=True)
    if not a.no_mail:
        from shared.report_mail import send_report
        send_report(BATCH, OUT / "diag.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-mail", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 runs x 1 chunk of 20 instances, no report")
    ARGS = ap.parse_args()
    if ARGS.smoke:
        CHUNK = 20
        rs = [r for r in runs() if r["res"]["scenario"] == "gt" and r["res"]["seed"] == 1 and r["res"]["mechanism"] == "repair"]
        for r in rs[:3]:
            for mk in (False, True):
                eid, _, _, ev = eval_chunk((r, 0, mk))
                print(r["res"]["learner"], mk, summarize(ev))
        sys.exit(0)
    from shared.batch_manifest import nontrain_batch      # docs/data_spec.md §6: listed by the v2 panel
    with nontrain_batch(BATCH, "eval", __file__,
                        {"zh": f"{VERSION}：Repair 为何帮 DQN/DDQN——v6.0.0 Repair 模型与 Mask 模型在 v6.0.4 固定测试集上逐步统计修正率、best-fit 比例、Q 差，并对比只在可行 ECU 中选的反事实（只评估，不训练）",
                         "en": f"{VERSION}: why Repair helps DQN/DDQN -- v6.0.0 Repair and Mask models on the fixed v6.0.4 test set: per-step repair rate, best-fit share, Q margin, and the feasible-only counterfactual (evaluation only, no training)"},
                        depends_on=["v6.0.0", "v6.0.0.2", "v6.0.4-testset"], report_dir=f"paper_contents/{BATCH}",
                        inputs=["data/testset_v6.0.4/{scen}.json"],
                        outputs=[f"paper_contents/{BATCH}/diag.json", f"paper_contents/{BATCH}/diag.md", f"paper_contents/{BATCH}/diag.en.md"],
                        logs=[f"logs/{BATCH}/diag.log"]):
        main()
