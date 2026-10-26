#!/usr/bin/env python3
"""v6.0.3: test-set evaluation of the v6 checkpoints (no training).

Every run of the v6.0.0 manifest plus the v6.0.1 Unconstraint runs is evaluated at its 5M / 10M / 15M
checkpoints with paper_rl.train.evaluate on its own 400 test instances (same settings as the final evaluation;
Lagrangian checkpoints with the lambda recorded in ckpt_lambda.json). The 20M point is the final evaluation
already stored in results.json. gap = 1 - AR / ILP AR, AR over legal placements.

    .venv/bin/python src/scripts/eval_ckpt_v6.py   -> paper_contents/v6.0.3/ckpt_eval.{json,md,en.md}, mailed
"""
import argparse
import glob
import json
import statistics as st
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "paper_contents/v6.0.3"
STEPS = [5_000_000, 10_000_000, 15_000_000]
SCENS = ["eq", "gt", "lt"]
LEARNERS = ["ppo", "dqn", "ddqn"]
ROWS = [("mask", "v6.0.0"), ("repair", "v6.0.0"), ("lagrange", "v6.0.0"), ("none", "v6.0.0"), ("none", "v6.0.1")]
NAME = {("mask", "v6.0.0"): "Mask", ("repair", "v6.0.0"): "Repair", ("lagrange", "v6.0.0"): "Lagrangian",
        ("none", "v6.0.0"): "Penalty", ("none", "v6.0.1"): "Unconstraint"}


def runs():
    out, seen = [], set()
    for v in ("v6.0.0", "v6.0.1"):
        for r in json.loads((ROOT / f"logs/{v}/manifest.json").read_text())["runs"]:
            if r["exp_id"] in seen:
                continue
            seen.add(r["exp_id"])
            d = Path(glob.glob(str(ROOT / f"results/unified/{r['scenario']}/{r['dir']}/{r['exp_id']}/results.json"))[0]).parent
            res = json.loads((d / "results.json").read_text())
            out.append(dict(dir=str(d), res=res))
    return out


def summarize(ev, res):
    ars = np.array([e["ar"] for e in ev])
    ilp = res["ilp"]["ar"]
    return dict(ar=float(ars.mean()), gap=float(1 - ars.mean() / ilp),
                cap=float(np.mean([e["cap_v"] > 0 for e in ev])), conf=float(np.mean([e["conf_v"] > 0 for e in ev])),
                exit=float(np.mean([e["exited"] for e in ev])))


def eval_ckpt(args):
    import torch
    torch.set_num_threads(1)
    from paper_rl.train import evaluate, lam_max_arg, model_class, split_instances
    run, step = args
    res, d = run["res"], Path(run["dir"])
    mech, learner = res["mechanism"], res["learner"]
    _, _, test = split_instances(res["scenario"], res["seed"])
    lam, lam_max = 0.0, 1.0
    if mech == "lagrange":
        lams = json.loads((d / "ckpt_lambda.json").read_text())
        lam, lam_max = lams[str(step)], lam_max_arg(str(lams["lambda_max"]))
    model = model_class(learner, mech).load(str(next(d.glob(f"ckpt_{step}_*.zip"))), device="cpu")
    kw = dict(lam_max=lam_max) if mech == "lagrange" else {}
    ev = evaluate(model, test, mech, res["reward_mode"], lam, learner, res["obs"], res["full_episode"], res["exit_action"],
                  res["action_mode"], res["order"], res["order_seed"], **kw)
    return res["exp_id"], step, summarize(ev, res)


def pct(v):
    return f"{100 * v:.3f}%"


def build(rows, lang, minutes):
    zh = lang == "zh"
    T = (lambda z, e_: z if zh else e_)
    L = ["# v6.0.3 " + T("checkpoint 测试评估", "checkpoint test evaluation"), ""]
    L.append(T(f"- 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/eval_ckpt_v6.py`；耗时 {minutes:.1f} 分钟；不训练。",
               f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; script `src/scripts/eval_ckpt_v6.py`; took {minutes:.1f} min; no training."))
    L.append(T("- v6.0.0 全部 run 与 v6.0.1 Unconstraint 的 5M / 10M / 15M checkpoint，在各自种子的 400 个测试实例上确定性评估（与最终评估相同设置；"
               "Lagrangian 用 checkpoint 时记录的 λ）。20M = results.json 中的最终评估。gap = 1 − AR / ILP AR，AR 只计合法放置。",
               "- The 5M / 10M / 15M checkpoints of every v6.0.0 run and the v6.0.1 Unconstraint runs, evaluated deterministically on each "
               "seed's 400 test instances (same settings as the final evaluation; Lagrangian with the λ recorded at the checkpoint). "
               "20M = the final evaluation in results.json. gap = 1 − AR / ILP AR, AR over legal placements."))
    L.append(T("- 表中为种子间均值 ± 标准差。", "- Mean ± std over seeds."))
    L.append("")
    for s in SCENS:
        L.append(f"## {s.upper()}")
        L.append("")
        L.append("| " + T("机制", "Mechanism") + " | " + T("算法", "Algo") + " | 5M | 10M | 15M | 20M | "
                 + T("20M 容量违约", "20M cap viol") + " | " + T("20M 隐私违约", "20M privacy viol") + " | 20M EXIT |")
        L.append("|---" * 9 + "|")
        for m, v in ROWS:
            for l in LEARNERS:
                g = [r for r in rows if (r["mech"], r["version"], r["scen"], r["learner"]) == (m, v, s, l)]
                if not g:
                    continue
                cell = lambda step, k: (lambda xs: pct(st.mean(xs)) + (f" ± {pct(st.stdev(xs))}" if len(xs) > 1 else ""))(
                    [r["points"][str(step)][k] for r in g])
                L.append(f"| {NAME[(m, v)]} | {l.upper()} | " + " | ".join(cell(e, "gap") for e in STEPS + [20_000_000]) + " | "
                         + " | ".join(cell(20_000_000, k) for k in ("cap", "conf", "exit")) + " |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--no-mail", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rs = runs()
    tasks = [(r, s) for r in rs for s in STEPS]
    print(f"{len(rs)} runs, {len(tasks)} checkpoint evaluations", flush=True)
    pts = {}
    with Pool(a.workers) as pool:
        for m, (eid, step, p) in enumerate(pool.imap_unordered(eval_ckpt, tasks), 1):
            pts.setdefault(eid, {})[str(step)] = p
            if m % 20 == 0 or m == len(tasks):
                print(f"[ckpt] {m}/{len(tasks)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    rows = []
    for r in rs:
        res = r["res"]
        al = res[res["algo"]]
        final = dict(ar=al["ar_mean"], gap=1 - al["ar_mean"] / res["ilp"]["ar"], cap=al["cap_viol_rate"],
                     conf=al["conflict_viol_rate"], exit=al.get("exit_rate", 0.0))
        rows.append(dict(exp_id=res["exp_id"], mech=res["mechanism"], version=res["version"], scen=res["scenario"],
                         learner=res["learner"], seed=res["seed"], ilp=res["ilp"]["ar"],
                         points={**pts[res["exp_id"]], str(20_000_000): final}))
    minutes = (time.time() - t0) / 60
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ckpt_eval.json").write_text(json.dumps(rows, indent=1))
    (OUT / "ckpt_eval.md").write_text(build(rows, "zh", minutes))
    (OUT / "ckpt_eval.en.md").write_text(build(rows, "en", minutes))
    print(OUT / "ckpt_eval.md", f"({minutes:.1f} min)", flush=True)
    if not a.no_mail:
        from shared.report_mail import send_report
        send_report("v6.0.3", OUT / "ckpt_eval.md")


if __name__ == "__main__":
    from shared.batch_manifest import nontrain_batch      # docs/data_spec.md §6: listed by the v2 panel
    with nontrain_batch("v6.0.3", "eval", __file__,
                        {"zh": "v6.0.3：checkpoint 测试评估（v6.0.0 全部 run 与 v6.0.1 Unconstraint 的 5M / 10M / 15M checkpoint，各种子 400 个测试实例，不训练）",
                         "en": "v6.0.3: checkpoint test evaluation (5M / 10M / 15M checkpoints of every v6.0.0 run and v6.0.1 Unconstraint, 400 test instances per seed, no training)"},
                        depends_on=['v6.0.0', 'v6.0.1'], report_dir="paper_contents/v6.0.3"):
        main()
