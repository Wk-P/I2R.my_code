#!/usr/bin/env python3
"""v6.0.4-test-50k: the v6.0.0 final models on the fixed v6.0.4 test set (CPU only, no training).

Every run of the v6.0.0 manifest (4 strategies x PPO / DQN / DDQN x 3 scenarios, seeds 1-3) loads its final model
(model_*.zip) and is evaluated deterministically with paper_rl.train.evaluate on data/testset_v6.0.4/<scen>.json
(5000 ILP-verified instances per scenario, the same for every seed), with the run's own settings as in its final
evaluation (Lagrangian: the final lambda and lambda_max of training). Each run is split into chunks of instances so
the work spreads over the worker pool.

Metrics, as the panel / results.json:
  gap      = 1 - mean(AR) / mean(ILP AR) over all test instances (AR over legal placements);
  vf gap   = ILP AR - AR (absolute) and (ILP AR - AR) / ILP AR (relative) over the violation-free instances;
  cap / privacy violation = share of instances with a capacity / conflict violation; EXIT for Maskable only.
The 400-instance column is the run's own final evaluation in results.json (same gap definition).

    .venv/bin/python src/scripts/eval_testset_v6_0_4.py [--workers 32]
        -> paper_contents/v6.0.4-test-50k/test_eval.{json,md,en.md}, mailed
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
BATCH = "v6.0.4-test-50k"
OUT = ROOT / "paper_contents" / BATCH
TEST = ROOT / "data/testset_v6.0.4/{scen}.json"
CHUNK = 500
SCENS = ["eq", "gt", "lt"]
LEARNERS = ["ppo", "dqn", "ddqn"]
MECHS = ["mask", "repair", "lagrange", "none"]
NAME = {"mask": "Mask", "repair": "Repair", "lagrange": "Lagrangian", "none": "Penalty"}


def runs():
    out, seen = [], set()
    man = json.loads((ROOT / "logs/v6.0.0/manifest.json").read_text())
    for r in man["runs"]:
        if not r.get("exp_id") or r["exp_id"] in seen:
            continue
        seen.add(r["exp_id"])
        d = ROOT / "results" / man["results_space"] / r["scenario"] / r["dir"] / r["exp_id"]
        out.append(dict(dir=str(d), res=json.loads((d / "results.json").read_text())))
    return out


def eval_chunk(args):
    import torch
    torch.set_num_threads(1)
    from paper_rl.train import evaluate, model_class
    run, lo = args
    res, d = run["res"], Path(run["dir"])
    mech, learner = res["mechanism"], res["learner"]
    test = json.loads(Path(str(TEST).format(scen=res["scenario"])).read_text())[lo:lo + CHUNK]
    tr = res.get("training") or {}
    lam = tr.get("final_lambda", 0.0) if mech == "lagrange" else 0.0
    lm = tr.get("lambda_max", 50.0)
    lm = float("inf") if lm == "none" else float(lm)
    model = model_class(learner, mech).load(str(next(d.glob("model_*"))), device="cpu")
    ev = evaluate(model, test, mech, res["reward_mode"], lam, learner, res["obs"], res["full_episode"], res["exit_action"],
                  res["action_mode"], res["order"], res["order_seed"], lam_max=lm)
    keep = ("ar", "ar_star", "success", "cap_v", "conf_v", "exited")
    return res["exp_id"], lo, [{k: e[k] for k in keep} for e in ev]


def summarize(ev):
    ar, star = np.array([e["ar"] for e in ev]), np.array([e["ar_star"] for e in ev])
    ok = [e for e in ev if e["success"]]
    ar_ok = float(np.mean([e["ar"] for e in ok])) if ok else float("nan")
    star_ok = float(np.mean([e["ar_star"] for e in ok])) if ok else float("nan")
    return dict(n=len(ev), ar=float(ar.mean()), ilp=float(star.mean()), gap=float(1 - ar.mean() / star.mean()),
                vf=len(ok) / len(ev), vf_abs=star_ok - ar_ok, vf_rel=(star_ok - ar_ok) / star_ok if ok else float("nan"),
                cap=float(np.mean([e["cap_v"] > 0 for e in ev])), conf=float(np.mean([e["conf_v"] > 0 for e in ev])),
                exit=float(np.mean([e["exited"] for e in ev])))


def old_gap(res):
    """The run's own 400-instance final evaluation, same gap definition."""
    al = res[res["algo"]]
    return 1 - al["ar_mean"] / res["ilp"]["ar"]


def pct(v):
    return "—" if v is None or v != v else f"{100 * v:.3f}%"


def cell(xs):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return "—"
    return pct(st.mean(xs)) + (f" ± {pct(st.stdev(xs))}" if len(xs) > 1 else "")


def build(rows, lang, minutes, n_test):
    zh = lang == "zh"
    T = (lambda z, e: z if zh else e)
    L = [f"# {BATCH} " + T("v6.0.0 模型在 v6.0.4 固定测试集上的评估", "v6.0.0 models on the fixed v6.0.4 test set"), ""]
    L.append(T(f"- 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/eval_testset_v6_0_4.py`；耗时 {minutes:.1f} 分钟；只用 CPU，不训练。",
               f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; script `src/scripts/eval_testset_v6_0_4.py`; took {minutes:.1f} min; CPU only, no training."))
    seeds = {m: sorted({r["seed"] for r in rows if r["mech"] == m}) for m in MECHS}
    sz = "；".join(f"{NAME[m]} 种子 {', '.join(map(str, v))}" for m, v in seeds.items())
    se = "; ".join(f"{NAME[m]} seeds {', '.join(map(str, v))}" for m, v in seeds.items())
    L.append(T(f"- **模型：v6.0.0 的最终模型（20M 步）**，全部 {len(rows)} 个 run（4 strategies × PPO / DQN / DDQN × 3 场景；{sz}，与 v6.0.0 本身一致），不重新训练。",
               f"- **Models: the v6.0.0 final models (20M steps)**, all {len(rows)} runs (4 strategies x PPO / DQN / DDQN x 3 scenarios; {se}, as in v6.0.0 itself), not retrained."))
    L.append(T(f"- 测试集：`data/testset_v6.0.4/<场景>.json`（v6.0.4-testset），每场景 {n_test} 个经 ILP 求得最优的实例，所有种子共用；"
               "确定性评估，设置与各 run 的最终评估相同（Lagrangian 用训练结束时的 λ 与 λ_max）。",
               f"- Test set: `data/testset_v6.0.4/<scenario>.json` (v6.0.4-testset), {n_test} ILP-optimal instances per scenario, shared by every seed; "
               "deterministic evaluation with each run's final-evaluation settings (Lagrangian: the λ and λ_max at the end of training)."))
    L.append(T("- **gap** = 1 − mean(AR) / mean(ILP AR)，全部测试实例，AR 只计合法放置（与看板、results.json 同一定义）；"
               "**400 实例** = 该 run 原来在各自种子 400 个测试实例上的最终评估（同一定义）。",
               "- **gap** = 1 − mean(AR) / mean(ILP AR) over all test instances, AR over legal placements (as the panel and results.json); "
               "**400 instances** = the run's own final evaluation on its seed's 400 test instances (same definition)."))
    L.append(T("- **无违约实例**：ILP AR − AR（绝对）与 (ILP AR − AR) / ILP AR（相对），只在无违约完成的实例上计算。EXIT 只属于 Mask。",
               "- **Violation-free instances**: ILP AR − AR (absolute) and (ILP AR − AR) / ILP AR (relative), over the instances completed without violation. EXIT exists for Mask only."))
    L.append(T("- 表中为种子间均值 ± 标准差。", "- Mean ± std over seeds."))
    L.append("")
    for s in SCENS:
        L.append(f"## {s.upper()}")
        L.append("")
        L.append("| " + " | ".join([T("机制", "Mechanism"), T("算法", "Algo"), T(f"gap（{n_test} 实例）", f"gap ({n_test} instances)"),
                                    T("gap（400 实例）", "gap (400 instances)"), T("无违约比例", "violation-free"),
                                    T("无违约实例 绝对 gap", "violation-free abs. gap"), T("无违约实例 相对 gap", "violation-free rel. gap"),
                                    T("容量违约", "cap viol"), T("隐私违约", "privacy viol"), "EXIT"]) + " |")
        L.append("|---" * 10 + "|")
        for m in MECHS:
            for l in LEARNERS:
                g = [r for r in rows if (r["mech"], r["scen"], r["learner"]) == (m, s, l)]
                if not g:
                    continue
                c = lambda k: cell([r["new"][k] for r in g])
                L.append(f"| {NAME[m]} | {l.upper()} | {c('gap')} | {cell([r['old_gap'] for r in g])} | {c('vf')} | "
                         f"{c('vf_abs')} | {c('vf_rel')} | {c('cap')} | {c('conf')} | {c('exit') if m == 'mask' else '—'} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-mail", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rs = runs()
    n_test = {s: len(json.loads(Path(str(TEST).format(scen=s)).read_text())) for s in SCENS}
    tasks = [(r, lo) for r in rs for lo in range(0, n_test[r["res"]["scenario"]], CHUNK)]
    print(f"{len(rs)} runs (v6.0.0 final models), {len(tasks)} chunks of {CHUNK} instances, "
          f"test set {n_test}, {a.workers} workers", flush=True)
    parts = {}
    with Pool(a.workers) as pool:
        for i, (eid, lo, ev) in enumerate(pool.imap_unordered(eval_chunk, tasks), 1):
            parts.setdefault(eid, {})[lo] = ev
            if i % 50 == 0 or i == len(tasks):
                print(f"[chunk] {i}/{len(tasks)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    rows = []
    for r in rs:
        res = r["res"]
        ev = [e for lo in sorted(parts[res["exp_id"]]) for e in parts[res["exp_id"]][lo]]
        rows.append(dict(exp_id=res["exp_id"], mech=res["mechanism"], scen=res["scenario"], learner=res["learner"],
                         seed=res["seed"], dir=r["dir"], old_gap=old_gap(res), new=summarize(ev)))
    minutes = (time.time() - t0) / 60
    assert len(set(n_test.values())) == 1, n_test
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_eval.json").write_text(json.dumps(rows, indent=1))
    (OUT / "test_eval.md").write_text(build(rows, "zh", minutes, n_test["eq"]))
    (OUT / "test_eval.en.md").write_text(build(rows, "en", minutes, n_test["eq"]))
    print(OUT / "test_eval.md", f"({minutes:.1f} min)", flush=True)
    if not a.no_mail:
        from shared.report_mail import send_report
        send_report(BATCH, OUT / "test_eval.md")


if __name__ == "__main__":
    from shared.batch_manifest import nontrain_batch      # docs/data_spec.md §6: listed by the v2 panel
    with nontrain_batch(BATCH, "eval", __file__,
                        {"zh": "v6.0.4-test-50k：使用 v6.0.0 的最终模型（20M 步，全部 90 个 run，不重新训练），在 v6.0.4 固定测试集（每场景 5000 个 ILP 最优实例）上只用 CPU 重新评估",
                         "en": "v6.0.4-test-50k: the v6.0.0 final models (20M steps, every run, not retrained) re-evaluated on CPU on the fixed v6.0.4 test set (5000 ILP-optimal instances per scenario)"},
                        depends_on=["v6.0.0", "v6.0.4-testset"], report_dir=f"paper_contents/{BATCH}",
                        inputs=["data/testset_v6.0.4/{scen}.json"],
                        outputs=[f"paper_contents/{BATCH}/test_eval.json", f"paper_contents/{BATCH}/test_eval.md", f"paper_contents/{BATCH}/test_eval.en.md"],
                        logs=[f"logs/{BATCH}/eval.log"]):
        main()
