"""v6.0.0 / v6.0.1 strict reports (zh + en), percentages to 3 decimals.

One report per version, each over the runs of that version's manifest: Mask / Repair / Lagrangian are the
v6.0.0 runs (v6.0.1 reuses them), the unconstrained row is v6.0.0 "Penalty" (ar_pen, constraint info in
the state) or v6.0.1 "Unconstraint" (no penalty, AR_obj reward, no constraint info).
Strict filter: a (mechanism, scenario, learner) cell qualifies only if, on every seed and every one of the
400 test instances, there is no capacity violation, no privacy violation and no EXIT (Mask giving up leaves
services unplaced). Counts come from cap_viol_total / conflict_viol_total / exit_rate in results.json.
Writes logs/<version>/report/strict.md + strict.en.md and mails them.
"""
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENS = ["eq", "gt", "lt"]
LEARNERS = ["ppo", "dqn", "ddqn"]
MECHS = ["mask", "repair", "lagrange", "none"]
NAME = {"v6.0.0": {"zh": ["Mask", "Repair", "Lagrangian", "Penalty（v6.0.0 Unconstraint）"],
                   "en": ["Mask", "Repair", "Lagrangian", "Penalty (v6.0.0 Unconstraint)"]},
        "v6.0.1": {"zh": ["Mask", "Repair", "Lagrangian", "Unconstraint（v6.0.1）"],
                   "en": ["Mask", "Repair", "Lagrangian", "Unconstraint (v6.0.1)"]}}


def load(version):
    data = defaultdict(list)
    for run in json.loads((ROOT / f"logs/{version}/manifest.json").read_text())["runs"]:
        f = glob.glob(str(ROOT / f"results/unified/*/*/{run['exp_id']}/results.json"))
        if not f:
            sys.exit(f"missing results.json for {run['exp_id']}")
        r = json.loads(Path(f[0]).read_text())
        a = r[r["algo"]]
        n = r["test_count"]
        data[(r["mechanism"], r["scenario"], r["learner"])].append(dict(
            seed=r["seed"], n=n, ilp=r["ilp"]["ar"], ar=a["ar_mean"], gap=1 - a["ar_mean"] / r["ilp"]["ar"],
            cap=a["cap_viol_rate"], conf=a["conflict_viol_rate"], exit=a.get("exit_rate", 0.0),
            cap_n=round(a["cap_viol_rate"] * n), conf_n=round(a["conflict_viol_rate"] * n),
            exit_n=round(a.get("exit_rate", 0.0) * n),
            cap_tot=a["cap_viol_total"], conf_tot=a["conflict_viol_total"]))
    for v in data.values():
        v.sort(key=lambda e: e["seed"])
    return data


def strict(v):
    return all(e["cap_tot"] == 0 and e["conf_tot"] == 0 and e["exit_n"] == 0 for e in v)


def pct(e):
    return f"{100 * e:.3f}%"


def cell(v, key):
    xs = [e[key] for e in v]
    return pct(st.mean(xs)) + (f" ± {pct(st.stdev(xs))}" if len(xs) > 1 else "")


def build(version, data, lang):
    zh = lang == "zh"
    names = NAME[version][lang]
    T = (lambda z, e_: z if zh else e_)
    L = [f"# {version} " + T("严格筛选报告", "strict report"), ""]
    L.append(T("20M 步，图网络，每个种子 400 个测试实例（测试集随种子划分，故各种子 ILP AR 略有不同），最终模型确定性评估。"
               "gap = 1 − AR / ILP AR，AR 只计合法放置。Mask / Repair 种子 1–3，Lagrangian 与无约束行种子 2–3。"
               "Mask / Repair / Lagrangian 来自 v6.0.0 的 run，v6.0.1 直接复用。百分比保留小数点后 3 位；"
               "400 个实例时 1 个实例 = 0.250%。",
               "20M steps, graph network, 400 test instances per seed (the test split depends on the seed, so ILP AR differs "
               "slightly across seeds), deterministic evaluation of the final model. gap = 1 − AR / ILP AR, AR counts legal "
               "placements only. Mask / Repair use seeds 1–3, Lagrangian and the unconstrained row seeds 2–3. Mask / Repair / "
               "Lagrangian are the v6.0.0 runs, reused by v6.0.1. Percentages to 3 decimals; with 400 instances one instance = 0.250%."))
    L.append("")
    L.append(T("**严格筛选**：某个（机制, 场景, 算法）组合只有在所有种子、全部 400 个测试实例上都满足"
               "容量违约 0 次、隐私违约 0 次、EXIT 0 次（Mask 放弃会留下未放置的服务）时才保留。",
               "**Strict filter**: a (mechanism, scenario, learner) combination is kept only if, on every seed and all 400 test "
               "instances, it has 0 capacity violations, 0 privacy violations and 0 EXITs (a Mask EXIT leaves services unplaced)."))
    L.append("")

    L.append("## " + T("严格筛选结果（按 gap 升序）", "Strict filter result (sorted by gap)"))
    L.append("")
    L.append("| " + T("场景 | 通过的组合（gap 均值 ± 种子间标准差）", "Scenario | Passing combinations (gap mean ± std over seeds)") + " |")
    L.append("|---|---|")
    for s in SCENS:
        ok = [(st.mean(e["gap"] for e in data[(m_, s, l)]), names[m], l, data[(m_, s, l)])
              for m, m_ in enumerate(MECHS) for l in LEARNERS if strict(data[(m_, s, l)])]
        ok.sort(key=lambda t: t[0])
        txt = " · ".join(f"{'**' if k == 0 else ''}{n} {l.upper()} {cell(v, 'gap')}{'**' if k == 0 else ''}"
                         for k, (_, n, l, v) in enumerate(ok)) or T("无", "none")
        L.append(f"| {s.upper()} | {txt} |")
    L.append("")
    L.append(T("对照：只要求容量与隐私违约均为 0、允许 EXIT 时，额外通过的组合（括号内为 EXIT 实例数 / 总实例数，各种子合计）：",
               "For reference, combinations that additionally pass when only zero capacity and privacy violations are "
               "required and EXIT is allowed (EXIT instances / total instances, summed over seeds, in brackets):"))
    L.append("")
    for s in SCENS:
        extra = [(st.mean(e["gap"] for e in data[(m_, s, l)]), names[m], l, data[(m_, s, l)])
                 for m, m_ in enumerate(MECHS) for l in LEARNERS
                 if not strict(data[(m_, s, l)]) and all(e["cap_tot"] == 0 and e["conf_tot"] == 0 for e in data[(m_, s, l)])]
        extra.sort(key=lambda t: t[0])
        txt = " · ".join(f"{n} {l.upper()} {cell(v, 'gap')}（EXIT {sum(e['exit_n'] for e in v)} / {sum(e['n'] for e in v)}）" if zh else
                         f"{n} {l.upper()} {cell(v, 'gap')} (EXIT {sum(e['exit_n'] for e in v)} / {sum(e['n'] for e in v)})" for _, n, l, v in extra)
        L.append(f"- {s.upper()}: " + (txt or T("无", "none")))
    L.append("")

    tables = (("AR gap（均值 ± 种子间标准差）", "AR gap (mean ± std over seeds)", "gap"),
              ("容量违约率（有容量违约的实例占比）", "Capacity violation rate (share of instances with a capacity violation)", "cap"),
              ("隐私违约率（有隐私违约的实例占比）", "Privacy violation rate (share of instances with a privacy violation)", "conf"),
              ("EXIT 率（Mask 放弃的实例占比）", "EXIT rate (share of instances where Mask gave up)", "exit"))
    for tz, te, key in tables:
        L.append(f"## {T(tz, te)}")
        L.append("")
        L.append("| " + T("机制", "Mechanism") + " | " + " | ".join(f"{s.upper()} {l.upper()}" for s in SCENS for l in LEARNERS) + " |")
        L.append("|---" * (1 + len(SCENS) * len(LEARNERS)) + "|")
        for m, m_ in enumerate(MECHS):
            L.append(f"| {names[m]} | " + " | ".join(cell(data[(m_, s, l)], key) for s in SCENS for l in LEARNERS) + " |")
        L.append("")

    L.append("## " + T("逐 run 明细", "Per run"))
    L.append("")
    L.append(T("实例数 = 400 个测试实例中出现该情况的实例个数；放置次数 = 违约放置的总次数（一个实例可有多次）。",
               "Instances = number of the 400 test instances where it happens; placements = total number of violating "
               "placements (one instance can have several)."))
    L.append("")
    L.append("| " + T("机制 | 场景 | 算法 | 种子 | ILP AR | AR | gap | 容量违约 实例 / 放置 | 隐私违约 实例 / 放置 | EXIT 实例 / 总实例 | 严格通过",
                       "Mechanism | Scen | Algo | Seed | ILP AR | AR | gap | Cap viol instances / placements | "
                       "Privacy viol instances / placements | EXIT instances / total | Strict pass") + " |")
    L.append("|---" * 11 + "|")
    for m, m_ in enumerate(MECHS):
        for s in SCENS:
            for l in LEARNERS:
                for e in data[(m_, s, l)]:
                    ok = e["cap_tot"] == 0 and e["conf_tot"] == 0 and e["exit_n"] == 0
                    L.append(f"| {names[m]} | {s.upper()} | {l.upper()} | {e['seed']} | {e['ilp']:.6f} | {e['ar']:.6f} | "
                             f"{pct(e['gap'])} | {e['cap_n']} / {e['cap_tot']} | {e['conf_n']} / {e['conf_tot']} | "
                             f"{e['exit_n']} / {e['n']} | {'✓' if ok else '✗'} |")
    L.append("")
    return "\n".join(L)


def main():
    for version in ("v6.0.0", "v6.0.1"):
        data = load(version)
        out = ROOT / f"logs/{version}/report"
        out.mkdir(parents=True, exist_ok=True)
        (out / "strict.md").write_text(build(version, data, "zh"))
        (out / "strict.en.md").write_text(build(version, data, "en"))
        print(out / "strict.md")
        if "--no-mail" not in sys.argv:
            sys.path.insert(0, str(ROOT / "src"))
            from shared.report_mail import send_report
            send_report(f"{version} strict", out / "strict.md", manifest=ROOT / f"logs/{version}/manifest.json")


if __name__ == "__main__":
    main()
