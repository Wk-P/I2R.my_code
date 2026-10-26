"""v6.0.0 + v6.0.1 combined report (zh + en).

Mask / Repair / Lagrangian come from v6.0.0 (G1-G8, reused by v6.0.1); Unconstraint exists in both:
v6.0.0 = ar_pen penalty + constraint info in the state ("Penalty"), v6.0.1 = no penalty, AR_obj reward,
no constraint info ("Unconstraint"). gap = 1 - AR / ILP AR on the 400 test instances (final model).
Writes logs/v6.0.1/report/report.md + report.en.md and mails them.
"""
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/v6.0.1/report"
SCENS = ["eq", "gt", "lt"]
LEARNERS = ["ppo", "dqn", "ddqn"]
ROWS = [("mask", "v6.0.0"), ("repair", "v6.0.0"), ("lagrange", "v6.0.0"), ("none", "v6.0.0"), ("none", "v6.0.1")]
NAME = {"zh": {("mask", "v6.0.0"): "Mask", ("repair", "v6.0.0"): "Repair", ("lagrange", "v6.0.0"): "Lagrangian",
               ("none", "v6.0.0"): "Penalty（v6.0.0 Unconstraint）", ("none", "v6.0.1"): "Unconstraint（v6.0.1）"},
        "en": {("mask", "v6.0.0"): "Mask", ("repair", "v6.0.0"): "Repair", ("lagrange", "v6.0.0"): "Lagrangian",
               ("none", "v6.0.0"): "Penalty (v6.0.0 Unconstraint)", ("none", "v6.0.1"): "Unconstraint (v6.0.1)"}}


def load():
    ids = {}
    for v in ("v6.0.0", "v6.0.1"):
        for r in json.loads((ROOT / f"logs/{v}/manifest.json").read_text())["runs"]:
            ids[r["exp_id"]] = v
    data = defaultdict(list)
    for m in ids:
        f = glob.glob(str(ROOT / f"results/unified/*/*/{m}/results.json"))
        if not f:
            sys.exit(f"missing results.json for {m}")
        r = json.loads(Path(f[0]).read_text())
        a = r[r["algo"]]
        data[(r["mechanism"], r["version"], r["scenario"], r["learner"])].append(dict(
            seed=r["seed"], ilp=r["ilp"]["ar"], ar=a["ar_mean"], ar_obj=a.get("ar_obj_mean"),
            gap=1 - a["ar_mean"] / r["ilp"]["ar"], cap=a["cap_viol_rate"], conf=a["conflict_viol_rate"],
            last50=r["training"]["ar_last50"]))
    return data


def pct(e):
    return f"{100 * e:.1f}%"


def cell(v, key):
    xs = [e[key] for e in v]
    m = st.mean(xs)
    return pct(m) + (f" ± {pct(st.stdev(xs))}" if len(xs) > 1 else "")


def build(data, lang):
    zh = lang == "zh"
    L = []
    L.append("# v6.0.0 + v6.0.1 合并报告" if zh else "# v6.0.0 + v6.0.1 combined report")
    L.append("")
    L.append(("Mask / Repair / Lagrangian 取自 v6.0.0（G1–G8，v6.0.1 直接复用）；Unconstraint 两版都列："
              "v6.0.0 = ar_pen 扣分 + 观测含约束信息（下文称 **Penalty**），v6.0.1 = 不扣分、终局给 AR_obj、观测无约束信息"
              "（**Unconstraint**）。20M 步，图网络，400 个测试实例，最终模型确定性评估；"
              "gap = 1 − AR / ILP AR，AR 按约束口径计算（违约实例记为不合法放置）。Mask/Repair 种子 1–3，其余种子 2–3。")
             if zh else
             ("Mask / Repair / Lagrangian come from v6.0.0 (G1–G8, reused unchanged by v6.0.1); Unconstraint is shown in both versions: "
              "v6.0.0 = ar_pen penalty + constraint information in the state (called **Penalty** below), v6.0.1 = no penalty, "
              "terminal AR_obj reward, no constraint information in the state (**Unconstraint**). 20M steps, graph network, 400 test "
              "instances, deterministic evaluation of the final model; gap = 1 − AR / ILP AR, AR computed under the constraints. "
              "Mask/Repair use seeds 1–3, the others seeds 2–3."))
    L.append("")
    for title_zh, title_en, key in (("AR gap（均值 ± 种子间标准差）", "AR gap (mean ± std over seeds)", "gap"),
                                    ("容量违约率", "Capacity violation rate", "cap"),
                                    ("隐私违约率", "Privacy violation rate", "conf")):
        L.append(f"## {title_zh if zh else title_en}")
        L.append("")
        L.append("| " + ("机制" if zh else "Mechanism") + " | " + " | ".join(f"{s.upper()} {l.upper()}" for s in SCENS for l in LEARNERS) + " |")
        L.append("|---" * (1 + len(SCENS) * len(LEARNERS)) + "|")
        for m, v in ROWS:
            L.append(f"| {NAME[lang][(m, v)]} | " + " | ".join(cell(data[(m, v, s, l)], key) for s in SCENS for l in LEARNERS) + " |")
        L.append("")
    L.append("## " + ("Unconstraint（v6.0.1）逐 run" if zh else "Unconstraint (v6.0.1) per run"))
    L.append("")
    L.append("| " + (" | ".join(["场景", "算法", "种子", "ILP AR", "AR_obj", "真实 AR", "训练 ar_last50", "容量违约", "隐私违约"]) if zh else
                    " | ".join(["Scen", "Algo", "Seed", "ILP AR", "AR_obj", "AR", "train ar_last50", "Cap viol", "Privacy viol"])) + " |")
    L.append("|---" * 9 + "|")
    for s in SCENS:
        for l in LEARNERS:
            for e in sorted(data[("none", "v6.0.1", s, l)], key=lambda e: e["seed"]):
                L.append(f"| {s.upper()} | {l.upper()} | {e['seed']} | {e['ilp']:.3f} | {e['ar_obj']:.3f} | {e['ar']:.3f} | "
                         f"{e['last50']:.3f} | {pct(e['cap'])} | {pct(e['conf'])} |")
    L.append("")
    L.append("## " + ("观察" if zh else "Observations"))
    L.append("")
    L += ([
        "- **Mask / Repair**：违约约为 0；EQ/GT 的 gap 为 3–7%，LT 为 7–16%，LT 上 Repair 仍有 1–6% 违约。",
        "- **Lagrangian**：PPO 在三个场景都在 5–7%；DQN/DDQN 在 EQ 和 LT 上 12–25%，LT 违约 2–6%。",
        "- **Penalty（v6.0.0）**：PPO 学成软约束，gap 4–8%、违约 ≤1.8%，接近 Mask；DQN/DDQN gap 13–31%，LT 隐私违约 52–97%。只靠扣分不能保证零违约。",
        "- **Unconstraint（v6.0.1）**：所有实例都违反容量和隐私约束；AR_obj ≈ 1.0 高于 ILP，这是超载被截断到 1 造成的，"
        "不能与 ILP 比较。按约束口径的真实 AR，PPO 的 gap 为 15–19%；DQN/DDQN 为 42–74%，且种子间差异极大（训练 ar_last50 一致，属于训练崩坏，不是评估问题）。",
        "- 约束越弱违约越多：硬约束 ≈ 0 → Lagrangian 少量 → Penalty 依算法而定 → 无约束 100%。",
    ] if zh else [
        "- **Mask / Repair**: violations ≈ 0; gap 3–7% on EQ/GT and 7–16% on LT; Repair still violates 1–6% on LT.",
        "- **Lagrangian**: PPO is at 5–7% in all three scenarios; DQN/DDQN reach 12–25% on EQ and LT, with 2–6% violations on LT.",
        "- **Penalty (v6.0.0)**: PPO learns a soft constraint, gap 4–8% with ≤1.8% violations, close to Mask; DQN/DDQN gap 13–31%, "
        "privacy violations 52–97% on LT. A penalty alone does not guarantee zero violations.",
        "- **Unconstraint (v6.0.1)**: every instance violates both capacity and privacy; AR_obj ≈ 1.0 exceeds ILP only because "
        "overloads are clipped at 1, so it cannot be compared with ILP. On the constrained AR, PPO's gap is 15–19%; DQN/DDQN 42–74% with "
        "very large seed-to-seed spread (training ar_last50 agrees, so training collapsed; this is not an evaluation artefact).",
        "- Weaker constraint handling means more violations: hard constraints ≈ 0 → Lagrangian small → Penalty learner-dependent → unconstrained 100%.",
    ])
    L.append("")
    return "\n".join(L)


def main():
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(build(data, "zh"))
    (OUT / "report.en.md").write_text(build(data, "en"))
    print(OUT / "report.md")
    if "--no-mail" not in sys.argv:
        sys.path.insert(0, str(ROOT / "src"))
        from shared.report_mail import send_report
        send_report("v6.0.0+v6.0.1", OUT / "report.md", manifest=ROOT / "logs/v6.0.1/manifest.json")


if __name__ == "__main__":
    main()
