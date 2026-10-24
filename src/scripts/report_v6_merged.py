"""v6 merged report (zh + en): one table set that takes every cell from the batch that fixed it.

  Mask / Repair / Lagrangian  v6.0.0 models, evaluated on the v6.0.4 fixed test set (v6.0.4-test-50k, 5000 / scenario)
  Unconstraint                v6.0.0 models (full ar_pen reward), same fixed test set
  Mask PPO, n_epochs=4        v6.0.0.2 models, own 400-instance eval (supplement row, PPO update-size fix)
Not used: v6.0.1 Unconstraint (no-penalty variant), v6.0.0.1 (target_kl, worse).
Writes paper_contents/v6.0.0-merge/report.md + report.en.md (report of the v6.0.0-merge batch).
"""
import glob
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_contents/v6.0.0-merge"
SCENS, LEARNERS = ["lt", "eq", "gt"], ["ppo", "dqn", "ddqn"]
LEARN = {"ppo": "PPO", "dqn": "DQN", "ddqn": "Double DQN"}
ROWS = [("mask", "v6.0.0"), ("mask", "v6.0.0.2"), ("repair", "v6.0.0"), ("lagrange", "v6.0.0"), ("none", "v6.0.0")]
MECH = {"zh": {"mask": "Mask", "repair": "Repair", "lagrange": "Lagrangian", "none": "Unconstraint"},
        "en": {"mask": "Mask", "repair": "Repair", "lagrange": "Lagrangian", "none": "Unconstraint"}}
LINE = re.compile(r"violation-free instances: AR=([\d.]+) ILP AR=([\d.]+)")


def from_test50k():
    out = defaultdict(list)
    for r in json.loads((ROOT / "paper_contents/v6.0.4-test-50k/test_eval.json").read_text()):
        n = r["new"]
        out[(r["mech"], "v6.0.0", r["scen"], r["learner"])].append(dict(
            seed=r["seed"], n=n["n"], gap=n["gap"], vf=n["vf"], vf_rel=n["vf_rel"], cap=n["cap"], conf=n["conf"],
            exit=n["exit"]))
    return out


def from_runs(version, keep):
    out = defaultdict(list)
    for run in json.loads((ROOT / f"logs/{version}/manifest.json").read_text())["runs"]:
        d = Path(glob.glob(str(ROOT / f"results/unified/*/*/{run['exp_id']}"))[0])
        r = json.loads((d / "results.json").read_text())
        if r["mechanism"] != keep:
            continue
        a = r[r["algo"]]
        log = next(iter(glob.glob(str(ROOT / f"logs/{version}/*_{run['exp_id']}.log"))), None)
        m = LINE.search(Path(log).read_text()) if log else None
        out[(keep, version, r["scenario"], r["learner"])].append(dict(
            seed=r["seed"], n=r["test_count"], gap=1 - a["ar_mean"] / r["ilp"]["ar"], vf=a["success_rate"],
            vf_rel=(1 - float(m[1]) / float(m[2])) if m else None, cap=a["cap_viol_rate"], conf=a["conflict_viol_rate"],
            exit=a.get("exit_rate", 0.0)))
    return out


def ms(xs, d=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    s = f"{100 * st.mean(xs):.{d}f}%"
    return s + (f" ± {100 * st.stdev(xs):.{d}f}%" if len(xs) > 1 else "")


def write(lang, data):
    zh = lang == "zh"
    L = []
    L.append("# v6 整合结果（定稿口径）" if zh else "# v6 merged results (final reward / parameter definition)")
    L.append("")
    if zh:
        L += ["每一格取自修正了该问题的批次，奖励函数与参数均为定稿口径：",
              "",
              "| 行 | 模型来源 | 测试集 | 种子 | 说明 |", "|---|---|---|---|---|",
              "| Mask / Repair / Lagrangian / Unconstraint | v6.0.0（20M 步） | v6.0.4 固定测试集，每场景 5000 个 ILP 最优实例（v6.0.4-test-50k） | Mask / Repair 1–3，Lagrangian / Unconstraint 2–3 | ar_pen（Unconstraint 同为完整 ar_pen，无额外约束机制）；Lagrangian 每步再减 λ·c_t；EXIT 只给 Mask |",
              "| Mask PPO（n_epochs=4） | v6.0.0.2（20M 步） | 各种子自己的 400 个实例 | 1–3 | 修正 PPO 更新幅度过大（v6.0.3.1 诊断）；**尚未在 5000 固定测试集上评估** |",
              "",
              "未使用：v6.0.1 的 Unconstraint（不罚分版本）、v6.0.0.1（target_kl=0.02，结果更差）。"
              "缺口：Lagrangian / Unconstraint 缺种子 1；v6.0.0.2 尚未在固定测试集上评估。",
              "",
              "- **gap** = 1 − mean(AR) / mean(ILP AR)，全部测试实例，AR 只计合法放置。",
              "- **无违约实例相对 gap** = (ILP AR − AR) / ILP AR，只在无违约完成的实例上计算。",
              "- 违约率 = 出现 ≥1 次该类违约的实例占比。EXIT 只属于 Mask，其余显示 —。",
              "- 表中为种子间均值 ± 标准差。"]
    else:
        L += ["Every cell comes from the batch that fixed it; reward function and parameters follow the final definition:",
              "",
              "| Row | Models | Test set | Seeds | Notes |", "|---|---|---|---|---|",
              "| Mask / Repair / Lagrangian / Unconstraint | v6.0.0 (20M steps) | v6.0.4 fixed test set, 5000 ILP-optimal instances per scenario (v6.0.4-test-50k) | Mask / Repair 1–3, Lagrangian / Unconstraint 2–3 | ar_pen (Unconstraint: the same full ar_pen, no constraint mechanism); Lagrangian −λ·c_t per step; EXIT for Mask only |",
              "| Mask PPO (n_epochs=4) | v6.0.0.2 (20M steps) | each seed's own 400 instances | 1–3 | fixes the oversized PPO updates (v6.0.3.1); **not yet evaluated on the 5000-instance test set** |",
              "",
              "Not used: v6.0.1 Unconstraint (no-penalty variant), v6.0.0.1 (target_kl=0.02, worse). "
              "Gaps: Lagrangian / Unconstraint lack seed 1; v6.0.0.2 not yet evaluated on the fixed test set.",
              "",
              "- **gap** = 1 − mean(AR) / mean(ILP AR) over all test instances, AR over legal placements.",
              "- **Violation-free relative gap** = (ILP AR − AR) / ILP AR over the instances completed without violation.",
              "- Violation rate = share of instances with ≥1 violation of that kind. EXIT exists for Mask only (— elsewhere).",
              "- Mean ± std over seeds."]
    hdr = (["机制", "算法", "来源", "测试实例", "种子", "gap", "无违约实例占比", "无违约实例相对 gap", "容量违约", "隐私违约", "EXIT"] if zh else
           ["Mechanism", "Algo", "Source", "Test instances", "Seeds", "gap", "Violation-free", "Violation-free rel. gap", "Cap viol", "Privacy viol", "EXIT"])
    for s in SCENS:
        L += ["", f"## {s.upper()}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for mech, ver in ROWS:
            for l in LEARNERS:
                xs = data.get((mech, ver, s, l))
                if not xs:
                    continue
                name = MECH[lang][mech] + (" (n_epochs=4)" if ver == "v6.0.0.2" else "")
                L.append("| " + " | ".join([
                    name, LEARN[l], ver, str(xs[0]["n"]), ",".join(str(x["seed"]) for x in sorted(xs, key=lambda x: x["seed"])),
                    ms([x["gap"] for x in xs]), ms([x["vf"] for x in xs]), ms([x["vf_rel"] for x in xs]),
                    ms([x["cap"] for x in xs]), ms([x["conf"] for x in xs]),
                    ms([x["exit"] for x in xs]) if mech == "mask" else "—"]) + " |")
    (OUT / ("report.md" if zh else "report.en.md")).write_text("\n".join(L) + "\n")


def main():
    data = from_test50k()
    data.update(from_runs("v6.0.0.2", "mask"))
    OUT.mkdir(parents=True, exist_ok=True)
    for lang in ("zh", "en"):
        write(lang, data)
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
