"""v6.0.0-merge 2-D summary: rows = learner, columns = mechanism, cell = mean gap over LT / GT / EQ.

Sources (all on the fixed v6.0.4 test set, 5000 ILP-optimal instances per scenario):
  Mask PPO   -> v6.0.0.2-test-50k (n_epochs=4)
  Lagrangian -> v6.0.1-lagrangian-test-50k (standard Lagrangian)
  others     -> v6.0.4-test-50k (v6.0.0 models)
Per scenario the gap is first averaged over seeds; a cell is the plain mean of the three scenario gaps.
Writes paper_contents/v6.0.0-merge/table_2d.md and table_2d.en.md.

    .venv/bin/python src/scripts/merge_2d_table.py
"""
import json
import statistics as st
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PC = ROOT / "paper_contents"
OUT = PC / "v6.0.0-merge"
MECHS = ["mask", "repair", "lagrange", "none"]
LEARNERS = ["ppo", "dqn", "ddqn"]
SCENS = ["lt", "gt", "eq"]
NAME = {"mask": "Mask", "repair": "Repair", "lagrange": "Lagrangian", "none": "Unconstraint",
        "ppo": "PPO", "dqn": "DQN", "ddqn": "Double DQN"}


def load():
    rd = lambda b: json.loads((PC / b / "test_eval.json").read_text())
    rows = [r for r in rd("v6.0.4-test-50k")
            if r["mech"] != "lagrange" and not (r["mech"] == "mask" and r["learner"] == "ppo")]
    return rows + rd("v6.0.0.2-test-50k") + rd("v6.0.1-lagrangian-test-50k")


def build(rows, lang):
    zh = lang == "zh"
    T = (lambda z, e: z if zh else e)
    sel = lambda m, l, s=None: [r for r in rows if (r["mech"], r["learner"]) == (m, l) and (s is None or r["scen"] == s)]
    gap = lambda m, l, s: st.mean(r["new"]["gap"] for r in sel(m, l, s))
    cell = {(m, l): st.mean(gap(m, l, s) for s in SCENS) for m in MECHS for l in LEARNERS}
    viol = {(m, l): st.mean(1 - r["new"]["vf"] - r["new"]["exit"] for r in sel(m, l)) for m in MECHS for l in LEARNERS}
    seeds = {(m, l): sorted({r["seed"] for r in sel(m, l)}) for m in MECHS for l in LEARNERS}
    p = lambda v: f"{100 * v:.2f}%"

    L = [T("# v6.0.0-merge 二维汇总：算法 × 机制（三场景平均 gap）", "# v6.0.0-merge 2-D summary: algorithm × mechanism (gap averaged over three scenarios)"), ""]
    L.append(T(f"- 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/merge_2d_table.py`；不训练、不重新评估。",
               f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; script `src/scripts/merge_2d_table.py`; no training, no re-evaluation."))
    L.append(T("- 测试集：v6.0.4 固定测试集，每场景 5000 个 ILP 最优实例。",
               "- Test set: the fixed v6.0.4 test set, 5000 ILP-optimal instances per scenario."))
    L.append(T("- 来源：Mask PPO = v6.0.0.2（n_epochs=4，`v6.0.0.2-test-50k`）；Lagrangian = v6.0.1-lagrangian（标准 Lagrangian，`v6.0.1-lagrangian-test-50k`）；其余 = v6.0.0（`v6.0.4-test-50k`）。",
               "- Sources: Mask PPO = v6.0.0.2 (n_epochs=4, `v6.0.0.2-test-50k`); Lagrangian = v6.0.1-lagrangian (standard Lagrangian, `v6.0.1-lagrangian-test-50k`); others = v6.0.0 (`v6.0.4-test-50k`)."))
    L.append(T("- **gap** = 1 − mean(AR) / mean(ILP AR)，AR 只计合法放置；每场景先对种子取均值，每格 = LT、GT、EQ 三个场景 gap 的算术平均；行/列平均为该行/列各格的算术平均。",
               "- **gap** = 1 − mean(AR) / mean(ILP AR), AR over legal placements; per scenario averaged over seeds, each cell = plain mean of the LT, GT, EQ gaps; row / column means are plain means of the cells."))
    L.append("")

    L.append(T("## 1. 三场景平均 gap", "## 1. Gap averaged over LT / GT / EQ"))
    L.append("")
    L.append("| " + T("算法 \\ 机制", "Algorithm \\ Mechanism") + " | " + " | ".join(NAME[m] for m in MECHS) + " | " + T("**行平均**", "**Row mean**") + " |")
    L.append("|---" * (len(MECHS) + 2) + "|")
    for l in LEARNERS:
        L.append(f"| {NAME[l]} | " + " | ".join(p(cell[m, l]) for m in MECHS) + f" | **{p(st.mean(cell[m, l] for m in MECHS))}** |")
    L.append("| " + T("**列平均**", "**Column mean**") + " | " + " | ".join(f"**{p(st.mean(cell[m, l] for l in LEARNERS))}**" for m in MECHS)
             + f" | **{p(st.mean(cell.values()))}** |")
    L.append("")

    L.append(T("## 2. 各格分场景明细", "## 2. Per-scenario breakdown"))
    L.append("")
    L.append("| " + " | ".join([T("机制", "Mechanism"), T("算法", "Algorithm"), T("种子", "Seeds"), "LT", "GT", "EQ",
                                T("三场景平均", "Mean"), T("违约率（容量或隐私）", "Violation rate (cap or privacy)")]) + " |")
    L.append("|---" * 8 + "|")
    for m in MECHS:
        for l in LEARNERS:
            L.append(f"| {NAME[m]} | {NAME[l]} | {','.join(map(str, seeds[m, l]))} | " + " | ".join(p(gap(m, l, s)) for s in SCENS)
                     + f" | {p(cell[m, l])} | {p(max(viol[m, l], 0.0))} |")
    L.append("")
    L.append(T("- 违约率 = 1 − 无违约比例 − EXIT 比例，三场景、种子平均；只有 Mask 为 0，其余机制的 gap 不能与 Mask 直接等价比较。",
               "- Violation rate = 1 − violation-free − EXIT, averaged over scenarios and seeds; only Mask is 0, so other mechanisms' gaps are not directly equivalent to Mask's."))
    L.append("")
    return "\n".join(L)


def main():
    rows = load()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table_2d.md").write_text(build(rows, "zh"))
    (OUT / "table_2d.en.md").write_text(build(rows, "en"))
    print(OUT / "table_2d.md")


if __name__ == "__main__":
    main()
