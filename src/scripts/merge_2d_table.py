"""v6.0.0-merge 2-D summary: rows = strategy (mechanism), columns = algorithm (learner), cell = mean gap over LT / GT / EQ.

Sources (all on the fixed v6.0.4 test set, 5000 ILP-optimal instances per scenario):
  Mask PPO   -> v6.0.5-PPO-hyparam-test-50k, n_epochs=6 runs only
  Lagrangian -> v6.0.1-lagrangian-test-50k (standard Lagrangian)
  Unconstraint seed 1 -> v6.0.1-unconstraint-test-50k (G12; seeds 2, 3 from v6.0.4-test-50k)
  others     -> v6.0.4-test-50k (v6.0.0 models)
Per scenario the gap is first averaged over seeds; a cell is the plain mean of the three scenario gaps.
Writes paper_contents/v6.0.0-merge/table_2d.md, table_2d.en.md, table_2d.png (heatmap of section 1) and
table_2d.json (the same numbers for the panel's summary page, /api/v2/summary).

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
MASK_PPO_SRC, MASK_PPO_EPOCHS = "v6.0.5-PPO-hyparam", 6
NAME = {"mask": "Mask", "repair": "Repair", "lagrange": "Lagrangian", "none": "Unconstraint",
        "ppo": "PPO", "dqn": "DQN", "ddqn": "Double DQN"}


def load():
    rd = lambda b: json.loads((PC / b / "test_eval.json").read_text())
    rows = [r for r in rd("v6.0.4-test-50k")
            if r["mech"] != "lagrange" and not (r["mech"] == "mask" and r["learner"] == "ppo")]
    man = json.loads((ROOT / f"logs/{MASK_PPO_SRC}/manifest.json").read_text())
    keep = {r["exp_id"] for r in man["runs"] if r["params"].get("n_epochs") == MASK_PPO_EPOCHS}
    mask_ppo = [r for r in rd(f"{MASK_PPO_SRC}-test-50k") if r["exp_id"] in keep]
    return rows + mask_ppo + rd("v6.0.1-lagrangian-test-50k") + rd("v6.0.1-unconstraint-test-50k")


def stats(rows):
    sel = lambda m, l, s=None: [r for r in rows if (r["mech"], r["learner"]) == (m, l) and (s is None or r["scen"] == s)]
    gap = lambda m, l, s: st.mean(r["new"]["gap"] for r in sel(m, l, s))
    cell = {(m, l): st.mean(gap(m, l, s) for s in SCENS) for m in MECHS for l in LEARNERS}
    viol = {(m, l): st.mean(1 - r["new"]["vf"] - r["new"]["exit"] for r in sel(m, l)) for m in MECHS for l in LEARNERS}
    return cell, gap, viol


def build(rows, lang):
    zh = lang == "zh"
    T = (lambda z, e: z if zh else e)
    cell, _, _ = stats(rows)
    _, gap, viol = stats(rows)
    seeds = {(m, l): sorted({r["seed"] for r in rows if (r["mech"], r["learner"]) == (m, l)}) for m in MECHS for l in LEARNERS}
    p = lambda v: f"{100 * v:.2f}%"

    L = [T("# v6.0.0-merge 二维汇总：机制 × 算法（三场景平均 gap）", "# v6.0.0-merge 2-D summary: mechanism × algorithm (gap averaged over three scenarios)"), ""]
    L.append(T(f"- 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/merge_2d_table.py`；不训练、不重新评估。",
               f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; script `src/scripts/merge_2d_table.py`; no training, no re-evaluation."))
    L.append(T("- 测试集：v6.0.4 固定测试集，每场景 5000 个 ILP 最优实例。",
               "- Test set: the fixed v6.0.4 test set, 5000 ILP-optimal instances per scenario."))
    L.append(T("- 来源：Mask PPO = v6.0.5-PPO-hyparam（n_epochs=6，`v6.0.5-PPO-hyparam-test-50k`）；Lagrangian = v6.0.1-lagrangian（标准 Lagrangian，`v6.0.1-lagrangian-test-50k`）；Unconstraint seed 1 = v6.0.1-unconstraint（`v6.0.1-unconstraint-test-50k`）；其余 = v6.0.0（`v6.0.4-test-50k`）。",
               "- Sources: Mask PPO = v6.0.5-PPO-hyparam (n_epochs=6, `v6.0.5-PPO-hyparam-test-50k`); Lagrangian = v6.0.1-lagrangian (standard Lagrangian, `v6.0.1-lagrangian-test-50k`); Unconstraint seed 1 = v6.0.1-unconstraint (`v6.0.1-unconstraint-test-50k`); others = v6.0.0 (`v6.0.4-test-50k`)."))
    L.append(T("- **gap** = 1 − mean(AR) / mean(ILP AR)，AR 只计合法放置；每场景先对种子取均值，每格 = LT、GT、EQ 三个场景 gap 的算术平均；行/列平均为该行/列各格的算术平均。",
               "- **gap** = 1 − mean(AR) / mean(ILP AR), AR over legal placements; per scenario averaged over seeds, each cell = plain mean of the LT, GT, EQ gaps; row / column means are plain means of the cells."))
    L.append("")

    L.append(T("## 1. 三场景平均 gap", "## 1. Gap averaged over LT / GT / EQ"))
    L.append("")
    L.append("| " + T("机制 \\ 算法", "Mechanism \\ Algorithm") + " | " + " | ".join(NAME[l] for l in LEARNERS) + " | " + T("**行平均**", "**Row mean**") + " |")
    L.append("|---" * (len(LEARNERS) + 2) + "|")
    for m in MECHS:
        L.append(f"| {NAME[m]} | " + " | ".join(p(cell[m, l]) for l in LEARNERS) + f" | **{p(st.mean(cell[m, l] for l in LEARNERS))}** |")
    L.append("| " + T("**列平均**", "**Column mean**") + " | " + " | ".join(f"**{p(st.mean(cell[m, l] for m in MECHS))}**" for l in LEARNERS)
             + f" | **{p(st.mean(cell.values()))}** |")
    L.append("")
    L.append("![table_2d](table_2d.png)")
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


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cell, _, _ = stats(rows)
    A = [[100 * cell[m, l] for l in LEARNERS] for m in MECHS]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(A, cmap="Blues")
    ax.set_xticks(range(len(LEARNERS)), [NAME[l] for l in LEARNERS])
    ax.set_yticks(range(len(MECHS)), [NAME[m] for m in MECHS])
    ax.set_xlabel("Algorithm"); ax.set_ylabel("Strategy")
    hi = max(max(r) for r in A)
    for i, r in enumerate(A):
        for j, v in enumerate(r):
            ax.text(j, i, f"{v:.2f}%", ha="center", va="center", color="white" if v > 0.6 * hi else "black")
    fig.colorbar(im, ax=ax, label="AR gap (%), mean of LT / GT / EQ")
    plt.tight_layout(); plt.savefig(OUT / "table_2d.png", dpi=150); plt.close()


def dump(rows):
    # names as docs/data_spec.md: strategy maskable / repair / lagrangian / unconstraint, RL_algorithm PPO / DQN / DDQN
    STRAT = {"mask": "maskable", "repair": "repair", "lagrange": "lagrangian", "none": "unconstraint"}
    _, gap, viol = stats(rows)
    sel = lambda m, l: [r for r in rows if (r["mech"], r["learner"]) == (m, l)]
    cells = [dict(strategy=STRAT[m], RL_algorithm=l.upper(), seeds=sorted({r["seed"] for r in sel(m, l)}),
                  gap={s: gap(m, l, s) for s in SCENS}, mean=st.mean(gap(m, l, s) for s in SCENS),
                  viol=max(viol[m, l], 0.0),
                  exit=st.mean(r["new"]["exit"] for r in sel(m, l)) if m == "mask" else None)
             for m in MECHS for l in LEARNERS]
    src = {"zh": "Mask PPO = v6.0.5-PPO-hyparam（n_epochs=6）；Lagrangian = v6.0.1-lagrangian（标准 Lagrangian）；Unconstraint seed 1 = v6.0.1-unconstraint；其余 = v6.0.0",
           "en": "Mask PPO = v6.0.5-PPO-hyparam (n_epochs=6); Lagrangian = v6.0.1-lagrangian (standard Lagrangian); Unconstraint seed 1 = v6.0.1-unconstraint; others = v6.0.0"}
    (OUT / "table_2d.json").write_text(json.dumps(dict(
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"), script="src/scripts/merge_2d_table.py",
        test_set="data/testset_v6.0.4/{scen}.json", n_test=5000, scenarios=SCENS,
        strategies=[STRAT[m] for m in MECHS], RL_algorithms=[l.upper() for l in LEARNERS], sources=src,
        batches=[f"{MASK_PPO_SRC}-test-50k", "v6.0.1-lagrangian-test-50k", "v6.0.1-unconstraint-test-50k", "v6.0.4-test-50k"], cells=cells), indent=1))


def main():
    rows = load()
    OUT.mkdir(parents=True, exist_ok=True)
    plot(rows)
    dump(rows)
    (OUT / "table_2d.md").write_text(build(rows, "zh"))
    (OUT / "table_2d.en.md").write_text(build(rows, "en"))
    print(OUT / "table_2d.md")


if __name__ == "__main__":
    main()
