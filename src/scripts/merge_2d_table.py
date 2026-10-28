"""v6.0.0-merge 2-D summary: rows = strategy (mechanism), columns = algorithm (learner), cell = mean gap over LT / GT / EQ.

Sources (all on the fixed v6.0.4 test set, 5000 ILP-optimal instances per scenario):
  Mask PPO   -> v6.0.5-PPO-hyparam-test-50k, n_epochs=6 runs only
  Lagrangian -> v6.0.1-lagrangian-test-50k (standard Lagrangian)
  Unconstraint seed 1 -> v6.0.1-unconstraint-test-50k (G12; seeds 2, 3 from v6.0.4-test-50k)
  others     -> v6.0.4-test-50k (v6.0.0 models)
Per scenario the gap is first averaged over seeds; a cell is the plain mean of the three scenario gaps.
Writes paper_contents/v6.0.0-merge/table_2d.md, table_2d.en.md, table_2d.png (heatmap of section 1),
table_2d.json (the same numbers for the panel's summary page, /api/v2/summary) and the paper versions:
  fig_gap_bars.{pdf,png}   gap per scenario, strategy groups x algorithm bars, error bar = std over seeds
  fig_gap_viol.{pdf,png}   mean gap vs violation rate per strategy x algorithm (lower-left is better)
  table_main.tex           booktabs table, mean +- std over seeds, best / second best per gap column marked
std is the sample std (n-1) over seeds of the per-seed gap; for "Mean" the per-seed gap is first averaged over scenarios.

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


def per_seed(rows):
    """{(mech, learner): {"gap": {scen: [per-seed gap]}, "mean": [per-seed mean over scenarios], "viol": {scen: mean}}}"""
    out = {}
    for m in MECHS:
        for l in LEARNERS:
            sel = [r for r in rows if (r["mech"], r["learner"]) == (m, l)]
            seeds = sorted({r["seed"] for r in sel})
            g = {s: {r["seed"]: r["new"]["gap"] for r in sel if r["scen"] == s} for s in SCENS}
            out[m, l] = dict(seeds=seeds, gap={s: [g[s][k] for k in seeds] for s in SCENS},
                             mean=[st.mean(g[s][k] for s in SCENS) for k in seeds],
                             viol={s: max(st.mean(1 - r["new"]["vf"] - r["new"]["exit"] for r in sel if r["scen"] == s), 0.0) for s in SCENS})
    return out


sd = lambda xs: st.stdev(xs) if len(xs) > 1 else 0.0
# Okabe-Ito (colour-blind safe); hatches keep the bars apart in grey-scale print
ALGO_STYLE = {"ppo": ("#0072B2", "", "o"), "dqn": ("#E69F00", "//", "^"), "ddqn": ("#009E73", "..", "s")}
STRAT_COLOR = {"mask": "#0072B2", "repair": "#009E73", "lagrange": "#E69F00", "none": "#D55E00"}
SCEN_TITLE = {"lt": "LT (scarce)", "gt": "GT (ample)", "eq": "EQ (balanced)"}


def paper(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    P = per_seed(rows)

    # 1. grouped bars, one panel per scenario (double-column width)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    w = 0.26
    for ax, s in zip(axes, SCENS):
        for j, l in enumerate(LEARNERS):
            c, h, _ = ALGO_STYLE[l]
            ys = [100 * st.mean(P[m, l]["gap"][s]) for m in MECHS]
            es = [100 * sd(P[m, l]["gap"][s]) for m in MECHS]
            ax.bar([i + (j - 1) * w for i in range(len(MECHS))], ys, w, yerr=es, color=c, hatch=h, edgecolor="black",
                   linewidth=0.4, error_kw=dict(elinewidth=0.6, capsize=1.5), label=NAME[l])
        ax.set_xticks(range(len(MECHS)), [NAME[m] for m in MECHS], rotation=20)
        ax.set_title(SCEN_TITLE[s]); ax.grid(axis="y", linewidth=0.3, alpha=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("AR gap (%)")
    axes[0].legend(frameon=False, loc="upper left")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(OUT / f"fig_gap_bars.{ext}", dpi=300)
    plt.close()

    # 2. gap vs violation rate (single-column width)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    # the methods that matter crowd the lower-left corner: repeat them in a zoomed inset
    ins = ax.inset_axes([0.40, 0.25, 0.40, 0.42])
    for a, ms in ((ax, 5), (ins, 4.5)):
        for m in MECHS:
            for l in LEARNERS:
                x = 100 * st.mean(P[m, l]["mean"]); y = 100 * st.mean(P[m, l]["viol"].values())
                a.errorbar(x, y, xerr=100 * sd(P[m, l]["mean"]), fmt=ALGO_STYLE[l][2], color=STRAT_COLOR[m], ms=ms,
                           mec="black", mew=0.4, elinewidth=0.6, capsize=1.5)
    ins.set_xlim(3.5, 8.6); ins.set_ylim(-0.3, 2.7)
    ins.tick_params(labelsize=6); ins.grid(linewidth=0.3, alpha=0.6)
    for sp in ins.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)
    ax.indicate_inset_zoom(ins, edgecolor="0.4", linewidth=0.5)
    ax.set_xlabel("AR gap (%), mean of LT / GT / EQ"); ax.set_ylabel("Violation rate (%)")
    ax.grid(linewidth=0.3, alpha=0.6); ax.set_axisbelow(True)
    h1 = [Line2D([], [], ls="", marker="s", color=STRAT_COLOR[m], mec="black", mew=0.4, label=NAME[m]) for m in MECHS]
    h2 = [Line2D([], [], ls="", marker=ALGO_STYLE[l][2], color="white", mec="black", mew=0.6, label=NAME[l]) for l in LEARNERS]
    ax.legend(handles=h1 + h2, frameon=False, loc="upper left", ncol=2, handletextpad=0.3, columnspacing=0.8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(OUT / f"fig_gap_viol.{ext}", dpi=300)
    plt.close()

    # 3. booktabs table
    cols = SCENS + ["mean"]
    val = lambda m, l, c: P[m, l]["mean"] if c == "mean" else P[m, l]["gap"][c]
    rank = {c: sorted(st.mean(val(m, l, c)) for m in MECHS for l in LEARNERS) for c in cols}
    def fmt(m, l, c):
        xs = val(m, l, c); v = f"{100 * st.mean(xs):.2f} $\\pm$ {100 * sd(xs):.2f}"
        return f"\\textbf{{{v}}}" if st.mean(xs) == rank[c][0] else f"\\underline{{{v}}}" if st.mean(xs) == rank[c][1] else v
    T = ["% generated by src/scripts/merge_2d_table.py -- do not edit by hand", "\\begin{table}[t]", "\\centering",
         "\\caption{AR gap (\\%, lower is better) on the test set (5000 ILP-optimal instances per scenario), mean $\\pm$ std over "
         "seeds; best in bold, second best underlined. Viol.: share of test instances with a capacity or privacy violation.}",
         "\\label{tab:main}", "\\small", "\\begin{tabular}{llccccc}", "\\toprule",
         "Strategy & Algorithm & LT & GT & EQ & Mean & Viol.\\ (\\%) \\\\", "\\midrule"]
    for i, m in enumerate(MECHS):
        if i:
            T.append("\\midrule")
        for j, l in enumerate(LEARNERS):
            v = 100 * st.mean(P[m, l]["viol"].values())
            T.append(f"{NAME[m] if j == 0 else ''} & {NAME[l]} & " + " & ".join(fmt(m, l, c) for c in cols) + f" & {v:.2f} \\\\")
    T += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (OUT / "table_main.tex").write_text("\n".join(T))


def dump(rows):
    # names as docs/data_spec.md: strategy maskable / repair / lagrangian / unconstraint, RL_algorithm PPO / DQN / DDQN
    STRAT = {"mask": "maskable", "repair": "repair", "lagrange": "lagrangian", "none": "unconstraint"}
    _, gap, viol = stats(rows)
    P = per_seed(rows)
    sel = lambda m, l: [r for r in rows if (r["mech"], r["learner"]) == (m, l)]
    cells = [dict(strategy=STRAT[m], RL_algorithm=l.upper(), seeds=sorted({r["seed"] for r in sel(m, l)}),
                  gap={s: gap(m, l, s) for s in SCENS}, mean=st.mean(gap(m, l, s) for s in SCENS),
                  viol=max(viol[m, l], 0.0),
                  exit=st.mean(r["new"]["exit"] for r in sel(m, l)) if m == "mask" else None,
                  gap_std={s: sd(P[m, l]["gap"][s]) for s in SCENS}, mean_std=sd(P[m, l]["mean"]), viol_by_scen=P[m, l]["viol"])
             for m in MECHS for l in LEARNERS]
    src = {"zh": "Mask PPO = v6.0.5-PPO-hyparam（n_epochs=6）；Lagrangian = v6.0.1-lagrangian（标准 Lagrangian）；Unconstraint seed 1 = v6.0.1-unconstraint；其余 = v6.0.0",
           "en": "Mask PPO = v6.0.5-PPO-hyparam (n_epochs=6); Lagrangian = v6.0.1-lagrangian (standard Lagrangian); Unconstraint seed 1 = v6.0.1-unconstraint; others = v6.0.0"}
    (OUT / "table_2d.json").write_text(json.dumps(dict(
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"), script="src/scripts/merge_2d_table.py",
        test_set="data/testset_v6.0.4/{scen}.json", n_test=5000, scenarios=SCENS,
        strategies=[STRAT[m] for m in MECHS], RL_algorithms=[l.upper() for l in LEARNERS], sources=src,
        batches=[f"{MASK_PPO_SRC}-test-50k", "v6.0.1-lagrangian-test-50k", "v6.0.1-unconstraint-test-50k", "v6.0.4-test-50k"], cells=cells,
        files=["fig_gap_bars.pdf", "fig_gap_bars.png", "fig_gap_viol.pdf", "fig_gap_viol.png", "table_main.tex", "table_2d.png"]), indent=1))


def main():
    rows = load()
    OUT.mkdir(parents=True, exist_ok=True)
    plot(rows)
    paper(rows)
    dump(rows)
    (OUT / "table_2d.md").write_text(build(rows, "zh"))
    (OUT / "table_2d.en.md").write_text(build(rows, "en"))
    print(OUT / "table_2d.md")


if __name__ == "__main__":
    main()
