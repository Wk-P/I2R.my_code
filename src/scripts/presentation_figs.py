#!/usr/bin/env python3
"""The /summary page as vector figures for presentation/ (no training, no evaluation).

Same data and the same order as app/web/src/pages/Summary.vue, in the paper style of merge_2d_table.py (serif,
Okabe-Ito colours), every file saved as PDF and EPS:
  1 fig1_gap_heatmap    AR gap card: strategy x RL_algorithm heat table, row / column means, bold = lowest,
                        dashed outline = below the 5% target
  2 fig2_ar_<strategy>  AR card: ILP / PPO / DQN / DDQN per scenario, one file per strategy (the page's switch)
  3 fig3_reward_curves  training reward curves, one panel per strategy, x-axis [0, 20M]
  4 fig4_time           timing card: per-instance time, log axis; fig4_time_table the table next to it
  5 fig_gap_bars, fig_gap_viol   paper-figures card, drawn by merge_2d_table.paper itself (PNG swapped for EPS)
  6 table_main          Table 1 (table_main.tex) rendered booktabs-style; the .tex is written as well
Data: paper_contents/v6.0.0-merge/table_2d.json; 5 / 6 re-read the runs through merge_2d_table.load.

    .venv/bin/python src/scripts/presentation_figs.py  ->  presentation/figs/*.{pdf,eps}
"""
import json
import statistics as st
import sys
from math import log
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))
import merge_2d_table as M  # noqa: E402

SRC = ROOT / "paper_contents/v6.0.0-merge/table_2d.json"
OUT = ROOT / "presentation/figs"
NAME = dict(maskable="Mask", repair="Repair", lagrangian="Lagrangian", unconstraint="Unconstraint",
            PPO="PPO", DQN="DQN", DDQN="Double DQN")
COLOR = dict(ILP="#7a7f8c", PPO="#0072B2", DQN="#E69F00", DDQN="#009E73")      # as Summary.vue COLOR
SCEN = dict(lt="LT", gt="GT", eq="EQ")
STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
         "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
         "axes.spines.top": False, "axes.spines.right": False}
GRID = dict(linewidth=0.3, color="#cccccc")                                      # opaque: EPS has no transparency


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "eps"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(name, flush=True)


def heatmap(d, cell):
    S, A = d["strategies"], d["RL_algorithms"]
    v = {k: c["mean"] for k, c in cell.items()}
    lo, hi = min(v.values()), max(v.values())
    fig, ax = plt.subplots(figsize=(4.8, 2.2))
    ax.set_xlim(-1.3, len(A) + 1); ax.set_ylim(len(S) + 1, -1); ax.axis("off")
    for j, h in enumerate([NAME[a] for a in A] + ["Row mean"]):
        ax.text(j + 0.5, -0.5, h, ha="center", va="center", weight="bold")
    for i, s in enumerate(S + ["Column mean"]):
        ax.text(-0.1, i + 0.5, NAME.get(s, s), ha="right", va="center", weight="bold")
    for i, s in enumerate(S):
        for j, a in enumerate(A):
            k = (log(v[s, a]) - log(lo)) / (log(hi) - log(lo))                   # white -> accent, log scale (page heat())
            rgb = [(x + (y - x) * k) / 255 for x, y in zip((238, 240, 255), (49, 46, 129))]
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=rgb, edgecolor="white", lw=1.5))
            if v[s, a] < 0.05:
                ax.add_patch(Rectangle((j + 0.05, i + 0.08), 0.9, 0.84, fill=False, ls="--", lw=0.8, edgecolor="black"))
            ax.text(j + 0.5, i + 0.5, f"{100 * v[s, a]:.2f}%", ha="center", va="center",
                    color="white" if k > 0.55 else "black", weight="bold" if v[s, a] == lo else "normal")
        ax.text(len(A) + 0.5, i + 0.5, f"{100 * st.mean(v[s, a] for a in A):.2f}%", ha="center", va="center")
    for j, a in enumerate(A):
        ax.text(j + 0.5, len(S) + 0.5, f"{100 * st.mean(v[s, a] for s in S):.2f}%", ha="center", va="center")
    ax.text(len(A) + 0.5, len(S) + 0.5, f"{100 * st.mean(v.values()):.2f}%", ha="center", va="center", weight="bold")
    ax.plot([-1.3, len(A) + 1], [len(S), len(S)], color="black", lw=0.5)
    save(fig, "fig1_gap_heatmap")


def ar_bars(d, cell):
    sc, rls = d["scenarios"], d["RL_algorithms"]
    for s in d["strategies"]:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        series = [("ILP", [st.mean(cell[s, r]["ilp_ar"][x] for r in rls) for x in sc])] + \
                 [(r, [cell[s, r]["ar"][x] for x in sc]) for r in rls]
        w, ys_all = 0.2, []
        for i, (k, ys) in enumerate(series):
            xs = [j + (i - 1.5) * w for j in range(len(sc))]
            ax.bar(xs, ys, w, color=COLOR[k], edgecolor="black", linewidth=0.4, label=NAME.get(k, k))
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.004, f"{y:.3f}", ha="center", va="bottom", fontsize=5.5, rotation=90)
            ys_all += ys
        ax.set_xticks(range(len(sc)), [SCEN[x] for x in sc])
        ax.set_ylim(max(0, int((min(ys_all) - 0.05) * 10) / 10), 1.0); ax.set_ylabel("AR")
        ax.grid(axis="y", **GRID); ax.set_axisbelow(True)
        ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1, columnspacing=0.8)
        save(fig, f"fig2_ar_{s}")


def curves(d, cell):
    steps, end = d["curve_steps"], d["curve_steps"][-1] + d["curve_steps"][0]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.4))
    for ax, s in zip(axs.flat, d["strategies"]):
        for r in d["RL_algorithms"]:
            ys = cell[s, r]["reward_curve"]
            ax.plot([x / 1e6 for x in steps[:len(ys)]], ys, color=COLOR[r], lw=0.9, label=NAME[r])
        ax.set_xlim(0, end / 1e6); ax.set_title(NAME[s]); ax.grid(**GRID); ax.set_axisbelow(True)
        ax.set_xlabel("Training steps (M)"); ax.set_ylabel("Episode reward")
    axs[0, 0].legend(frameon=False)
    save(fig, "fig3_reward_curves")


def timing(d):
    tb, sc = d["time_bench"], d["scenarios"]
    ilp = {x: tb["ilp"][x]["mean_ms"] for x in sc}
    bars, trows = [("ILP (Dinkelbach)", st.mean(ilp.values()), "ILP")], []
    for s in d["strategies"]:
        for r in d["RL_algorithms"]:
            v = tb["rl"][f"{s}|{r}"]
            bars.append((f"{NAME[s]} {NAME[r]}", st.mean(v.values()), r))
            trows.append([NAME[s], NAME[r]] + [f"{v[x]:.1f}" for x in sc]
                         + [f"{st.mean(v.values()):.1f}", f"{st.mean(ilp[x] / v[x] for x in sc):.0f}×"])
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    ax.barh(range(len(bars)), [b[1] for b in bars], color=[COLOR[b[2]] for b in bars], edgecolor="black", linewidth=0.4)
    for i, b in enumerate(bars):
        ax.text(b[1] * 1.1, i, f"{b[1]:.0f}", va="center", fontsize=6.5)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars]); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlim(10, 10000); ax.set_xlabel("ms per instance (log)")
    ax.grid(axis="x", **GRID); ax.set_axisbelow(True)
    save(fig, "fig4_time")
    trows.insert(0, ["ILP (Dinkelbach)", ""] + [f"{ilp[x]:.1f}" for x in sc] + [f"{st.mean(ilp.values()):.1f}", "1×"])
    booktabs(["Strategy", "Algorithm"] + [f"{SCEN[x]} (ms)" for x in sc] + ["Mean (ms)", "Speed-up"], trows,
             "fig4_time_table", widths=[1.1, 1.0, 0.8, 0.8, 0.8, 0.9, 0.8], groups=[1, 4, 7, 10])


def booktabs(header, rows, name, widths, groups=(), fmt=None):
    """A table in booktabs style (top / mid / bottom rules, no vertical lines); fmt[(row, col)] = 'bold' | 'under'."""
    n, W = len(rows), sum(widths)
    x0 = [sum(widths[:j]) for j in range(len(widths))]
    fig, ax = plt.subplots(figsize=(W + 0.2, 0.2 * (n + 1) + 0.2))
    ax.set_xlim(0, W); ax.set_ylim(n, -1.5); ax.axis("off")
    for y, lw in ((-1.4, 0.9), (-0.1, 0.5), (n - 0.1, 0.9)):
        ax.plot([0, W], [y, y], color="black", lw=lw)
    for g in groups:
        ax.plot([0.05, W - 0.05], [g - 0.1, g - 0.1], color="black", lw=0.3)
    cx = lambda j: x0[j] + widths[j] / 2
    for j, h in enumerate(header):
        ax.text(cx(j), -0.75, h, ha="center", va="center", weight="bold")
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            f = (fmt or {}).get((i, j))
            ax.text(cx(j), i + 0.4, v, ha="center", va="center", weight="bold" if f == "bold" else "normal")
            if f == "under":
                half = 0.026 * len(v)
                ax.plot([cx(j) - half, cx(j) + half], [i + 0.72, i + 0.72], color="black", lw=0.5)
    save(fig, name)


def main_table(rows):
    P = M.per_seed(rows)
    rank = sorted(st.mean(P[m, l]["mean"]) for m in M.MECHS for l in M.LEARNERS)
    body, fmt = [], {}
    for m in M.MECHS:
        for j, l in enumerate(M.LEARNERS):
            xs = P[m, l]["mean"]
            body.append([M.NAME[m] if j == 0 else "", M.NAME[l], f"{100 * st.mean(xs):.2f} ± {100 * M.sd(xs):.2f}",
                         f"{100 * st.mean(P[m, l]['viol'].values()):.2f}"])
            if st.mean(xs) in rank[:2]:
                fmt[len(body) - 1, 2] = "bold" if st.mean(xs) == rank[0] else "under"
    booktabs(["Strategy", "Algorithm", "AR gap (%)", "Viol. (%)"], body, "table_main", widths=[1.0, 1.0, 1.1, 0.8],
             groups=[3, 6, 9], fmt=fmt)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    d = json.loads(SRC.read_text())
    cell = {(c["strategy"], c["RL_algorithm"]): c for c in d["cells"]}
    heatmap(d, cell); ar_bars(d, cell); curves(d, cell); timing(d)
    # paper-figures card: the same code as the paper; PNG swapped for EPS (opaque grid), written here, not v6.0.0-merge
    rows = M.load()
    M.OUT, save_orig = OUT, plt.savefig
    plt.savefig = lambda p, **k: save_orig(str(p).replace(".png", ".eps"))
    try:
        M.paper(rows)
    finally:
        plt.savefig = save_orig
    print("fig_gap_bars, fig_gap_viol, table_main.tex", flush=True)
    plt.rcParams.update(STYLE)
    main_table(rows)


if __name__ == "__main__":
    main()
