"""Per-scenario figures for the paper -> figs/<name>.{pdf,eps,png} (Times, 8 pt, 7 in wide, double column).
The slide versions (and the rest of the slide set) come from plot_slides.py -> ../slides_figs/.

  fig_gap_scenarios   dot plot: one row per variant (12), the relative AR gap on LT / EQ / GT as three markers joined
                      by a grey range line, the three-scenario mean as a black tick with its value on the right.
                      Gap on violation-free instances, mean over seeds 1-3 (data: table_2d.json = Tables 7 and 8).
  fig_time_scenarios  mean online time per instance (ms, log scale) per scenario and over all 15,000 instances:
                      ILP vs RL, where the RL bar is the mean of the 12 variants and its whisker their min-max;
                      the speed-up range (ILP / slowest variant - ILP / fastest variant) is written above each group.
                      Data: time_full.json (= Table 6).

    .venv/bin/python paper_contents/20261122/experiments/plot_scenario_figs.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs"
SCENS = [("lt", "LT"), ("eq", "EQ"), ("gt", "GT")]
SCEN_STYLE = {"lt": ("#6250d6", "o"), "eq": ("#eda100", "^"), "gt": ("#008300", "s")}   # same as plot_slides.py
MECHS = [("maskable", "M"), ("repair", "R"), ("lagrangian", "L"), ("unconstraint", "U")]
LEARNERS = ["PPO", "DQN", "DDQN"]
INK, MUTED, GRID, RANGE, C_ILP, C_RL = "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c3", "#8a8984", "#2a78d6"

STYLES = {
    "paper": dict(rc={"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                      "mathtext.fontset": "stix", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42},
                  gap_size=(7.0, 3.3), time_size=(7.0, 2.6), ms=5, lw=0.8, out=FIGS, exts=("pdf", "eps", "png")),
}


def setup(style):
    plt.rcdefaults()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                         "axes.linewidth": 0.6, "savefig.facecolor": "white", **STYLES[style]["rc"]})
    return STYLES[style]


def save(fig, S, name):
    S["out"].mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in S["exts"]:                            # EPS: opaque, no transparency anywhere
        fig.savefig(S["out"] / f"{name}.{ext}", dpi=300 if ext == "png" else None, transparent=False)
    plt.close(fig)


def gap_fig(style):
    S = setup(style)
    cells = {(c["strategy"], c["RL_algorithm"]): c for c in json.loads((HERE / "table_2d.json").read_text())["cells"]}
    rows = []                                        # top to bottom, a blank row between mechanisms
    for i, (key, tag) in enumerate(MECHS):
        rows += ([None] if i else []) + [(f"{l}-{tag}", cells[key, l]) for l in LEARNERS]
    top = len(rows) - 1
    fig, ax = plt.subplots(figsize=S["gap_size"])
    ticks = {}
    for k, r in enumerate(rows):
        yy = top - k
        if r is None:
            ax.axhline(yy, color=GRID, lw=S["lw"] * 0.8)
            continue
        name, c = r
        ticks[name] = yy
        g = [100 * c["gap"][s] for s, _ in SCENS]
        ax.plot([min(g), max(g)], [yy, yy], color=RANGE, lw=S["lw"] * 2.5, solid_capstyle="round", zorder=1)
        for (s, _), v in zip(SCENS, g):
            col, m = SCEN_STYLE[s]
            ax.plot(v, yy, m, color=col, ms=S["ms"], mec="white", mew=S["lw"] * 0.8, zorder=3)
        mean = 100 * c["mean"]
        ax.plot(mean, yy, "|", color=INK, ms=S["ms"] * 1.9, mew=S["lw"] * 2, zorder=4)
        ax.text(1.005, yy, f"{mean:.2f}%", transform=ax.get_yaxis_transform(), va="center", ha="left", color=INK,
                fontweight="bold" if name == "PPO-M" else "normal")
    ax.set_yticks(list(ticks.values()), list(ticks.keys()))
    for t in ax.get_yticklabels():
        t.set_fontweight("bold" if t.get_text() == "PPO-M" else "normal")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, top + 0.7)
    ax.set_xlim(0, 32)
    ax.grid(axis="x", color=GRID, lw=S["lw"] * 0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Relative gap (%)")
    ax.text(1.005, top + 0.7, "Mean", transform=ax.get_yaxis_transform(), ha="left", va="bottom", color=MUTED)
    hs = [Line2D([], [], ls="", marker=SCEN_STYLE[s][1], color=SCEN_STYLE[s][0], ms=S["ms"], mec="white", label=n)
          for s, n in SCENS]
    hs.append(Line2D([], [], ls="", marker="|", color=INK, ms=S["ms"] * 1.6, mew=S["lw"] * 2,
                     label="Mean of 3 scenarios"))
    ax.legend(handles=hs, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, handletextpad=0.3,
              borderaxespad=0.2)
    save(fig, S, "fig_gap_scenarios")


def time_fig(style):
    S = setup(style)
    d = json.loads((HERE / "time_full.json").read_text())
    groups = SCENS + [(None, "All")]
    mean = lambda v, s: float(np.mean(v[s]["ms"] if s else np.concatenate([v[k]["ms"] for k, _ in SCENS])))
    ilp = [mean(d["ilp"], s) for s, _ in groups]
    rl = [[mean(v, s) for v in d["rl"].values()] for s, _ in groups]
    assert all(len(r) == 12 for r in rl)
    rm = [float(np.mean(r)) for r in rl]
    x, w = np.arange(len(groups)), 0.36
    fig, ax = plt.subplots(figsize=S["time_size"])
    ax.bar(x - w / 2 - 0.01, ilp, w, color=C_ILP, label="ILP")
    ax.bar(x + w / 2 + 0.01, rm, w, color=C_RL, label="RL (mean of 12 variants, whisker = min–max)",
           yerr=[[m - min(r) for m, r in zip(rm, rl)], [max(r) - m for m, r in zip(rm, rl)]],
           error_kw=dict(ecolor=INK, elinewidth=S["lw"], capsize=S["ms"] * 0.5))
    for i in range(len(groups)):
        ax.text(x[i] - w / 2, ilp[i] * 1.12, f"{ilp[i]:.0f}", ha="center", va="bottom", color=INK)
        ax.text(x[i] + w / 2, max(rl[i]) * 1.15, f"{rm[i]:.0f}", ha="center", va="bottom", color=INK)
        ax.text(x[i], 1.2e4, f"{ilp[i] / max(rl[i]):.1f}–{ilp[i] / min(rl[i]):.1f}× faster",
                ha="center", va="bottom", color=INK, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(10, 3e4)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(x, [n for _, n in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Mean online time per instance (ms)")
    ax.grid(axis="y", color=GRID, lw=S["lw"] * 0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    save(fig, S, "fig_time_scenarios")


if __name__ == "__main__":
    for style in STYLES:
        gap_fig(style)
        time_fig(style)
