"""v4.2.0 figures from ../final_summary_data.json -> ../figs/

Visual style follows paper_contents/campaign_5seed_figs/ (matplotlib defaults,
tab10 colour per algorithm, black capped error bars, percent axes), one metric
per figure:

  ar_{lt,eq,gt}.png  success_{lt,eq,gt}.png  violation_{lt,eq,gt}.png
  timing_{lt,eq,gt}.png  summary_table.png  timing_table.png

Usage: python paper_contents/v4.2.0/tools/make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PKG = Path(__file__).resolve().parents[1]
FIGDIR = PKG / "figs"
FIGDIR.mkdir(exist_ok=True)
d = json.loads((PKG / "final_summary_data.json").read_text())

ALGO_ORDER = ["ppo", "ppo_mask", "ppo_lagrangian", "ppo_opt", "dqn", "ddqn"]
DISPLAY = {"ppo": "PPO", "ppo_mask": "Mask-PPO", "ppo_lagrangian": "Lagrange-PPO",
           "ppo_opt": "Repair-PPO", "dqn": "DQN", "ddqn": "DDQN"}
TICK = {"ppo": "PPO", "ppo_mask": "Mask-\nPPO", "ppo_lagrangian": "Lagrange-\nPPO",
        "ppo_opt": "Repair-\nPPO", "dqn": "DQN", "ddqn": "DDQN"}
# same colour per algorithm as campaign_5seed_figs
COLOR = {"ppo_mask": "#1f77b4", "ppo_lagrangian": "#ff7f0e", "ppo_opt": "#2ca02c",
         "ppo": "#d62728", "dqn": "#9467bd", "ddqn": "#8c564b"}
CAP_C, CONF_C = "#e07b39", "#3b6fa0"
SCENARIOS = [
    ("lt", "LT scenario (resource-scarce, N=10, M=15)"),
    ("eq", "EQ scenario (balanced, N=10, M=10)"),
    ("gt", "GT scenario (resource-rich, N=15, M=10)"),
]
NM = {"lt": "N=10,M=15", "eq": "N=10,M=10", "gt": "N=15,M=10"}

x = np.arange(len(ALGO_ORDER))
labels = [TICK[a] for a in ALGO_ORDER]
colors = [COLOR[a] for a in ALGO_ORDER]


def save(fig, name):
    fig.tight_layout()
    out = FIGDIR / name
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


for scen, title in SCENARIOS:
    S = d["scenarios"][scen]
    A = [S["algos"][a] for a in ALGO_ORDER]

    # Allocation ratio
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, [v["ar_mean"] for v in A], yerr=[v["ar_std"] for v in A], capsize=4, color=colors)
    ax.axhline(S["ilp_ar"], ls="--", color="black", lw=1, label=f"ILP (Optimal) = {S['ilp_ar']:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AR (mean ± std over seeds)")
    ax.set_title(f"{title}\nAllocation Ratio (AR)")
    ax.legend(loc="lower right")
    save(fig, f"ar_{scen}.png")

    # Success rate (%)
    fig, ax = plt.subplots(figsize=(7, 5))
    sm = [v["success_mean"] * 100 for v in A]
    ss = [v["success_std"] * 100 for v in A]
    ax.bar(x, sm, yerr=ss, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Success rate % (mean ± std over seeds)")
    ax.set_title(f"{title}\nTest-set Success Rate")
    ax.set_ylim(0, 105)
    save(fig, f"success_{scen}.png")

    # Violations (%)
    fig, ax = plt.subplots(figsize=(7, 5))
    w = 0.35
    ax.bar(x - w / 2, [v["cap_viol_mean"] * 100 for v in A], width=w, color=CAP_C, label="CapViol%")
    ax.bar(x + w / 2, [v["conflict_viol_mean"] * 100 for v in A], width=w, color=CONF_C, label="ConflictViol%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    top = max([v["cap_viol_mean"] for v in A] + [v["conflict_viol_mean"] for v in A]) * 100
    ax.set_ylim(0, max(top * 1.35, 10))  # headroom so the legend does not cover bars
    ax.set_ylabel("Violation rate %")
    ax.set_title(f"{title}\nTest-set Violation Rates (delivered placement)\n"
                 "(Repair-PPO is always 0 by construction — repair replaces\n"
                 "the action before delivery)", fontsize=10)
    ax.legend(loc="upper left")
    save(fig, f"violation_{scen}.png")

    # Solve time (ms, log scale)
    names = ["ILP"] + [TICK[a] for a in ALGO_ORDER]
    ms = [S["ilp_timing_ms"]["ms_mean"]] + [S["algos"][a]["ms_per_episode_mean"] for a in ALGO_ORDER]
    err = [S["ilp_timing_ms"]["ms_std"]] + [S["algos"][a]["ms_per_episode_std"] for a in ALGO_ORDER]
    lower = [min(e, m * 0.6) for m, e in zip(ms, err)]  # keep lower whisker visible on log axis
    xt = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(xt, ms, yerr=[lower, err], capsize=4, color=["#7f7f7f"] + colors)
    ax.set_yscale("log")
    for xi, m, e in zip(xt, ms, err):
        ax.text(xi, (m + e) * 1.12, f"{m:.1f}", ha="center", fontsize=9)
    ax.set_ylim(1, max(m + e for m, e in zip(ms, err)) * 4)
    ax.set_xticks(xt)
    ax.set_xticklabels(names)
    ax.set_ylabel("Time per test instance (ms, log scale)")
    ax.set_title(f"{title}\nSolve Time: ILP vs RL (1 CPU thread)")
    save(fig, f"timing_{scen}.png")


def styled_table(rows, col_labels, highlight, title, name, figsize, col_widths=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=20)
    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center",
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#c9d6ea")
        table[(0, j)].set_text_props(fontweight="bold")
    for i, hl in enumerate(highlight, start=1):
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor("#fdf1c7" if hl else "white")
            if hl:
                table[(i, j)].set_text_props(fontweight="bold")
    fig.tight_layout()
    out = FIGDIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------- summary_table.png ----------
rows, hl = [], []
for scen, _ in SCENARIOS:
    S = d["scenarios"][scen]
    rows.append([f"{scen.upper()}\n({NM[scen]})", "ILP (Optimal)", f"{S['ilp_ar']:.4f}", "100.0",
                 "0.00", "0.00", "—", "—"])
    hl.append(True)
    for a in ALGO_ORDER:
        v = S["algos"][a]
        trig = (f"{v['repair_trigger_cap_mean'] * 100:.2f} / {v['repair_trigger_conflict_mean'] * 100:.2f}"
                if "repair_trigger_cap_mean" in v else "—")
        rows.append(["", DISPLAY[a], f"{v['ar_mean']:.4f}±{v['ar_std']:.4f}",
                     f"{v['success_mean'] * 100:.1f}±{v['success_std'] * 100:.1f}",
                     f"{v['cap_viol_mean'] * 100:.2f}", f"{v['conflict_viol_mean'] * 100:.2f}",
                     trig, str(v["n_seeds"])])
        hl.append(False)
styled_table(
    rows,
    ["Scenario", "Algorithm", "AR (mean±std)", "Success%\n(mean±std)", "CapViol%", "ConflictViol%",
     "Repair trigger %\n(cap / conflict)", "Seeds"],
    hl,
    "v4.2.0 — test-set results (5M steps, single deterministic pass per test instance)\n"
    "CapViol% / ConflictViol% = violation present in the DELIVERED placement, same definition for every algorithm\n"
    "(ILP AR: seed-1 test set)",
    "summary_table.png", (20, 12.5),
    [0.09, 0.14, 0.14, 0.12, 0.1, 0.11, 0.16, 0.07])

# ---------- timing_table.png ----------
rows, hl = [], []
for scen, _ in SCENARIOS:
    S = d["scenarios"][scen]
    il = S["ilp_timing_ms"]
    rows.append([f"{scen.upper()}\n({NM[scen]})", "ILP (Dinkelbach + CBC)",
                 f"{il['ms_mean']:.1f}±{il['ms_std']:.1f}", "—", "1.0×"])
    hl.append(True)
    for a in ALGO_ORDER:
        v = S["algos"][a]
        rows.append(["", DISPLAY[a], f"{v['ms_per_episode_mean']:.2f}±{v['ms_per_episode_std']:.2f}",
                     f"{v['ms_per_decision_mean']:.2f}", f"{il['ms_mean'] / v['ms_per_episode_mean']:.1f}×"])
        hl.append(False)
styled_table(
    rows,
    ["Scenario", "Method", "Time per instance\n(ms, mean±std)", "Time per decision\n(ms)", "Speed-up\nvs ILP"],
    hl,
    "v4.2.0 — solve time per test instance (1 CPU thread)\n"
    "ILP: mean±std over 400 test instances; RL: mean±std across seeds (each seed averaged over its 400 test instances)",
    "timing_table.png", (14, 12.5),
    [0.14, 0.26, 0.22, 0.2, 0.15])
