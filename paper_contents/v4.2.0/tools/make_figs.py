"""v4.2.0 figures from ../final_summary_data.json -> ../figs/

  {ar,success,violation}_{lt,eq,gt}.png   one metric per figure
  timing_{lt,eq,gt}.png                   ILP vs RL time per test instance
  summary_table.png, timing_table.png

Usage: python paper_contents/v4.2.0/tools/make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.unicode_minus"] = False

PKG = Path(__file__).resolve().parents[1]
FIGDIR = PKG / "figs"
FIGDIR.mkdir(exist_ok=True)
d = json.loads((PKG / "final_summary_data.json").read_text())

ALGO_ORDER = ["ppo", "ppo_mask", "ppo_lagrangian", "ppo_opt", "dqn", "ddqn"]
DISPLAY = {
    "ppo_mask": "Mask-PPO",
    "ppo_lagrangian": "Lagrange-PPO",
    "ppo": "PPO",
    "ppo_opt": "Repair-PPO",
    "dqn": "DQN",
    "ddqn": "DDQN",
}
COLORS = ["#1aaf7a", "#2a7ad5", "#eb6834", "#eda100", "#e87ba4", "#008300"]
SCENARIOS = [
    ("lt", "LT (resource-scarce, N=10 < M=15)"),
    ("eq", "EQ (balanced, N=10 = M=10)"),
    ("gt", "GT (resource-rich, N=15 > M=10)"),
]


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.8)
    ax.set_axisbelow(True)


def new_fig(title):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.patch.set_facecolor("#fcfcfa")
    ax.set_facecolor("#fcfcfa")
    ax.set_title(title, fontsize=13, pad=12)
    return fig, ax


def finish(fig, ax, x, labels, out, note=None):
    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
    if note:
        fig.text(0.5, 0.015, note, ha="center", fontsize=8.5, color="#444444")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", out)


for scen, title in SCENARIOS:
    S = d["scenarios"][scen]
    A = [S["algos"][a] for a in ALGO_ORDER]
    labels = [DISPLAY[a] for a in ALGO_ORDER]
    x = np.arange(len(ALGO_ORDER))

    # Allocation ratio
    fig, ax = new_fig(f"{title}\nAllocation Ratio (mean ± std)")
    ar = [v["ar_mean"] for v in A]
    ax.bar(x, ar, yerr=[v["ar_std"] for v in A], capsize=4, color=COLORS, width=0.62,
           error_kw={"ecolor": "#555555"})
    ax.axhline(S["ilp_ar"], ls="--", color="black", lw=1.6)
    ax.text(-0.3, S["ilp_ar"] + 0.012, f"ILP optimum {S['ilp_ar']:.3f}", fontsize=10)
    for xi, m in zip(x, ar):
        ax.text(xi, m - 0.045, f"{m:.3f}", ha="center", color="white", fontsize=10)
    ax.set_ylim(0, max(0.8, S["ilp_ar"] + 0.06))
    finish(fig, ax, x, labels, FIGDIR / f"ar_{scen}.png")

    # Success rate
    fig, ax = new_fig(f"{title}\nSuccess Rate (mean ± std)")
    sm = [v["success_mean"] for v in A]
    ss = [v["success_std"] for v in A]
    ax.bar(x, sm, yerr=ss, capsize=4, color=COLORS, width=0.62, error_kw={"ecolor": "#555555"})
    for xi, m, sd in zip(x, sm, ss):
        ax.text(xi, m + sd + 0.03, f"{m:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.12)
    finish(fig, ax, x, labels, FIGDIR / f"success_{scen}.png")

    # Violations
    fig, ax = new_fig(f"{title}\nFraction of test episodes with ≥1 violation")
    w = 0.36
    cap = [v["cap_viol_mean"] for v in A]
    conf = [v["conflict_viol_mean"] for v in A]
    ax.bar(x - w / 2, cap, width=w, color="#2a7ad5", label="Capacity violation")
    ax.bar(x + w / 2, conf, width=w, color="#eb6834", label="Conflict violation")
    for xi, c, f in zip(x, cap, conf):
        ax.text(xi - w / 2, c + 0.015, f"{c:.2f}", ha="center", fontsize=8.5)
        ax.text(xi + w / 2, f + 0.015, f"{f:.2f}", ha="center", fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=10)
    finish(fig, ax, x, labels, FIGDIR / f"violation_{scen}.png",
           note="Repair-PPO: 0 by construction (invalid actions are repaired before execution).")

# ---------- timing_{scen}.png ----------
for scen, title in SCENARIOS:
    S = d["scenarios"][scen]
    names = ["ILP"] + [DISPLAY[a] for a in ALGO_ORDER]
    ms = [S["ilp_timing_ms"]["ms_mean"]] + [S["algos"][a]["ms_per_episode_mean"] for a in ALGO_ORDER]
    err = [S["ilp_timing_ms"]["ms_std"]] + [S["algos"][a]["ms_per_episode_std"] for a in ALGO_ORDER]
    x = np.arange(len(names))
    fig, ax = new_fig(f"{title}\nSolve time per test instance (ms, log scale)")
    # log axis: clip the lower whisker so it stays visible (ILP std > mean on LT)
    lower = [min(e, m * 0.6) for m, e in zip(ms, err)]
    ax.bar(x, ms, yerr=[lower, err], capsize=4, color=["#555555"] + COLORS, width=0.62,
           error_kw={"ecolor": "#333333"})
    ax.set_yscale("log")
    for xi, m, e in zip(x, ms, err):
        ax.text(xi, (m + e) * 1.15, f"{m:.1f}", ha="center", fontsize=10)
    ax.set_ylim(1, max(m + e for m, e in zip(ms, err)) * 4)
    finish(fig, ax, x, names, FIGDIR / f"timing_{scen}.png",
           note="ILP: mean ± std over 400 test instances. RL: mean over seeds (± std across seeds); 1 CPU thread.")

# ---------- timing_table.png ----------
t_rows = []
for scen, _ in SCENARIOS:
    S = d["scenarios"][scen]
    il = S["ilp_timing_ms"]
    t_rows.append([scen.upper(), "ILP (Dinkelbach + CBC)", f"{il['ms_mean']:.1f} ± {il['ms_std']:.1f}", "—", "1.0×"])
    for a in ALGO_ORDER:
        v = S["algos"][a]
        t_rows.append([scen.upper(), DISPLAY[a], f"{v['ms_per_episode_mean']:.2f} ± {v['ms_per_episode_std']:.2f}",
                       f"{v['ms_per_decision_mean']:.2f}", f"{il['ms_mean'] / v['ms_per_episode_mean']:.1f}×"])
t_labels = ["Scenario", "Method", "Time per instance (ms)", "Time per decision (ms)", "Speed-up vs ILP"]
fig, ax = plt.subplots(figsize=(13, 0.42 * (len(t_rows) + 1) + 1.0))
fig.patch.set_facecolor("#fcfcfa")
ax.axis("off")
ax.set_title("Solve time: ILP vs RL (3 scenarios)", fontsize=15, pad=10)
table = ax.table(cellText=t_rows, colLabels=t_labels, cellLoc="center", loc="upper center",
                 colWidths=[0.1, 0.25, 0.22, 0.22, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.8)
for (i, j), cell in table.get_celld().items():
    cell.set_edgecolor("#dddddd")
    if i == 0:
        cell.set_facecolor("#eef1f7")
        cell.set_text_props(fontweight="bold")
    elif ((i - 1) // 7) % 2 == 1:
        cell.set_facecolor("#f7f7f4")
fig.tight_layout()
out = FIGDIR / "timing_table.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print("wrote", out)

# ---------- summary_table.png ----------
col_labels = ["Scenario", "Algorithm", "AR (mean ± std)", "Success rate", "Cap. viol.", "Conflict viol.",
              "Repair trigger (cap. / conflict)", "Seeds"]
rows = []
for scen, _ in SCENARIOS:
    for a in ALGO_ORDER:
        v = d["scenarios"][scen]["algos"][a]
        rows.append([scen.upper(), DISPLAY[a], f"{v['ar_mean']:.4f}±{v['ar_std']:.4f}",
                     f"{v['success_mean']:.4f}±{v['success_std']:.4f}",
                     f"{v['cap_viol_mean']:.4f}", f"{v['conflict_viol_mean']:.4f}",
                     (f"{v['repair_trigger_cap_mean']:.4f} / {v['repair_trigger_conflict_mean']:.4f}"
                      if "repair_trigger_cap_mean" in v else "—"),
                     str(v["n_seeds"])])

fig, ax = plt.subplots(figsize=(19, 0.42 * (len(rows) + 1) + 1.0))
fig.patch.set_facecolor("#fcfcfa")
ax.axis("off")
ax.set_title("Test-set results, v4.2.0 (3 scenarios × 6 algorithms)", fontsize=15, pad=10)
table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="upper center",
                 colWidths=[0.08, 0.13, 0.15, 0.15, 0.10, 0.11, 0.20, 0.08])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.8)
for (i, j), cell in table.get_celld().items():
    cell.set_edgecolor("#dddddd")
    if i == 0:
        cell.set_facecolor("#eef1f7")
        cell.set_text_props(fontweight="bold")
    elif ((i - 1) // 6) % 2 == 1:
        cell.set_facecolor("#f7f7f4")
fig.tight_layout()
out = FIGDIR / "summary_table.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print("wrote", out)
