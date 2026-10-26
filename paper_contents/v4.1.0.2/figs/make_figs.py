"""Regenerate bars_{lt,eq,gt}.png and summary_table.png from final_summary_data.json.

Usage: python paper_contents/v4.1.0.2/figs/make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.unicode_minus"] = False

FIGDIR = Path(__file__).resolve().parent
d = json.loads((FIGDIR / "final_summary_data.json").read_text())

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
    ("lt", "LT scenario (resource-scarce, N=10 < M=15)"),
    ("eq", "EQ scenario (balanced, N=10 = M=10)"),
    ("gt", "GT scenario (resource-rich, N=15 > M=10)"),
]
FOOTNOTE = ("PPO / Mask-PPO / Lagrange-PPO: 5M steps, 5 seeds (GT Lagrange-PPO: 3 seeds). Repair-PPO / DQN / DDQN: 5M steps, 3 seeds. "
            "Repair-PPO violations are 0 by construction (invalid actions are repaired before execution); "
            "its repair-trigger rate is reported separately in the table.")


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.8)
    ax.set_axisbelow(True)


for scen, title in SCENARIOS:
    S = d["scenarios"][scen]
    A = [S["algos"][a] for a in ALGO_ORDER]
    labels = [DISPLAY[a] for a in ALGO_ORDER]
    x = np.arange(len(ALGO_ORDER))

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("#fcfcfa")
    fig.suptitle(title, fontsize=16)

    ax = axes[0]
    ar = [v["ar_mean"] for v in A]
    ax.bar(x, ar, yerr=[v["ar_std"] for v in A], capsize=4, color=COLORS, width=0.62,
           error_kw={"ecolor": "#555555"})
    ax.axhline(S["ilp_ar"], ls="--", color="black", lw=1.8)
    ax.text(-0.3, S["ilp_ar"] + 0.012, f"ILP optimum {S['ilp_ar']:.3f}", fontsize=11)
    for xi, m in zip(x, ar):
        ax.text(xi, m - 0.04, f"{m:.3f}", ha="center", color="white", fontsize=11)
    ax.set_ylim(0, max(0.8, S["ilp_ar"] + 0.06))
    ax.set_title("Allocation Ratio (mean ± std)", fontsize=14, pad=14)

    ax = axes[1]
    sm = [v["success_mean"] for v in A]
    ss = [v["success_std"] for v in A]
    ax.bar(x, sm, yerr=ss, capsize=4, color=COLORS, width=0.62, error_kw={"ecolor": "#555555"})
    for xi, m, s in zip(x, sm, ss):
        ax.text(xi, m + s + 0.03, f"{m:.3f}", ha="center", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Success Rate (mean ± std)", fontsize=14, pad=14)

    ax = axes[2]
    w = 0.36
    cap = [v["cap_viol_mean"] for v in A]
    conf = [v["conflict_viol_mean"] for v in A]
    ax.bar(x - w / 2, cap, width=w, color="#2a7ad5", label="Capacity violation")
    ax.bar(x + w / 2, conf, width=w, color="#eb6834", label="Conflict violation")
    for xi, c, f in zip(x, cap, conf):
        ax.text(xi - w / 2, c + 0.015, f"{c:.2f}", ha="center", fontsize=10)
        ax.text(xi + w / 2, f + 0.015, f"{f:.2f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=11)
    ax.set_title("Fraction of test episodes with ≥1 violation", fontsize=14, pad=14)

    for ax in axes:
        style(ax)
        ax.set_facecolor("#fcfcfa")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)

    fig.text(0.5, 0.015, FOOTNOTE, ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = FIGDIR / f"bars_{scen}.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
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
ax.set_title("Test-set results (3 scenarios × 6 algorithms)", fontsize=15, pad=10)
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
