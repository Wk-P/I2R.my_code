#!/usr/bin/env python3
"""Generate the final (v4.1.0.2) result figures from
paper_contents/v4.1.0.2/figs/final_summary_data.json.

Each (scenario, algo) cell in that JSON has mixed provenance (some algos keep
the v4.1.0 reward fix and were rerun at 5M/3seed; ppo_lagrangian was reverted
and reuses the v4.0.0 5M/5seed frozen numbers for lt/eq, with a fresh
5M/3seed gt-only run) -- see the JSON's "source" field per cell and
paper_contents/v4.1.0_changelog.md for the full story. This script only
renders what's already been decided; it does not aggregate raw run data.

Categorical palette follows the dataviz skill's validated 8-slot default
order (references/palette.md), first 6 slots for algorithm identity.

Usage:
    .venv/bin/python scripts/generate_final_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "paper_contents" / "v4.1.0.2" / "figs" / "final_summary_data.json"
OUT_DIR = PROJECT_ROOT / "paper_contents" / "v4.1.0.2" / "figs"

# dataviz skill validated categorical order (light mode), slots 1-6 for the
# 6 algorithms -- fixed order, never cycled/reassigned per chart.
ALGO_ORDER = ["ppo_mask", "ppo_lagrangian", "ppo", "ppo_opt", "dqn", "ddqn"]
ALGO_LABELS = {
    "ppo_mask": "Mask-PPO (P4)",
    "ppo_lagrangian": "Lagrange-PPO (P5)",
    "ppo": "PPO (P3)",
    "ppo_opt": "Repair-PPO (P6)",
    "dqn": "DQN",
    "ddqn": "DDQN",
}
ALGO_COLORS = {
    "ppo_mask":       "#2a78d6",  # slot 1 blue
    "ppo_lagrangian": "#eb6834",  # slot 2 orange
    "ppo":            "#1baf7a",  # slot 3 aqua
    "ppo_opt":        "#eda100",  # slot 4 yellow
    "dqn":            "#e87ba4",  # slot 5 magenta
    "ddqn":           "#008300",  # slot 6 green
}
# Violation-type encoding (a different semantic axis, its own 2-color pair)
VIOL_COLORS = {"cap": "#2a78d6", "conflict": "#eb6834"}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e3e2dd"

SCENARIO_LABELS = {"lt": "lt (资源紧缺, N=10<M=15)", "eq": "eq (供需均衡, N=M=10)", "gt": "gt (资源充裕, N=15>M=10)"}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def bar_with_labels(ax, x, values, errs, colors, fmt="{:.3f}", y_range=None, labels_inside=False):
    bars = ax.bar(x, values, yerr=errs, color=colors, width=0.62,
                   capsize=3, error_kw={"ecolor": TEXT_SECONDARY, "linewidth": 1.2}, zorder=3)
    if labels_inside:
        # AR bars can sit close to the ILP reference line -- an above-bar label
        # then collides with the line/its own annotation when several bars are
        # near-tied in height. Placing labels INSIDE the bar (near its top, in
        # white) keeps them clear of the line entirely regardless of how
        # tightly the bars cluster.
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.95,
                     fmt.format(v), ha="center", va="top", fontsize=8.5, color="#ffffff", zorder=5)
    else:
        offset = (y_range if y_range else max(values)) * 0.03
        for bar, v, err in zip(bars, values, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + offset,
                     fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=TEXT_PRIMARY)
    return bars


def make_scenario_figure(scenario: str, data: dict, out_path: Path):
    algos = ALGO_ORDER
    ar = [data["algos"][a]["ar_mean"] for a in algos]
    ar_err = [data["algos"][a]["ar_std"] for a in algos]
    succ = [data["algos"][a]["success_mean"] for a in algos]
    succ_err = [data["algos"][a]["success_std"] for a in algos]
    cap_v = [data["algos"][a]["cap_viol_mean"] for a in algos]
    conf_v = [data["algos"][a]["conflict_viol_mean"] for a in algos]
    colors = [ALGO_COLORS[a] for a in algos]
    x = np.arange(len(algos))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_facecolor("#fcfcfb")

    # Panel 1: AR with ILP reference line
    ax = axes[0]
    ax.set_facecolor("#fcfcfb")
    style_axis(ax)
    ilp = data["ilp_ar"]
    y_top = max(ar + [ilp]) * 1.20
    bar_with_labels(ax, x, ar, ar_err, colors, y_range=y_top, labels_inside=True)
    ax.axhline(ilp, color=TEXT_PRIMARY, linestyle="--", linewidth=1.4, zorder=4)
    ax.text(0.0, ilp + y_top * 0.02, f"ILP最优 {ilp:.3f}", fontsize=8.5,
            color=TEXT_PRIMARY, ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in algos], fontsize=8, rotation=20, ha="right")
    ax.set_ylim(0, y_top)
    ax.set_title("资源利用率 AR（均值±标准差）", fontsize=11, color=TEXT_PRIMARY, pad=10)

    # Panel 2: success_rate
    ax = axes[1]
    ax.set_facecolor("#fcfcfb")
    style_axis(ax)
    bar_with_labels(ax, x, succ, succ_err, colors, fmt="{:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in algos], fontsize=8, rotation=20, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_title("Success Rate（均值±标准差）", fontsize=11, color=TEXT_PRIMARY, pad=10)

    # Panel 3: violation rates (grouped, 2-color = violation TYPE, not algo identity)
    ax = axes[2]
    ax.set_facecolor("#fcfcfb")
    style_axis(ax)
    w = 0.35
    b1 = ax.bar(x - w / 2, cap_v, width=w, color=VIOL_COLORS["cap"], zorder=3, label="容量违规率")
    b2 = ax.bar(x + w / 2, conf_v, width=w, color=VIOL_COLORS["conflict"], zorder=3, label="冲突违规率")
    for bar, v in zip(list(b1) + list(b2), cap_v + conf_v):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in algos], fontsize=8, rotation=20, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("违规率（episode 级，出现过≥1次即计入）", fontsize=11, color=TEXT_PRIMARY, pad=10)

    fig.suptitle(f"场景：{SCENARIO_LABELS[scenario]} — v4.1.0.2 最终定案数据", fontsize=13,
                 color=TEXT_PRIMARY, y=1.03)
    fig.text(0.5, -0.04,
              "数据来源混合：部分算法沿用v4.1.0修复重跑数据(5M步/3种子)，ppo_lagrangian回退后复用v4.0.0冻结数据(5M步/5种子，gt除外)。"
              "各单元格具体来源见 final_summary_data.json。",
              ha="center", fontsize=7.5, color=TEXT_SECONDARY, wrap=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out_path}")


def make_summary_table(all_data: dict, out_path: Path):
    scenarios = ["lt", "eq", "gt"]
    algos = ALGO_ORDER
    rows = []
    for s in scenarios:
        for a in algos:
            d = all_data["scenarios"][s]["algos"][a]
            rows.append([
                s, ALGO_LABELS[a].replace("\n", " "),
                f"{d['ar_mean']:.4f}±{d['ar_std']:.4f}",
                f"{d['success_mean']:.4f}±{d['success_std']:.4f}",
                f"{d['cap_viol_mean']:.4f}", f"{d['conflict_viol_mean']:.4f}",
                f"{d['n_seeds']}",
            ])

    fig, ax = plt.subplots(figsize=(12, 0.42 * len(rows) + 1.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.axis("off")
    col_labels = ["场景", "算法", "AR(均值±标准差)", "success_rate", "cap_viol", "conflict_viol", "种子数"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.4)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if r == 0:
            cell.set_facecolor("#eef1f6")
            cell.set_text_props(weight="bold", color=TEXT_PRIMARY)
        else:
            cell.set_facecolor("#ffffff" if (r - 1) % 6 < 3 else "#fbfbfa")
            cell.set_text_props(color=TEXT_PRIMARY)
    ax.set_title("v4.1.0.2 最终定案数据汇总表（3场景×6算法）", fontsize=12, color=TEXT_PRIMARY, pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    all_data = json.loads(DATA_PATH.read_text())
    for scenario in ["lt", "eq", "gt"]:
        make_scenario_figure(scenario, all_data["scenarios"][scenario], OUT_DIR / f"bars_{scenario}.png")
    make_summary_table(all_data, OUT_DIR / "summary_table.png")


if __name__ == "__main__":
    main()
