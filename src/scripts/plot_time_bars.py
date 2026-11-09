"""Simple solve-time bar charts on the full test set, all scenarios pooled (docs/ilp_vs_saferl.md section 7).

  fig_time_bar_ms      : mean time per instance, ILP + the 12 variants (one bar each, value on the bar)
  fig_time_bar_speedup : speed-up = ILP mean time / variant mean time, the 12 variants
Means are over every instance of LT, EQ and GT together. Data: paper_contents/ilp_vs_saferl/time_full.json.

    .venv/bin/python src/scripts/plot_time_bars.py   # -> paper_contents/ilp_vs_saferl/fig_time_bar_{ms,speedup}.{pdf,eps,svg,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_contents" / "ilp_vs_saferl"
SRC = OUT / "time_full.json"
SCENS = ("lt", "eq", "gt")
MECH = [("none", "U"), ("mask", "M"), ("repair", "R"), ("lagrange", "L")]
LEARN = [("ppo", "PPO", "#54A24B"), ("dqn", "DQN", "#E8A33D"), ("ddqn", "DDQN", "#B279A2")]
C_ILP = "#4C78A8"


def pooled(v):
    return float(np.mean(np.concatenate([v[s]["ms"] for s in SCENS])))


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "eps", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8)


def main():
    d = json.loads(SRC.read_text())
    n = sum(len(d["ilp"][s]["ms"]) for s in SCENS)
    ilp = pooled(d["ilp"])
    names, vals, cols = [], [], []
    for l, ln, c in LEARN:
        for m, mn in MECH:
            names.append(f"{ln}-{mn}"); vals.append(pooled(d["rl"][f"{m}|{l}"])); cols.append(c)
    note = f"{n:,} test instances (LT+EQ+GT), 1 CPU thread"

    # mean time
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(names) + 1)
    bars = ax.bar(x, [ilp] + vals, color=[C_ILP] + cols, ec="black", lw=0.5)
    for b, v in zip(bars, [ilp] + vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(["ILP"] + names, rotation=45, ha="right")
    ax.set_yscale("log"); ax.set_ylim(top=ilp * 3)
    ax.set_ylabel("mean time per instance (ms, log scale)", fontsize=8.5)
    ax.set_title(f"Mean solve time — {note}", fontsize=8.5, loc="left")
    style(ax)
    save(fig, "fig_time_bar_ms")

    # speed-up
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    sp = [ilp / v for v in vals]
    x = np.arange(len(names))
    bars = ax.bar(x, sp, color=cols, ec="black", lw=0.5)
    for b, v in zip(bars, sp):
        ax.text(b.get_x() + b.get_width() / 2, v + max(sp) * 0.01, f"{v:.0f}×", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, max(sp) * 1.15)
    ax.set_ylabel("speed-up over ILP (×)", fontsize=8.5)
    ax.set_title(f"Speed-up = ILP mean / variant mean time — {note}", fontsize=8.5, loc="left")
    style(ax)
    save(fig, "fig_time_bar_speedup")

    print(f"ILP {ilp:.1f} ms;", ", ".join(f"{a} {v:.1f} ms ({ilp / v:.1f}x)" for a, v in zip(names, vals)))


if __name__ == "__main__":
    main()
