"""Main-text solve-time figure: ILP vs PPO-M on the full test set, two questions only (docs/ilp_vs_saferl.md section 7).

  (a) How much faster?  per scenario, the per-instance time of ILP and PPO-M (box: IQR, whiskers 5th-95th pct,
      white line: median), median written next to each box, speed-up of the medians on the right.
  (b) At what cost?     mean AR of ILP and PPO-M per scenario, relative AR gap written above.
The 12-variant detail stays in fig_time_full (appendix). Data: paper_contents/ilp_vs_saferl/time_full.json.

    .venv/bin/python src/scripts/plot_time_simple.py   # -> paper_contents/ilp_vs_saferl/fig_time_simple.{pdf,eps,svg,png}
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
SCEN = [("lt", "LT"), ("eq", "EQ"), ("gt", "GT")]
C_ILP, C_RL = "#4C78A8", "#54A24B"


def fmt(ms):
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def hbox(ax, y, v, col):
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    lo, hi = np.percentile(v, [5, 95])
    ax.plot([lo, hi], [y, y], color="black", lw=0.8, zorder=1)
    for x in (lo, hi):
        ax.plot([x, x], [y - 0.12, y + 0.12], color="black", lw=0.8, zorder=1)
    ax.add_patch(plt.Rectangle((q1, y - 0.3), q3 - q1, 0.6, fc=col, ec="black", lw=0.6, zorder=2))
    ax.plot([med, med], [y - 0.3, y + 0.3], color="white", lw=2, zorder=3)
    return med, hi


def main():
    d = json.loads(SRC.read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1]))

    # (a) how much faster
    ticks, labels = [], []
    for j, (s, sl) in enumerate(SCEN):
        y0 = j * 3
        ilp, rl = np.asarray(d["ilp"][s]["ms"]), np.asarray(d["rl"]["mask|ppo"][s]["ms"])
        m_ilp, hi_ilp = hbox(a, y0 + 0.5, ilp, C_ILP)
        m_rl, hi_rl = hbox(a, y0 - 0.5, rl, C_RL)
        a.text(hi_ilp * 1.15, y0 + 0.5, f"median {fmt(m_ilp)}", va="center", fontsize=7.5)
        a.text(hi_rl * 1.15, y0 - 0.5, f"median {fmt(m_rl)}", va="center", fontsize=7.5)
        a.text(1.02, y0, f"{m_ilp / m_rl:.0f}× faster", transform=a.get_yaxis_transform(), va="center",
               fontsize=9, weight="bold", color=C_RL)
        ticks += [y0 + 0.5, y0 - 0.5]; labels += [f"{sl}  ILP", f"{sl}  PPO-M"]
    a.set_yticks(ticks); a.set_yticklabels(labels, fontsize=8)
    a.set_xscale("log"); a.set_xlabel("time per instance (ms, log scale)", fontsize=8); a.tick_params(axis="x", labelsize=7)
    a.set_ylim(-1.3, 7.3); a.invert_yaxis()
    a.set_title("(a) How much faster?", fontsize=9.5, loc="left", weight="bold")
    a.text(0.0, -0.2, "box: middle 50% of instances, white line: median, whiskers: 5th–95th percentile",
           transform=a.transAxes, fontsize=6.5, color="dimgray")
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

    # (b) at what cost
    w = 0.36
    for j, (s, sl) in enumerate(SCEN):
        y_ilp, y_rl = float(np.mean(d["ilp"][s]["ar"])), float(np.mean(d["rl"]["mask|ppo"][s]["ar"]))
        b.bar(j - w / 2, y_ilp, w, color=C_ILP, ec="black", lw=0.5, label="ILP (optimal)" if j == 0 else None)
        b.bar(j + w / 2, y_rl, w, color=C_RL, ec="black", lw=0.5, label="PPO-M" if j == 0 else None)
        b.text(j - w / 2, y_ilp + 0.004, f"{y_ilp:.3f}", ha="center", va="bottom", fontsize=7)
        b.text(j + w / 2, y_rl + 0.004, f"{y_rl:.3f}", ha="center", va="bottom", fontsize=7)
        b.text(j, max(y_ilp, y_rl) + 0.03, f"gap {100 * (y_ilp - y_rl) / y_ilp:.1f}%", ha="center", fontsize=9,
               weight="bold", color="#B23A48")
    b.set_xticks(range(len(SCEN))); b.set_xticklabels([sl for _, sl in SCEN], fontsize=8.5)
    b.set_ylim(0.6, 1.02); b.set_ylabel("mean AR", fontsize=8); b.tick_params(axis="y", labelsize=7)
    b.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2)
    b.set_title("(b) At what cost?", fontsize=9.5, loc="left", weight="bold")
    b.text(0.0, -0.13, "gap = (ILP AR − PPO-M AR) / ILP AR; y-axis starts at 0.6", transform=b.transAxes,
           fontsize=6.5, color="dimgray")
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)

    n = d["n_inst"]
    fig.suptitle(f"ILP vs PPO-M on the full test set ({n['lt']} instances per scenario, 1 CPU thread)",
                 fontsize=9, x=0.01, ha="left", color="dimgray")
    fig.tight_layout()
    for ext in ("pdf", "eps", "svg"):
        fig.savefig(OUT / f"fig_time_simple.{ext}", facecolor="white")
    fig.savefig(OUT / "fig_time_simple.png", dpi=300, facecolor="white")
    print(OUT / "fig_time_simple.pdf")


if __name__ == "__main__":
    main()
