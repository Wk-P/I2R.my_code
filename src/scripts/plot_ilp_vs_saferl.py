"""ILP vs Safe RL comparison figure (docs/ilp_vs_saferl.md).

(a) who does what: human design / solver search / learned by training, offline vs online.
(b) online work per instance: log10 of the worst-case search space N^M versus the inference cost
    M L (T^2 d + T d^2) of the Transformer policy (T = 1+N+M, L=3, d=128). Worst-case orders, not measured time.

    python src/scripts/plot_ilp_vs_saferl.py   # -> paper_contents/ilp_vs_saferl/fig_ilp_vs_saferl.{pdf,eps}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]

OUT = ROOT / "paper_contents" / "ilp_vs_saferl"
COL = {"human": "#E8A33D", "solver": "#4C78A8", "learned": "#54A24B", "data": "#BBBBBB"}


def box(ax, x, y, w, h, text, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=COL[kind], ec="black", lw=0.6, alpha=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2,
            color="white" if kind in ("solver", "learned") else "black")


def arrow(ax, x0, x1, y):
    ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black"))


def panel_a(ax):
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.axvline(5.55, ymin=0.04, ymax=0.9, ls="--", lw=0.7, color="gray")
    ax.text(2.7, 4.35, "offline (once)", ha="center", fontsize=8, style="italic")
    ax.text(8.0, 4.35, "online (every instance)", ha="center", fontsize=8, style="italic")
    h = 0.85
    # ILP row
    ax.text(0.0, 3.25, "ILP", fontsize=9, weight="bold", va="center")
    box(ax, 0.8, 2.85, 2.2, h, "write model\n(variables, objective,\nconstraints)", "human")
    box(ax, 3.3, 2.85, 2.0, h, "Dinkelbach\nlinearisation", "human")
    box(ax, 5.8, 2.85, 1.3, h, "instance\n$c_n, e_m, C_k$", "data")
    box(ax, 7.4, 2.85, 2.9, h, "branch & bound search\n(exponential worst case)", "solver")
    arrow(ax, 7.1, 7.4, 3.27)
    # RL row
    ax.text(0.0, 1.25, "Safe\nRL", fontsize=9, weight="bold", va="center")
    box(ax, 0.8, 1.5, 2.2, 0.75, "MDP, reward,\nconstraint mechanism", "human")
    box(ax, 0.8, 0.55, 2.2, 0.75, "network, features,\nhyperparameters", "human")
    box(ax, 3.3, 0.85, 2.0, 1.1, "PPO training\n→ policy $\\pi_\\theta$\n(all parameters)", "learned")
    arrow(ax, 3.0, 3.3, 1.4)
    box(ax, 5.8, 0.98, 1.3, h, "instance\n$c_n, e_m, C_k$", "data")
    box(ax, 7.4, 0.98, 2.9, h, "$M$ forward passes\n(polynomial)", "learned")
    arrow(ax, 7.1, 7.4, 1.4)
    # legend
    for i, (k, t) in enumerate([("human", "human design"), ("solver", "solver search"),
                                ("learned", "learned / run by the policy"), ("data", "input data")]):
        ax.add_patch(FancyBboxPatch((0.8 + 2.6 * i, 0.02), 0.25, 0.22, boxstyle="round,pad=0.01", fc=COL[k], ec="black", lw=0.5))
        ax.text(1.12 + 2.6 * i, 0.13, t, fontsize=7, va="center")
    ax.set_title("(a) Who does the work", fontsize=9, loc="left")


def panel_b(ax):
    import math
    L, d = 3, 128
    sizes = list(range(5, 101, 5))                       # N = M
    space = [s * math.log10(s) for s in sizes]           # log10 N^M: assignments an exact method may have to rule out
    ops = [math.log10(s * L * ((1 + 2 * s) ** 2 * d + (1 + 2 * s) * d * d)) for s in sizes]   # T = 1 + N + M
    ax.plot(sizes, space, "-", lw=1.8, color=COL["solver"], label="ILP: worst-case search space $N^M$")
    ax.plot(sizes, ops, "-", lw=1.8, color=COL["learned"], label="Safe RL: inference $M\\,L\\,(T^2d+Td^2)$")
    ax.axvspan(10, 15, color="#E6E6E6", lw=0, zorder=0)
    ax.text(12.5, 150, "our\nscenarios", fontsize=6.5, ha="center", color="dimgray")
    ax.set_xlabel("problem size $N=M$", fontsize=8); ax.set_ylabel("$\\log_{10}$ (count)", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6.8, loc="upper left", frameon=False)
    ax.set_title("(b) Online work per instance", fontsize=9, loc="left")
    ax.text(0.98, 0.22, "exponential vs. polynomial\n(worst case; $L=3$, $d=128$)", transform=ax.transAxes,
            fontsize=6.5, ha="right", va="bottom", color="dimgray")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw=dict(width_ratios=[1.75, 1]))
    panel_a(a); panel_b(b)
    fig.tight_layout()
    for ext in ("pdf", "eps", "svg"):
        fig.savefig(OUT / f"fig_ilp_vs_saferl.{ext}", facecolor="white")
    fig.savefig(OUT / "fig_ilp_vs_saferl.png", dpi=300, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
