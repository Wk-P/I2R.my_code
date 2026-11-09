"""Full evaluation of solve time on the whole test set: ILP vs the 12 Safe RL variants (docs/ilp_vs_saferl.md section 7).

Data: paper_contents/ilp_vs_saferl/time_full.json (src/scripts/bench_time_full.py, every instance timed separately).
  (a) per-instance time distribution of every method per scenario (box: quartiles, whiskers 1st-99th pct, + mean)
  (b) empirical CDF of the time, ILP vs PPO-M
  (c) per-instance speed-up ILP / PPO-M (same instance)
  (d) AR vs time per instance (quality-speed trade-off), ILP vs PPO-M
also writes fig_time_full_table.{md,en.md}: mean / median / p90 / p99 / max per method and scenario.

    .venv/bin/python src/scripts/plot_time_full.py   # -> paper_contents/ilp_vs_saferl/fig_time_full.{pdf,eps,svg,png}
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
MECH = [("none", "U"), ("mask", "M"), ("repair", "R"), ("lagrange", "L")]
LEARN = [("ppo", "PPO"), ("dqn", "DQN"), ("ddqn", "DDQN")]
C_ILP, C_MAIN = "#4C78A8", "#2E7D32"
C_LEARN = {"ppo": "#54A24B", "dqn": "#E8A33D", "ddqn": "#B279A2"}
C_SCEN = {"lt": "#4C78A8", "eq": "#E45756", "gt": "#72B7B2"}


def methods(d):
    yield "ILP", "ILP", C_ILP, lambda s: d["ilp"][s]
    for l, ln in LEARN:
        for m, mn in MECH:
            yield f"{m}|{l}", f"{ln}-{mn}", C_LEARN[l], (lambda s, k=f"{m}|{l}": d["rl"][k][s])


def panel_box(ax, d):
    ms = list(methods(d))
    step = len(ms) + 2
    for j, (s, sl) in enumerate(SCEN):
        for i, (_, name, col, get) in enumerate(ms):
            x = j * step + i
            v = np.asarray(get(s)["ms"])
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            lo, hi = np.percentile(v, [1, 99])
            ax.plot([x, x], [lo, hi], color="black", lw=0.5, zorder=1)
            ax.add_patch(plt.Rectangle((x - 0.38, q1), 0.76, q3 - q1, fc=col, ec="black", lw=0.4, zorder=2))
            ax.plot([x - 0.38, x + 0.38], [med, med], color="white", lw=1.0, zorder=3)
            ax.plot(x, v.mean(), "x", color="black", ms=3, mew=0.7, zorder=4)
        ax.text(j * step + (len(ms) - 1) / 2, 1.0, sl, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, weight="bold")
    ax.set_xticks([j * step + i for j in range(len(SCEN)) for i in range(len(ms))])
    ax.set_xticklabels([n for _ in SCEN for _, n, _, _ in ms], rotation=90, fontsize=5.5)
    ax.set_yscale("log"); ax.set_ylabel("time per instance (ms)", fontsize=7.5); ax.tick_params(axis="y", labelsize=6.5)
    ax.set_xlim(-1, len(SCEN) * step - 2)
    ax.set_title("(a) Per-instance time, every method (box: IQR, line: median, ×: mean, whiskers: 1st–99th pct)",
                 fontsize=8, loc="left", pad=12)


def panel_cdf(ax, d):
    for s, sl in SCEN:
        for v, ls, lab in ((d["ilp"][s]["ms"], "-", "ILP"), (d["rl"]["mask|ppo"][s]["ms"], "--", "PPO-M")):
            x = np.sort(v)
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), ls, color=C_SCEN[s], lw=1.2, label=f"{lab} {sl}")
    ax.set_xscale("log"); ax.set_xlabel("time per instance (ms)", fontsize=7.5); ax.set_ylabel("fraction of instances", fontsize=7.5)
    ax.tick_params(labelsize=6.5); ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower right")
    ax.set_title("(b) CDF of time: ILP (solid) vs PPO-M (dashed)", fontsize=8, loc="left")


def panel_speedup(ax, d):
    bins = np.logspace(0, 3.5, 50)
    for s, sl in SCEN:
        r = np.asarray(d["ilp"][s]["ms"]) / np.asarray(d["rl"]["mask|ppo"][s]["ms"])
        ax.hist(r, bins=bins, histtype="step", lw=1.2, color=C_SCEN[s],
                label=f"{sl}: median {np.median(r):.0f}×, {100 * (r > 1).mean():.1f}% > 1×")
    ax.axvline(1, color="black", lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("speed-up ILP / PPO-M (same instance)", fontsize=7.5); ax.set_ylabel("instances", fontsize=7.5)
    ax.tick_params(labelsize=6.5); ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.set_title("(c) Per-instance speed-up", fontsize=8, loc="left")


def panel_scatter(ax, d):
    rng = np.random.default_rng(0)
    for s, sl in SCEN:
        n = len(d["ilp"][s]["ms"])
        idx = rng.choice(n, size=min(n, 1500), replace=False)       # thinned for readability; statistics use all
        ax.scatter(np.asarray(d["ilp"][s]["ms"])[idx], np.asarray(d["ilp"][s]["ar"])[idx], s=2, marker="o",
                   color=C_SCEN[s], alpha=1.0, lw=0, label=f"ILP {sl}")
        ax.scatter(np.asarray(d["rl"]["mask|ppo"][s]["ms"])[idx], np.asarray(d["rl"]["mask|ppo"][s]["ar"])[idx], s=2,
                   marker="^", color=C_SCEN[s], lw=0, label=f"PPO-M {sl}")
    ax.set_xscale("log"); ax.set_xlabel("time per instance (ms)", fontsize=7.5); ax.set_ylabel("AR", fontsize=7.5)
    ax.tick_params(labelsize=6.5); ax.legend(fontsize=6, frameon=False, ncol=2, markerscale=3, loc="lower right")
    ax.set_title("(d) AR vs time per instance (○ ILP, △ PPO-M)", fontsize=8, loc="left")


def table(d, lang):
    T = (lambda z, e: z if lang == "zh" else e)
    L = [T("# 完整测试集单实例耗时", "# Per-instance time on the full test set"), "",
         T(f"- 数据 `paper_contents/ilp_vs_saferl/time_full.json`（{d['generated_at']}，`bench_time_full.py`），"
           f"每场景 {d['n_inst']} 个实例，逐实例计时，CPU 单线程，{d['workers']} 个进程并行。单位 ms。",
           f"- Data `paper_contents/ilp_vs_saferl/time_full.json` ({d['generated_at']}, `bench_time_full.py`), "
           f"{d['n_inst']} instances per scenario, each timed separately, one CPU thread, {d['workers']} processes in parallel. Unit ms."), ""]
    for s, sl in SCEN:
        L += [f"## {sl}", "", "| " + T("方法", "Method") + " | mean | median | p90 | p99 | max | " + T("均值加速比", "speed-up (mean)") + " |",
              "|---|---|---|---|---|---|---|"]
        ilp_mean = float(np.mean(d["ilp"][s]["ms"]))
        for _, name, _, get in methods(d):
            v = np.asarray(get(s)["ms"])
            p = np.percentile(v, [50, 90, 99])
            L.append(f"| {name} | {v.mean():.1f} | {p[0]:.1f} | {p[1]:.1f} | {p[2]:.1f} | {v.max():.1f} | {ilp_mean / v.mean():.1f}× |")
        L.append("")
    return "\n".join(L)


def main():
    d = json.loads(SRC.read_text())
    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    panel_box(fig.add_subplot(gs[0, :]), d)
    panel_cdf(fig.add_subplot(gs[1, 0]), d)
    panel_speedup(fig.add_subplot(gs[1, 1]), d)
    panel_scatter(fig.add_subplot(gs[1, 2]), d)
    n = d["n_inst"]
    fig.suptitle(f"Solve time on the full test set (LT/EQ/GT: {n['lt']}/{n['eq']}/{n['gt']} instances, 1 CPU thread each)",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("pdf", "eps", "svg"):
        fig.savefig(OUT / f"fig_time_full.{ext}", facecolor="white")
    fig.savefig(OUT / "fig_time_full.png", dpi=300, facecolor="white")
    (OUT / "fig_time_full_table.md").write_text(table(d, "zh"))
    (OUT / "fig_time_full_table.en.md").write_text(table(d, "en"))
    print(OUT / "fig_time_full.pdf")


if __name__ == "__main__":
    main()
