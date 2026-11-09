"""Measured per-instance wall time: ILP vs Safe RL (docs/ilp_vs_saferl.md section 6).

Data: paper_contents/v6.0.0-merge/time_bench.json (src/scripts/bench_time_v6.py: the same 200 test instances per
scenario, one CPU thread; ILP = solve_ilp_max_ar incl. model building and every Dinkelbach iteration; RL = full
deterministic episode). No new runs.

    .venv/bin/python src/scripts/plot_time_measured.py   # -> paper_contents/ilp_vs_saferl/fig_time_measured.{pdf,eps,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper_contents" / "v6.0.0-merge" / "time_bench.json"
OUT = ROOT / "paper_contents" / "ilp_vs_saferl"
SCEN = [("lt", "LT ($N$=10, $M$=15)"), ("eq", "EQ ($N$=10, $M$=10)"), ("gt", "GT ($N$=15, $M$=10)")]
C_ILP, C_RL, C_MAIN = "#4C78A8", "#9CCB94", "#2E7D32"


def main():
    d = json.loads(SRC.read_text())
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    w = 0.34
    for i, (s, lab) in enumerate(SCEN):
        ilp = d["ilp"][s]
        rl = [v[s] for v in d["rl"].values()]
        ppo_m = d["rl"]["mask|ppo"][s]
        ax.bar(i - w / 2, ilp["mean_ms"], w, color=C_ILP, label="ILP (mean)" if i == 0 else None)
        ax.plot(i - w / 2, ilp["median_ms"], "D", ms=4, color="white", mec="black", mew=0.6,
                label="ILP (median)" if i == 0 else None)
        ax.bar(i + w / 2, max(rl), w, color=C_RL, label=f"Safe RL, {len(rl)} variants (max)" if i == 0 else None)
        ax.plot(i + w / 2, ppo_m, "o", ms=4.5, color=C_MAIN, label="PPO-M" if i == 0 else None)
        ax.annotate(f"{ilp['mean_ms'] / ppo_m:.0f}×", (i, ilp["mean_ms"] * 1.6), ha="center", fontsize=8, weight="bold")
    ax.set_yscale("log"); ax.set_ylim(10, 2e4)
    ax.set_xticks(range(len(SCEN))); ax.set_xticklabels([l for _, l in SCEN], fontsize=7.5)
    ax.set_ylabel("wall time per instance (ms, log)", fontsize=8); ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6.8, frameon=False, ncol=2, loc="upper right")
    ax.set_title(f"Measured solve time ({d['n_inst']} test instances / scenario, 1 CPU thread)", fontsize=8.5, loc="left")
    ax.set_xlabel("bold ×: ILP mean / PPO-M mean", fontsize=6.8, color="dimgray")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps", "svg"):
        fig.savefig(OUT / f"fig_time_measured.{ext}", facecolor="white")
    fig.savefig(OUT / "fig_time_measured.png", dpi=300, facecolor="white")
    print(OUT / "fig_time_measured.pdf")


if __name__ == "__main__":
    main()
