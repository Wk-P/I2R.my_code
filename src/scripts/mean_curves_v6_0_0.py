#!/usr/bin/env python3
"""Seed-mean training curves of v6.0.0 (3 learners x 4 strategies x 3 scenarios).
Each run's training_curve.csv is binned on the step axis; per bin we take the run's mean, then mean ± std
over seeds. Output: logs/v6.0.0/report/mean_curves/{lt,eq,gt}.png + mean_curves.csv + README(.en).md
    .venv/bin/python src/scripts/mean_curves_v6_0_0.py [--bins 100]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
LOG = ROOT / "logs" / "v6.0.0"
OUT = LOG / "report" / "mean_curves"
SCEN, LEARN = ("lt", "eq", "gt"), ("ppo", "dqn", "ddqn")
STRAT = ("maskable", "repair", "lagrangian", "unconstraint")
LABEL = {"ppo": "PPO", "dqn": "DQN", "ddqn": "Double DQN",
         "maskable": "Mask", "repair": "Repair", "lagrangian": "Lagrangian", "unconstraint": "Penalty (Unconstraint)"}
COLOR = {"maskable": "#2a6fdb", "repair": "#e07b00", "lagrangian": "#2b9a48", "unconstraint": "#c23b3b"}
COLS = ["timestep", "episode_ar", "episode_cap_violated", "episode_conflict_violated", "episode_exited"]


def runs():
    """{(scen, learner, strat): [run_dir, ...]} from the 'saved ->' line of each log."""
    out = {}
    for f in sorted(LOG.glob("seed*.log")):
        m = re.match(r"seed(\d)_(\w\w)_(\w+?)_(\w+)_[0-9a-f]{8}\.log", f.name)
        saved = [ln.split("->")[-1].strip() for ln in f.read_text().splitlines() if "saved ->" in ln]
        if m and saved:
            out.setdefault((m[2], m[3], m[4]), []).append(Path(saved[-1]))
    return out


def binned(d: Path, edges):
    c = pd.read_csv(d / "training_curve.csv", usecols=COLS)
    c["viol"] = ((c.episode_cap_violated > 0) | (c.episode_conflict_violated > 0)).astype(float)
    b = np.clip(np.searchsorted(edges, c.timestep, side="right") - 1, 0, len(edges) - 2)
    g = c.assign(b=b).groupby("b")[["episode_ar", "viol", "episode_exited"]].mean()
    return g.reindex(range(len(edges) - 1)), json.loads((d / "results.json").read_text())["ilp"]["ar"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=100)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    total = 20_000_000
    edges = np.linspace(0, total, a.bins + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    R, rows, ilp = runs(), [], {}
    for (s, l, k), ds in sorted(R.items()):
        per = [binned(d, edges) for d in ds]
        ilp.setdefault(s, []).extend(x[1] for x in per)
        for metric in ("episode_ar", "viol", "episode_exited"):
            M = np.vstack([p[0][metric].to_numpy() for p in per])
            for i in range(a.bins):
                rows.append(dict(scenario=s, learner=l, strategy=k, metric=metric, step=int(mid[i]),
                                 mean=np.nanmean(M[:, i]), std=np.nanstd(M[:, i]), n_seeds=len(ds)))
        print(f"{s} {l:4s} {k:12s} seeds={len(ds)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mean_curves.csv", index=False)

    for s in SCEN:
        fig, ax = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
        for j, l in enumerate(LEARN):
            for i, metric in enumerate(("episode_ar", "viol")):
                A = ax[i, j]
                for k in STRAT:
                    q = df[(df.scenario == s) & (df.learner == l) & (df.strategy == k) & (df.metric == metric)]
                    if q.empty:
                        continue
                    A.plot(q.step, q["mean"], color=COLOR[k], lw=1.6, label=f"{LABEL[k]} (n={q.n_seeds.iat[0]})")
                    A.fill_between(q.step, q["mean"] - q["std"], q["mean"] + q["std"], color=COLOR[k], alpha=.15, lw=0)
                if metric == "episode_ar":
                    A.axhline(np.mean(ilp[s]), color="k", ls="--", lw=1, label="ILP AR")
                    A.set_title(LABEL[l])
                A.set_xlim(0, total)
                A.set_ylabel("training episode AR" if metric == "episode_ar" else "violation rate (cap or privacy)")
                A.grid(alpha=.3)
                if i == 1:
                    A.set_xlabel("timestep")
            ax[0, j].legend(fontsize=8, loc="lower right")
        fig.suptitle(f"v6.0.0 — {s.upper()} — seed mean ± std ({a.bins} bins over 20M steps)")
        fig.tight_layout()
        fig.savefig(OUT / f"{s}.png", dpi=130)
        plt.close(fig)

    note = {"zh": ("# v6.0.0 种子均值训练曲线\n\n每个 run 的 training_curve.csv 沿步数分 {b} 段，段内取均值，再对种子取均值 ± 标准差（阴影）。"
                   "Mask / Repair 为种子 1–3，Lagrangian / Penalty（Unconstraint）为种子 2–3（图例 n=种子数）。上排：训练回合 AR"
                   "（含违约回合，虚线为测试集 ILP AR 均值，仅作参考）；下排：违约回合占比（容量或隐私）。ddqn = Double DQN。"
                   "数据：mean_curves.csv。\n\n![LT](lt.png)\n![EQ](eq.png)\n![GT](gt.png)\n"),
            "en": ("# v6.0.0 seed-mean training curves\n\nEach run's training_curve.csv is split into {b} step bins, averaged per bin, "
                   "then mean ± std over seeds (shaded). Mask / Repair use seeds 1–3, Lagrangian / Penalty (Unconstraint) seeds 2–3 "
                   "(legend n = seeds). Top: training-episode AR (violating episodes included; dashed = mean test-set ILP AR, "
                   "reference only); bottom: share of episodes with a capacity or privacy violation. ddqn = Double DQN. "
                   "Data: mean_curves.csv.\n\n![LT](lt.png)\n![EQ](eq.png)\n![GT](gt.png)\n")}
    (OUT / "README.md").write_text(note["zh"].format(b=a.bins))
    (OUT / "README.en.md").write_text(note["en"].format(b=a.bins))
    print("->", OUT)


if __name__ == "__main__":
    sys.exit(main())
