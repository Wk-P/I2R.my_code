#!/usr/bin/env python3
"""Redraw training_curve.png of pre-paper_rl runs (scenarios/*/run_all.py, shared/dqn_variant_runner.py;
results/{main,add_states,final_paper_experiments,paper-verfication,pretrain,...}) from their
training_curve.csv + results.json, with the step axis fixed to [0, total steps] (2026-10-11).
Their CSVs come in several layouts (timestep, episode_ar, episode_success, [episode_valid_placed],
episode_placed, [episode_reward | episode_lambda | episode_repair_rate]); every column present is plotted.
Runs without training_curve.csv cannot be redrawn and are listed instead.
    .venv/bin/python src/scripts/replot_legacy_curves.py [--dry-run]
"""
import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

PAPER_RL_HEADER = ["timestep", "episode_reward", "episode_ar", "episode_success", "episode_valid_placed",
                   "episode_cap_violated", "episode_conflict_violated", "lambda"]   # handled by replot_curves.py
PANELS = [("episode_reward", "Episode reward", "steelblue"), ("episode_ar", "AR", "seagreen"),
          ("episode_success", "success_rate", "mediumseagreen"), ("episode_placed", "services placed", "slategray"),
          ("episode_lambda", "lambda", "purple"), ("episode_repair_rate", "repair rate", "darkorange")]


def one(d: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paper_rl import config as C
    from shared.ilp_utils import moving_avg
    try:
        r = json.loads((d / "results.json").read_text())
        with open(d / "training_curve.csv") as f:
            rows = list(csv.reader(f))
        head, body = rows[0], rows[1:]
        if not body:
            return f"{d}: empty csv"
        a = np.array([[float(x) for x in row] for row in body])
        col = {h: a[:, i] for i, h in enumerate(head)}
        total = float(r["training"]["total_steps"])
        ts = col["timestep"]
        w = max(1, min(C.SMOOTH_W, len(a) // 10))
        panels = [p for p in PANELS if p[0] in col]
        fig, ax = plt.subplots(len(panels), 1, figsize=(10, 3 * len(panels)), sharex=True)
        ax = np.atleast_1d(ax)
        for x, (k, label, color) in zip(ax, panels):
            x.plot(ts, moving_avg(col[k], w)[0], color=color, label=label)
            if k == "episode_placed" and "episode_valid_placed" in col:
                x.plot(ts, moving_avg(col["episode_valid_placed"], w)[0], color="tomato", label="valid placed")
                x.legend()
            if k == "episode_ar":
                x.axhline(r["ilp"]["ar"], color="red", ls="--", label=f"ILP AR* = {r['ilp']['ar']:.4f}")
                x.legend()
            if k == "episode_success":
                x.set_ylim(-0.02, 1.02)
            x.set_ylabel(label); x.grid(alpha=0.3)
            x.set_xlim(0, total)                         # step axis: 0 .. total steps
        rel = d.relative_to(ROOT / "results") if d.is_relative_to(ROOT / "results") else d.relative_to(ROOT)
        ax[0].set_title(f"{rel} — {int(total):,} steps")
        ax[-1].set_xlabel("training steps")
        plt.tight_layout(); plt.savefig(d / "training_curve.png", dpi=120); plt.close()
        return None
    except Exception as e:
        return f"{d}: {e!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    todo, no_csv = [], []
    for root in (ROOT / "results", ROOT / "archive"):
        for p in root.rglob("training_curve.png"):
            d = p.parent
            c = d / "training_curve.csv"
            if not c.exists():
                no_csv.append(d)
                continue
            with open(c) as f:
                if (next(csv.reader(f), None) or [])[:len(PAPER_RL_HEADER)] == PAPER_RL_HEADER:   # v5.0.0 appends episode_exited
                    continue
            if (d / "results.json").exists():
                todo.append(d)
    print(f"legacy runs with csv: {len(todo)}; without csv (cannot redraw): {len(no_csv)}")
    if a.dry_run:
        return
    with Pool(16) as pool:
        errs = [e for e in pool.map(one, todo) if e]
    print(f"replotted {len(todo) - len(errs)}/{len(todo)}")
    for e in errs[:20]:
        print("  ", e)
    out = ROOT / "logs" / "replot_legacy_no_csv.txt"
    out.write_text("\n".join(str(d.relative_to(ROOT)) for d in sorted(no_csv)) + "\n")
    print(f"no-csv list -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
