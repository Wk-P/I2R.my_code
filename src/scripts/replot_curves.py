#!/usr/bin/env python3
"""Redraw training_curve.png of finished paper_rl runs from training_curve.csv + results.json (2026-10-11:
step axis fixed to [0, total steps], trailing-mean smoothing that starts at the first episode).
    .venv/bin/python src/scripts/replot_curves.py [--since 2026-10-01]
"""
import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


def one(d: Path):
    from paper_rl.train import algo_label, curve_history, plot_curve
    try:
        r = json.loads((d / "results.json").read_text())
        with open(d / "training_curve.csv") as f:
            rows = [tuple(float(e) for e in row) for row in list(csv.reader(f))[1:]]
        tr, res = r["training"], r.get("resumed")
        start = int(res["at_step"]) if res else 0
        title = (f"{algo_label(r['algo'])} — {r['scenario'].upper()} — reward={r['reward_mode']} — {tr['total_steps']:,} steps"
                 + (f" (resumed at {start:,})" if res else ""))
        plot_curve(rows, r["ilp"]["ar"], d / "training_curve.png", title, history=curve_history(d) if res else None,
                   x_max=tr["total_steps"])
        return None
    except Exception as e_:
        return f"{d}: {e_}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="")
    a = ap.parse_args()
    dirs = []
    for p in (ROOT / "results").rglob("training_curve.csv"):
        try:
            r = json.loads((p.parent / "results.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue                                   # still training, or not a paper_rl run
        if "training" in r and "ilp" in r and "reward_mode" in r and r.get("created_at", "") >= a.since:
            dirs.append(p.parent)
    with Pool(16) as pool:
        errs = [e for e in pool.map(one, dirs) if e]
    print(f"replotted {len(dirs) - len(errs)}/{len(dirs)}")
    for e in errs[:20]:
        print("  ", e)


if __name__ == "__main__":
    main()
