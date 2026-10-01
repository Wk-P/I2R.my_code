"""Evaluate every v4.2.0 model and time ILP vs RL, then write the final data.

Steps:
  1. eval_model.py on each run in manifest.json (single deterministic pass,
     4 processes in parallel, 1 torch thread each) -> results/v4.2.0/eval_raw.jsonl
  2. ilp_timing.py per scenario (seed-1 test set, 400 instances)
     -> results/v4.2.0/ilp_timing_<scen>.json
  3. aggregate -> paper_contents/v4.2.0/final_summary_data.json

usage: python paper_contents/v4.2.0/tools/run_eval.py [--skip-eval] [--skip-ilp]
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
PKG = TOOLS.parent
ROOT = TOOLS.parents[2]
OUT = ROOT / "results" / "v4.2.0"
ALGOS = ["ppo", "ppo_mask", "ppo_lagrangian", "ppo_opt", "dqn", "ddqn"]
SCENS = ["lt", "eq", "gt"]
M = {"lt": 15, "eq": 10, "gt": 10}


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode != 0 or not lines:
        raise RuntimeError(f"{' '.join(cmd)}\n{p.stderr[-2000:]}")
    return json.loads(lines[-1])


def evaluate(manifest):
    jobs = [(cell, r) for cell, c in manifest["cells"].items() for r in c["runs"]]

    def one(job):
        cell, r = job
        scen, algo = cell.split("/")
        res = run([sys.executable, str(TOOLS / "eval_model.py"), scen, algo,
                   str(ROOT / r["path"]), str(r["seed"]), "1", "1"])
        print(f"  {cell:18} seed={r['seed']} success={res['success']:.4f} "
              f"(stored {res['stored_success']:.4f}) {res['ms_per_episode']:.2f} ms/ep", flush=True)
        return res

    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(one, jobs))
    with open(OUT / "eval_raw.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def ilp_timing():
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(lambda s: run([sys.executable, str(TOOLS / "ilp_timing.py"), s]), SCENS))
    for r in res:
        (OUT / f"ilp_timing_{r['scen']}.json").write_text(json.dumps(r, indent=1))
    return {r["scen"]: r for r in res}


def sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def aggregate(manifest, rows, ilp):
    camp = json.loads((ROOT / "paper_contents/campaign_5seed_figs/summary_data.json").read_text())
    data = {"version": "v4.2.0",
            "description": "v4.2.0 best-model set (mixed batches, see manifest.json). All cells evaluated "
                           "with a single deterministic pass per test scenario; timing on 1 CPU thread.",
            "eval_protocol": manifest["eval_protocol"], "scenarios": {}}
    for scen in SCENS:
        S = {"ilp_ar": round(camp[f"{scen}_ILP"]["per_seed"][0], 6),  # seed-1 test set
             "ilp_timing_ms": {k: round(ilp[scen][k], 3) for k in ("ms_mean", "ms_std", "ms_median", "ms_max")},
             "algos": {}}
        for algo in ALGOS:
            rs = [r for r in rows if r["scen"] == scen and r["algo"] == algo]
            cell = manifest["cells"][f"{scen}/{algo}"]
            v = {
                "ar_mean": round(float(np.mean([r["ar"] for r in rs])), 4),
                "ar_std": round(sd([r["ar"] for r in rs]), 4),
                "success_mean": round(float(np.mean([r["success"] for r in rs])), 4),
                "success_std": round(sd([r["success"] for r in rs]), 4),
                "cap_viol_mean": round(float(np.mean([r["cap_viol"] or 0.0 for r in rs])), 4),
                "conflict_viol_mean": round(float(np.mean([r["conflict_viol"] or 0.0 for r in rs])), 4),
                "ms_per_episode_mean": round(float(np.mean([r["ms_per_episode"] for r in rs])), 3),
                "ms_per_episode_std": round(sd([r["ms_per_episode"] for r in rs]), 3),
                "ms_per_decision_mean": round(float(np.mean([r["ms_per_episode"] for r in rs])) / M[scen], 3),
                "n_seeds": len(rs),
                "source": cell["batch"],
            }
            if algo == "ppo_opt":
                # eval counts violations of the raw policy action (= repair triggers);
                # the executed placement never violates (see v4.1.0_changelog.md)
                v["repair_trigger_cap_mean"] = v["cap_viol_mean"]
                v["repair_trigger_conflict_mean"] = v["conflict_viol_mean"]
                v["cap_viol_mean"] = v["conflict_viol_mean"] = 0.0
            S["algos"][algo] = v
        data["scenarios"][scen] = S
    (PKG / "final_summary_data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print("wrote", PKG / "final_summary_data.json")


def main():
    manifest = json.loads((PKG / "manifest.json").read_text())
    if "--skip-eval" in sys.argv:
        rows = [json.loads(l) for l in open(OUT / "eval_raw.jsonl")]
    else:
        print("[1/3] evaluating models ...")
        rows = evaluate(manifest)
    if "--skip-ilp" in sys.argv:
        ilp = {s: json.loads((OUT / f"ilp_timing_{s}.json").read_text()) for s in SCENS}
    else:
        print("[2/3] ILP timing ...")
        ilp = ilp_timing()
    print("[3/3] aggregating ...")
    aggregate(manifest, rows, ilp)


if __name__ == "__main__":
    main()
