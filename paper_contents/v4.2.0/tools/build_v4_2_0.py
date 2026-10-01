"""Assemble the v4.2.0 "best models" set.

v4.2.0 is a mixed set: for every (scenario, algorithm) cell it takes the model
batch with the best test-set result under the unified evaluation protocol
(single deterministic pass, EVAL_BEST_OF_N=1). The models were trained by
different code versions; only the reward differed between those versions, the
observation/transition used at evaluation time is identical, so all models are
evaluated with the current scenarios/ code.

Copies each selected run directory to results/v4.2.0/<scen>/<algo>/<exp_id>/
(results/ is git-ignored) and writes manifest.json next to this script's parent.

usage: python paper_contents/v4.2.0/tools/build_v4_2_0.py
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v4.2.0"
MANIFEST = Path(__file__).resolve().parents[1] / "manifest.json"

# source batch -> (results subdir, training code reference)
BATCHES = {
    "v4.0.0":   ("add_states", "v4.0.0 campaign (code at ce1dab0, frozen as tag v4.0.0_final_1 / 82513ac)"),
    "v4.1.0":   ("final_paper_experiments", "v4.1.0 5M campaign (tag v4.1.0 / d0899cc + ILP cache fix ece29ae)"),
    "v4.1.0.1": ("final_paper_experiments", "v4.1.0.1 ppo_lagrangian ablation (4757d10)"),
}

def campaign_runs(scen, algo):
    camp = json.loads((ROOT / "paper_contents/campaign_5seed_figs/summary_data.json").read_text())
    c = camp[f"{scen}_{algo}"]
    return list(zip(c["seeds"], c["exp_ids"]))


def build_selection():
    """(scenario, algo) -> (batch, [(seed, exp_id), ...] or None = resolve from results.json)."""
    sel = {}
    for scen in ("lt", "eq", "gt"):
        for algo in ("ppo", "ppo_mask"):
            sel[(scen, algo)] = ("v4.0.0", campaign_runs(scen, algo))
        sel[("lt", "ppo_lagrangian")] = ("v4.0.0", campaign_runs("lt", "ppo_lagrangian"))
        for algo in ("ppo_opt", "dqn", "ddqn"):
            sel[(scen, algo)] = ("v4.1.0", None)  # resolved below from results.json
    sel[("eq", "ppo_lagrangian")] = ("v4.1.0", [(1, "69c94039"), (2, "aa878d42"), (3, "cb6d38b6")])
    sel[("gt", "ppo_lagrangian")] = ("v4.1.0.1", [(1, "6e4f9730"), (2, "3987b950"), (3, "78534789")])
    return sel


def seed_of(scen, ilp_ar):
    camp = json.loads((ROOT / "paper_contents/campaign_5seed_figs/summary_data.json").read_text())
    per = [round(v, 6) for v in camp[f"{scen}_ILP"]["per_seed"]]
    return per.index(round(ilp_ar, 6)) + 1


def resolve_v410(scen, algo):
    """The three 5M-step v4.1.0 runs of this cell (seed from the test-set ILP fingerprint)."""
    runs = []
    for f in sorted((ROOT / "results/final_paper_experiments" / scen / algo).glob("*/results.json")):
        d = json.loads(f.read_text())
        if d["training"]["total_steps"] == 5_000_000:
            runs.append((seed_of(scen, d["ilp"]["ar"]), f.parent.name))
    assert len(runs) == 3, (scen, algo, runs)
    return sorted(runs)


def main():
    manifest = {"version": "v4.2.0",
                "eval_protocol": "single deterministic pass per test scenario (EVAL_BEST_OF_N=1)",
                "cells": {}}
    for (scen, algo), (batch, runs) in sorted(build_selection().items()):
        if runs is None:
            runs = resolve_v410(scen, algo)
        subdir, code_ref = BATCHES[batch]
        entries = []
        for seed, exp_id in runs:
            src = ROOT / "results" / subdir / scen / algo / exp_id
            dst = OUT / scen / algo / exp_id
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file() and (f.name.startswith("model_") or f.suffix in (".json", ".csv")):
                    shutil.copy2(f, dst / f.name)
            entries.append({"seed": seed, "exp_id": exp_id,
                            "source": str(src.relative_to(ROOT)),
                            "path": str(dst.relative_to(ROOT))})
        manifest["cells"][f"{scen}/{algo}"] = {"batch": batch, "training_code": code_ref, "runs": entries}
        print(f"{scen}/{algo:15} {batch:9} seeds={[e['seed'] for e in entries]}")
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("wrote", MANIFEST)


if __name__ == "__main__":
    main()
