#!/usr/bin/env python3
"""One-off (2026-10-13): re-choose v5.0.3's lambda_max with the tie-aware rule of run_v5.select_lam_max.

v5.0.3 ran with the first rule (lowest mean effective gap only), which picked PPO lambda_max = 1 over 2 on a
0.07 pp difference although 2 violates less. The current rule puts violations first (a 3M sweep says little about the gap). This script re-applies the current rule to the sweep evaluations already saved in
paper_contents/v5.0.3/lam_sweep_<learner>.json (no training, no re-evaluation), rewrites lam_sweep_<learner>.md /
.en.md / .json and report.md / .en.md, and writes the new choice into logs/v5.0.3/manifest.json ("lambda_max", which
v5.0.4 reads) together with a "lambda_max_revision" record. Refuses to run while the v5.0.3 driver is alive (its
last manifest write would undo the change).

    .venv/bin/python src/scripts/oneoff_v5_0_3_reselect.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("run_v5", ROOT / "src" / "scripts" / "run_v5.py")
R = importlib.util.module_from_spec(spec)
sys.argv = sys.argv[:1]
spec.loader.exec_module(R)

VERSION = "5.0.3"
MANIFEST = ROOT / "logs" / f"v{VERSION}" / "manifest.json"
OUT = ROOT / "paper_contents" / f"v{VERSION}"


def main():
    man = json.loads(MANIFEST.read_text())
    pid = man.get("driver_pid")
    try:
        alive = pid and "run_v5.py" in Path(f"/proc/{int(pid)}/cmdline").read_bytes().decode(errors="ignore")
    except OSError:
        alive = False
    if alive:
        sys.exit(f"the v{VERSION} driver (pid {pid}) is still running; run this after it has finished")
    cfg = R.CONFIG[VERSION]
    R.SEEDS = cfg.get("seeds", R.SEEDS)
    V = {"version": VERSION, "cfg": cfg, "out": OUT, "manifest": MANIFEST, "report": OUT / "report.md", "lam_max": {}}
    old = dict(man.get("lambda_max") or {})
    for learner in cfg["lam_sweep"]["algos"]:
        p = OUT / f"lam_sweep_{learner}.json"
        if not p.exists():
            sys.exit(f"missing {p.relative_to(ROOT)}: the {learner} sweep was not evaluated")
        rows = json.loads(p.read_text())["rows"]
        V["lam_max"][learner] = R.write_lam_sweep(V, learner, rows)
    R.write_sweep_report(V)
    changes = {l: {"from": old.get(l), "to": v} for l, v in V["lam_max"].items() if old.get(l) != v}
    man["lambda_max"] = V["lam_max"]
    man["lambda_max_revision"] = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "by": "src/scripts/oneoff_v5_0_3_reselect.py (user decision 2026-10-13)",
        "rule": {"zh": f"违约优先：三场景平均无违约比例最高；相差 {100 * R.LAM_TIE:g} 个百分点以内视为持平，持平时选平均有效 gap 更低的",
                 "en": f"violations first: highest mean violation-free share; within {100 * R.LAM_TIE:g} pp is a tie, broken by the lower mean effective gap"},
        "driver_choice": old, "changes": changes}
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=2))
    os.replace(tmp, MANIFEST)
    print(f"lambda_max: {old} -> {V['lam_max']} | changed: {changes or 'none'}")


if __name__ == "__main__":
    main()
