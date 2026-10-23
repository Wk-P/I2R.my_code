"""Manifest of a batch that trains nothing -- an evaluation, a diagnosis or a data build (docs/data_spec.md §6).

The v2 panel lists a batch only through logs/<batch>/manifest.json; a script that writes a report or a data set but
launches no training run wraps its main() in `nontrain_batch(...)` so it shows up there with its status, inputs,
report and log:

    with nontrain_batch("v6.0.3", "eval", __file__, {"zh": "...", "en": "..."}, depends_on=["v6.0.0", "v6.0.1"],
                        report_dir="paper_contents/v6.0.3"):
        main()
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs"
PURPOSES = ("eval", "diag", "data", "tune")   # v6.1.0: tune = Optuna search (runs live in results/tune)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _commit(script: str) -> tuple[str | None, bool]:
    """(short HEAD, whether the script differs from HEAD or is untracked)."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain", "--", script], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return head or None, bool(st)
    except OSError:
        return None, False


def write(batch: str, /, **fields) -> Path:
    """Create or update logs/<batch>/manifest.json (atomic); fields are merged into what is there."""
    d = LOGS / batch
    d.mkdir(parents=True, exist_ok=True)
    p = d / "manifest.json"
    try:
        man = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        man = {}
    man.update(fields)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    tmp.replace(p)
    return p


def plan(batch: str, purpose: str, script: str, title: dict, depends_on=(), report_dir: str | None = None,
         outputs=(), logs=None, inputs=(), version: str | None = None, links=()) -> dict:
    """The fixed fields of a non-training manifest (docs/data_spec.md §6)."""
    assert purpose in PURPOSES, purpose
    script = str(Path(script).resolve().relative_to(ROOT))
    commit, dirty = _commit(script)
    man = {"schema": 3, "batch": batch, "version": version or batch, "purpose": purpose, "script": script,
           "commit": commit, "script_uncommitted": dirty, "title": title, "depends_on": list(depends_on),
           "inputs": list(inputs), "report_dir": report_dir, "outputs": list(outputs), "runs": [], "references": [],
           "links": list(links)}   # v6.1.0: [{"zh", "en", "url"}] shown on the batch page
    if logs is not None:
        man["logs"] = list(logs)
    return man


@contextmanager
def nontrain_batch(batch: str, purpose: str, script: str, title: dict, **kw):
    """Write the manifest as running (driver_pid = this process), then finished / failed when the block ends."""
    t0 = time.time()
    write(batch, **plan(batch, purpose, script, title, **kw), status="running", driver_pid=os.getpid(),
          created_at=_now(), finished_at=None)
    try:
        yield
    except BaseException as e:
        write(batch, status="failed", error=f"{e.__class__.__name__}: {e}"[:500], finished_at=_now(),
              minutes=round((time.time() - t0) / 60, 1))
        raise
    write(batch, status="finished", finished_at=_now(), minutes=round((time.time() - t0) / 60, 1))
