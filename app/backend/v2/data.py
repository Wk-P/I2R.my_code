"""File layer of the v2 panel backend (docs/data_spec.md): batches (logs/<batch>/manifest.json), runs
(results/<space>/<scenario>/<algo>/<exp_id>/{run.json, results.json, training_curve.csv}), report data
(report_data.json, lam_sweep_<learner>.json) and result spaces / stages (app/backend/result_spaces.json).

Read-only; never imports training code. Old manifests (no schema) are converted by the legacy backend's
_manifest_runs (docs/data_spec.md §5) -- the only thing borrowed from it here."""
from __future__ import annotations

import json
import math
import re
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
PAPER = ROOT / "paper_contents"
SPACES_META = Path(__file__).resolve().parents[1] / "result_spaces.json"
SCENARIOS = ["lt", "eq", "gt"]
LEARNERS = ["ppo", "dqn", "ddqn"]
MECHANISMS = ["none", "lagrange", "mask", "repair"]
OLD_PPO = {"ppo_lagrangian": "lagrange", "ppo_mask": "mask", "ppo_opt": "repair"}     # pre-v4.3.1.3 dir names


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def clean(v):
    """NaN / inf -> None (JSON has neither)."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def split_algo(algo: str) -> tuple[str, str]:
    """(learner, mechanism) of an algo directory name, e.g. lagrange_ddqn -> (ddqn, lagrange)."""
    if algo in OLD_PPO:
        return "ppo", OLD_PPO[algo]
    parts = algo.split("_")
    if len(parts) > 1 and parts[0] in MECHANISMS and parts[-1] in LEARNERS:
        return parts[-1], parts[0]
    if algo in LEARNERS:
        return algo, "none"
    return algo, "none"


# ── spaces and stages ─────────────────────────────────────────────────────────
def spaces() -> list[dict]:
    """Every results/<space>/ with its label and stage; smoke* / check_* spaces belong to the newest stage."""
    meta = {k: v for k, v in (read_json(SPACES_META) or {}).items() if not k.startswith("_")}
    newest = max((m.get("stage") or 0 for m in meta.values()), default=0)
    out = []
    names = sorted(p.name for p in RESULTS.iterdir() if p.is_dir()) if RESULTS.is_dir() else []
    for n in names:
        m = meta.get(n)
        test = n.startswith("smoke") or n.startswith("check_")
        if m is None and not test:
            m = {"label": n, "kind": "other", "stage": None, "order": 99}
        if m is None:
            m = {"label": n, "kind": "test", "stage": newest, "order": 50}
        out.append({"name": n, "label": m.get("label", n), "kind": m.get("kind", "branch"), "stage": m.get("stage"),
                    "versions": m.get("versions"), "order": m.get("order", 99)})
    return sorted(out, key=lambda s: (s["order"], s["name"]))


def stages() -> list[dict]:
    seen = {}
    for s in spaces():
        if s["stage"] is not None and s["kind"] == "branch" and s["stage"] not in seen:
            seen[s["stage"]] = {"stage": s["stage"], "label": s["label"], "versions": s["versions"]}
    newest = max(seen, default=None)
    return [dict(v, current=k == newest) for k, v in sorted(seen.items(), reverse=True)]


def stage_of(space: str | None) -> int | None:
    return next((s["stage"] for s in spaces() if s["name"] == space), None)


# ── runs ──────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=4096)
def _results_cached(path: str, mtime: float):
    return read_json(Path(path))


def results_json(run_dir: Path) -> dict | None:
    p = run_dir / "results.json"
    try:
        return _results_cached(str(p), p.stat().st_mtime)
    except OSError:
        return None


def metrics(data: dict | None, algo: str) -> dict | None:
    """The numbers the panel shows for a finished run. Relative gap = (ILP AR - AR) / ILP AR, AR over the test
    instances without violation (as the training script reports it); violation rates = share of test instances
    with a capacity / privacy violation."""
    if not data:
        return None
    key = next((k for k in (algo, data.get("algo")) if k and isinstance(data.get(k), dict)), None)
    if key is None:     # old results.json: the first block holding ar_mean
        key = next((k for k, v in data.items() if isinstance(v, dict) and "ar_mean" in v), None)
    ev = data.get(key) or {}
    tr = data.get("training") or {}
    ar, ilp = clean(ev.get("ar_mean")), clean((data.get("ilp") or {}).get("ar"))
    gap = (ilp - ar) / ilp if ar is not None and ilp else None
    return {"ar": ar, "ar_std": clean(ev.get("ar_std")), "ilp_ar": ilp, "rel_gap": gap,
            "success": clean(ev.get("success_rate")), "cap_viol": clean(ev.get("cap_viol_rate")),
            "priv_viol": clean(ev.get("conflict_viol_rate")), "exit": clean(ev.get("exit_rate")),
            "dead_end": clean(ev.get("dead_end_rate")),
            "final_lambda": clean(tr.get("final_lambda")), "lambda_max": clean(tr.get("lambda_max")),
            "train_seconds": clean(tr.get("train_seconds")), "episodes": tr.get("n_episodes"),
            "train_ar_last50": clean(tr.get("ar_last50")), "total_steps": tr.get("total_steps"),
            "created_at": data.get("created_at"), "device": data.get("device"), "net": data.get("net")}


def run_dir(space: str, scenario: str, algo: str, exp_id: str) -> Path:
    for part in (space, scenario, algo, exp_id):
        if not re.fullmatch(r"[\w.\-]+", part or ""):
            raise ValueError(part)
    return RESULTS / space / scenario / algo / exp_id


def find_space(scenario: str, algo: str, exp_id: str, prefer: str | None = None) -> str | None:
    """The space holding a run (old manifests do not say which)."""
    cands = ([prefer] if prefer else []) + [s["name"] for s in spaces()]
    return next((s for s in cands if s and (RESULTS / s / scenario / algo / exp_id).is_dir()), None)


# ── training trace (training_curve.csv) ───────────────────────────────────────
TRACE_COLS = {"episode_reward": "reward", "episode_ar": "ar", "episode_success": "success",
              "episode_cap_violated": "cap_viol", "episode_conflict_violated": "priv_viol", "lambda": "lambda",
              "episode_valid_placed": "valid_placed"}


TRACE_CACHE = Path(__file__).resolve().parent / ".trace_cache"       # small JSON per (file, mtime, points, window)


def _trace_compute(path: str, points: int, window: int) -> dict:
    import numpy as np
    import pandas as pd
    df = pd.read_csv(path, engine="c")
    n = len(df)
    if not n:
        return {"n_episodes": 0, "window": window, "step": [], "series": {}}
    idx = np.unique(np.round(np.linspace(0, n - 1, min(points, n))).astype(int))
    lo = np.maximum(0, idx + 1 - window)
    out = {}
    for c, name in TRACE_COLS.items():
        if c not in df:
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        ok = ~np.isnan(v)
        cs = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])     # moving average over the last `window`
        cn = np.concatenate([[0], np.cumsum(ok)])                          # episodes, NaN skipped
        cnt = cn[idx + 1] - cn[lo]
        avg = np.where(cnt > 0, (cs[idx + 1] - cs[lo]) / np.maximum(cnt, 1), np.nan)
        out[name] = [None if x != x else round(float(x), 6) for x in avg]
    return {"n_episodes": int(n), "window": window, "step": [int(x) for x in df["timestep"].to_numpy()[idx]], "series": out}


def trace(rdir: Path, points: int = 400, window: int = 1000) -> dict | None:
    """Moving averages of training_curve.csv at `points` evenly spaced episodes. 20M-step runs have >1M rows
    (~80 MB), so the result is cached on disk per (file, mtime) -- computed once per finished run."""
    import hashlib
    p = rdir / "training_curve.csv"
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    key = hashlib.sha1(f"{p}|{mtime}|{points}|{window}".encode()).hexdigest()
    cf = TRACE_CACHE / f"{key}.json"
    hit = read_json(cf)
    if hit is not None:
        return hit
    out = _trace_compute(str(p), points, window)
    TRACE_CACHE.mkdir(exist_ok=True)
    tmp = cf.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(cf)
    return out


# ── batches ───────────────────────────────────────────────────────────────────
def manifest(batch: str) -> dict | None:
    if not re.fullmatch(r"[\w.\-]+", batch):
        return None
    return read_json(LOGS / batch / "manifest.json")


def batch_names() -> list[str]:
    return [d.name for d in LOGS.iterdir() if d.is_dir()] if LOGS.is_dir() else []


def plan(batch: str, man: dict | None) -> list[dict]:
    """The planned runs of a batch in the v2 shape (docs/data_spec.md §3); old manifests (and batches known only
    from their log file names) via the legacy converter (§5)."""
    if man and man.get("schema") == 2:
        return [dict(r, role=r.get("role") or "main", depends_on=r.get("depends_on") or [], params=r.get("params") or {})
                for r in man.get("runs", [])]
    from .. import main as legacy
    out = []
    for r in legacy._batch_runs(LOGS / batch)[0]:
        learner, mech = split_algo(r["algo"])
        out.append({"key": None, "role": "main", "group": None, "depends_on": [], "scenario": r["scenario"],
                    "seed": r["seed"], "algo": r["algo"], "learner": learner, "mechanism": mech, "net": None,
                    "steps": (man or {}).get("steps"), "params": r.get("params") or {}, "exp_id": r["exp_id"]})
    return out


def report_dir(batch: str, man: dict) -> Path | None:
    """Where the batch's report lives: logs/<batch>/report/ (smoke) or paper_contents/v<version>/."""
    d = LOGS / batch / "report"
    if d.is_dir():
        return d
    v = man.get("version")
    if v and man.get("purpose", "main") == "main" and (PAPER / f"v{v}").is_dir():
        return PAPER / f"v{v}"
    return None


def report_files(rdir: Path | None) -> list[dict]:
    if rdir is None:
        return []
    out = []
    for p in sorted(rdir.glob("*.md")):
        if p.name.endswith(".en.md"):
            continue
        first = next((l for l in p.read_text(errors="replace").splitlines() if l.startswith("# ")), p.stem)
        out.append({"file": p.name, "title": first.lstrip("# ").strip(), "has_en": p.with_suffix("").with_suffix(".en.md").is_file()
                    or (rdir / (p.stem + ".en.md")).is_file()})
    return sorted(out, key=lambda f: (f["file"] != "report.md", f["file"]))


def age(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None
