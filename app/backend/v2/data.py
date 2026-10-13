"""File layer of the v2 panel backend (docs/data_spec.md): batches (logs/<batch>/manifest.json), runs
(results/<space>/<scenario>/<algo>/<exp_id>/{run.json, results.json, training_curve.csv}), report data
(report_data.json, lam_sweep_<learner>.json).

Only batches written to the v2 data spec (manifest "schema": 2, v5.0.3 on) exist for this panel; older batches stay
in the classic panel (/classic/, app/backend/main.py). Read-only; never imports training code or the old backend."""
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
SCENARIOS = ["lt", "eq", "gt"]
RL_ALGORITHMS = ["PPO", "DQN", "DDQN"]                                   # docs/data_spec.md §1
STRATEGIES = ["unconstraint", "lagrangian", "maskable", "repair"]
_PREFIX = {"unconstraint": "", "lagrangian": "lagrange_", "maskable": "mask_", "repair": "repair_"}
_FROM_MECH = {"none": "unconstraint", "lagrange": "lagrangian", "mask": "maskable", "repair": "repair"}


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


def result_dir(rl: str, strategy: str) -> str:
    """Results directory name of an (RL_algorithm, strategy), e.g. (DQN, lagrangian) -> lagrange_dqn."""
    return _PREFIX[strategy] + rl.lower()


def from_dir(d: str) -> tuple[str | None, str | None]:
    """(RL_algorithm, strategy) of a results directory name."""
    pre, _, rl = d.rpartition("_")
    strategy = next((k for k, v in _PREFIX.items() if v == (pre + "_" if pre else "")), None)
    return (rl.upper() if rl.upper() in RL_ALGORITHMS else None), strategy


def normalize(man: dict) -> dict:
    """A schema-2 or schema-3 manifest in the schema-3 shape (docs/data_spec.md §2, §5): RL_algorithm, strategies,
    seeds, scenarios, params at the batch level; runs with RL_algorithm, strategy, seed, scenario, params, dir."""
    if man.get("schema") == 3:
        runs = [dict(r, depends_on=r.get("depends_on") or [], params=r.get("params") or {}) for r in man.get("runs", [])]
        return dict(man, runs=runs)
    up = lambda d: {k.upper(): v for k, v in d.items()} if isinstance(d, dict) else d
    params = {k: man[k] for k in ("steps", "checkpoints", "data", "reward", "gamma", "full_episode", "exit_action")
              if man.get(k) is not None}
    if man.get("lam_sweep"):
        sw = man["lam_sweep"]
        params["lambda_max_sweep"] = {"values": sw.get("values")}
        if all(r.get("role") == "sweep" for r in man.get("runs", [])):    # a sweep-only batch: its own steps
            params.update(steps=sw.get("steps") if man.get("purpose") != "smoke" else man.get("steps"),
                          checkpoints=sw.get("ckpts") if man.get("purpose") != "smoke" else man.get("checkpoints"))
        else:
            params["lambda_max_sweep"].update(steps=sw.get("steps"), checkpoints=sw.get("ckpts"))
    if man.get("lambda_max") is not None:
        params["lambda_max"] = up(man["lambda_max"])
    if man.get("lambda_max_from"):
        params["lambda_max_from"] = man["lambda_max_from"]
    grp = lambda g: f"lambda_max_sweep_{g.split('_')[-1].upper()}" if g and g.startswith("lam_sweep_") else g
    runs = []
    for r in man.get("runs", []):
        rp = dict(r.get("params") or {})
        if r.get("steps") is not None:
            rp["steps"] = r["steps"]
        runs.append({"key": r.get("key"), "role": r.get("role") or "main", "group": grp(r.get("group")),
                     "depends_on": [grp(g) for g in r.get("depends_on") or []],
                     "RL_algorithm": (r.get("learner") or "").upper(), "strategy": _FROM_MECH.get(r.get("mechanism"), r.get("mechanism")),
                     "seed": r.get("seed"), "scenario": r["scenario"], "params": rp, "dir": r["algo"], "exp_id": r.get("exp_id")})
    out = {k: v for k, v in man.items() if k not in ("runs", "exp_ids", "lam_sweep", "lambda_max", "lambda_max_from", "steps",
                                                      "checkpoints", "data", "reward", "gamma", "full_episode", "exit_action",
                                                      "algos", "seeds", "scenarios")}
    out.update(RL_algorithm=[a for a in RL_ALGORITHMS if any(r["RL_algorithm"] == a for r in runs)],
               strategies=[x for x in STRATEGIES if any(r["strategy"] == x for r in runs)],
               seeds=sorted({r["seed"] for r in runs if r["seed"] is not None}),
               scenarios=[x for x in SCENARIOS if any(r["scenario"] == x for r in runs)],
               params=params, runs=runs)
    return out


def run_info(rj: dict, d: str) -> dict:
    """run.json (schema 1 = v5.0.3: learner / mechanism; schema 3) -> RL_algorithm, strategy, params."""
    rl, st = from_dir(d)
    if rj.get("schema") == 3:
        return {"RL_algorithm": rj.get("RL_algorithm") or rl, "strategy": rj.get("strategy") or st, "params": rj.get("params") or {}}
    p = dict(rj.get("params") or {})
    if rj.get("steps") is not None:
        p.setdefault("steps", rj["steps"])
    if rj.get("net"):
        p.setdefault("net", rj["net"])
    return {"RL_algorithm": (rj.get("learner") or "").upper() or rl,
            "strategy": _FROM_MECH.get(rj.get("mechanism"), st), "params": p}


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
    """Batches of this panel: logs/<batch>/manifest.json with "schema" 2 (v5.0.3) or 3."""
    if not LOGS.is_dir():
        return []
    return [d.name for d in LOGS.iterdir() if d.is_dir() and ((read_json(d / "manifest.json") or {}).get("schema") in (2, 3))]


def report_dir(batch: str, man: dict) -> Path | None:
    """Where the batch's report lives: logs/<batch>/report/ (smoke) or paper_contents/v<version>/."""
    d = LOGS / batch / "report"
    if d.is_dir():
        return d
    v = man.get("version")
    if v and man.get("purpose") == "main" and (PAPER / f"v{v}").is_dir():
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
