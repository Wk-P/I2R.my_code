"""v2 panel API (/api/v2/...), read-only and independent of the old backend (app/backend/main.py). Built on
docs/data_spec.md: only batches with a schema-2 manifest (v5.0.3 on) exist here -- older batches are the classic
panel's (/classic/). A batch is its manifest's plan, a run is its results directory; status comes from the files plus
the live processes, metrics from results.json, charts from training_curve.csv and the report's report_data.json /
lam_sweep_<algorithm>.json. Field names follow docs/data_spec.md (schema 3: RL_algorithm, strategy, seed, params);
v5.0.3's schema-2 manifest is converted by data.normalize.

    GET /api/v2/overview                                     running / reporting / recent batches, queue, machine
    GET /api/v2/batches?tests=                               batch summaries, newest first (tests: also smoke)
    GET /api/v2/batches/{name}                               summary + config + runs (with metrics) + references + reports
    GET /api/v2/batches/{name}/report?file=&lang=            one report markdown file
    GET /api/v2/batches/{name}/report_data                   report_data.json + lam_sweep_*.json
    GET /api/v2/runs/{space}/{scen}/{algo}/{exp_id}          run.json, metrics, files, batch
    GET /api/v2/runs/{space}/{scen}/{algo}/{exp_id}/trace    training trace (moving averages, thinned)
    GET /api/v2/runs/{space}/{scen}/{algo}/{exp_id}/file/{f} a png / csv / json of the run
    GET /api/v2/summary                                      strategy x RL_algorithm summary (gap over LT / GT / EQ)
    GET /api/v2/summary/file/{f}                             its paper figures / LaTeX table (listed in table_2d.json)
    GET /api/v2/system, /api/v2/gpu_history, /api/v2/queue   machine and queue
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from . import data as D
from . import live as L
from . import versions as VT

router = APIRouter(prefix="/api/v2")
RANK = {"done": 0, "running": 1, "waiting": 2, "queued": 2, "stopped": 3}


def _epoch(s: str | None) -> float | None:
    try:
        return datetime.fromisoformat(s).timestamp() if s else None
    except ValueError:
        return None


def _batch_eta(runs: list[dict], batch_steps: int | None) -> dict | None:
    """Rough time to the end of training: the running runs finish at their own speed; the queued / waiting ones
    take the free slots in plan order (as the driver does), each at the mean speed of the running runs of the same
    RL_algorithm (else of all). Slots = the runs running now. Report time not included."""
    run = [r for r in runs if r["status"] == "running" and (r["live"] or {}).get("eta_seconds") is not None]
    if not run:
        return None
    speed = {}
    for r in run:
        speed.setdefault(r["RL_algorithm"], []).append(r["live"]["steps_per_sec"])
    mean = lambda xs: sum(xs) / len(xs)
    every = mean([x for v in speed.values() for x in v])
    slots = sorted(r["live"]["eta_seconds"] for r in run)
    for r in runs:
        if r["status"] in ("queued", "waiting"):
            st = r["params"].get("steps") or batch_steps or 0
            v = mean(speed[r["RL_algorithm"]]) if r["RL_algorithm"] in speed else every
            t0 = slots.pop(0)
            slots.append(t0 + st / v)
            slots.sort()
    return {"seconds": max(slots), "running_done_seconds": max(r["live"]["eta_seconds"] for r in run)}


def build_batch(name: str, detail: bool = False) -> dict | None:
    """One batch: summary (always) and, with detail, every run with its metrics."""
    log_dir = D.LOGS / name
    raw = D.manifest(name)
    if not raw or raw.get("schema") not in (2, 3):
        return None
    man = D.normalize(raw)
    runs = man["runs"]
    space = man.get("results_space")
    live = L.live_exp_ids()
    for r in runs:
        r["space"] = space
        rdir = D.RESULTS / space / r["scenario"] / r["dir"] / r["exp_id"] if r.get("exp_id") and space else None
        if rdir is None:
            r["status"] = "queued"
        elif (rdir / "results.json").is_file():
            r["status"] = "done"
        elif r["exp_id"] in live:
            r["status"] = "running"
        elif rdir.is_dir():
            r["status"] = "stopped"
        else:
            r["status"] = "queued"
    open_groups = {r["group"] for r in runs if r.get("group") and r["status"] in ("running", "queued")}
    for r in runs:
        if r["status"] == "queued" and any(g in open_groups for g in r["depends_on"]):
            r["status"] = "waiting"
    n = lambda st: sum(1 for r in runs if r["status"] == st)
    training = man.get("purpose") not in D.NON_TRAINING
    if (log_dir / ".cancelled").is_file():
        status = "cancelled"
    elif not training:                                # no runs: the script's own status (docs/data_spec.md §6)
        status = "running" if L.script_alive(raw) else {"failed": "stopped"}.get(man.get("status"), "finished")
    elif n("running") or (n("done") < len(runs) and L.driver_alive(raw)):
        status = "running"
    elif n("done") == len(runs):
        status = "finished"
    else:
        status = "stopped"
    if status != "running":                           # nothing will launch them any more
        for r in runs:
            if r["status"] in ("queued", "waiting"):
                r["status"] = "skipped"
    counts = {st: n(st) for st in ("done", "running", "waiting", "queued", "stopped", "skipped")}
    frac = counts["done"]
    for r in runs:
        r["live"] = L.run_live(name, r["exp_id"]) if r["status"] == "running" else None
        r["progress"] = r["live"]["pct"] if r["live"] else None
        frac += (r["progress"] or 0) / 100
    eta = _batch_eta(runs, man.get("params", {}).get("steps")) if status == "running" else None
    logs = list(log_dir.glob("*.log"))
    updated = max([f.stat().st_mtime for f in logs] + [log_dir.stat().st_mtime])
    started = _epoch(man.get("created_at")) or log_dir.stat().st_mtime
    if not training and _epoch(man.get("finished_at")):
        updated = _epoch(man["finished_at"])
    rdir = D.report_dir(name, man)
    total = len(runs)
    version = str(man["version"]).removeprefix("v") if man.get("version") else None    # v6 manifests write "v6.0.0.2"
    out = {"name": name, "version": version, "purpose": man.get("purpose") or "main",
           "title": man.get("title"), "label": man.get("label"), "description": man.get("description"), "commit": man.get("commit"), "status": status, "space": space,
           "total": total, "counts": counts,
           "pct": round(100 * frac / total, 1) if total else (100.0 if status == "finished" else 0.0), "eta": eta,
           "training": training, "script": man.get("script"), "depends_on": man.get("depends_on") or [],
           "steps_per_sec": sum((r["live"] or {}).get("steps_per_sec") or 0 for r in runs) or None,
           "started_at": started, "updated_at": updated,
           "elapsed": (time.time() if status == "running" else updated) - started,
           "report": L.report_stage(name) if status != "running" else None,
           "has_report": bool(rdir and any(rdir.glob("*.md"))),
           "RL_algorithm": man.get("RL_algorithm", []), "strategies": man.get("strategies", []), "seeds": man.get("seeds", []),
           "scenarios": man.get("scenarios", []), "params": man.get("params", {}), "links": man.get("links") or []}
    if not detail:
        return out
    for r in runs:
        r["metrics"] = (D.metrics(D.results_json(D.RESULTS / space / r["scenario"] / r["dir"] / r["exp_id"]), r["dir"])
                        if r["status"] == "done" else None)
    refs = []
    for ref in man.get("references") or []:
        sp = ref.get("results_space")
        d = ref.get("dir") or ref.get("algo")
        m = D.metrics(D.results_json(D.RESULTS / (sp or "_") / ref["scenario"] / d / str(ref.get("exp_id"))), d)
        rl, st = D.from_dir(d)
        refs.append({"role": "reference", "label": ref.get("label"), "RL_algorithm": rl, "strategy": st, "seed": ref.get("seed"),
                     "scenario": ref["scenario"], "params": ref.get("params") or {}, "dir": d, "exp_id": ref.get("exp_id"),
                     "space": sp, "status": "done" if m else "stopped", "metrics": m})
    out["runs"] = runs
    out["references"] = refs
    out["reports"] = D.report_files(rdir)
    if not training:                                  # the script's log, shown next to its report
        out["reports"] += [{"file": p.name, "title": p.name, "has_en": False, "log": True} for p in D.batch_logs(name, man)]
    out["config"] = {k: v for k, v in man.items() if k not in ("runs", "references")}
    return out


def warm_traces():
    """Background, at startup: compute the trace cache (default chart settings) of every finished run of the
    schema-2 batches, so the charts of a 20M batch do not parse ~80 MB per run on first view."""
    def work():
        for n in D.batch_names():
            raw = D.manifest(n)
            if not raw:
                continue
            man = D.normalize(raw)
            for r in man["runs"]:
                sp = man.get("results_space")
                if r.get("exp_id") and sp:
                    p = D.RESULTS / sp / r["scenario"] / r["dir"] / r["exp_id"]
                    if (p / "results.json").is_file():
                        try:
                            D.trace(p, 240, 1000)
                        except Exception:       # a broken csv must not stop the others
                            pass
    threading.Thread(target=work, daemon=True, name="v2-trace-warm").start()


_all_cache: dict = {"t": 0.0, "v": []}
_all_lock = threading.Lock()


def _all_batches() -> list[dict]:
    """Every batch summary, newest first; cached 5 s (the overview, the batch list and the sidebar all poll it)."""
    with _all_lock:
        if time.time() - _all_cache["t"] > 5:
            out = [b for b in (build_batch(n) for n in D.batch_names()) if b]
            _all_cache.update(t=time.time(), v=sorted(out, key=lambda b: b["started_at"] or 0, reverse=True))
        return _all_cache["v"]


@router.get("/batches")
def batches(tests: bool = False):
    """Every schema-2 batch; smoke / test batches only with tests=true (a running one always)."""
    bs = _all_batches()
    if not tests:
        bs = [b for b in bs if b["purpose"] in ("main", *D.NON_TRAINING) or b["status"] == "running"]
    return {"batches": bs}


@router.get("/overview")
def overview():
    bs = _all_batches()
    reporting = [b for b in bs if b["status"] != "running" and b.get("report") and not b["report"]["finished"]
                 and time.time() - b["report"]["updated"] < 3600]
    recent = [b for b in bs if b["status"] != "running" and b["purpose"] == "main"][:6]
    sysd = L.system()
    running, runs = _running_runs()
    for b in running:
        b.pop("runs", None)
        b.pop("references", None)
    return {"running": running, "running_runs": runs, "reporting": reporting, "recent": recent,
            "queue": L.queue(),
            "machine": {"cpu_count": sysd["cpu_count"], "load_avg": sysd["load_avg"], "memory": sysd["memory"],
                        "gpus": sysd["gpus"], "training_procs": sum(1 for p in sysd["processes"] if p["kind"] == "train")}}


def _running_runs() -> tuple[list[dict], list[dict]]:
    """(running batches with detail, their running runs each tagged with its batch)."""
    bs = [build_batch(b["name"], detail=True) for b in _all_batches() if b["status"] == "running"]
    bs = [b for b in bs if b]
    runs = [dict(r, batch=b["name"]) for b in bs for r in b["runs"] if r["status"] == "running"]
    return bs, runs


@router.get("/live")
def live():
    """Everything the live monitor shows: running batches (progress, ETA, throughput), every running run with its live
    progress, training processes that belong to no v2 batch, report stages, the machine and the queue."""
    bs, runs = _running_runs()
    sysd = L.system()
    known = {r["exp_id"] for r in runs}
    other_train = [p for p in sysd["processes"] if p["kind"] == "train" and p.get("exp_id") not in known]
    reporting = [b for b in _all_batches() if b["status"] != "running" and b.get("report") and not b["report"]["finished"]
                 and time.time() - b["report"]["updated"] < 3600]
    for b in bs:
        b.pop("runs", None)
        b.pop("references", None)
    return {"batches": bs, "runs": runs, "other_training": other_train, "reporting": reporting,
            "system": sysd, "queue": L.queue(), "now": time.time()}


@router.get("/batches/{name}")
def batch(name: str):
    b = build_batch(name, detail=True)
    if b is None:
        raise HTTPException(404, f"no such batch (v2 panel: schema-2 manifests only): {name}")
    return b


@router.get("/batches/{name}/report")
def report(name: str, file: str = "report.md", lang: str = "zh"):
    man = D.manifest(name) or {}
    if "/" not in file and file.endswith(".log"):     # a non-training batch's log
        p = next((x for x in D.batch_logs(name, man) if x.name == file), None)
        if p is None:
            raise HTTPException(404)
        return {"file": p.name, "dir": str(p.parent.relative_to(D.ROOT)), "text": p.read_text(errors="replace"),
                "updated_at": p.stat().st_mtime, "log": True}
    rdir = D.report_dir(name, man)
    if rdir is None or "/" in file or not file.endswith(".md"):
        raise HTTPException(404)
    p = rdir / file
    if lang == "en" and (rdir / (p.stem + ".en.md")).is_file():
        p = rdir / (p.stem + ".en.md")
    if not p.is_file():
        raise HTTPException(404)
    return {"file": p.name, "dir": str(rdir.relative_to(D.ROOT)), "text": p.read_text(errors="replace"),
            "updated_at": p.stat().st_mtime}


@router.get("/batches/{name}/report_data")
def report_data(name: str):
    rdir = D.report_dir(name, D.manifest(name) or {})
    if rdir is None:
        return {"report": None, "sweeps": []}
    return {"report": D.read_json(rdir / "report_data.json"),
            "sweeps": [dict(n, RL_algorithm=(n.get("RL_algorithm") or n.get("learner") or "").upper())
                       for p in sorted(rdir.glob("lam_sweep_*.json")) if (n := D.read_json(p))]}


def _run_path(space, scen, algo, exp_id) -> Path:
    try:
        p = D.run_dir(space, scen, algo, exp_id)
    except ValueError:
        raise HTTPException(400)
    if not p.is_dir():
        raise HTTPException(404, "no such run")
    return p


_batch_index: dict = {"t": 0.0, "idx": {}}


def _batch_of(exp_id: str) -> dict | None:
    if time.time() - _batch_index["t"] > 60:
        idx = {}
        for n in D.batch_names():
            man = D.normalize(D.manifest(n))
            for r in man["runs"]:
                if r.get("exp_id"):
                    idx[r["exp_id"]] = {"batch": n, "version": man.get("version"), "key": r.get("key"),
                                        "role": r.get("role"), "group": r.get("group")}
        _batch_index.update(t=time.time(), idx=idx)
    return _batch_index["idx"].get(exp_id)


@router.get("/runs/{space}/{scen}/{algo}/{exp_id}")
def run(space: str, scen: str, algo: str, exp_id: str):
    b = _batch_of(exp_id)
    if b is None:                                    # a run of an old batch: classic panel only
        raise HTTPException(404, "not a run of a v2 batch")
    p = _run_path(space, scen, algo, exp_id)
    res = D.results_json(p)
    rj = D.read_json(p / "run.json") or {}
    info = D.run_info(rj, algo)
    files = sorted(f.name for f in p.iterdir() if f.is_file() and f.suffix in (".png", ".csv", ".json"))
    ck = sorted(int(f.name.split("_")[1]) for f in p.glob("ckpt_*_*.zip") if f.name.split("_")[1].isdigit())
    live = exp_id in L.live_exp_ids()
    return {"space": space, "scenario": scen, "dir": algo, "exp_id": exp_id, **info, "seed": rj.get("seed"),
            "role": rj.get("role"), "group": b.get("group"), "key": rj.get("key"), "batch": b["batch"],
            "status": "done" if res else ("running" if live else "stopped"),
            "live": L.run_live(b["batch"], exp_id) if live else None,
            "log_series": L.log_series(b["batch"], exp_id) if not (p / "training_curve.csv").exists() else None,
            "metrics": D.metrics(res, algo),
            "config": {k: v for k, v in (res or {}).items() if not isinstance(v, dict) or k in ("data",)},
            "evaluation": next((v for k, v in (res or {}).items() if isinstance(v, dict) and "ar_mean" in v), None),
            "training": (res or {}).get("training"), "checkpoints": ck,
            "ckpt_lambda": D.read_json(p / "ckpt_lambda.json"), "files": files}


@router.get("/runs/{space}/{scen}/{algo}/{exp_id}/trace")
def run_trace(space: str, scen: str, algo: str, exp_id: str, points: int = 400, window: int = 1000):
    t = D.trace(_run_path(space, scen, algo, exp_id), max(20, min(points, 2000)), max(1, min(window, 20000)))
    if t is None:
        raise HTTPException(404, "no training_curve.csv")
    return t


@router.get("/runs/{space}/{scen}/{algo}/{exp_id}/file/{fname}")
def run_file(space: str, scen: str, algo: str, exp_id: str, fname: str):
    p = _run_path(space, scen, algo, exp_id) / fname
    if "/" in fname or p.suffix not in (".png", ".csv", ".json") or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "no-cache"})


@router.get("/summary")
def summary():
    # written by src/scripts/merge_2d_table.py
    d = D.read_json(D.PAPER / "v6.0.0-merge" / "table_2d.json")
    if d is None:
        raise HTTPException(404, "paper_contents/v6.0.0-merge/table_2d.json not found")
    return d


@router.get("/summary/file/{fname}")
def summary_file(fname: str):
    d = D.read_json(D.PAPER / "v6.0.0-merge" / "table_2d.json") or {}
    p = D.PAPER / "v6.0.0-merge" / fname
    if fname not in d.get("files", []) or not p.is_file():
        raise HTTPException(404, fname)
    media = {".tex": "text/plain; charset=utf-8"}.get(p.suffix)
    return FileResponse(p, media_type=media, filename=fname if p.suffix in (".pdf", ".tex") else None)


@router.get("/version_trend")
def version_trend():
    """Test relative gap by version (versions.py; the comparable versions are listed in version_trend.json)."""
    return VT.trend()


@router.get("/system")
def system():
    return L.system()


@router.get("/gpu_history")
def gpu_history(since: float = 0.0):
    return L.gpu_history(since)


@router.get("/queue")
def queue():
    return L.queue()
