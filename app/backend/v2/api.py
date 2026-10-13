"""v2 panel API (/api/v2/...), read-only. Built on docs/data_spec.md: a batch is its manifest's plan, a run is its
results directory; status comes from the files plus the live processes, metrics from results.json, charts from
training_curve.csv and the report's report_data.json / lam_sweep_<learner>.json.

    GET /api/v2/meta                                         stages, spaces
    GET /api/v2/overview                                     running / reporting / recent batches, queue, machine
    GET /api/v2/batches?stage=&tests=                        batch summaries, newest first
    GET /api/v2/batches/{name}                               summary + config + runs (with metrics) + references + reports
    GET /api/v2/batches/{name}/report?file=&lang=            one report markdown file
    GET /api/v2/batches/{name}/report_data                   report_data.json + lam_sweep_*.json
    GET /api/v2/runs/{space}/{scen}/{algo}/{exp_id}          run.json, metrics, files, batch
    GET /api/v2/runs/{space}/{scen}/{algo}/{exp_id}/trace    training trace (moving averages, thinned)
    GET /api/v2/runs/{space}/{scen}/{algo}/{exp_id}/file/{f} a png / csv / json of the run
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

router = APIRouter(prefix="/api/v2")
RANK = {"done": 0, "running": 1, "waiting": 2, "queued": 2, "stopped": 3}


def _epoch(s: str | None) -> float | None:
    try:
        return datetime.fromisoformat(s).timestamp() if s else None
    except ValueError:
        return None


def build_batch(name: str, detail: bool = False) -> dict | None:
    """One batch: summary (always) and, with detail, every run with its metrics."""
    log_dir = D.LOGS / name
    if not log_dir.is_dir():
        return None
    man = D.manifest(name)
    if man and man.get("kind") == "eval":
        st = L.eval_state(name, man)
        st.pop("results_root", None)
        runs = st.pop("runs")
        out = {"name": name, "kind": "eval", "version": st.get("version"), "purpose": "test", "title": None,
               "description": st.get("description"), "status": st["status"], "space": None, "stage": None,
               "total": st["total_runs"], "counts": {k: st.get(k, 0) for k in ("done", "running", "queued", "stopped", "skipped")},
               "pct": st["overall_pct"], "started_at": st["started_at"], "updated_at": st["last_updated"]}
        if detail:
            out["runs"] = runs
            out["config"] = {k: v for k, v in man.items() if k != "jobs"}
        return out
    runs = D.plan(name, man)
    if not runs:
        return None
    man = man or {}
    space = man.get("results_space") or man.get("branch")
    if not space:
        first = next((r for r in runs if r.get("exp_id")), None)
        space = D.find_space(first["scenario"], first["algo"], first["exp_id"]) if first else None
    live = L.live_exp_ids()
    for r in runs:
        r["space"] = space
        if not r.get("exp_id"):
            r["status"] = "queued"
            continue
        rdir = D.RESULTS / (space or "_") / r["scenario"] / r["algo"] / r["exp_id"]
        if not rdir.is_dir():
            sp = D.find_space(r["scenario"], r["algo"], r["exp_id"])
            if sp:
                r["space"], rdir = sp, D.RESULTS / sp / r["scenario"] / r["algo"] / r["exp_id"]
        if (rdir / "results.json").is_file():
            r["status"] = "done"
        elif r["exp_id"] in live:
            r["status"] = "running"
        elif rdir.is_dir():
            r["status"] = "stopped"
        else:
            r["status"] = "queued"
    # a slot can hold several attempts (a crash and its retry): keep the best one
    best: dict = {}
    for r in runs:
        slot = (r.get("key"), r["scenario"], r["algo"], r.get("role"), json.dumps(r.get("params") or {}, sort_keys=True),
                r.get("steps"), r.get("seed"))
        if slot not in best or RANK[r["status"]] < RANK[best[slot]["status"]]:
            best[slot] = r
    runs = list(best.values())
    open_groups = {r["group"] for r in runs if r.get("group") and r["status"] in ("running", "queued", "waiting")}
    for r in runs:
        if r["status"] == "queued" and any(g in open_groups for g in r.get("depends_on") or []):
            r["status"] = "waiting"
    total = max(len(man.get("runs") or []) if man.get("schema") == 2 else 0, len(runs))
    n = lambda s: sum(1 for r in runs if r["status"] == s)
    if (log_dir / ".cancelled").is_file():
        status = "cancelled"
    elif n("running") or (n("done") < len(runs) and L.driver_alive(man)):
        status = "running"
    elif n("done") == len(runs):
        status = "finished"
    else:
        status = "stopped"
    if status != "running":                           # nothing will launch them any more
        for r in runs:
            if r["status"] in ("queued", "waiting"):
                r["status"] = "skipped"
    counts = {s: n(s) for s in ("done", "running", "waiting", "queued", "stopped", "skipped")}
    frac = counts["done"]
    for r in runs:
        r["progress"] = None
        if r["status"] == "running":
            r["progress"] = L.train_pct(name, r["exp_id"])
            frac += (r["progress"] or 0) / 100
    logs = list(log_dir.glob("*.log"))
    updated = max([f.stat().st_mtime for f in logs] + [log_dir.stat().st_mtime])
    started = _epoch(man.get("created_at")) or L.birth_time(logs + [log_dir / "manifest.json"]) or log_dir.stat().st_ctime
    rdir = D.report_dir(name, man)
    rep = L.report_stage(name) if status != "running" else None
    out = {"name": name, "kind": "train", "version": man.get("version"),
           "purpose": man.get("purpose") or ("smoke" if "smoke" in name else "main"),
           "title": man.get("title"), "commit": man.get("commit"), "status": status, "space": space,
           "stage": D.stage_of(space), "steps": man.get("steps"), "total": total, "counts": counts,
           "pct": round(100 * frac / total, 1) if total else 0.0,
           "started_at": started, "updated_at": updated,
           "elapsed": (time.time() if status == "running" else updated) - started,
           "report": rep, "has_report": bool(rdir and (rdir / "report.md").is_file()),
           "learners": sorted({r.get("learner") or D.split_algo(r["algo"])[0] for r in runs}, key=lambda l: (l not in D.LEARNERS, D.LEARNERS.index(l) if l in D.LEARNERS else 0, l)),
           "mechanisms": [m for m in D.MECHANISMS if any((r.get("mechanism") or D.split_algo(r["algo"])[1]) == m for r in runs)],
           "scenarios": [s for s in D.SCENARIOS if any(r["scenario"] == s for r in runs)],
           "seeds": sorted({r["seed"] for r in runs if r.get("seed") is not None}),
           "lambda_max": man.get("lambda_max"), "lam_sweep": man.get("lam_sweep")}
    if not detail:
        return out
    for r in runs:
        r.setdefault("learner", D.split_algo(r["algo"])[0])
        r.setdefault("mechanism", D.split_algo(r["algo"])[1])
        r["metrics"] = (D.metrics(D.results_json(D.RESULTS / r["space"] / r["scenario"] / r["algo"] / r["exp_id"]), r["algo"])
                        if r["status"] == "done" and r.get("space") else None)
    refs = []
    for ref in man.get("references") or []:
        sp = ref.get("results_space") or D.find_space(ref["scenario"], ref["algo"], str(ref.get("exp_id")))
        m = D.metrics(D.results_json(D.RESULTS / (sp or "_") / ref["scenario"] / ref["algo"] / str(ref.get("exp_id"))), ref["algo"])
        refs.append(dict(ref, role="reference", space=sp, status="done" if m else "stopped", metrics=m,
                         learner=ref.get("learner") or D.split_algo(ref["algo"])[0],
                         mechanism=ref.get("mechanism") or D.split_algo(ref["algo"])[1]))
    out["runs"] = runs
    out["references"] = refs
    out["reports"] = D.report_files(rdir)
    out["config"] = {k: v for k, v in man.items() if k not in ("runs", "exp_ids", "references")}
    return out


def warm_traces():
    """Background, at startup: compute the trace cache (default chart settings) of every finished run of the
    schema-2 batches, so the charts of a 20M batch do not parse ~80 MB per run on first view."""
    def work():
        for n in D.batch_names():
            man = D.manifest(n)
            if not man or man.get("schema") != 2:
                continue
            for r in man.get("runs", []):
                sp = man.get("results_space")
                if r.get("exp_id") and sp:
                    p = D.RESULTS / sp / r["scenario"] / r["algo"] / r["exp_id"]
                    if (p / "results.json").is_file():
                        try:
                            D.trace(p, 240, 1000)
                        except Exception:       # a broken csv must not stop the others
                            pass
    threading.Thread(target=work, daemon=True, name="v2-trace-warm").start()


@router.get("/meta")
def meta():
    return {"stages": D.stages(), "spaces": D.spaces(), "now": time.time()}


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
def batches(stage: str = "current", tests: bool = False):
    """stage: a stage number, "current" or "all"; tests: also smoke / test batches."""
    bs = _all_batches()
    if stage != "all":
        want = next((s["stage"] for s in D.stages() if s["current"]), None) if stage == "current" else int(stage)
        bs = [b for b in bs if b["stage"] == want or (b["stage"] is None and b["status"] == "running")]
    if not tests:
        bs = [b for b in bs if b["purpose"] == "main" or b["status"] == "running"]
    return {"batches": bs}


@router.get("/overview")
def overview():
    bs = _all_batches()
    cur = next((s["stage"] for s in D.stages() if s["current"]), None)
    reporting = [b for b in bs if b["status"] != "running" and b.get("report") and not b["report"]["finished"]
                 and time.time() - b["report"]["updated"] < 3600]
    recent = [b for b in bs if b["status"] != "running" and b["purpose"] == "main" and b["stage"] == cur][:6]
    sysd = L.system()
    procs = [p for p in sysd.get("processes", []) if p.get("scenario") and p.get("algo")]
    return {"running": [b for b in bs if b["status"] == "running"], "reporting": reporting, "recent": recent,
            "queue": L.queue(),
            "machine": {"cpu_count": sysd.get("cpu_count"), "load_avg": sysd.get("load_avg"), "host": sysd.get("host"),
                        "gpus": sysd.get("gpus", []),
                        "training_procs": len(procs)}}


@router.get("/batches/{name}")
def batch(name: str):
    if not D.manifest(name) and not (D.LOGS / name).is_dir():
        raise HTTPException(404, f"no such batch: {name}")
    b = build_batch(name, detail=True)
    if b is None:
        raise HTTPException(404, f"batch {name} has no runs")
    return b


@router.get("/batches/{name}/report")
def report(name: str, file: str = "report.md", lang: str = "zh"):
    man = D.manifest(name) or {}
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
            "sweeps": [j for p in sorted(rdir.glob("lam_sweep_*.json")) if (j := D.read_json(p))]}


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
            man = D.manifest(n)
            if man and man.get("kind") == "eval":
                continue
            for r in D.plan(n, man):
                if r.get("exp_id"):
                    idx[r["exp_id"]] = {"batch": n, "version": (man or {}).get("version"), "key": r.get("key"),
                                        "role": r.get("role"), "group": r.get("group")}
        _batch_index.update(t=time.time(), idx=idx)
    return _batch_index["idx"].get(exp_id)


@router.get("/runs/{space}/{scen}/{algo}/{exp_id}")
def run(space: str, scen: str, algo: str, exp_id: str):
    p = _run_path(space, scen, algo, exp_id)
    res = D.results_json(p)
    rj = D.read_json(p / "run.json") or {}
    learner, mech = D.split_algo(algo)
    files = sorted(f.name for f in p.iterdir() if f.is_file() and f.suffix in (".png", ".csv", ".json"))
    ck = sorted(int(f.name.split("_")[1]) for f in p.glob("ckpt_*_*.zip") if f.name.split("_")[1].isdigit())
    live = exp_id in L.live_exp_ids()
    b = _batch_of(exp_id)
    prog = D.read_json(p / ".progress.json")
    return {"space": space, "scenario": scen, "algo": algo, "exp_id": exp_id, "learner": rj.get("learner") or learner,
            "mechanism": rj.get("mechanism") or mech, "run": rj, "batch": rj.get("batch") or (b or {}).get("batch"),
            "batch_info": b, "status": "done" if res else ("running" if live else "stopped"),
            "progress": L.train_pct(b["batch"], exp_id) if live and b else None, "live_progress": prog if live else None,
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


@router.get("/system")
def system():
    return L.system()


@router.get("/gpu_history")
def gpu_history(since: float = 0.0):
    return L.gpu_history(since)


@router.get("/queue")
def queue():
    return L.queue()
