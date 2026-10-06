"""
my-code experiment dashboard — read-only view over
results/<branch>/<scenario>/<algo>/<run>/ for the currently checked-out git
branch, plus best-effort process/log introspection for live training
progress.

This module never imports or executes anything under scenarios/ or shared/ —
it only reads files (results.json, PNGs, log files) and OS process state
(ps, /proc, git). It cannot affect running or future training runs.

Serves:
  GET  /api/results                                  summary table (latest run per scenario/algo)
  GET  /api/results/{scenario}/{algo}/{run}/{file}    training_curve.png / comparison.png
  GET  /api/history/{scenario}/{algo}                 all historical runs for one algo
  GET  /api/progress                                  live training progress per scenario
  GET  /api/batches                                    auto-discovered ad-hoc batch names under scripts/logs/
  GET  /api/batch_progress/{batch_name}                per-run status/results for one ad-hoc batch
  GET  /api/branch                                    current git branch + whether it has BC support
  GET  /api/system                                    CPU/load info, grouped by pinned core range
  GET  /                                              single-page dashboard (vanilla JS, no build step)
"""

import itertools
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

APP_DIR           = Path(__file__).parent
PROJECT_ROOT      = APP_DIR.parent.parent
RESULTS_ROOT_BASE = PROJECT_ROOT / "results"
SCENARIOS         = ["eq", "gt", "lt"]

app = FastAPI(title="my-code experiment dashboard")
app.mount("/assets", StaticFiles(directory=APP_DIR / "static" / "assets"), name="assets")


# ── Git branch awareness ────────────────────────────────────────────────────
#
# results/<branch>/... mirrors shared/paths.py's own branch-scoped
# results_dir() — see that module's docstring. This backend is a long-lived
# server (unlike the one-shot training scripts), so the branch has to be
# re-resolved on every request rather than cached at import time: a `git
# checkout` in another terminal must show up here without restarting uvicorn.

def _git_current_branch() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_branch_list() -> list[str]:
    try:
        r = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads/"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        return [b for b in r.stdout.splitlines() if b] if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        return []


def _active_space() -> str:
    """results/<space>/ that new training runs write to: $RESULTS_SPACE, else
    RESULTS_SPACE in shared/version_config.py (read as text -- this module
    never imports shared/), else the checked-out branch (pre-v4.3.1 layout)."""
    if os.environ.get("RESULTS_SPACE"):
        return os.environ["RESULTS_SPACE"]
    try:
        m = re.search(r'^RESULTS_SPACE\s*=\s*"([^"]+)"',
                      (PROJECT_ROOT / "shared" / "version_config.py").read_text(), re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    return _git_current_branch() or "unknown"


def _results_root(branch: str | None = None) -> Path:
    """results/<space>/ -- `branch` (the ?branch= query param, kept under
    that name for URL compatibility) may name any existing results/
    subdirectory, whether or not a git branch of that name still exists;
    anything else falls back to the active space."""
    if branch and "/" not in branch and ".." not in branch and (RESULTS_ROOT_BASE / branch).is_dir():
        return RESULTS_ROOT_BASE / branch
    return RESULTS_ROOT_BASE / _active_space()


@app.get("/api/branch")
def get_branch():
    return {
        # "current" = the results space new runs write to (see _active_space)
        "current":      _active_space(),
        "git_branch":   _git_current_branch(),
        "branches":     _git_branch_list(),
        "bc_supported": (PROJECT_ROOT / "shared" / "bc_pretrain.py").is_file(),
        # Branches that have a results/<branch>/ tree, with their run count
        # (results.json files), so the UI can offer only browsable branches.
        "result_branches": _result_branches(),
    }


def _result_branches() -> list[dict]:
    """Every results/<space>/ with runs, labelled from result_spaces.json
    (stage name, version range, branch vs. model collection) and the actual
    date range of its runs; newest stage first."""
    if not RESULTS_ROOT_BASE.is_dir():
        return []
    meta = _read_json(APP_DIR / "result_spaces.json") or {}
    out = []
    for b in RESULTS_ROOT_BASE.iterdir():
        if not b.is_dir():
            continue
        files = list(b.glob("*/*/*/results.json"))
        if not files:
            continue
        mtimes = [f.stat().st_mtime for f in files]
        m = meta.get(b.name, {})
        out.append({
            "name": b.name, "runs": len(files),
            "label": m.get("label", b.name), "versions": m.get("versions"),
            "kind": m.get("kind", "branch"), "order": m.get("order", 99),
            "first": min(mtimes), "last": max(mtimes),
        })
    out.sort(key=lambda x: (x["order"], -x["last"]))
    return out


def _algo_key(data: dict) -> str | None:
    # Any results.json top-level key that isn't the algo's own result block
    # must be listed here, or _algo_key() mistakes it for one and every
    # .get("ar_mean")-style read below blows up on a str/int instead of the
    # algo eval dict. "bc" and "exp_id" are both metadata added by
    # run_all_bc.py — see shared/bc_pretrain.py.
    reserved = ("scenario", "prototype_scenario", "scenario_count",
                "train_count", "test_count", "N", "M", "ilp", "training",
                "feasibility", "created_at", "bc", "exp_id",
                # paper_rl (v4.3.1.3+) metadata
                "version", "commit", "algo", "mechanism", "learner", "reward_mode",
                "reward_norm", "obs", "gamma", "full_episode", "seed", "data")
    # The algo block is always a dict; skipping scalars keeps a newly added
    # metadata field from taking down /api/experiments again.
    return next((k for k, v in data.items() if k not in reserved and isinstance(v, dict)), None)


def _nan_to_none(v):
    """A crashed/degenerate training run can leave NaN in results.json
    (e.g. ar_mean over zero successful episodes). Python's json module
    round-trips NaN fine, but Starlette's JSONResponse.render() calls
    json.dumps(..., allow_nan=False) and raises ValueError on it, which
    previously took down /api/experiments (and everything downstream of it,
    including the whole batch/results dashboard) for ALL runs the instant
    a single bad run's NaN was in the list."""
    return None if isinstance(v, float) and math.isnan(v) else v


def _row_from_run(scenario: str, algo: str, run_dir: Path) -> dict | None:
    try:
        data = json.loads((run_dir / "results.json").read_text())
    except (json.JSONDecodeError, OSError):
        return None
    algo_key  = _algo_key(data)
    algo_eval = data.get(algo_key, {}) if algo_key else {}
    training  = data.get("training", {})
    ilp       = data.get("ilp", {})
    # run_all_bc.py (ILP behavior-cloning pretrain) writes its result key as
    # "<algo>_bc" (e.g. "maskable_ppo_bc") — see shared/bc_pretrain.py and
    # each scenarios/<scenario>/<algo>/run_all_bc.py. That's the single
    # source of truth for "is this a BC run", since it comes straight from
    # the training script itself rather than a directory-naming convention.
    is_bc = bool(algo_key) and algo_key.endswith("_bc")
    # "run" (run_dir.name) and "exp_id" (the whole eq+gt+lt batch id, see
    # shared/paths.new_exp_id) are different things that happened to always
    # be equal — until run_all_bc.py started suffixing the run dir with
    # "_bc" to keep it separate from the baseline run in the same algo
    # folder. run_all_bc.py now writes "exp_id" into results.json explicitly
    # so batch grouping (ExperimentTree) can use the real id instead of the
    # directory name. Older baseline runs never had a reason to diverge, so
    # falling back to run_dir.name for them is exact, not approximate.
    exp_id = data.get("exp_id") or run_dir.name
    return {
        "scenario":      scenario,
        "algo":          algo,
        "display_algo":  f"{algo}+bc" if is_bc else algo,
        "is_bc":         is_bc,
        "run":           run_dir.name,
        "exp_id":        exp_id,
        "created_at":    data.get("created_at"),
        "N":             data.get("N"),
        "M":             data.get("M"),
        "train_count":   data.get("train_count"),
        "test_count":    data.get("test_count"),
        "ilp_ar":        _nan_to_none(ilp.get("ar")),
        "test_ar_mean":  _nan_to_none(algo_eval.get("ar_mean")),
        "test_ar_std":   _nan_to_none(algo_eval.get("ar_std")),
        "test_success_rate":        _nan_to_none(algo_eval.get("success_rate")),
        "test_cap_viol_rate":       _nan_to_none(algo_eval.get("cap_viol_rate")),
        "test_conflict_viol_rate":  _nan_to_none(algo_eval.get("conflict_viol_rate")),
        "test_cap_viol_total":      algo_eval.get("cap_viol_total"),
        "test_conflict_viol_total": algo_eval.get("conflict_viol_total"),
        "train_ar_last50": _nan_to_none(training.get("ar_last50")),
        "train_steps":     training.get("total_steps"),
        "train_episodes":  training.get("n_episodes"),
    }


_TS_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$")


def _sort_key(run_dir: Path) -> str:
    """Run dirs used to be named as timestamps (lexicographic == chronological);
    now they're random exp_id hashes, so chronology has to come from
    results.json's created_at field, with a fallback that normalizes old
    timestamp dir names into the same sortable ISO-ish shape."""
    try:
        data = json.loads((run_dir / "results.json").read_text())
    except (json.JSONDecodeError, OSError):
        data = {}
    if data.get("created_at"):
        return data["created_at"]
    m = _TS_RE.match(run_dir.name)
    if m:
        y, mo, d, h, mi, s = m.groups()
        return f"{y}-{mo}-{d}T{h}:{mi}:{s}"
    return run_dir.name


def _run_dirs(algo_dir: Path) -> list[Path]:
    runs = [d for d in algo_dir.iterdir() if d.is_dir() and (d / "results.json").exists()]
    return sorted(runs, key=_sort_key)


def _collect_results(branch: str | None = None) -> list[dict]:
    rows = []
    results_root = _results_root(branch)
    for scenario in SCENARIOS:
        scenario_dir = results_root / scenario
        if not scenario_dir.is_dir():
            continue
        for algo_dir in sorted(scenario_dir.iterdir()):
            if not algo_dir.is_dir() or algo_dir.name == "ilp":
                continue
            runs = _run_dirs(algo_dir)
            if not runs:
                continue
            row = _row_from_run(scenario, algo_dir.name, runs[-1])
            if row:
                rows.append(row)
    return rows


def _collect_all_experiments(branch: str | None = None) -> list[dict]:
    """Every historical run across every scenario/algo, not just the latest
    one per algo — powers the EXP_ID -> scenario -> algo tree, which needs
    the full history to group by batch (exp_id)."""
    rows = []
    results_root = _results_root(branch)
    for scenario in SCENARIOS:
        scenario_dir = results_root / scenario
        if not scenario_dir.is_dir():
            continue
        for algo_dir in sorted(scenario_dir.iterdir()):
            if not algo_dir.is_dir() or algo_dir.name == "ilp":
                continue
            for run_dir in _run_dirs(algo_dir):
                row = _row_from_run(scenario, algo_dir.name, run_dir)
                if row:
                    rows.append(row)
    return rows


@app.get("/api/results")
def get_results(branch: str | None = None):
    return _collect_results(branch)


def _exp_batch_index() -> dict[str, dict]:
    """exp_id -> {batch, variant, seed} over every batch under scripts/logs/."""
    index = {}
    if not LOGS_ROOT.is_dir():
        return index
    for d in LOGS_ROOT.iterdir():
        if d.is_dir():
            runs, manifest = _batch_runs(d)
            for r in runs:
                if r["exp_id"] is None:
                    continue
                index[r["exp_id"]] = {"batch": d.name, "variant": r["variant"], "seed": r["seed"],
                                      "version": manifest.get("version")}
    return index


@app.get("/api/experiments")
def get_experiments(branch: str | None = None):
    rows = _collect_all_experiments(branch)
    index = _exp_batch_index()
    for r in rows:
        info = index.get(r["exp_id"], {})
        r["batch"] = info.get("batch")
        r["variant"] = info.get("variant") or ""
        r["seed"] = info.get("seed")
        r["version"] = info.get("version")
    return rows


def _git_tag_dates() -> dict[str, str]:
    try:
        r = subprocess.run(
            ["git", "for-each-ref", "--sort=-creatordate",
             "--format=%(refname:short)|%(creatordate:short)", "refs/tags"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        out = {}
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "|" in line:
                    tag, date = line.split("|", 1)
                    out[tag] = date
        return out
    except (OSError, subprocess.SubprocessError):
        return {}


# VERSION.md table rows look like: "| v1.2.3 | 2026-07-16 | 摘要文字 | [doc.md](doc.md) |"
_VERSION_TABLE_ROW = re.compile(
    r"^\|\s*(v[\d.]+)\s*\|\s*([\d-]+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$"
)


def _parse_version_md() -> dict[str, dict]:
    """Reads version/VERSION.md's index table for a one-line summary (and,
    if present, a linked vX.Y.Z.md doc) per tag — this is the single source
    both this endpoint and humans editing the changelog read from, so the
    dashboard never drifts out of sync with what's written there."""
    path = PROJECT_ROOT / "version" / "VERSION.md"
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _VERSION_TABLE_ROW.match(line)
        if not m:
            continue
        tag, date, summary, doc_cell = m.groups()
        if tag == "版本":  # header row
            continue
        doc_m = re.search(r"\(([^)]+\.md)\)", doc_cell)
        out[tag] = {
            "date": date,
            "summary": summary,
            "doc_file": doc_m.group(1) if doc_m else None,
        }
    return out


def _tag_sort_key(tag: str):
    """Newest first for any tag shape: v4.1.0, v4.1.0.2, v4.0.0_final_1,
    lt-1M-bestof32-partial-14of30. Version tags sort by their numeric parts
    (suffix after the numbers ranks after the bare version); other tags go
    last, by name."""
    import re as _re
    m = _re.match(r"^v(\d+(?:\.\d+)*)(.*)$", tag)
    if not m:
        return (0, [], tag)
    return (1, [int(x) for x in m.group(1).split(".")], m.group(2))


def _tag_doc_path(tag: str) -> Path | None:
    """version/<doc>.md (from VERSION.md or <tag>.md), else the paper
    package's README (paper_contents/<tag>/ or paper_contents/[<tag>]/)."""
    info = _parse_version_md().get(tag) or {}
    doc_file = info.get("doc_file") or f"{tag}.md"
    if ".." not in Path(doc_file).parts and (PROJECT_ROOT / "version" / doc_file).is_file():
        return PROJECT_ROOT / "version" / doc_file
    if "/" in tag or ".." in tag:
        return None
    for d in (tag, tag.removeprefix("v"), f"[{tag}]"):
        readme = PROJECT_ROOT / "paper_contents" / d / "README.md"
        if readme.is_file():
            return readme
    return None


@app.get("/api/tags")
def get_tags():
    """All git tags with a description for the frontend: VERSION.md's summary
    line, plus whether a standalone version/vX.Y.Z.md doc exists (fetch its
    body via /api/tags/{tag}/doc). Doesn't require a `git checkout` — reads
    tag metadata and the changelog file as committed on the current branch."""
    dates = _git_tag_dates()
    parsed = _parse_version_md()
    docs_dir = PROJECT_ROOT / "version"
    tags = []
    for tag in sorted(dates, key=_tag_sort_key, reverse=True):
        info = parsed.get(tag, {})
        doc_file = info.get("doc_file") or f"{tag}.md"
        has_doc = _tag_doc_path(tag) is not None
        tags.append({
            "tag":       tag,
            "date":      info.get("date") or dates.get(tag),
            "summary":   info.get("summary") or "",
            "has_doc":   has_doc,
            "doc_file":  doc_file if has_doc else None,
        })
    return tags


@app.get("/api/tags/{tag}/doc")
def get_tag_doc(tag: str, lang: str = "zh"):
    path = _tag_doc_path(tag)
    if path is None:
        raise HTTPException(404)
    if lang == "en":                                   # English copy next to the doc: <name>.en.md
        en = path.with_name(path.stem + ".en.md")
        if en.is_file():
            path = en
    return {"tag": tag, "doc_file": str(path.relative_to(PROJECT_ROOT)), "content": path.read_text(encoding="utf-8")}


@app.get("/api/history/{scenario}/{algo}")
def get_history(scenario: str, algo: str, branch: str | None = None):
    algo_dir = _results_root(branch) / scenario / algo
    if not algo_dir.is_dir():
        raise HTTPException(404)
    rows = [_row_from_run(scenario, algo, d) for d in _run_dirs(algo_dir)]
    return [r for r in rows if r]


@app.get("/api/results/{scenario}/{algo}/{run}/{filename}")
def get_result_file(scenario: str, algo: str, run: str, filename: str, branch: str | None = None):
    if filename not in ("training_curve.png", "comparison.png"):
        raise HTTPException(404)
    path = _results_root(branch) / scenario / algo / run / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


MONITOR_STATE_PATH = APP_DIR / "monitor_state.json"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _read_monitor_state() -> dict:
    """Written by app/backend/monitor.py (a separate long-running watchdog
    process, typically under systemd — see my-code-monitor.service) every
    ~30s. Missing file just means the watchdog isn't running yet; callers
    treat that the same as "no stuck/idle info available" rather than
    erroring, so /api/progress keeps working even before the watchdog is
    installed."""
    try:
        return json.loads(MONITOR_STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


LOG_SOURCES_PATH = APP_DIR / "log_sources.json"
PHASE_RE = re.compile(r"===\s+\[(\w+)\]\s+(starting|finished)\s+(\w+)\s+at\s+(.+?)\s+===")


def _read_live_progress(scenario: str, algo: str) -> dict | None:
    """Read results/<branch>/<scenario>/<algo>/.progress.json, written
    directly by the training callback (shared.paths.write_progress) on every
    progress tick. Not scraped from stdout: stdout is block-buffered whenever
    it's piped to a file instead of a TTY, so log-tailing for live progress
    lagged for minutes at a time — this file is overwritten fresh on every
    tick instead.

    KNOWN LIMITATION: run_all.py and run_all_bc.py share the same C.OUTDIR
    (and thus the same .progress.json path) for a given branch/scenario/algo,
    since the BC variant only diverges at the run-directory level (exp_id +
    "_bc"), not the algo dir. If baseline and BC training run concurrently
    for the same algo on the same branch checkout, whichever writes last
    "wins" the progress file — this display can transiently show the wrong
    run's step count. Final results.json are unaffected (each run gets its
    own directory). Running them on different branches no longer collides,
    since results/<branch>/... physically separates them."""
    algo = algo.removesuffix("+bc")
    path = _results_root() / scenario / algo / ".progress.json"
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

# Order the sequential shell loop launches algorithms in, per scenario — used to
# turn "which algo is currently running" into "N/6 models done" overall progress.
ALGO_ORDER = ["ppo", "ppo_mask", "ppo_lagrangian", "ppo_opt", "dqn", "ddqn"]


def _load_log_sources() -> dict:
    """scenario -> log file path. Best-effort: this file just points at wherever
    each scenario's sequential-run stdout is being captured; update it any time
    the launch method changes. Missing/stale entries just mean no live progress."""
    try:
        return json.loads(LOG_SOURCES_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _proc_stdout_log_path(pid: int) -> str | None:
    """Resolve the regular file a running process's stdout (fd 1) is
    redirected to, straight from /proc — independent of log_sources.json.
    Returns None if fd 1 isn't a regular file (TTY, pipe, closed, or the
    process/permission lookup fails), so callers can fall back to the
    static log_sources.json pointer for historical/no-longer-running runs."""
    try:
        target = os.readlink(f"/proc/{pid}/fd/1")
    except OSError:
        return None
    if not target.startswith("/") or not Path(target).is_file():
        return None
    return target


def _tail_log_progress(path_str: str) -> dict:
    path = Path(path_str)
    if not path.is_file():
        return {}
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return {}

    completed_algos = []
    for line in lines:
        m = PHASE_RE.search(line)
        if not m:
            continue
        _, event, algo, _ = m.groups()
        if event == "finished" and algo not in completed_algos:
            completed_algos.append(algo)

    return {
        "completed_algos":   completed_algos,
        "log_tail":          lines[-12:],
    }


def _ps_all() -> list[dict]:
    """Read-only process table lookup — never touches the processes themselves.
    Unfiltered; callers narrow down for their own purpose."""
    out = subprocess.run(
        ["ps", "-eo", "pid,psr,etimes,pcpu,cmd", "--no-headers"],
        capture_output=True, text=True, timeout=5,
    ).stdout
    procs = []
    for line in out.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid, psr, etimes, pcpu, cmd = parts
        procs.append({"pid": int(pid), "psr": int(psr), "etimes": int(etimes),
                       "pcpu": float(pcpu), "cmd": cmd})
    return procs


def _ps_snapshot() -> list[dict]:
    """Narrow the full process table down to the canonical training-pipeline
    shape (scenarios/<scenario>/<algo>/run_all.py or its self-imitation
    sibling) that /api/progress's one-process-per-scenario model expects."""
    return [
        p for p in _ps_all()
        if re.search(r"(?:^|[\s/])\w+/run_all(_bc)?\.py", p["cmd"]) or "self_imitation_finetune_v2.py" in p["cmd"]
        or "paper_rl.train" in p["cmd"]
    ]


PROJECT_VENV_PYTHON_MARKER = str(PROJECT_ROOT / ".venv" / "bin")


def _proc_cwd(pid: int) -> str | None:
    try:
        return str(Path(f"/proc/{pid}/cwd").resolve())
    except OSError:
        return None


# Human-readable English titles for the project's own long-lived infra
# scripts/modules, keyed by the "-m <module>" or "<script>.py" name
# _project_related_procs() extracts. Anything not listed here (any ad-hoc
# exploratory script) falls back to a title-cased version of its filename —
# see _friendly_label() — so this dict only needs entries worth a nicer
# name than that fallback would produce on its own.
KNOWN_SCRIPT_LABELS = {
    "app.backend.monitor": "Backend Monitor",
    "uvicorn": "Dashboard Server",
    "run_full_5M_campaign.py": "Full 5M-Step Campaign",
}


def _friendly_label(scenario: str | None, algo: str | None, script_name: str | None, script_args: list[str]) -> str:
    """One human-readable English label per process, for the frontend's
    "Running now" list — never the raw cmd line, which is unreadable at a
    glance (venv path, flags, etc.) and was the complaint that prompted
    this. Priority: recognized training pipeline > a name from
    KNOWN_SCRIPT_LABELS > a title-cased guess from the script's own
    filename, with its leading args appended for context (e.g. which algo
    a generic sweep script was launched with)."""
    if scenario and algo:
        return f"Training: {scenario}/{algo}"
    if not script_name:
        return "Unrecognized process"
    if script_name in KNOWN_SCRIPT_LABELS:
        return KNOWN_SCRIPT_LABELS[script_name]  # args are internal (e.g. uvicorn's own flags), not worth surfacing
    stem = re.sub(r"\.py$", "", script_name)
    base = stem.replace("_", " ").replace("-", " ").title()
    if script_args:
        return f"{base} ({' '.join(script_args)})"
    return base


def _project_related_procs() -> list[dict]:
    """Broader than _ps_snapshot(): every live process tied to this project,
    not just the canonical run_all.py training pipeline. Two independent
    signals, either one qualifies:
      - the process was launched with THIS project's venv interpreter
        (.venv/bin/python...) — catches ad-hoc/one-off scripts too, even
        ones living outside the repo (e.g. a scratch script under /tmp that
        imports scenarios/shared via sys.path), since what matters is which
        Python env it's running under, not where the .py file sits;
      - its cwd resolves under PROJECT_ROOT — catches a system-python
        invocation run from within the repo.
    Best-effort process/log introspection only; this never touches the
    processes it finds (see module docstring)."""
    out = []
    for p in _ps_all():
        if p["cmd"].startswith("ps ") or " ps -eo" in p["cmd"]:
            continue  # don't report this endpoint's own `ps` subprocess
        if not re.search(r"(?:^|/)python[\d.]*(?:\s|$)", p["cmd"]):
            continue  # cwd can match for a plain shell/editor helper too; require an actual python invocation
        if "multiprocessing" in p["cmd"] or "resource_tracker" in p["cmd"]:
            continue  # uvicorn --reload's own internal worker/tracker helpers, not a project script
        cwd = None
        related = PROJECT_VENV_PYTHON_MARKER in p["cmd"]
        if not related:
            cwd = _proc_cwd(p["pid"])
            related = bool(cwd and cwd.startswith(str(PROJECT_ROOT)))
        if not related:
            continue
        scenario, algo = _match_scenario_algo(p["cmd"], p["pid"])
        # cmd's last token(s) after the interpreter path make a readable
        # label for ad-hoc scripts that don't match the run_all.py shape
        # (_match_scenario_algo returns (None, None) for those) — covers
        # both `python foo.py args...` and `python -m pkg.module args...`.
        tokens = p["cmd"].split()
        script_name, script_args = None, []
        for i, tok in enumerate(tokens):
            if tok.endswith(".py"):
                script_name, script_args = Path(tok).name, tokens[i + 1:i + 3]
                break
            if tok == "-m" and i + 1 < len(tokens):
                script_name, script_args = tokens[i + 1], tokens[i + 2:i + 4]
                break
        script_label = " ".join([script_name] + script_args) if script_name else None
        out.append({
            "pid": p["pid"], "core": p["psr"], "elapsed_seconds": p["etimes"], "cpu_percent": p["pcpu"],
            "cmd": p["cmd"], "cwd": cwd if cwd is not None else _proc_cwd(p["pid"]),
            "scenario": scenario, "algo": algo, "script": script_label,
            "label": _friendly_label(scenario, algo, script_name, script_args),
        })
    out.sort(key=lambda r: -r["elapsed_seconds"])
    return out


def _match_scenario_algo(cmd: str, pid: int | None = None):
    """Returns (scenario, algo) where algo gets a "+bc" suffix when the
    process is running run_all_bc.py (ILP behavior-cloning pretrain variant),
    so it doesn't get conflated with the plain run_all.py baseline in the
    live-progress display.

    Two invocation shapes are supported:
      - the standard scripts/start_experiment.sh launcher, which always uses
        the absolute .../scenarios/<scenario>/<algo>/run_all.py form;
      - an ad-hoc `cd scenarios/<scenario> && python3 <algo>/run_all[_bc].py`
        invocation (e.g. a manually backgrounded comparison run), whose cmd
        only contains "<algo>/run_all.py" — the scenario is recovered from
        /proc/<pid>/cwd instead.
    """
    # unified trainer (v4.3.1.3+): python -m paper_rl.train --scen <s> --algo <a>
    if "paper_rl.train" in cmd:
        ms, ma = re.search(r"--scen\s+(\w+)", cmd), re.search(r"--algo\s+(\w+)", cmd)
        return (ms.group(1) if ms else None, ma.group(1) if ma else None)

    m = re.search(r"scenarios/(\w+)/(\w+)/run_all(_bc)?\.py", cmd)
    if m:
        scenario, algo, is_bc = m.group(1), m.group(2), m.group(3)
        return (scenario, f"{algo}+bc" if is_bc else algo)

    # scripts/self_imitation_finetune_v2.py (see version/v2.1.0.md) is a
    # lt/ppo_mask-only side experiment, not part of the standard run_all(_bc)
    # pipeline — hardcoded scenario/algo since the script itself is
    # hardcoded to lt/ppo_mask (imports scenarios/lt/ppo_mask/config.py).
    # Writes its own results/<branch>/lt/ppo_mask_selfimit/.progress.json,
    # distinct from lt/ppo_mask's, so it can't collide with the real
    # training run's progress display.
    if re.search(r"self_imitation_finetune_v2\.py", cmd):
        return ("lt", "ppo_mask_selfimit")

    m = re.search(r"(?:^|[\s/])(\w+)/run_all(_bc)?\.py", cmd)
    if not m or pid is None:
        return (None, None)
    algo, is_bc = m.group(1), m.group(2)
    try:
        cwd = Path(f"/proc/{pid}/cwd").resolve()
    except OSError:
        return (None, None)
    scenario = cwd.name
    if scenario not in SCENARIOS:
        return (None, None)
    return (scenario, f"{algo}+bc" if is_bc else algo)


@app.get("/api/progress")
def get_progress():
    procs = _ps_snapshot()
    log_sources = _load_log_sources()
    monitor_state = _read_monitor_state()
    result = {}
    for scenario in SCENARIOS:
        proc = next((p for p in procs if _match_scenario_algo(p["cmd"], p["pid"])[0] == scenario), None)
        entry = {"scenario": scenario, "running": proc is not None}

        # Prefer the actually-running process's own stdout target (works no
        # matter how it was launched) over the log_sources.json pointer,
        # which only gets updated by resume_scenario.sh/start_experiment.sh
        # and otherwise goes stale — showing a previous run's log forever.
        live_log_path = _proc_stdout_log_path(proc["pid"]) if proc else None
        log_info = _tail_log_progress(live_log_path or log_sources.get(scenario, ""))
        completed_algos = log_info.pop("completed_algos", [])
        entry["completed_algos"] = completed_algos
        entry["log_tail"] = log_info.get("log_tail", [])

        current_algo = None
        latest_progress = None
        if proc:
            _, current_algo = _match_scenario_algo(proc["cmd"], proc["pid"])
            entry.update({
                "current_algo":    current_algo,
                "pid":             proc["pid"],
                "core":            proc["psr"],
                "cpu_percent":     proc["pcpu"],
                "elapsed_seconds": proc["etimes"],
            })
            latest_progress = _read_live_progress(scenario, current_algo)
        entry["latest_progress"] = latest_progress

        total = len(ALGO_ORDER)
        done = len(completed_algos)
        within_current = (latest_progress["pct"] / 100.0) if (latest_progress and current_algo not in completed_algos) else 0.0
        entry["models_done"]  = done
        entry["models_total"] = total
        entry["overall_pct"]  = round(min(100.0, (done + within_current) / total * 100), 1)

        # "running": true just means a matching process exists — it says
        # nothing about whether it's actually making progress. monitor.py
        # tracks step-count movement across polls (this endpoint is
        # stateless per-request, so it can't) and flags stalls itself.
        mon = monitor_state.get(scenario)
        entry["monitor_status"]  = (mon or {}).get("status", "unknown")
        entry["stalled_seconds"] = (mon or {}).get("stalled_seconds")

        result[scenario] = entry
    return result


def _training_proc_info(pid: int) -> dict:
    """exp_id / seed / reward mode (from the launcher-provided environment),
    owning batch (log directory) and training progress (log tail)."""
    info = {"exp_id": None, "seed": None, "variant": None, "batch": None, "progress_pct": None}
    try:
        env = dict(kv.split(b"=", 1) for kv in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in kv)
        info["exp_id"] = env.get(b"EXP_ID", b"").decode() or None
        info["seed"] = env.get(b"TRAIN_SEED", b"").decode() or None
        info["variant"] = env.get(b"REWARD_MODE", b"").decode() or None
    except OSError:
        pass
    log_path = _proc_stdout_log_path(pid)
    if log_path:
        lp = Path(log_path)
        if lp.parent.parent == LOGS_ROOT:
            info["batch"] = lp.parent.name
        try:
            with open(lp, "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 8192))
                hits = _TRAIN_PCT_RE.findall(fh.read().decode(errors="ignore"))
            info["progress_pct"] = float(hits[-1]) if hits else 0.0
        except OSError:
            pass
    return info


@app.get("/api/system")
def get_system():
    """Live, generic "what's actually running right now" view — independent
    of /api/progress's one-process-per-scenario assumption and /api/batches'
    scripts/logs/ naming convention, so it also picks up ad-hoc/exploratory
    runs (a hyperparameter sweep, a one-off timing probe, anything launched
    by hand) that neither of those endpoints know how to parse. A process
    counts as "related" if it runs under this project's venv interpreter or
    its cwd is inside the repo — see _project_related_procs()."""
    procs = _project_related_procs()
    for p in procs:
        if p.get("scenario") and p.get("algo"):
            p.update(_training_proc_info(p["pid"]))
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = None
    return {
        "cpu_count": os.cpu_count(),
        "load_avg": {"1m": load1, "5m": load5, "15m": load15},
        "processes": procs,
    }


def _live_exp_ids(procs: list[dict]) -> set[str]:
    """exp_id of every live training process. Two sources, either is enough:
    the EXP_ID environment variable every campaign launcher passes to
    run_all.py, and the trailing _<exp_id>.log of the process's stdout log
    (older launchers). Matching on exp_id rather than (scenario, algo) keeps
    a stale entry of a killed batch from borrowing liveness from an
    unrelated later campaign's process for the same scenario/algo."""
    exp_ids = set()
    for p in procs:
        try:
            env = Path(f"/proc/{p['pid']}/environ").read_bytes().split(b"\0")
            for kv in env:
                if kv.startswith(b"EXP_ID="):
                    exp_ids.add(kv[7:].decode(errors="ignore"))
        except OSError:
            pass
        log_path = _proc_stdout_log_path(p["pid"])
        m = LOG_EXP_ID_RE.search(Path(log_path).name) if log_path else None
        if m:
            exp_ids.add(m["exp_id"])
    return exp_ids


# Old-style batch log name: {scenario}_{algo}_{steps}_seed{seed}_{exp_id}.log
BATCH_LOG_DIR_RE = re.compile(
    r"^(?P<scenario>eq|gt|lt)_(?P<algo>\w+)_(?P<steps>\d+)_seed(?P<seed>\d+)_(?P<exp_id>[0-9a-f]+)\.log$"
)
LOG_EXP_ID_RE = re.compile(r"_(?P<exp_id>[0-9a-f]{8})\.log$")
LOGS_ROOT = PROJECT_ROOT / "scripts" / "logs"


def _manifest_runs(manifest: dict) -> list[dict]:
    """Flatten a campaign manifest's nested exp_ids dict into runs. Launchers
    nest it differently ({scen: {algo: {seed: id}}} for seed campaigns,
    {mode: {scen: {algo: id}}} for the reward pilot), so each key on the path
    is classified instead of assuming one order: lt/eq/gt is the scenario,
    a member of manifest["algos"] is the algo, a bare integer is the seed,
    and anything else is a variant label (e.g. a reward mode)."""
    algos = set(manifest.get("algos") or [])
    runs = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + [str(k)])
            return
        if not isinstance(node, str):
            return
        run = {"scenario": None, "algo": None, "seed": None, "variant": "", "exp_id": node}
        variant = []
        for k in path:
            if k in SCENARIOS and run["scenario"] is None:
                run["scenario"] = k
            elif k in algos and run["algo"] is None:
                run["algo"] = k
            elif k.isdigit() and run["seed"] is None:
                run["seed"] = int(k)
            else:
                variant.append(k)
        if run["scenario"] and run["algo"]:
            if run["seed"] is None and manifest.get("seed") is not None:
                run["seed"] = int(manifest["seed"])
            run["variant"] = "/".join(variant)
            runs.append(run)

    walk(manifest.get("exp_ids") or {}, [])

    # Planned but not yet launched: the manifest only lists launched runs, so
    # fill in the rest of scenarios x algos x modes x seeds as queued
    # (exp_id None) -- otherwise queued jobs are invisible until they start.
    scens = manifest.get("scenarios") or []
    seeds = manifest.get("seeds") or ([manifest["seed"]] if manifest.get("seed") is not None else [None])
    modes = manifest.get("modes") or [""]
    if scens and algos:
        have = {(r["scenario"], r["algo"], r["variant"], r["seed"]) for r in runs}
        for mode, scen, algo, seed in itertools.product(modes, scens, manifest["algos"], seeds):
            seed = int(seed) if seed is not None else None
            if (scen, algo, mode, seed) not in have:
                runs.append({"scenario": scen, "algo": algo, "seed": seed, "variant": mode, "exp_id": None})
    return runs


def _batch_runs(log_dir: Path) -> tuple[list[dict], dict]:
    """(runs, manifest) of one batch. manifest.json wins when present;
    otherwise runs come from old-style log file names."""
    manifest_path = log_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("kind") == "eval":       # no training runs / exp_ids, see _eval_batch_state
                return [], manifest
            return _manifest_runs(manifest), manifest
        except (json.JSONDecodeError, OSError):
            pass
    runs = []
    for f in sorted(log_dir.glob("*.log")):
        m = BATCH_LOG_DIR_RE.match(f.name)
        if m:
            runs.append({"scenario": m["scenario"], "algo": m["algo"], "seed": int(m["seed"]),
                         "variant": "", "exp_id": m["exp_id"]})
    return runs, {}


def _branch_holding(runs: list[dict]) -> str | None:
    """Branch whose results/ tree holds this batch's runs (old batches have
    no manifest saying which branch they ran on). None = current branch."""
    if not RESULTS_ROOT_BASE.is_dir():
        return None
    for r in runs:
        for bdir in RESULTS_ROOT_BASE.iterdir():
            if r["exp_id"] and (bdir / r["scenario"] / r["algo"] / r["exp_id"]).is_dir():
                return bdir.name
    return None


_TRAIN_PCT_RE = re.compile(r"\[train\] step=[\d,]+/[\d,]+ \(\s*([\d.]+)%\)")


def _log_train_pct(log_dir: Path, exp_id: str) -> float | None:
    """Latest "[train] ... (P%)" in this run's own log -- per run, unlike
    .progress.json, which concurrent runs of one algo overwrite in turn."""
    for f in log_dir.glob(f"*{exp_id}*.log"):
        try:
            with open(f, "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 8192))
                tail = fh.read().decode(errors="ignore")
        except OSError:
            continue
        hits = _TRAIN_PCT_RE.findall(tail)
        if hits:
            return float(hits[-1])
    return None


def _birth_time(paths: list[Path]) -> float | None:
    """Earliest creation time of these files. ctime is no good for a batch's start:
    it moves on every write to a log, so it tracked the oldest *last write*."""
    paths = [str(p) for p in paths if p.is_file()]
    if not paths:
        return None
    try:
        out = subprocess.run(["stat", "-c", "%W", *paths], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    times = [int(t) for t in out.split() if t.lstrip("-").isdigit() and int(t) > 0]
    return float(min(times)) if times else None


def _elapsed(st: dict) -> float:
    end = time.time() if st["status"] == "running" else st["last_updated"]
    return round(max(end - st["started_at"], 0.0), 1)


# ── evaluation batches ──────────────────────────────────────────────────────
# A batch whose manifest.json has "kind": "eval" re-evaluates saved models (no
# training, no exp_id of its own). Its manifest lists every job up front:
#   {"kind": "eval", "version", "script", "log", "workers", "started_at",
#    "jobs": [{"scenario", "algo", "seed"}],
#    "raw": "<per-run jsonl, written when the script ends>",
#    "raw_format": "best_of_k" | "single", "ks": [...], "report", "description"}
# Metrics per K: success rate; AR and ILP AR averaged over the same successful
# instances; AR gap = ILP AR - AR.
# A job is done once its run is in the raw file or the log has
# "done <scen> <algo> seed<N> (<ms> ms/episode)". While the script is alive,
# the first `workers` unfinished jobs are running (Pool.map hands jobs out in
# order, one at a time), the rest are queued.
EVAL_DONE_RE = re.compile(r"done (?P<scen>lt|eq|gt) (?P<algo>\S+) seed(?P<seed>\d+) \((?P<ms>[\d.]+) ms/episode\)")
_eval_raw_cache: dict[str, tuple[float, dict]] = {}


def _script_alive(script: str) -> bool:
    for pd in Path("/proc").iterdir():
        if not pd.name.isdigit():
            continue
        try:
            if script.encode() in (pd / "cmdline").read_bytes():
                return True
        except OSError:
            continue
    return False


def _eval_raw_summary(manifest: dict) -> dict:
    """(scen, algo, seed) -> {"ms": ms/episode, "k": {K: [success rate, AR, ILP AR]}},
    from the raw jsonl; cached on its mtime (it can be tens of MB)."""
    path = PROJECT_ROOT / manifest.get("raw", "")
    if not manifest.get("raw") or not path.is_file():
        return {}
    mtime = path.stat().st_mtime
    hit = _eval_raw_cache.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    ks = manifest.get("ks") or [1]
    out = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:          # a run being appended right now
                continue
            key = (r["scen"], r["algo"], int(r["seed"]))
            rows = r["rows"]
            per_k = {}
            for k in ks:
                n_ok, ar_sum, ilp_sum = 0, 0.0, 0.0
                for x in rows:
                    if manifest.get("raw_format") == "best_of_k":
                        ok, ar = max(x["samples"][:k], key=lambda s: (s[0], s[1]))
                    else:
                        ok, ar = x["success"], x["ar"]
                    if ok:
                        n_ok, ar_sum, ilp_sum = n_ok + 1, ar_sum + ar, ilp_sum + x["ar_star"]
                # [success rate, AR, ILP AR] -- AR and ILP AR over the same successful instances
                per_k[str(k)] = [n_ok / len(rows), ar_sum / n_ok if n_ok else None, ilp_sum / n_ok if n_ok else None]
            ms = r.get("ms_episode")
            if ms is None and rows and "ms" in rows[0]:
                ms = sum(x["ms"] for x in rows) / len(rows)
            out[key] = {"ms": ms, "k": per_k}
    _eval_raw_cache[str(path)] = (mtime, out)
    return out


def _eval_batch_state(batch_name: str, manifest: dict) -> dict:
    log_dir = LOGS_ROOT / batch_name
    log = PROJECT_ROOT / manifest["log"] if manifest.get("log") else None
    done_ms = {}
    if log and log.is_file():
        for m in EVAL_DONE_RE.finditer(log.read_text(errors="ignore")):
            done_ms[(m["scen"], m["algo"], int(m["seed"]))] = float(m["ms"])
    summary = _eval_raw_summary(manifest)
    alive = bool(manifest.get("script")) and _script_alive(manifest["script"])
    free = int(manifest.get("workers") or 1) if alive else 0
    runs = []
    for j in manifest.get("jobs", []):
        key = (j["scenario"], j["algo"], int(j["seed"]))
        if key in summary or key in done_ms:
            status = "done"
        elif free:
            status, free = "running", free - 1
        else:
            status = "queued" if alive else "skipped"
        runs.append({"scenario": j["scenario"], "algo": j["algo"], "seed": int(j["seed"]), "variant": "",
                     "exp_id": None, "status": status,
                     "ms": (summary.get(key) or {}).get("ms") or done_ms.get(key),
                     "metrics": (summary.get(key) or {}).get("k")})
    counts = {s: sum(1 for r in runs if r["status"] == s) for s in ("done", "running", "queued", "stopped", "skipped")}
    total = len(runs)
    if (log_dir / ".cancelled").is_file():
        status = "cancelled"
    elif alive:
        status = "running"
    elif total and counts["done"] == total:
        status = "finished"
    else:
        status = "stopped"
    files = [p for p in (log, PROJECT_ROOT / manifest.get("raw", "_")) if p and p.is_file()]
    started = manifest.get("started_at") or _birth_time(files) or log_dir.stat().st_ctime
    return {
        "batch_name": batch_name, "kind": "eval", "status": status,
        "version": manifest.get("version"), "branch": _active_space(), "steps": None,
        "description": manifest.get("description"), "report": manifest.get("report"),
        "ks": manifest.get("ks") or [1], "workers": manifest.get("workers"),
        "total_runs": total, **counts,
        "overall_pct": round(100.0 * counts["done"] / total, 1) if total else 0.0,
        "started_at": started,
        "last_updated": max([p.stat().st_mtime for p in files] or [log_dir.stat().st_mtime]),
        "runs": runs, "results_root": None, "ilp": manifest.get("ilp"),
    }


def _eval_batch_progress(st: dict) -> dict:
    import statistics
    groups: dict[tuple, list[dict]] = {}
    for r in st["runs"]:
        groups.setdefault((r["scenario"], r["algo"]), []).append(r)

    def agg(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return [None, None]
        return [statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0]

    rows = []
    for (scenario, algo), rs in groups.items():
        done = [r for r in rs if r["status"] == "done"]
        metrics = {}
        for k in map(str, st["ks"]):
            got = [r["metrics"][k] for r in done if r["metrics"] and k in r["metrics"]]
            metrics[k] = {"success": agg([g[0] for g in got]), "ar": agg([g[1] for g in got]),
                          "ilp_ar": agg([g[2] for g in got]),
                          "gap": agg([g[2] - g[1] for g in got if g[1] is not None]), "n": len(got)}
        rows.append({"scenario": scenario, "algo": algo, "variant": "", "n_done": len(done), "n_total": len(rs),
                     "ms": agg([r["ms"] for r in done]), "metrics": metrics,
                     "seeds": sorted(({"seed": r["seed"], "status": r["status"], "exp_id": None,
                                       "progress_pct": None} for r in rs), key=lambda x: x["seed"])})
    st.pop("runs")
    st.pop("results_root")
    st["rows"] = rows
    st["elapsed_seconds"] = _elapsed(st)
    return st


def _batch_state(batch_name: str, live_exp_ids: set[str]) -> dict | None:
    log_dir = LOGS_ROOT / batch_name
    runs, manifest = _batch_runs(log_dir)
    if manifest.get("kind") == "eval":
        return _eval_batch_state(batch_name, manifest)
    if not runs:
        return None
    branch = manifest.get("branch") or _branch_holding(runs)
    results_root = _results_root(branch)
    for r in runs:
        if r["exp_id"] is None:
            r["status"] = "queued"
            continue
        run_dir = results_root / r["scenario"] / r["algo"] / r["exp_id"]
        if (run_dir / "results.json").is_file():
            r["status"] = "done"
        elif r["exp_id"] in live_exp_ids:
            r["status"] = "running"
        elif (run_dir / ".progress.json").is_file() or run_dir.is_dir():
            r["status"] = "stopped"   # started, never finished, not alive
        else:
            r["status"] = "queued"

    # A slot can hold several attempts after a crash/retry: keep the best one.
    rank = {"done": 0, "running": 1, "queued": 2, "stopped": 3}
    best: dict[tuple, dict] = {}
    for r in runs:
        slot = (r["scenario"], r["algo"], r["variant"], r["seed"])
        if slot not in best or rank[r["status"]] < rank[best[slot]["status"]]:
            best[slot] = r
    runs = list(best.values())

    counts = {s: sum(1 for r in runs if r["status"] == s) for s in rank}
    # A manifest only lists runs launched so far; the planned total is the
    # product of its scenarios / algos / seeds / modes lists.
    total = 1
    for k in ("scenarios", "algos", "seeds", "modes"):
        if isinstance(manifest.get(k), list) and manifest[k]:
            total *= len(manifest[k])
    total = max(total if manifest else 0, len(runs))
    if (log_dir / ".cancelled").is_file():
        status = "cancelled"
    elif counts["running"]:
        status = "running"
    elif counts["done"] == len(runs):
        status = "finished"
    else:
        status = "stopped"
    if status in ("stopped", "cancelled", "finished"):
        # nothing is going to launch them any more
        for r in runs:
            if r["status"] == "queued":
                r["status"] = "skipped"
        counts["skipped"] = counts.pop("queued")
        counts["queued"] = 0
    else:
        counts["skipped"] = 0
    log_files = list(log_dir.glob("*.log"))
    mtimes = [f.stat().st_mtime for f in log_files] or [log_dir.stat().st_mtime]
    ctimes = [f.stat().st_ctime for f in log_files] or [log_dir.stat().st_ctime]
    return {
        "batch_name": batch_name,
        "status": status,
        "version": manifest.get("version"),
        "branch": branch or _active_space(),
        "steps": manifest.get("steps"),
        "total_runs": total,
        **counts,
        "queued": counts["queued"] + (total - len(runs)),
        "overall_pct": round(100.0 * counts["done"] / total, 1),
        "started_at": _birth_time(log_files + [log_dir / "manifest.json"]) or min(ctimes),
        "last_updated": max(mtimes),
        "runs": runs,
        "results_root": results_root,
    }


@app.get("/api/batches")
def list_batches(all: bool = False, branch: str | None = None):
    """Batches under scripts/logs/ (one subdirectory each, described by a
    manifest.json or by old-style run log names). Default: only batches with
    a live run (what "currently training" needs). all=true: every batch with
    its status (running / finished / stopped / cancelled), newest first."""
    if not LOGS_ROOT.is_dir():
        return {"batches": []}
    live = _live_exp_ids(_ps_snapshot())
    batches = []
    for d in LOGS_ROOT.iterdir():
        if not d.is_dir():
            continue
        st = _batch_state(d.name, live)
        if st is None or (not all and st["status"] != "running"):
            continue
        if branch and st["branch"] != branch:
            continue
        st.pop("runs")
        st.pop("results_root")
        batches.append(st)
    batches.sort(key=lambda b: b["last_updated"], reverse=True)
    return {"batches": batches}


@app.get("/api/batch_progress/{batch_name}")
def get_batch_progress(batch_name: str):
    """Per-run status and metrics of one batch, plus mean/std per
    (scenario, algo, variant) over the finished seeds."""
    import statistics
    if not (LOGS_ROOT / batch_name).is_dir():
        raise HTTPException(status_code=404, detail=f"no such batch: {batch_name}")
    st = _batch_state(batch_name, _live_exp_ids(_ps_snapshot()))
    if st is None:
        raise HTTPException(status_code=404, detail=f"batch {batch_name} has no runs")
    if st.get("kind") == "eval":
        return _eval_batch_progress(st)
    results_root = st.pop("results_root")

    for r in st["runs"]:
        r["progress_pct"] = None
        run_dir = results_root / r["scenario"] / r["algo"] / (r["exp_id"] or "_")
        if r["status"] == "done":
            row = _row_from_run(r["scenario"], r["algo"], run_dir)
            if row:
                for k in ("test_success_rate", "test_ar_mean", "ilp_ar",
                          "test_cap_viol_rate", "test_conflict_viol_rate"):
                    r[k] = row[k]
        elif r["status"] == "running":
            r["progress_pct"] = _log_train_pct(LOGS_ROOT / batch_name, r["exp_id"])

    def agg(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        return round(statistics.fmean(vals), 6), (round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0)

    groups: dict[tuple, list[dict]] = {}
    for r in st["runs"]:
        groups.setdefault((r["scenario"], r["algo"], r["variant"]), []).append(r)
    rows = []
    for (scenario, algo, variant), rs in groups.items():
        done = [r for r in rs if r["status"] == "done"]
        row = {"scenario": scenario, "algo": algo, "variant": variant,
               "n_done": len(done), "n_total": len(rs),
               "seeds": sorted(({"seed": r["seed"], "status": r["status"], "exp_id": r["exp_id"],
                                 "progress_pct": r["progress_pct"]} for r in rs),
                               key=lambda x: (x["seed"] is None, x["seed"] or 0))}
        for k in ("test_success_rate", "test_ar_mean", "ilp_ar", "test_cap_viol_rate", "test_conflict_viol_rate"):
            row[k + "_mean"], row[k + "_std"] = agg([r.get(k) for r in done])
        rows.append(row)
    scen_order = {"lt": 0, "eq": 1, "gt": 2}
    rows.sort(key=lambda x: (scen_order.get(x["scenario"], 9), x["algo"], x["variant"]))
    st["rows"] = rows
    st["elapsed_seconds"] = _elapsed(st)
    return st


@app.get("/", response_class=HTMLResponse)
def dashboard():
    html = (APP_DIR / "static" / "index.html").read_text()
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
