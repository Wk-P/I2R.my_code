"""Live state for the v2 panel, independent of the old backend: training processes (from /proc, identified by the
EXP_ID / RUN_* environment the v2 drivers pass, docs/data_spec.md §2), GPUs (nvidia-smi), the machine, the experiment
queue (logs/queue.json) and a driver's report stage (logs/run_<batch>_driver.log). Read-only."""
from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from .data import LOGS, ROOT

VENV = str(ROOT / ".venv")


# ── processes ─────────────────────────────────────────────────────────────────
def _environ(pid: int) -> dict:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return {}
    return {k.decode(errors="ignore"): v.decode(errors="ignore") for k, _, v in (x.partition(b"=") for x in raw) if k}


def _arg(tokens: list[str], flag: str) -> str | None:
    return tokens[tokens.index(flag) + 1] if flag in tokens and tokens.index(flag) + 1 < len(tokens) else None


_ps_cache: dict = {"t": 0.0, "v": []}
_ps_lock = threading.Lock()


def processes() -> list[dict]:
    """Every live python process of this project (its venv interpreter, or its cwd inside the repo); training runs
    carry their batch / key / exp_id from the environment. Cached 3 s."""
    with _ps_lock:
        if time.time() - _ps_cache["t"] < 3:
            return _ps_cache["v"]
        try:
            out = subprocess.run(["ps", "-eo", "pid=,psr=,etimes=,pcpu=,stat=,args="], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        procs = []
        for line in out.splitlines():
            f = line.split(None, 5)
            if len(f) < 6 or not re.search(r"(?:^|/)python[\d.]*(?:\s|$)", f[5]):
                continue
            pid, cmd = int(f[0]), f[5]
            if "multiprocessing" in cmd or "resource_tracker" in cmd:
                continue
            try:
                cwd = os.readlink(f"/proc/{pid}/cwd")
            except OSError:
                cwd = ""
            if VENV not in cmd and not cwd.startswith(str(ROOT)):
                continue
            tok = cmd.split()
            p = {"pid": pid, "core": int(f[1]), "elapsed_seconds": int(f[2]), "cpu_percent": float(f[3]),
                 "frozen": f[4].startswith("T"), "cmd": cmd, "kind": "other"}
            if "paper_rl.train" in cmd:
                env = _environ(pid)
                p.update(kind="train", exp_id=env.get("EXP_ID"), batch=env.get("RUN_BATCH") or None, key=env.get("RUN_KEY") or None,
                         scenario=_arg(tok, "--scen"), algo=_arg(tok, "--algo"), seed=_arg(tok, "--seed"),
                         device=("GPU " + env.get("CUDA_VISIBLE_DEVICES", "?")) if _arg(tok, "--device") == "cuda" else "CPU",
                         mps=bool(env.get("CUDA_MPS_PIPE_DIRECTORY")))
                if p["batch"] and p["exp_id"]:
                    p["live"] = run_live(p["batch"], p["exp_id"])
                p["label"] = " · ".join(x for x in (p["batch"], (p["scenario"] or "").upper(), p["key"] or p["algo"]) if x)
            else:
                at = next((m for m, x in enumerate(tok) if x.endswith(".py") or x == "-m"), None)
                if at is None:
                    p["label"] = cmd[:80]
                else:
                    script = Path(tok[at]).name if tok[at].endswith(".py") else (tok[at + 1] if at + 1 < len(tok) else "")
                    rest = tok[at + 1:] if tok[at].endswith(".py") else tok[at + 2:]
                    p["label"] = " ".join([script, *rest[:2]])
                    if script.startswith("run_"):
                        p["kind"] = "driver"
            procs.append(p)
        procs.sort(key=lambda r: (r["kind"] != "train", -r["elapsed_seconds"]))
        _ps_cache.update(t=time.time(), v=procs)
        return procs


def live_exp_ids() -> set[str]:
    return {p["exp_id"] for p in processes() if p["kind"] == "train" and p.get("exp_id")}


def driver_alive(man: dict | None) -> bool:
    """docs/data_spec.md §3 rule 5: the manifest's driver_pid still runs a driver script."""
    pid = (man or {}).get("driver_pid")
    try:
        return bool(pid) and "src/scripts/run_" in Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (OSError, ValueError):
        return False


def script_alive(man: dict | None) -> bool:
    """A non-training batch (docs/data_spec.md §6): its driver_pid still runs the manifest's script."""
    pid, script = (man or {}).get("driver_pid"), (man or {}).get("script")
    try:
        return bool(pid and script) and script in Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (OSError, ValueError):
        return False


_TRAIN = re.compile(r"\[train\] step=([\d,]+)/([\d,]+) \(\s*([\d.]+)%\)(.*)")
_FIELD = re.compile(r"([\w/]+(?:\(\w+\))?)=([-\d.,]+)")


def _tail(path: Path, n: int = 16384) -> str:
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - n))
        return fh.read().decode(errors="ignore")


def run_live(batch: str, exp_id: str) -> dict | None:
    """Live progress of a training run from its own log (logs/<batch>/*_<exp_id>.log), last "[train]" line:
    step / total / pct, episodes, success(last500), the cumulative steps/s, lambda (Lagrangian); the recent speed
    from the last few lines (each carries the cumulative average a_k = s_k / T_k, so T_k = s_k / a_k) and the ETA."""
    for f in (LOGS / batch).glob(f"*{exp_id}*.log"):
        try:
            tail = _tail(f)
            mtime = f.stat().st_mtime
        except OSError:
            continue
        rows = []
        for m in _TRAIN.finditer(tail):
            kv = {k: float(v.replace(",", "")) for k, v in _FIELD.findall(m.group(4))}
            rows.append((int(m.group(1).replace(",", "")), int(m.group(2).replace(",", "")), float(m.group(3)), kv))
        if not rows:
            return {"step": 0, "total": None, "pct": 0.0, "updated": mtime}
        step, total, pct, kv = rows[-1]
        avg = kv.get("steps/s")
        rate = avg
        pts = [(s, k["steps/s"]) for s, _, _, k in rows if k.get("steps/s")]
        if len(pts) >= 2:
            (s1, a1), (s2, a2) = pts[-min(4, len(pts))], pts[-1]
            dt = s2 / a2 - s1 / a1
            rate = (s2 - s1) / dt if dt > 0 else avg
        return {"step": step, "total": total, "pct": pct, "episodes": kv.get("eps"), "success_last500": kv.get("success(last500)"),
                "lambda": kv.get("lambda"), "steps_per_sec": round(rate) if rate else None, "steps_per_sec_avg": avg,
                "eta_seconds": (total - step) / rate if rate else None, "updated": mtime}
    return None


def log_series(batch: str, exp_id: str) -> dict | None:
    """Every "[train]" line of a run's log (one per logging interval, e.g. 100k steps): step, success(last500),
    lambda, cumulative steps/s -- the training curve while training_curve.csv does not exist yet."""
    for f in (LOGS / batch).glob(f"*{exp_id}*.log"):
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        out = {"step": [], "success_last500": [], "lambda": [], "steps_per_sec": []}
        for m in _TRAIN.finditer(text):
            kv = {k: float(v.replace(",", "")) for k, v in _FIELD.findall(m.group(4))}
            out["step"].append(int(m.group(1).replace(",", "")))
            out["success_last500"].append(kv.get("success(last500)"))
            out["lambda"].append(kv.get("lambda"))
            out["steps_per_sec"].append(kv.get("steps/s"))
        return out
    return None


def train_pct(batch: str, exp_id: str) -> float | None:
    lv = run_live(batch, exp_id)
    return lv["pct"] if lv else None


def report_stage(batch: str) -> dict | None:
    """After training ("=== done" in logs/run_<batch>_driver.log): the last "[report] ..." step, and whether the
    report (report.md) has been written. None before training ends."""
    log = LOGS / f"run_{batch}_driver.log"
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 200_000))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    done = max((m for m, l in enumerate(lines) if l.startswith("=== done")), default=None)
    if done is None:
        return None
    after = lines[done + 1:]
    finished = any(l.startswith("report -> ") and l.rstrip().endswith("/report.md") for l in after)
    step = next((l[len("[report] "):] for l in reversed(after) if l.startswith("[report] ")), None)
    return {"finished": finished, "step": step, "updated": log.stat().st_mtime}


# ── machine ───────────────────────────────────────────────────────────────────
GPU_FIELDS = ["index", "name", "uuid", "utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw",
              "power.limit", "clocks.sm", "clocks.max.sm", "clocks_event_reasons.active"]
# clocks_event_reasons bits (nvml.h) -> (zh, en); 0x1 (idle) is not a slowdown
THROTTLE = {0x2: ("应用时钟设置", "applications clocks setting"), 0x4: ("软件功耗墙", "SW power cap"),
            0x8: ("硬件降频", "HW slowdown"), 0x10: ("同步加速", "sync boost"),
            0x20: ("软件热降频（温度目标）", "SW thermal slowdown"), 0x40: ("硬件热降频", "HW thermal slowdown"),
            0x80: ("硬件电源制动", "HW power brake"), 0x100: ("显示时钟设置", "display clock setting")}


def _throttle(hexbits: str) -> list[dict]:
    try:
        bits = int(hexbits, 16)
    except ValueError:
        return []
    return [{"bit": b, "zh": z, "en": e} for b, (z, e) in THROTTLE.items() if bits & b]


def _smi(args: list[str]) -> str:
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _num(x):
    try:
        return float(x)
    except ValueError:
        return None


def gpus() -> list[dict]:
    out, by_uuid = [], {}
    for line in _smi([f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"]).strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) != len(GPU_FIELDS):
            continue
        r = dict(zip(GPU_FIELDS, f))
        g = {"index": int(r["index"]), "name": r["name"], "util": _num(r["utilization.gpu"]), "mem_used": _num(r["memory.used"]),
             "mem_total": _num(r["memory.total"]), "temp": _num(r["temperature.gpu"]), "power": _num(r["power.draw"]),
             "power_limit": _num(r["power.limit"]), "clock_sm": _num(r["clocks.sm"]), "clock_sm_max": _num(r["clocks.max.sm"]),
             "throttle": _throttle(r["clocks_event_reasons.active"]), "procs": 0, "pids": []}
        out.append(g)
        by_uuid[r["uuid"]] = g
    for line in _smi(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]).strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 2 and f[0] in by_uuid and f[1].isdigit():
            by_uuid[f[0]]["procs"] += 1
            by_uuid[f[0]]["pids"].append(int(f[1]))
    return out


def _meminfo() -> dict:
    try:
        kv = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
        kb = lambda k: int(kv[k].split()[0]) * 1024
        return {"total": kb("MemTotal"), "available": kb("MemAvailable")}
    except (OSError, KeyError, ValueError):
        return {}


def system() -> dict:
    procs = [dict(p) for p in processes()]
    gs = gpus()
    on = {pid: g["index"] for g in gs for pid in g["pids"]}
    for p in procs:
        p["gpu"] = on.get(p["pid"])
    try:
        l1, l5, l15 = os.getloadavg()
    except OSError:
        l1 = l5 = l15 = None
    cpu_model = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                      if l.startswith("model name")), None) if Path("/proc/cpuinfo").exists() else None
    with _hist_lock:                                     # latest CPU sample of the sampler thread (rates need two samples)
        cpu = _hist[-1]["cpu"] if _hist else None
    return {"hostname": os.uname().nodename, "cpu_model": cpu_model, "cpu_count": os.cpu_count(), "cpu": cpu,
            "rapl_domains": [_read(d / "name") for d in _rapl_domains()],
            "load_avg": {"1m": l1, "5m": l5, "15m": l15}, "memory": _meminfo(), "gpus": gs, "processes": procs}


# ── CPU: utilisation (/proc/stat), mean frequency (cpufreq), package temperatures (coretemp), package / DRAM power
# (RAPL energy counters; root-only on recent kernels -- None until made readable, see the panel's note) ─────────────
_RAPL = Path("/sys/class/powercap")
_cpu_prev: dict = {}


def _read(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def _cpu_times() -> tuple[int, int] | None:
    line = (_read(Path("/proc/stat")) or "").splitlines()[0:1]
    if not line:
        return None
    f = [int(x) for x in line[0].split()[1:]]
    idle = f[3] + (f[4] if len(f) > 4 else 0)
    return sum(f), idle


def _rapl_domains() -> list[Path]:
    return sorted(p for p in _RAPL.glob("intel-rapl:*") if (p / "energy_uj").exists())


def cpu_sample() -> dict:
    """One CPU sample; utilisation and power are rates since the previous call."""
    now = time.time()
    out = {"util": None, "freq": None, "freq_max": None, "temps": [], "power": [], "power_readable": None}
    t = _cpu_times()
    if t and "times" in _cpu_prev:
        dt, di = t[0] - _cpu_prev["times"][0], t[1] - _cpu_prev["times"][1]
        out["util"] = round(100 * (1 - di / dt), 1) if dt > 0 else None
    if t:
        _cpu_prev["times"] = t
    freqs = [int(x) for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq") if (x := _read(p))]
    if freqs:
        out["freq"] = round(sum(freqs) / len(freqs) / 1000)
        out["freq_max"] = round(int(_read(Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")) or 0) / 1000) or None
    for h in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        if _read(h / "name") == "coretemp":
            for lab in sorted(h.glob("temp*_label")):
                if (_read(lab) or "").startswith("Package"):
                    v = _read(lab.with_name(lab.name.replace("_label", "_input")))
                    if v:
                        out["temps"].append({"name": _read(lab), "c": int(v) / 1000})
    readable = True
    for d in _rapl_domains():
        e = _read(d / "energy_uj")
        name = _read(d / "name") or d.name
        if e is None:
            readable = False
            continue
        key, e = f"rapl:{d.name}", int(e)
        prev = _cpu_prev.get(key)
        _cpu_prev[key] = (e, now)
        if prev and now > prev[1]:
            de = e - prev[0]
            if de < 0:                                   # counter wrapped
                de += int(_read(d / "max_energy_range_uj") or 0)
            out["power"].append({"name": name if ":" not in d.name[11:] else f"{name} {d.name.split(':')[1]}", "w": round(de / 1e6 / (now - prev[1]), 1)})
    out["power_readable"] = readable and bool(_rapl_domains())
    return out


# GPU history: a sampler thread (started on first use) keeps the last 30 minutes, one sample every 2 s
_hist: collections.deque = collections.deque(maxlen=900)
_hist_lock = threading.Lock()
_hist_started = False


def _sampler():
    while True:
        out = _smi(["--query-gpu=index,utilization.gpu,temperature.gpu,power.draw,memory.used,clocks.sm,clocks.max.sm,"
                    "clocks_event_reasons.active", "--format=csv,noheader,nounits"])
        row = []
        for line in out.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) == 8:
                row.append({"i": int(f[0]), "util": _num(f[1]), "temp": _num(f[2]), "power": _num(f[3]), "mem": _num(f[4]),
                            "clock": _num(f[5]), "max": _num(f[6]), "thr": [t["bit"] for t in _throttle(f[7])]})
        cpu = cpu_sample()
        with _hist_lock:
            _hist.append({"t": time.time(), "gpus": row, "cpu": cpu})
        time.sleep(2)


def gpu_history(since: float) -> dict:
    global _hist_started
    with _hist_lock:
        if not _hist_started:
            _hist_started = True
            threading.Thread(target=_sampler, daemon=True, name="v2-gpu-sampler").start()
        return {"interval": 2, "samples": [s for s in _hist if s["t"] > since]}


# ── experiment queue (src/scripts/queue_runner.py) ────────────────────────────
def queue() -> dict:
    try:
        q = json.loads((LOGS / "queue.json").read_text())
    except (OSError, json.JSONDecodeError):
        q = {}
    try:
        runner = subprocess.run(["systemctl", "--user", "is-active", "my-code-queue.service"],
                                capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        runner = "unknown"
    return {"paused": bool(q.get("paused")), "updated": q.get("updated"), "runner": runner,
            "items": [{k: it.get(k) for k in ("id", "status", "wait_for", "added", "started", "finished", "exit", "cmd")}
                      for it in q.get("items", [])]}
