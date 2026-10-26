"""Logging, state and email for src/scripts/run_campaign.py (2026-10-17).

Everything a campaign batch writes lives in logs/<batch>/:
  driver.log          (stdout of the driver, every line time-stamped)
  events.jsonl        one JSON object per event: launch / finish / check / retry / gpu_alert / gate_wait / mail / error
  state.json          the live state of every run (status, attempts, exp_ids, checks, eval numbers) -- used by --resume
                      and by rerun_campaign.py
  gpu_monitor.csv     one row per GPU sample: time, gpu, util %, memory %, runs on that GPU
  manifest.json       schema 3 (docs/data_spec.md), written by run_campaign.py
  seed<k>_<scen>_<key>_<exp_id>.log   one training log per attempt
Email goes through shared.notify (.env.email); a failed email never stops the campaign, it is logged as an event.
"""
from __future__ import annotations

import csv
import html as _html
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

TAG = "[my_code]"


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(path)


STYLE = ("<style>body{font-family:-apple-system,Segoe UI,Helvetica,Arial,'PingFang SC','Microsoft YaHei',sans-serif;font-size:14px;"
         "color:#1f2633}table{border-collapse:collapse;margin:6px 0 12px;font-size:13px}th,td{border:1px solid #cfd6e1;padding:4px 8px;"
         "white-space:nowrap}th{background:#f3f5f8}.bad{color:#b91c1c}.ok{color:#15803d}pre{background:#f7f7f9;padding:8px;"
         "font-size:12px;white-space:pre-wrap}</style>")


def html_table(header: list[str], rows: list[list]) -> str:
    h = "".join(f"<th>{_html.escape(str(c))}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{_html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def text_table(header: list[str], rows: list[list]) -> str:
    return "\n".join([" | ".join(map(str, header))] + [" | ".join(map(str, r)) for r in rows])


class CampaignLog:
    def __init__(self, log_dir: Path, batch: str, smoke: bool):
        self.dir, self.batch, self.smoke = log_dir, batch, smoke
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events = self.dir / "events.jsonl"
        self.state_path = self.dir / "state.json"
        self.gpu_csv = self.dir / "gpu_monitor.csv"
        self._lock = threading.Lock()
        try:
            self.state = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            self.state = {"batch": batch, "created_at": ts(), "runs": {}, "groups": {}}
        if not self.gpu_csv.exists():
            with open(self.gpu_csv, "w", newline="") as f:
                csv.writer(f).writerow(["time", "gpu", "util_pct", "mem_pct", "mem_used_mib", "runs"])

    # ── text log + structured events ───────────────────────────────────────
    def info(self, msg: str) -> None:
        print(f"[{ts()}] {msg}", flush=True)

    def event(self, kind: str, msg: str | None = None, **kw) -> None:
        rec = {"t": ts(), "kind": kind, **kw}
        if msg:
            rec["msg"] = msg
        with self._lock, open(self.events, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if msg:
            self.info(f"{kind}: {msg}")

    # ── state ──────────────────────────────────────────────────────────────
    def save(self) -> None:
        with self._lock:
            self.state["updated_at"] = ts()
            _atomic_json(self.state_path, self.state)

    def run(self, label: str) -> dict:
        return self.state["runs"].setdefault(label, {"status": "planned", "attempts": []})

    def group(self, gid: str) -> dict:
        return self.state["groups"].setdefault(gid, {"status": "planned"})

    def gpu_rows(self, stats: dict, runs_on: dict) -> None:
        with open(self.gpu_csv, "a", newline="") as f:
            w = csv.writer(f)
            for g, s in sorted(stats.items()):
                w.writerow([ts(), g, f"{s['util']:.0f}", f"{s['mem_pct']:.1f}", f"{s['mem_used']:.0f}", runs_on.get(g, 0)])

    # ── email ──────────────────────────────────────────────────────────────
    def mail(self, subject: str, lines: list[str], tables: list[tuple[str, list[str], list[list]]] = (),
             pre: str | None = None, attachments: list[Path] = ()) -> bool:
        """Never raises. subject gets the tag, the batch and [SMOKE] for a smoke batch."""
        full = f"{TAG} {self.batch}{' [SMOKE]' if self.smoke else ''} {subject}"
        foot = f"主机 {socket.gethostname()} · 日志目录 logs/{self.batch}/ · {ts()} · src/scripts/run_campaign.py"
        plain = "\n".join(lines) + "".join(f"\n\n== {t}\n{text_table(h, r)}" for t, h, r in tables) + (f"\n\n{pre}" if pre else "") + f"\n\n{foot}"
        html = (f"<html><head>{STYLE}</head><body><h2>{_html.escape(subject)}</h2>"
                + "".join(f"<p>{_html.escape(e)}</p>" for e in lines)
                + "".join(f"<h3>{_html.escape(t)}</h3>{html_table(h, r)}" for t, h, r in tables)
                + (f"<pre>{_html.escape(pre)}</pre>" if pre else "")
                + f"<hr><p style='color:#98a1b0;font-size:12px'>{_html.escape(foot)}</p></body></html>")
        try:
            from shared.notify import send_notification
            ok = bool(send_notification(full, plain, html=html, attachments=[p for p in attachments if Path(p).is_file()]))
        except Exception as e_:                       # never let email break the campaign
            ok = False
            self.event("mail", f"邮件发送异常：{e_!r}", subject=full, ok=False)
            return False
        self.event("mail", None, subject=full, ok=ok)
        if not ok:
            self.info(f"mail: 发送失败（{full}）")
        return ok


def tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def pid_cpu_seconds(pid: int) -> float | None:
    """utime + stime of a process (all threads), None if it is gone."""
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return None
