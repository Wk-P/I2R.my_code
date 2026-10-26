#!/usr/bin/env python3
"""Serial experiment queue (2026-10-11).

The queue lives in logs/queue.json; one item runs at a time, in order. Each item may wait for a process
(regex on the command line, e.g. "run_v4.4.17.py") to be gone before it starts. If an item fails, the
queue stops (later items stay queued) and an email is sent; nothing is retried automatically. The runner
is meant to run as the systemd user service my-code-queue.service: after a restart it keeps waiting for
an item whose process is still alive, and marks one whose process is gone as "interrupted" and stops.

    .venv/bin/python src/scripts/queue_runner.py add v4.5.1 --wait-for "run_v4.4.17.py" -- \\
        .venv/bin/python src/scripts/run_v4.5.py --version 4.5.1
    .venv/bin/python src/scripts/queue_runner.py list
    .venv/bin/python src/scripts/queue_runner.py remove v4.5.3        # queued items only
    .venv/bin/python src/scripts/queue_runner.py resume               # interrupted / failed head -> queued again
    .venv/bin/python src/scripts/queue_runner.py move v4.5.3 up       # reorder queued items (up / down)
    .venv/bin/python src/scripts/queue_runner.py pause | unpause      # do not start the next item / allow it again
    .venv/bin/python src/scripts/queue_runner.py run                  # the service's entry point
The queue is managed from the command line only; the panel (app/backend /api/queue) just reads logs/queue.json.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
QUEUE = ROOT / "logs" / "queue.json"
LOCK = ROOT / "logs" / "queue.lock"
POLL = 60


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_all() -> dict:
    try:
        d = json.loads(QUEUE.read_text())
        return {"paused": bool(d.get("paused")), "items": d.get("items", [])}
    except (OSError, json.JSONDecodeError):
        return {"paused": False, "items": []}


def load() -> list[dict]:
    return load_all()["items"]


def save(items: list[dict], paused: bool | None = None) -> None:
    paused = load_all()["paused"] if paused is None else paused
    tmp = QUEUE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"updated": now(), "paused": paused, "items": items}, indent=2, ensure_ascii=False))
    tmp.replace(QUEUE)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return stat.exists() and stat.read_text().split()[2] != "Z"


def proc_running(pattern: str) -> bool:
    """Any live process (other than this runner) whose command line matches `pattern`."""
    rx, me = re.compile(pattern), os.getpid()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if rx.search(cmd) and "queue_runner.py" not in cmd:
            return True
    return False


def mail(subject: str, body: str) -> None:
    try:
        from shared.notify import send_notification
        send_notification(f"[my_code] {subject}", body)
    except Exception as e:                      # never let email break the queue
        print(f"[queue] email failed: {e}", flush=True)


EDIT_LOCK = ROOT / "logs" / "queue.edit.lock"


class locked:
    """Exclusive lock around every read-modify-write of queue.json (runner and CLI / panel alike)."""

    def __enter__(self):
        self.fh = open(EDIT_LOCK, "w")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def update(item_id: str, **fields) -> dict:
    with locked():
        items = load()
        it = next(e for e in items if e["id"] == item_id)
        it.update(fields)
        save(items)
        return dict(it)


def pick() -> dict | None:
    """The item to act on: a running / waiting one, else the first queued one (unless paused or stopped
    behind a failed / interrupted item)."""
    d = load_all()
    items = d["items"]
    cur = next((it for it in items if it["status"] in ("running", "waiting")), None)
    if cur is not None:
        return cur
    cur = next((it for it in items if it["status"] == "queued"), None)
    if cur is None or d["paused"]:
        return None
    if any(it["status"] in ("failed", "interrupted") for it in items[:items.index(cur)]):
        return None
    return cur


def run() -> None:
    with open(LOCK, "w") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[queue] another runner holds the lock; exiting", flush=True)
            return
        print(f"[queue] runner started {now()}", flush=True)
        while True:
            cur = pick()
            if cur is None:
                time.sleep(POLL)
                continue
            if cur["status"] == "running":      # after a restart: still alive?
                if pid_alive(cur.get("pid")):
                    time.sleep(POLL)
                    continue
                update(cur["id"], status="interrupted", finished=now())
                mail(f"队列：{cur['id']} 中断", f"{cur['id']} 的进程已不在（pid {cur.get('pid')}），队列已停止。\n命令：{' '.join(cur['cmd'])}")
                continue
            if cur.get("wait_for") and proc_running(cur["wait_for"]):
                if cur["status"] != "waiting":
                    update(cur["id"], status="waiting")
                    print(f"[queue] {cur['id']}: waiting for /{cur['wait_for']}/", flush=True)
                time.sleep(POLL)
                continue
            if load_all()["paused"]:            # paused while waiting: do not start
                if cur["status"] == "waiting":
                    update(cur["id"], status="queued")
                time.sleep(POLL)
                continue
            log = ROOT / cur["log"]
            log.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(log, "w") as fh:
                    proc = subprocess.Popen(cur["cmd"], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                            start_new_session=True)
            except OSError as e_:                # bad command: fail this item and stop, do not crash the runner
                update(cur["id"], status="failed", finished=now(), exit=None, error=str(e_))
                mail(f"队列：{cur['id']} 无法启动，队列已停止", f"{cur['id']} 启动失败：{e_}\n命令：{' '.join(cur['cmd'])}")
                continue
            update(cur["id"], status="running", pid=proc.pid, started=now())
            print(f"[queue] {now()} start {cur['id']} pid={proc.pid}", flush=True)
            mail(f"队列：开始 {cur['id']}", f"{cur['id']} 已开始（pid {proc.pid}）。\n命令：{' '.join(cur['cmd'])}\n日志：{cur['log']}")
            ret = proc.wait()
            it = update(cur["id"], status="done" if ret == 0 else "failed", finished=now(), exit=ret)
            print(f"[queue] {now()} {it['id']} exited {ret}", flush=True)
            left = [e["id"] for e in load() if e["status"] == "queued"]
            if ret == 0:
                mail(f"队列：完成 {it['id']}", f"{it['id']} 已完成。\n接下来：{', '.join(left) or '队列已空'}")
            else:
                mail(f"队列：{it['id']} 失败，队列已停止", f"{it['id']} 退出码 {ret}，后面的 {', '.join(left) or '（无）'} 不会自动开始。\n"
                     f"日志：{it['log']}\n处理后执行：queue_runner.py resume")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_add = sub.add_parser("add")
    a_add.add_argument("id")
    a_add.add_argument("--wait-for", default=None, help="regex: start only when no process command line matches it")
    a_add.add_argument("--log", default=None)
    # the experiment command is everything after "--" (split by hand: argparse.REMAINDER after a positional
    # also swallows options such as --wait-for)
    a_rm = sub.add_parser("remove")
    a_rm.add_argument("id")
    a_mv = sub.add_parser("move")
    a_mv.add_argument("id")
    a_mv.add_argument("direction", choices=["up", "down"])
    sub.add_parser("pause")
    sub.add_parser("unpause")
    sub.add_parser("list")
    sub.add_parser("resume")
    sub.add_parser("run")
    argv = sys.argv[1:]
    command = argv[argv.index("--") + 1:] if "--" in argv else []
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    a.command = command
    if a.cmd == "run":
        return run()
    with locked():
        edit(a)
    print(f"paused: {load_all()['paused']}")
    for it in load():
        print(f"{it['id']:10s} {it['status']:12s} wait_for={it.get('wait_for') or '-':18s} started={it.get('started') or '-':20s} "
              f"finished={it.get('finished') or '-':20s} exit={it.get('exit')}  {' '.join(it['cmd'])}")


def edit(a) -> None:
    items = load()
    if a.cmd == "add":
        cmd = a.command
        assert cmd, "no command given (put it after --)"
        assert all(it["id"] != a.id for it in items), f"{a.id} already in the queue"
        items.append({"id": a.id, "cmd": cmd, "wait_for": a.wait_for, "status": "queued", "log": a.log or f"logs/run_{a.id}_driver.log",
                      "added": now(), "started": None, "finished": None, "pid": None, "exit": None})
    elif a.cmd == "remove":
        it = next((e for e in items if e["id"] == a.id), None)
        assert it and it["status"] in ("queued", "waiting"), f"{a.id}: only queued items can be removed"
        items.remove(it)
    elif a.cmd == "move":
        q = [e for e in items if e["status"] == "queued"]
        it = next((e for e in q if e["id"] == a.id), None)
        assert it, f"{a.id}: only queued items can be moved"
        n = q.index(it) + (-1 if a.direction == "up" else 1)
        if 0 <= n < len(q):
            i1, i2 = items.index(it), items.index(q[n])
            items[i1], items[i2] = items[i2], items[i1]
    elif a.cmd in ("pause", "unpause"):
        save(items, paused=a.cmd == "pause")
    elif a.cmd == "resume":
        for it in items:
            if it["status"] in ("failed", "interrupted"):
                it.update(status="queued", started=None, finished=None, pid=None, exit=None)
                break
    if a.cmd not in ("list", "pause", "unpause"):
        save(items)


if __name__ == "__main__":
    main()
