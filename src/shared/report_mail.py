"""Email interface for experiment reports (v4.4.14): detailed HTML mail when a version's report is written,
and a failure mail listing the failed jobs with the tail of their logs. Uses shared.notify (.env.email).

    from shared.report_mail import send_report, send_failure
    send_report("v4.4.14", "paper_contents/v4.4.14/report.md", manifest="logs/v4.4.14/manifest.json",
                driver_log="logs/run_v4.4.14_driver.log")
    send_failure("v4.4.14", ["seed1 lt mask_ppo_pool"], log_dir="logs/v4.4.14")

Command line (also for re-sending by hand):
    .venv/bin/python -m shared.report_mail report  --version v4.4.14 --report paper_contents/v4.4.14/report.md \\
        [--manifest logs/v4.4.14/manifest.json] [--driver-log logs/run_v4.4.14_driver.log]
    .venv/bin/python -m shared.report_mail failure --version v4.4.14 --log-dir logs/v4.4.14 --failed "seed1 lt key" ...
    .venv/bin/python -m shared.report_mail test

Mail content: run overview (version, commit, start / end / duration, jobs OK / failed, manifest, report paths),
the headline test gaps of the new variant in the subject, the full report rendered as HTML tables, and
report.md + report.en.md attached.
"""
from __future__ import annotations

import argparse
import datetime
import html as _html
import json
import re
import socket
import subprocess
from pathlib import Path

from shared.notify import send_notification

ROOT = Path(__file__).resolve().parent.parent.parent
TAG = "[my_code]"


# ── markdown subset (headings, lists, tables, `code`, **bold**) -> HTML ─────
def _inline(s: str) -> str:
    s = _html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)


def md_to_html(text: str) -> str:
    out, i, lines = [], 0, text.splitlines()
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                head = "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
                body = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
                out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            continue
        if ln.startswith("- ") or ln.startswith("  "):
            items = []
            while i < len(lines) and (lines[i].startswith("- ") or (lines[i].startswith("  ") and items)):
                if lines[i].startswith("- "):
                    items.append(lines[i][2:])
                else:
                    items[-1] += "<br>" + lines[i].strip()
                i += 1
            out.append("<ul>" + "".join(f"<li>{_inline(x) if '<br>' not in x else '<br>'.join(_inline(y) for y in x.split('<br>'))}</li>"
                                        for x in items) + "</ul>")
            continue
        m = re.match(r"^(#{1,4}) (.*)", ln)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n + 1}>{_inline(m.group(2))}</h{n + 1}>")
        elif ln.strip():
            out.append(f"<p>{_inline(ln)}</p>")
        i += 1
    return "\n".join(out)


STYLE = """<style>
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,'PingFang SC','Microsoft YaHei',sans-serif;font-size:14px;color:#1f2633;line-height:1.5}
h2{font-size:20px;margin:18px 0 8px} h3{font-size:16px;margin:16px 0 6px;border-bottom:1px solid #e3e7ee;padding-bottom:3px} h4{font-size:14px}
table{border-collapse:collapse;margin:6px 0 12px;font-size:13px} th,td{border:1px solid #cfd6e1;padding:4px 8px;text-align:right;white-space:nowrap}
th{background:#f3f5f8} td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}
code{background:#f3f5f8;padding:0 3px;border-radius:3px;font-size:12px} .ov td{text-align:left;white-space:normal}
.bad{color:#b91c1c} pre{background:#f7f7f9;padding:8px;font-size:12px;overflow-x:auto}
</style>"""


# ── run information ─────────────────────────────────────────────────────────
def _rel(p) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def run_info(report: Path | None, manifest: Path | None, driver_log: Path | None) -> dict:
    info = {"host": socket.gethostname()}
    if manifest and Path(manifest).is_file():
        m = json.loads(Path(manifest).read_text())
        info["commit"] = m.get("commit")
        info["steps"] = m.get("steps")
        info["seeds"] = m.get("seeds")
        info["n_jobs"] = sum(1 for sd in m.get("exp_ids", {}).values() for sc in sd.values() for _ in sc.values())
    if driver_log and Path(driver_log).is_file():
        txt = Path(driver_log).read_text(errors="replace")
        st = re.search(r"start (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", txt)
        dn = re.search(r"=== done (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", txt)
        info["started"] = st.group(1) if st else None
        info["training_done"] = dn.group(1) if dn else None
        info["n_ok"] = len(re.findall(r"finished .* \(OK\)", txt))
        info["failed"] = re.findall(r"finished (.*) \(EXIT=", txt)
    if report and Path(report).is_file():
        info["report_written"] = datetime.datetime.fromtimestamp(Path(report).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    if info.get("started") and info.get("report_written"):
        d = (datetime.datetime.fromisoformat(info["report_written"]) - datetime.datetime.fromisoformat(info["started"])).total_seconds()
        info["duration"] = f"{int(d // 3600)} h {int(d % 3600 // 60)} min"
    return info


def headline(report_text: str) -> str:
    """Test gaps of the last (new) variant per scenario from the first summary table, e.g. 'LT 9.1% · EQ 7.0% · GT 6.8%'."""
    m = re.search(r"\n(\|[^\n]*\n\|[-| ]+\n(?:\|[^\n]*\n)+)", report_text)
    if not m:
        return ""
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in m.group(1).splitlines() if not re.fullmatch(r"\|[-| ]+", r)]
    last = {}
    for r in rows[1:]:
        if len(r) > 2 and r[0] in ("LT", "EQ", "GT"):
            last[r[0]] = r[2].split(" ±")[0]
    return " · ".join(f"{k} {last[k]}" for k in ("LT", "EQ", "GT") if k in last)


# ── public interface ────────────────────────────────────────────────────────
def send_report(version: str, report, manifest=None, driver_log=None, note: str | None = None) -> bool:
    report = Path(report)
    if not report.is_absolute():
        report = ROOT / report
    text = report.read_text()
    en = report.with_suffix(".en.md")
    info = run_info(report, Path(ROOT / manifest) if manifest else None, Path(ROOT / driver_log) if driver_log else None)
    head = headline(text)
    subject = f"{TAG} {version} 报告已生成" + (f" · 测试 gap {head}" if head else "")
    ov = [("版本", version), ("git commit", info.get("commit")), ("主机", info.get("host")),
          ("开始", info.get("started")), ("训练结束", info.get("training_done")), ("报告写出", info.get("report_written")),
          ("总耗时", info.get("duration")), ("任务", f"{info.get('n_ok', '—')} 个成功 / {len(info.get('failed') or [])} 个失败"
                                                  + (f"（共 {info['n_jobs']} 个）" if info.get("n_jobs") else "")),
          ("步数 / 种子", f"{info.get('steps', '—')} / {info.get('seeds', '—')}"),
          ("manifest", _rel(ROOT / manifest) if manifest else None), ("报告", _rel(report) + (f"，英文版 {_rel(en)}" if en.is_file() else ""))]
    rows = "".join(f"<tr><td><b>{_html.escape(k)}</b></td><td>{_html.escape(str(v))}</td></tr>" for k, v in ov if v)
    failed = info.get("failed") or []
    fail_html = (f"<p class='bad'><b>失败的任务：</b>{_html.escape(', '.join(failed))}</p>" if failed else "")
    html = (f"<html><head>{STYLE}</head><body><h2>{_html.escape(version)} 实验报告</h2>"
            + (f"<p>{_html.escape(note)}</p>" if note else "")
            + f"<table class='ov'>{rows}</table>{fail_html}<hr>{md_to_html(text)}"
            + "<hr><p style='color:#98a1b0;font-size:12px'>附件：中文报告 report.md 与英文报告 report.en.md。由 shared/report_mail.py 自动发送。</p>"
            + "</body></html>")
    plain = "\n".join(f"{k}: {v}" for k, v in ov if v) + ("\n失败的任务: " + ", ".join(failed) if failed else "") + "\n\n" + text
    return send_notification(subject, plain, html=html, attachments=[report, en])


def send_failure(version: str, failed: list[str], log_dir=None, manifest=None, tail: int = 30) -> bool:
    log_dir = Path(ROOT / log_dir) if log_dir else None
    parts = []
    for label in failed:
        sd, scen, key = (label.split() + ["", "", ""])[:3]
        logs = sorted(log_dir.glob(f"{sd}_{scen}_{key}_*.log")) if log_dir else []
        body = "\n".join(logs[-1].read_text(errors="replace").splitlines()[-tail:]) if logs else "(no log found)"
        parts.append((label, _rel(logs[-1]) if logs else None, body))
    subject = f"{TAG} {version} 有 {len(failed)} 个任务失败"
    html = (f"<html><head>{STYLE}</head><body><h2 class='bad'>{_html.escape(version)}：{len(failed)} 个任务失败</h2>"
            + "".join(f"<h3>{_html.escape(lb)}</h3><p>日志：<code>{_html.escape(str(lp))}</code>（最后 {tail} 行）</p>"
                      f"<pre>{_html.escape(b)}</pre>" for lb, lp, b in parts) + "</body></html>")
    plain = "\n\n".join(f"== {lb} ({lp})\n{b}" for lb, lp, b in parts)
    return send_notification(subject, plain, html=html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["report", "failure", "test"])
    ap.add_argument("--version")
    ap.add_argument("--report")
    ap.add_argument("--manifest")
    ap.add_argument("--driver-log")
    ap.add_argument("--log-dir")
    ap.add_argument("--failed", nargs="*", default=[])
    ap.add_argument("--note")
    a = ap.parse_args()
    if a.cmd == "test":
        ok = send_notification(f"{TAG} 邮件接口测试", "shared/report_mail.py 测试邮件。",
                               html=f"<html><head>{STYLE}</head><body><h2>邮件接口测试</h2><p>如果能看到这封邮件，说明 .env.email 的配置有效。</p></body></html>")
    elif a.cmd == "report":
        ok = send_report(a.version, a.report, a.manifest, a.driver_log, a.note)
    else:
        ok = send_failure(a.version, a.failed, a.log_dir, a.manifest)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
