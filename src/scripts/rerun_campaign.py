#!/usr/bin/env python3
"""Re-training of failed campaign runs (2026-10-17), used by run_campaign.py and by hand.

retry(C, L, sup, failed): called by run_campaign.py when every run of a group has ended. Each failed run is
re-trained under a new exp_id (the old ones stay in state.json / manifest "attempts" and their log files), with the
same GPU gate and supervision as the group; up to MAX_RETRY rounds. Emails: the list of runs being re-trained,
then the outcome. Returns the runs that still fail.

By hand (the driver of that batch must not be running; refuses while other paper_rl.train processes exist):
    .venv/bin/python src/scripts/rerun_campaign.py --batch <batch> --list                       # status of every run
    .venv/bin/python src/scripts/rerun_campaign.py --batch <batch> --failed                     # every failed / interrupted run
    .venv/bin/python src/scripts/rerun_campaign.py --batch <batch> --label "seed1 eq ppo_maskable" [--label ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))


def retry(C, L, sup, failed):
    from run_campaign import MAX_RETRY
    for rnd in range(1, MAX_RETRY + 1):
        if not failed:
            break
        why = {r["label"]: "；".join(L.state["runs"].get(r["label"], {}).get("problems") or ["?"]) for r in failed}
        L.event("retry", f"第 {rnd}/{MAX_RETRY} 轮重跑 {len(failed)} 个 run：{', '.join(why)}", round=rnd, labels=list(why))
        L.mail(f"开始第 {rnd} 轮重跑（{len(failed)} 个 run）", [f"第 {rnd}/{MAX_RETRY} 轮，新 exp_id，GPU 规则与正式 run 相同。"],
               tables=[("待重跑", ["run", "上次失败原因"], [[k, v] for k, v in why.items()])])
        for r in failed:
            L.run(r["label"]).update(status="retrying")
        L.save()
        sup.drive(failed)
        still = [r for r in failed if L.state["runs"].get(r["label"], {}).get("status") != "ok"]
        rows = [[r["label"], L.state["runs"][r["label"]]["status"], L.state["runs"][r["label"]].get("exp_id"),
                 "；".join(L.state["runs"][r["label"]].get("problems") or []) or "—"] for r in failed]
        L.event("retry_done", f"第 {rnd} 轮重跑结束：{len(failed) - len(still)} 个通过，{len(still)} 个仍失败", round=rnd)
        L.mail(f"第 {rnd} 轮重跑结束：{len(failed) - len(still)} 通过 / {len(still)} 仍失败",
               ["仍失败的会进入下一轮重跑。" if still and rnd < MAX_RETRY else ("全部通过。" if not still else "已用完重跑次数。")],
               tables=[("重跑结果", ["run", "状态", "exp_id", "问题"], rows)])
        failed = still
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--failed", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    log_dir = ROOT / "logs" / a.batch
    state = json.loads((log_dir / "state.json").read_text())
    man = json.loads((log_dir / "manifest.json").read_text())
    if a.list:
        for lab, st in sorted(state["runs"].items()):
            print(f"{lab:32s} {st['status']:12s} attempts={len(st['attempts'])} exp_id={st.get('exp_id')} "
                  f"{'；'.join(st.get('problems') or [])}")
        return
    import run_campaign as RC
    smoke = man["purpose"] == "smoke"
    ns = argparse.Namespace(smoke=smoke, batch=a.batch, version=man["version"], groups=",".join(man["groups"]),
                            per_gpu=man["params"]["per_gpu"], no_mps=False)
    C = RC.Cfg(ns)
    C.inject = {}
    os.environ["RESULTS_SPACE"] = C.space
    pid = state.get("driver_pid")
    if pid and Path(f"/proc/{pid}").exists():
        sys.exit(f"the driver of {a.batch} (pid {pid}) is still running")
    if RC.foreign_training():
        sys.exit("other paper_rl.train processes are running; wait (queue rule)")
    from campaign_log import CampaignLog
    L = CampaignLog(log_dir, a.batch, smoke)
    allr = {r["label"]: r for g in C.groups for r in RC.plan_runs(C, g)}
    labels = a.label or [k for k, st in state["runs"].items() if a.failed and st["status"] in ("failed", "interrupted", "retrying", "running")]
    bad = [x for x in labels if x not in allr]
    if bad or not labels:
        sys.exit(f"nothing to re-run / unknown labels {bad}")
    sup = RC.Supervisor(C, L)
    if C.mps:
        RC.mps_start(L)
    try:
        left = retry(C, L, sup, [allr[x] for x in labels])
    finally:
        if C.mps:
            RC.mps_stop(L)
    print(f"{len(labels) - len(left)} OK, {len(left)} still failing", flush=True)
    sys.exit(1 if left else 0)


if __name__ == "__main__":
    main()
