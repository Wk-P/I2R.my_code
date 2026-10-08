#!/usr/bin/env python3
"""v4.3.1.6 formal campaign: 12 models x 3 scenarios x seeds {1, 2, 3} x 5M steps.

Seed 1 was already run with identical code and settings (scripts/run_v4.3.1.6.py,
manifest scripts/logs/v4.3.1.6_run/manifest.json, commit ede2187) and is reused;
this script launches seeds 2 and 3 (72 jobs) and writes a report that
aggregates all three seeds (mean +- sample std).

Settings (v4.3.1.6): p = 0.6 data; legacy reward for all 12 models; Lagrangian
terminal cost -lambda * total violations, lambda_max 50; Maskable / Repair dead
end = failure; PPO clip 0.1.

    nohup .venv/bin/python scripts/run_v4.3.1.6_campaign.py > scripts/logs/run_v4.3.1.6_campaign_driver.log 2>&1 &
    nohup .venv/bin/python scripts/run_v4.3.1.6_campaign.py --resume >> scripts/logs/run_v4.3.1.6_campaign_driver.log 2>&1 &
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
LOG_DIR = ROOT / "scripts" / "logs" / "v4.3.1.6_campaign"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = LOG_DIR / "manifest.json"
SEED1_MANIFEST = ROOT / "scripts" / "logs" / "v4.3.1.6_run" / "manifest.json"
REPORT = ROOT / "paper_contents" / "v4.3.1" / "v4.3.1.6" / "campaign_report.md"
RESULTS = ROOT / "results" / "unified"

CAPACITY = int(56 * 0.93)
SCENARIOS = ["lt", "eq", "gt"]
MECHS = ["", "lagrange", "mask", "repair"]
LEARNERS = ["ppo", "dqn", "ddqn"]
ALGOS = [f"{m}_{l}" if m else l for l in LEARNERS for m in MECHS]
WEIGHT = {a: (4 if a.endswith("ppo") else 2) for a in ALGOS}
SEEDS = [1, 2, 3]
NEW_SEEDS = [2, 3]
STEPS = 5_000_000
REWARD = "legacy"
MECH_LABEL = {"": "无约束（对照）", "lagrange": "Lagrangian", "mask": "Maskable", "repair": "Repair"}


def split(a):
    return ("", a) if a in LEARNERS else tuple(a.split("_", 1))


def new_exp_id():
    return subprocess.run([PY, "-c", "from shared.paths import new_exp_id; print(new_exp_id())"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def seed1_ids():
    d = json.loads(SEED1_MANIFEST.read_text())["exp_ids"][REWARD]
    return {s: dict(v) for s, v in d.items()}


def write_manifest(exp_ids):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": "4.3.1.6", "commit": commit, "branch": "unified", "steps": STEPS,
                               "reward": REWARD, "scenarios": SCENARIOS, "algos": ALGOS, "seeds": SEEDS,
                               "note": "seed 1 reused from scripts/logs/v4.3.1.6_run (commit ede2187)",
                               "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST)


def launch(seed, scen, algo, exp_ids):
    exp_id = new_exp_id()
    log = open(LOG_DIR / f"seed{seed}_{scen}_{algo}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "REWARD_MODE": REWARD,
           "TRAIN_SEED": str(seed), "PAPER_VERSION": "4.3.1.6"}
    proc = subprocess.Popen([PY, "-u", "-m", "paper_rl.train", "--scen", scen, "--algo", algo, "--reward", REWARD,
                             "--steps", str(STEPS), "--seed", str(seed)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    exp_ids.setdefault(str(seed), {}).setdefault(scen, {})[algo] = exp_id
    write_manifest(exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched seed{seed} {scen}/{algo} exp_id={exp_id} pid={proc.pid}", flush=True)
    return proc, log


def result(scen, algo, exp_id):
    p = RESULTS / scen / algo / exp_id / "results.json"
    return json.loads(p.read_text())[algo] if p.exists() else None


def write_report(exp_ids):
    def ms(vals, f):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        return f"{f(np.mean(vals))} ± {f(sd)}" + ("" if len(vals) == 3 else f" (n={len(vals)})")
    f3 = lambda v: f"{v:.3f}"
    f4 = lambda v: f"{v:.4f}"
    lines = ["# v4.3.1.6 正式实验（12 个模型 × 3 场景 × 3 种子 × 5M 步）", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 设定：p = 0.6 数据；12 个模型统一 legacy 奖励；Lagrangian 终局代价 λ·违规总数、λ_max 50；"
             "Maskable / Repair 死局判失败；PPO clip 0.1。种子 1 复用 `v4.3.1.6_run`，种子 2、3 为本批次。",
             "- 每个种子在各自的测试集（400 个实例）上单次确定性评估；表中为 3 个种子的均值 ± 样本标准差。",
             "- success_rate = M 个服务全部合法放置；AR = average resource utilization（只计合法放置），全部测试 episode 均值；"
             "AR/AR* 只在成功 episode 上计算；AR/AR*×success = 失败 episode 计 0 的综合指标；违规率 = max(容量违规率, 冲突违规率)，即至少一项违规的 episode 比例的下界。",
             "- manifest：`scripts/logs/v4.3.1.6_campaign/manifest.json`", ""]
    order = sorted(ALGOS, key=lambda a: (LEARNERS.index(split(a)[1]), MECHS.index(split(a)[0])))
    for s in SCENARIOS:
        lines += [f"## {s.upper()}", "",
                  "| 模型 | 约束处理 | success_rate | AR | AR/AR* | AR/AR*×success | 违规率 | 死局率 |",
                  "|---|---|---|---|---|---|---|---|"]
        for a in order:
            rs = [result(s, a, exp_ids.get(str(sd), {}).get(s, {}).get(a, "")) for sd in SEEDS]
            rs = [r for r in rs if r]
            g = lambda k: [r.get(k) for r in rs]
            comb = [(r["ar_ratio_successful"] or 0.0) * r["success_rate"] for r in rs]
            viol = [max(r["cap_viol_rate"], r["conflict_viol_rate"]) for r in rs]
            mech, learner = split(a)
            lines.append("| " + " | ".join([learner.upper(), MECH_LABEL[mech], ms(g("success_rate"), f3), ms(g("ar_mean"), f4),
                                             ms(g("ar_ratio_successful"), f4), ms(comb, f4), ms(viol, f3),
                                             ms(g("dead_end_rate"), f3)]) + " |")
        lines.append("")
    REPORT.write_text("\n".join(lines))
    print(f"report -> {REPORT}", flush=True)


def live_pids():
    live = {}
    for pd in Path("/proc").iterdir():
        if not pd.name.isdigit():
            continue
        try:
            if b"paper_rl.train" not in (pd / "cmdline").read_bytes():
                continue
            env = dict(kv.split(b"=", 1) for kv in (pd / "environ").read_bytes().split(b"\0") if b"=" in kv)
            live[env.get(b"EXP_ID", b"").decode()] = int(pd.name)
        except OSError:
            continue
    return live


def main():
    exp_ids = {"1": seed1_ids()}
    queue = [(sd, s, a) for sd in NEW_SEEDS for a in ALGOS for s in SCENARIOS]
    adopted = []
    if "--resume" in sys.argv and MANIFEST.exists():
        exp_ids = json.loads(MANIFEST.read_text())["exp_ids"]
        launched = {(int(sd), s, a): e for sd, d in exp_ids.items() if int(sd) in NEW_SEEDS
                    for s, d2 in d.items() for a, e in d2.items()}
        queue = [j for j in queue if j not in launched]
        live = live_pids()
        adopted = [(live[e], WEIGHT[a]) for (sd, s, a), e in launched.items() if e in live]
        print(f"=== resume: {len(launched)} launched, {len(adopted)} running, {len(queue)} to launch ===", flush=True)
    write_manifest(exp_ids)
    print(f"=== v4.3.1.6 campaign start {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(queue)} jobs (seeds {NEW_SEEDS}) ===", flush=True)
    active, failed, done = [], [], 0
    while queue or active or adopted:
        adopted = [x for x in adopted if Path(f"/proc/{x[0]}").exists()]
        load = sum(w for *_, w in active) + sum(w for _, w in adopted)
        i = 0
        while i < len(queue):
            sd, s, a = queue[i]
            if load + WEIGHT[a] <= CAPACITY:
                queue.pop(i)
                p, f = launch(sd, s, a, exp_ids)
                active.append((p, f, f"seed{sd} {s}/{a}", WEIGHT[a]))
                load += WEIGHT[a]
            else:
                i += 1
        still = []
        for p, f, label, w in active:
            ret = p.poll()
            if ret is None:
                still.append((p, f, label, w))
                continue
            f.close()
            done += 1
            print(f"[{time.strftime('%H:%M:%S')}] finished {label} ({'OK' if ret == 0 else f'EXIT={ret}'}) "
                  f"— {done} done, {len(queue)} queued", flush=True)
            if ret != 0:
                failed.append(label)
        active = still
        if queue or active or adopted:
            time.sleep(15)
    print(f"=== campaign done {time.strftime('%Y-%m-%d %H:%M:%S')} — {done} completed, {len(failed)} failed ===", flush=True)
    if failed:
        print("FAILED:", failed, flush=True)
    write_report(exp_ids)


if __name__ == "__main__":
    main()
