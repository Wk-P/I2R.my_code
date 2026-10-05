#!/usr/bin/env python3
"""v4.3.1.9: success reward M(2AR-1) -> M(1 + 2AR) (reward_mode succ_first), all 12 models.

12 models ({PPO, DQN, DDQN} x {none, Lagrangian, Maskable, Repair}) x LT/EQ/GT x
seeds {1, 2, 3} x 5M = 108 jobs. Everything else is v4.3.1.6 (p = 0.6 data, same
splits / hyper-parameters, failure branch -M(1 - valid/M), no reward
normalisation). Baseline = v4.3.1.6 models re-evaluated against the true AR* in
v4.3.1.7 (paper_contents/v4.3.1/v4.3.1.7/summary.json).

Load is counted over every running paper_rl.train process on the machine (PPO 4,
DQN / DDQN 2 threads), so this campaign shares the CPU with v4.3.1.8.

    nohup .venv/bin/python scripts/run_v4.3.1.9.py > scripts/logs/run_v4.3.1.9_driver.log 2>&1 &
    nohup .venv/bin/python scripts/run_v4.3.1.9.py --resume >> scripts/logs/run_v4.3.1.9_driver.log 2>&1 &
    .venv/bin/python scripts/run_v4.3.1.9.py --report
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
LOG_DIR = ROOT / "scripts" / "logs" / "v4.3.1.9"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = LOG_DIR / "manifest.json"
REPORT = ROOT / "paper_contents" / "v4.3.1" / "v4.3.1.9" / "report.md"
BASELINE = ROOT / "paper_contents" / "v4.3.1" / "v4.3.1.7" / "summary.json"
RESULTS = ROOT / "results" / "unified"

CAPACITY = int(56 * 0.93)
SCENARIOS = ["lt", "eq", "gt"]
MECHS = ["", "lagrange", "mask", "repair"]
LEARNERS = ["ppo", "dqn", "ddqn"]
ALGOS = [f"{m}_{l}" if m else l for l in LEARNERS for m in MECHS]
SEEDS = [1, 2, 3]
STEPS = 5_000_000
REWARD = "succ_first"
MECH_LABEL = {"": "无约束（对照）", "lagrange": "Lagrangian", "mask": "Maskable", "repair": "Repair"}


def split(a):
    return ("", a) if a in LEARNERS else tuple(a.split("_", 1))


def weight(algo):
    return 4 if algo.endswith("ppo") else 2


def new_exp_id():
    return subprocess.run([PY, "-c", "from shared.paths import new_exp_id; print(new_exp_id())"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def write_manifest(exp_ids):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": "4.3.1.9", "commit": commit, "steps": STEPS, "reward": REWARD,
                               "reward_norm": "none", "scenarios": SCENARIOS, "algos": ALGOS, "seeds": SEEDS,
                               "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST)


def launch(seed, scen, algo, exp_ids):
    exp_id = new_exp_id()
    log = open(LOG_DIR / f"seed{seed}_{scen}_{algo}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(seed),
           "PAPER_VERSION": "4.3.1.9"}
    proc = subprocess.Popen([PY, "-u", "-m", "paper_rl.train", "--scen", scen, "--algo", algo, "--reward", REWARD,
                             "--reward-norm", "none", "--steps", str(STEPS), "--seed", str(seed)],
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
    base = json.loads(BASELINE.read_text())["scenarios"]

    def ms(vals, f):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        return f"{f(np.mean(vals))} ± {f(sd)}" + ("" if len(vals) == 3 else f" (n={len(vals)})")

    def bs(b, k, f):
        return f"{f(b[k][0])} ± {f(b[k][1])}" if b and k in b else "—"

    f3 = lambda v: f"{v:.3f}"
    f4 = lambda v: f"{v:.4f}"
    lines = ["# v4.3.1.9 成功奖励 M(2AR-1) → M(1+2AR)（12 模型 × 3 场景 × 3 种子 × 5M）", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 唯一改动：成功分支 M(2AR-1) → M(1+2AR)（= 旧公式 + 2M，AR 斜率不变，任何成功严格高于任何失败）；"
             "失败分支 -M(1-M_v/M) 不变；其余与 v4.3.1.6 相同（p = 0.6 数据、同一划分与超参、无奖励归一化）。",
             "- 「旧」= v4.3.1.6 模型，取自 v4.3.1.7 重评（最优 AR 由 ILP 求得）；「新」= 本版本。",
             "- 相对最优 AR = 模型 AR ÷ 同一实例的最优 AR（ILP 求得）。「成功回合」只在成功的 episode 上平均；「失败计 0」把失败 episode 记为 0 再平均。均值 ± 样本标准差（3 种子）。",
             "- manifest：`scripts/logs/v4.3.1.9/manifest.json`", ""]
    order = sorted(ALGOS, key=lambda a: (LEARNERS.index(split(a)[1]), MECHS.index(split(a)[0])))
    for s in SCENARIOS:
        lines += [f"## {s.upper()}", "",
                  "| 模型 | 约束处理 | 奖励 | success_rate | 相对最优 AR（成功回合） | 相对最优 AR（失败计 0） | 死局率 |",
                  "|---|---|---|---|---|---|---|"]
        for a in order:
            mech, learner = split(a)
            b = base[s]["models"].get(a)
            lines.append("| " + " | ".join([learner.upper(), MECH_LABEL[mech], "旧", bs(b, "success", f3),
                                             bs(b, "ratio_success", f4), bs(b, "ratio_all", f4),
                                             bs(b, "dead_end", f3)]) + " |")
            rs = [result(s, a, exp_ids.get(str(sd), {}).get(s, {}).get(a, "")) for sd in SEEDS]
            rs = [r for r in rs if r]
            g = lambda k: [r.get(k) for r in rs]
            comb = [(r["ar_ratio_successful"] or 0.0) * r["success_rate"] for r in rs]
            lines.append("| " + " | ".join(["", "", "**新**", ms(g("success_rate"), f3),
                                             ms(g("ar_ratio_successful"), f4), ms(comb, f4),
                                             ms(g("dead_end_rate"), f3)]) + " |")
        lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"report -> {REPORT}", flush=True)


def train_procs():
    """{pid: (EXP_ID, algo)} of every running paper_rl.train process (any campaign)."""
    procs = {}
    for pd in Path("/proc").iterdir():
        if not pd.name.isdigit():
            continue
        try:
            cmd = (pd / "cmdline").read_bytes().split(b"\0")
            if b"paper_rl.train" not in cmd or b"--algo" not in cmd:
                continue
            env = dict(kv.split(b"=", 1) for kv in (pd / "environ").read_bytes().split(b"\0") if b"=" in kv)
            procs[int(pd.name)] = (env.get(b"EXP_ID", b"").decode(), cmd[cmd.index(b"--algo") + 1].decode())
        except OSError:
            continue
    return procs


def machine_load():
    return sum(weight(a) for _, a in train_procs().values())


def main():
    if "--report" in sys.argv:
        write_report(json.loads(MANIFEST.read_text())["exp_ids"])
        return
    exp_ids = {}
    queue = [(sd, s, a) for sd in SEEDS for a in ALGOS for s in SCENARIOS]
    if "--resume" in sys.argv and MANIFEST.exists():
        exp_ids = json.loads(MANIFEST.read_text())["exp_ids"]
        launched = {(int(sd), s, a) for sd, d in exp_ids.items() for s, d2 in d.items() for a in d2}
        queue = [j for j in queue if j not in launched]
        print(f"=== resume: {len(launched)} launched, {len(queue)} to launch ===", flush=True)
    write_manifest(exp_ids)
    print(f"=== v4.3.1.9 start {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(queue)} jobs, "
          f"machine load {machine_load()}/{CAPACITY} ===", flush=True)
    mine = {e for d in exp_ids.values() for d2 in d.values() for e in d2.values()}
    active, failed, done = [], [], 0
    while True:
        load = machine_load()
        i = 0
        while i < len(queue):
            sd, s, a = queue[i]
            if load + weight(a) <= CAPACITY:
                queue.pop(i)
                p, f = launch(sd, s, a, exp_ids)
                mine.add(exp_ids[str(sd)][s][a])
                active.append((p, f, f"seed{sd} {s}/{a}"))
                load += weight(a)
            else:
                i += 1
        still = []
        for p, f, label in active:
            ret = p.poll()
            if ret is None:
                still.append((p, f, label))
                continue
            f.close()
            done += 1
            print(f"[{time.strftime('%H:%M:%S')}] finished {label} ({'OK' if ret == 0 else f'EXIT={ret}'}) "
                  f"— {done} done, {len(queue)} queued", flush=True)
            if ret != 0:
                failed.append(label)
        active = still
        adopted = any(e in mine for e, _ in train_procs().values()) if not active else False
        if not (queue or active or adopted):
            break
        time.sleep(15)
    print(f"=== done {time.strftime('%Y-%m-%d %H:%M:%S')} — {done} completed, {len(failed)} failed ===", flush=True)
    if failed:
        print("FAILED:", failed, flush=True)
    write_report(exp_ids)


if __name__ == "__main__":
    main()
