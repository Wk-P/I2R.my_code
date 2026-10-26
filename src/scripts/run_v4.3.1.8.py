#!/usr/bin/env python3
"""v4.3.1.8: DQN / DDQN with the reward normalised by M (--reward-norm m).

8 models ({DQN, DDQN} x {none, Lagrangian, Maskable, Repair}) x LT/EQ/GT x seeds
{1, 2, 3} x 5M = 72 jobs. Everything else is v4.3.1.6 (legacy reward, p = 0.6
data, same splits / hyper-parameters); the only change is that the learner sees
reward / M, so the terminal reward lies in [-1, 1] instead of [-M, M].
Baseline = the same models without normalisation, re-evaluated against the true
AR* in v4.3.1.7 (paper_contents/v4.3.1/v4.3.1.7/summary.json).

    nohup .venv/bin/python src/scripts/run_v4.3.1.8.py > logs/run_v4.3.1.8_driver.log 2>&1 &
    nohup .venv/bin/python src/scripts/run_v4.3.1.8.py --resume >> logs/run_v4.3.1.8_driver.log 2>&1 &
    .venv/bin/python src/scripts/run_v4.3.1.8.py --report
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
LOG_DIR = ROOT / "logs" / "v4.3.1.8"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = LOG_DIR / "manifest.json"
REPORT = ROOT / "paper_contents" / "v4.3.1" / "v4.3.1.8" / "report.md"
BASELINE = ROOT / "paper_contents" / "v4.3.1" / "v4.3.1.7" / "summary.json"
RESULTS = ROOT / "results" / "unified"

CAPACITY = int(56 * 0.93)
WEIGHT = 2                                  # DQN_TORCH_THREADS
SCENARIOS = ["lt", "eq", "gt"]
MECHS = ["", "lagrange", "mask", "repair"]
LEARNERS = ["dqn", "ddqn"]
ALGOS = [f"{m}_{l}" if m else l for l in LEARNERS for m in MECHS]
SEEDS = [1, 2, 3]
STEPS = 5_000_000
REWARD, NORM = "legacy", "m"
MECH_LABEL = {"": "无约束（对照）", "lagrange": "Lagrangian", "mask": "Maskable", "repair": "Repair"}


def split(a):
    return ("", a) if a in LEARNERS else tuple(a.split("_", 1))


def new_exp_id():
    return subprocess.run([PY, "-c", "from shared.paths import new_exp_id; print(new_exp_id())"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def write_manifest(exp_ids):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": "4.3.1.8", "commit": commit, "steps": STEPS, "reward": REWARD,
                               "reward_norm": NORM, "scenarios": SCENARIOS, "algos": ALGOS, "seeds": SEEDS,
                               "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST)


def launch(seed, scen, algo, exp_ids):
    exp_id = new_exp_id()
    log = open(LOG_DIR / f"seed{seed}_{scen}_{algo}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(seed),
           "PAPER_VERSION": "4.3.1.8"}
    proc = subprocess.Popen([PY, "-u", "-m", "paper_rl.train", "--scen", scen, "--algo", algo, "--reward", REWARD,
                             "--reward-norm", NORM, "--steps", str(STEPS), "--seed", str(seed)],
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
    lines = ["# v4.3.1.8 DQN / DDQN 奖励归一化（reward / M）", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 唯一改动：学习器看到的奖励除以 M（终止奖励从 [-M, M] 缩到 [-1, 1]，Lagrangian 的 λ 代价同比缩放）；"
             "其余与 v4.3.1.6 相同（legacy 奖励、p = 0.6 数据、同一划分与超参、3 种子 × 5M）。",
             "- 「原始」= 未归一化的同一模型，取自 v4.3.1.7 重评（最优 AR 由 ILP 求得）；「归一化」= 本版本。",
             "- 相对最优 AR = 模型 AR ÷ 同一实例的最优 AR（ILP 求得）。「成功回合」只在成功的 episode 上平均；「失败计 0」把失败 episode 记为 0 再平均。均值 ± 样本标准差（3 种子）。",
             "- manifest：`logs/v4.3.1.8/manifest.json`", ""]
    order = sorted(ALGOS, key=lambda a: (LEARNERS.index(split(a)[1]), MECHS.index(split(a)[0])))
    for s in SCENARIOS:
        lines += [f"## {s.upper()}", "",
                  "| 模型 | 约束处理 | 奖励 | success_rate | 相对最优 AR（成功回合） | 相对最优 AR（失败计 0） | 死局率 |",
                  "|---|---|---|---|---|---|---|"]
        for a in order:
            mech, learner = split(a)
            b = base[s]["models"].get(a)
            lines.append("| " + " | ".join([learner.upper(), MECH_LABEL[mech], "原始", bs(b, "success", f3),
                                             bs(b, "ratio_success", f4), bs(b, "ratio_all", f4),
                                             bs(b, "dead_end", f3)]) + " |")
            rs = [result(s, a, exp_ids.get(str(sd), {}).get(s, {}).get(a, "")) for sd in SEEDS]
            rs = [r for r in rs if r]
            g = lambda k: [r.get(k) for r in rs]
            comb = [(r["ar_ratio_successful"] or 0.0) * r["success_rate"] for r in rs]
            lines.append("| " + " | ".join(["", "", "**归一化**", ms(g("success_rate"), f3),
                                             ms(g("ar_ratio_successful"), f4), ms(comb, f4),
                                             ms(g("dead_end_rate"), f3)]) + " |")
        lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
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
    if "--report" in sys.argv:
        write_report(json.loads(MANIFEST.read_text())["exp_ids"])
        return
    exp_ids = {}
    queue = [(sd, s, a) for sd in SEEDS for a in ALGOS for s in SCENARIOS]
    adopted = []
    if "--resume" in sys.argv and MANIFEST.exists():
        exp_ids = json.loads(MANIFEST.read_text())["exp_ids"]
        launched = {(int(sd), s, a): e for sd, d in exp_ids.items() for s, d2 in d.items() for a, e in d2.items()}
        queue = [j for j in queue if j not in launched]
        live = live_pids()
        adopted = [live[e] for e in launched.values() if e in live]
        print(f"=== resume: {len(launched)} launched, {len(adopted)} running, {len(queue)} to launch ===", flush=True)
    write_manifest(exp_ids)
    print(f"=== v4.3.1.8 start {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(queue)} jobs ===", flush=True)
    active, failed, done = [], [], 0
    while queue or active or adopted:
        adopted = [pid for pid in adopted if Path(f"/proc/{pid}").exists()]
        load = WEIGHT * (len(active) + len(adopted))
        while queue and load + WEIGHT <= CAPACITY:
            sd, s, a = queue.pop(0)
            p, f = launch(sd, s, a, exp_ids)
            active.append((p, f, f"seed{sd} {s}/{a}"))
            load += WEIGHT
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
        if queue or active or adopted:
            time.sleep(15)
    print(f"=== done {time.strftime('%Y-%m-%d %H:%M:%S')} — {done} completed, {len(failed)} failed ===", flush=True)
    if failed:
        print("FAILED:", failed, flush=True)
    write_report(exp_ids)


if __name__ == "__main__":
    main()
