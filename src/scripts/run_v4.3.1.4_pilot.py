#!/usr/bin/env python3
"""v4.3.1.4 pilot on the unified implementation (src/paper_rl/), p = 0.6 data.

12 models x 3 scenarios x reward `objective` x seed 1, 1M steps = 36 jobs,
each `python -m paper_rl.train`. objective = M * AR (all executed placements)
for every model, constraints handled only by the mechanism; dead end =
failure + penalty + termination. (Derived from run_v4.3.1.4_pilot.py.) Results: results/unified/<scen>/<algo>/<exp_id>/;
manifest logs/v4.3.1.4_pilot/manifest.json; report
paper_contents/v4.3.1/v4.3.1.4/pilot_report.md.

Jobs are interleaved across scenarios so partial results cover all three.

    nohup .venv/bin/python src/scripts/run_v4.3.1.4_pilot.py > logs/run_v4.3.1.4_pilot_driver.log 2>&1 &
    # after stopping the driver (training processes keep running):
    nohup .venv/bin/python src/scripts/run_v4.3.1.4_pilot.py --resume >> logs/run_v4.3.1.4_pilot_driver.log 2>&1 &
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
LOG_DIR = ROOT / "logs" / "v4.3.1.4_pilot"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = LOG_DIR / "manifest.json"
REPORT = ROOT / "paper_contents" / "v4.3.1" / "v4.3.1.4" / "pilot_report.md"
RESULTS = ROOT / "results" / "unified"

CAPACITY = int(56 * 0.93)
MODES = ["objective"]
SCENARIOS = ["lt", "eq", "gt"]
MECHS = ["", "lagrange", "mask", "repair"]      # 无约束（对照）, Lagrangian, Maskable, Repair
LEARNERS = ["ppo", "dqn", "ddqn"]
ALGOS = [f"{m}_{l}" if m else l for l in LEARNERS for m in MECHS]
WEIGHT = {a: (4 if a.endswith("ppo") else 2) for a in ALGOS}   # measured cores per job (ps: PPO ~3.9, DQN ~1.4)
SEED = 1
STEPS = 1_000_000
MECH_LABEL = {"": "无约束（对照）", "lagrange": "Lagrangian", "mask": "Maskable", "repair": "Repair"}


def split(algo):
    return ("", algo) if algo in LEARNERS else tuple(algo.split("_", 1))


def new_exp_id() -> str:
    return subprocess.run([PY, "-c", "from shared.paths import new_exp_id; print(new_exp_id())"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def write_manifest(exp_ids):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": "4.3.1.4-pilot", "commit": commit, "branch": "unified",
                               "steps": STEPS, "seed": SEED, "modes": MODES, "scenarios": SCENARIOS,
                               "algos": ALGOS, "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST)


def launch(mode, scen, algo, exp_ids):
    exp_id = new_exp_id()
    log = open(LOG_DIR / f"{mode}_{scen}_{algo}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "REWARD_MODE": mode,
           "TRAIN_SEED": str(SEED), "PAPER_VERSION": "4.3.1.4-pilot"}
    proc = subprocess.Popen([PY, "-u", "-m", "paper_rl.train", "--scen", scen, "--algo", algo,
                             "--reward", mode, "--steps", str(STEPS), "--seed", str(SEED)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    exp_ids.setdefault(mode, {}).setdefault(scen, {})[algo] = exp_id
    write_manifest(exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched {mode} {scen}/{algo} exp_id={exp_id} pid={proc.pid}", flush=True)
    return proc, log


def row(scen, algo, exp_id):
    p = RESULTS / scen / algo / exp_id / "summary.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    return rows[1] if len(rows) > 1 else None


def write_report(exp_ids):
    mode = MODES[0]

    def cell(s, a, k, f):
        r = row(s, a, exp_ids.get(mode, {}).get(s, {}).get(a, ""))
        return f(r[k]) if r and r[k] not in ("", "None") else "—"
    f3 = lambda v: f"{float(v):.3f}"
    f4 = lambda v: f"{float(v):.4f}"
    lines = ["# v4.3.1.4 试跑报告（统一实现，p = 0.6，objective 奖励；12 个模型，1M 步，种子 1）", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 奖励（objective）：放完 M 个服务时 M·AR（AR 计入所有执行的放置，不惩罚违规）；"
             "Lagrangian 每步另减 λ·c_t；Maskable / Repair 死局判失败，−M(1−valid/M) 并终止。",
             "- 测试集 400 个实例，单次确定性评估。success_rate = M 个服务全部合法放置的比例；"
             "AR = average resource utilization（只计合法放置），全部测试 episode 的均值；"
             "AR/AR* = 成功 episode 上 AR 与 ILP 最优值之比的均值；违规率 = 出现 ≥1 次违规的 episode 比例；死局率 = 因死局终止的 episode 比例。",
             f"- manifest：`logs/v4.3.1.4_pilot/manifest.json`", ""]
    order = sorted(ALGOS, key=lambda a: (LEARNERS.index(split(a)[1]), MECHS.index(split(a)[0])))
    for s in SCENARIOS:
        lines += [f"## {s.upper()}", "",
                  "| 模型 | 约束处理 | success_rate | AR | AR/AR* | 容量违规率 | 冲突违规率 | 死局率 |",
                  "|---|---|---|---|---|---|---|---|"]
        for a in order:
            mech, learner = split(a)
            lines.append("| " + " | ".join([learner.upper(), MECH_LABEL[mech],
                cell(s, a, "success_rate", f3), cell(s, a, "ar_mean", f4), cell(s, a, "ar_ratio_successful", f4),
                cell(s, a, "cap_viol_rate", f3), cell(s, a, "conflict_viol_rate", f3), cell(s, a, "dead_end_rate", f3)]) + " |")
        lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"report -> {REPORT}", flush=True)


def live_pids_by_exp():
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
    # interleave scenarios: lt, eq, gt for each (mode, algo)
    queue = [(m, s, a) for a in ALGOS for m in MODES for s in SCENARIOS]
    exp_ids, adopted = {}, []
    if "--resume" in sys.argv and MANIFEST.exists():
        exp_ids = json.loads(MANIFEST.read_text())["exp_ids"]
        launched = {(m, s, a): e for m, d in exp_ids.items() for s, d2 in d.items() for a, e in d2.items()}
        queue = [j for j in queue if j not in launched]
        live = live_pids_by_exp()
        adopted = [(live[e], f"{m} {s}/{a}", WEIGHT[a]) for (m, s, a), e in launched.items() if e in live]
        print(f"=== resume: {len(launched)} launched, {len(adopted)} running, {len(queue)} to launch ===", flush=True)
    print(f"=== v4.3.1.4 pilot start {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(queue)} jobs ===", flush=True)
    active, failed, done = [], [], 0
    alive = lambda pid: Path(f"/proc/{pid}").exists()
    while queue or active or adopted:
        adopted = [x for x in adopted if alive(x[0])]
        load = sum(x[-1] for x in active) + sum(x[-1] for x in adopted)
        i = 0
        while i < len(queue):
            m, s, a = queue[i]
            if load + WEIGHT[a] <= CAPACITY:
                queue.pop(i)
                p, f = launch(m, s, a, exp_ids)
                active.append((p, f, f"{m} {s}/{a}", WEIGHT[a]))
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
    print(f"=== pilot done {time.strftime('%Y-%m-%d %H:%M:%S')} — {done} completed, {len(failed)} failed ===", flush=True)
    if failed:
        print("FAILED:", failed, flush=True)
    write_report(exp_ids)


if __name__ == "__main__":
    main()
