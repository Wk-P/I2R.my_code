#!/usr/bin/env python3
"""v4.3.1.10: v4.3.1.6 with 10M training steps instead of 5M (only change).

12 models ({PPO, DQN, DDQN} x {none, Lagrangian, Maskable, Repair}) x LT/EQ/GT x seeds
{1, 2, 3} x 10M = 108 jobs. Legacy reward, no reward normalisation, p = 0.6 data,
same splits / hyper-parameters as v4.3.1.6. DQN's epsilon schedule is a fraction
of the total steps, so it now decays over 5M steps instead of 2.5M.
The report compares with the 5M models (v4.3.1.7 re-evaluation,
paper_contents/v4.3.1/v4.3.1.7/summary.json) and the greedy baseline (src/paper_rl/greedy.py).

    nohup .venv/bin/python src/scripts/run_v4.3.1.10.py > logs/run_v4.3.1.10_driver.log 2>&1 &
    nohup .venv/bin/python src/scripts/run_v4.3.1.10.py --resume >> logs/run_v4.3.1.10_driver.log 2>&1 &
    .venv/bin/python src/scripts/run_v4.3.1.10.py --report
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
PY = str(ROOT / ".venv" / "bin" / "python")
VERSION = "4.3.1.10"
LOG_DIR = ROOT / "logs" / f"v{VERSION}"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = LOG_DIR / "manifest.json"
REPORT = ROOT / "paper_contents" / "v4.3.1" / f"v{VERSION}" / "report.md"
BASELINE = ROOT / "paper_contents" / "v4.3.1" / "v4.3.1.7" / "summary.json"
RESULTS = ROOT / "results" / "unified"

CAPACITY = int(56 * 0.93)
SCENARIOS = ["lt", "eq", "gt"]
MECHS = ["", "lagrange", "mask", "repair"]
LEARNERS = ["ppo", "dqn", "ddqn"]
ALGOS = [f"{m}_{l}" if m else l for l in LEARNERS for m in MECHS]
WEIGHT = {a: (4 if a.endswith("ppo") else 2) for a in ALGOS}      # as in v4.3.1.6
SEEDS = [1, 2, 3]
STEPS = 10_000_000
REWARD, NORM = "legacy", "none"
MECH_LABEL = {"": "无约束（对照）", "lagrange": "Lagrangian", "mask": "Maskable", "repair": "Repair"}


def split(a):
    return ("", a) if a in LEARNERS else tuple(a.split("_", 1))


def new_exp_id():
    return subprocess.run([PY, "-c", "from shared.paths import new_exp_id; print(new_exp_id())"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def write_manifest(exp_ids):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": VERSION, "commit": commit, "steps": STEPS, "reward": REWARD,
                               "reward_norm": NORM, "scenarios": SCENARIOS, "algos": ALGOS, "seeds": SEEDS,
                               "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST)


def launch(seed, scen, algo, exp_ids):
    exp_id = new_exp_id()
    log = open(LOG_DIR / f"seed{seed}_{scen}_{algo}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(seed),
           "PAPER_VERSION": VERSION}
    proc = subprocess.Popen([PY, "-u", "-m", "paper_rl.train", "--scen", scen, "--algo", algo, "--reward", REWARD,
                             "--reward-norm", NORM, "--steps", str(STEPS), "--seed", str(seed)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    exp_ids.setdefault(str(seed), {}).setdefault(scen, {})[algo] = exp_id
    write_manifest(exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched seed{seed} {scen}/{algo} exp_id={exp_id} pid={proc.pid}", flush=True)
    return proc, log


def eval_run(job):
    """Deterministic re-evaluation of one saved model, per test instance (same as v4.3.1.7)."""
    import torch
    torch.set_num_threads(1)
    from paper_rl.train import evaluate, model_class, split_algo, split_instances
    scen, algo, seed, exp_id = job
    run_dir = RESULTS / scen / algo / exp_id
    if not (run_dir / "results.json").exists():
        return job, None
    mech, learner = split_algo(algo)
    model = model_class(learner, mech).load(str(next(run_dir.glob("model_*"))), device="cpu")
    lam = json.loads((run_dir / "results.json").read_text())["training"].get("final_lambda", 0.0)
    _, _, test = split_instances(scen, seed)
    return job, evaluate(model, test, mech, REWARD, lam, learner)


def stats(ev):
    """success rate, AR / ILP AR over the same successful instances, AR gap, dead-end rate."""
    ok = [e for e in ev if e["success"]]
    ar = float(np.mean([e["ar"] for e in ok])) if ok else None
    ilp = float(np.mean([e["ar_star"] for e in ok])) if ok else None
    return {"success": len(ok) / len(ev), "ar": ar, "ilp_ar": ilp,
            "gap": None if ar is None else ilp - ar, "dead_end": float(np.mean([e["dead_end"] for e in ev]))}


def write_report(exp_ids):
    from multiprocessing import Pool
    from paper_rl.greedy import greedy_action
    from paper_rl.env import PlacementEnv
    from paper_rl.train import split_instances
    base = json.loads(BASELINE.read_text())["scenarios"]
    jobs = [(s, a, sd, exp_ids[str(sd)][s][a]) for sd in SEEDS for s in SCENARIOS for a in ALGOS
            if exp_ids.get(str(sd), {}).get(s, {}).get(a)]
    with Pool(16) as pool:
        got = {(s, a, sd): stats(ev) for (s, a, sd, _), ev in pool.map(eval_run, jobs, chunksize=1) if ev}

    def greedy_stats(scen, seed):
        _, _, test = split_instances(scen, seed)
        env, ev = PlacementEnv(test, "mask", REWARD), []
        for k in range(len(test)):
            env.use_instance(k)
            env.reset()
            done = False
            while not done:
                _, _, done, _, info = env.step(greedy_action(env))
            ev.append({"success": info["valid_placed"] == env.M, "ar": info["ar"], "ar_star": test[k]["ar_star"],
                       "dead_end": info["dead_end"]})
        return stats(ev)

    def ms(vals, f):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        return f"{f(np.mean(vals))} ± {f(sd)}" + ("" if len(vals) == 3 else f" (n={len(vals)})")

    def row(label, mech, steps_, st):
        return "| " + " | ".join([label, mech, steps_, ms([x["success"] for x in st], pc),
                                  ms([x["ar"] for x in st], f4), ms([x["ilp_ar"] for x in st], f4),
                                  ms([x["gap"] for x in st], f4), ms([x["dead_end"] for x in st], pc)]) + " |"

    f4 = lambda v: f"{v:.4f}"
    pc = lambda v: f"{100 * v:.1f}%"
    lines = [f"# v{VERSION} 训练步数 5M → 10M", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 唯一改动：训练步数 10M（DQN 的 ε 衰减随之变为前 5M 步）；其余与 v4.3.1.6 相同"
             "（legacy 奖励、不归一化、p = 0.6 数据、同一划分与超参、3 种子）。",
             "- 「5M」= v4.3.1.6 的模型，取自 v4.3.1.7 重评；「10M」= 本版本，生成报告时重新评估；「贪心」= `src/paper_rl/greedy.py`"
             "（可行 ECU 中先选已开的，再选需求/容量最大的；不违规，无可行 ECU 即失败）。均为每个测试实例 1 次确定性输出。",
             "- AR 与 ILP AR 都只在该方法成功的测试实例上平均（同一批实例）；AR gap = ILP AR − AR。"
             "均值 ± 样本标准差（3 种子的测试集）。",
             f"- manifest：`logs/v{VERSION}/manifest.json`", ""]
    order = sorted(ALGOS, key=lambda a: (LEARNERS.index(split(a)[1]), MECHS.index(split(a)[0])))
    for s in SCENARIOS:
        lines += [f"## {s.upper()}", "",
                  "| 模型 | 约束处理 | 步数 | 成功率 | AR | ILP AR | AR gap | 死局率 |",
                  "|---|---|---|---|---|---|---|---|",
                  row("贪心", "—", "—", [greedy_stats(s, sd) for sd in SEEDS])]
        for a in order:
            mech, learner = split(a)
            b = base[s]["models"].get(a)
            if b:
                ar, gap = b["ar_success"], b["gap_success"]
                lines.append("| " + " | ".join([learner.upper(), MECH_LABEL[mech], "5M",
                                                 f"{pc(b['success'][0])} ± {pc(b['success'][1])}",
                                                 f"{f4(ar[0])} ± {f4(ar[1])}", f4(ar[0] + gap[0]),
                                                 f"{f4(gap[0])} ± {f4(gap[1])}",
                                                 f"{pc(b['dead_end'][0])} ± {pc(b['dead_end'][1])}"]) + " |")
            lines.append(row("", "", "**10M**", [got[(s, a, sd)] for sd in SEEDS if (s, a, sd) in got]))
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
    # PPO first: they are the longest jobs, so the DQN ones fill in the tail.
    queue = [(sd, s, a) for sd in SEEDS for a in ALGOS for s in SCENARIOS]
    queue.sort(key=lambda j: not j[2].endswith("ppo"))
    adopted = []
    if "--resume" in sys.argv and MANIFEST.exists():
        exp_ids = json.loads(MANIFEST.read_text())["exp_ids"]
        launched = {(int(sd), s, a): e for sd, d in exp_ids.items() for s, d2 in d.items() for a, e in d2.items()}
        queue = [j for j in queue if j not in launched]
        live = live_pids()
        adopted = [(live[e], WEIGHT[a]) for (sd, s, a), e in launched.items() if e in live]
        print(f"=== resume: {len(launched)} launched, {len(adopted)} running, {len(queue)} to launch ===", flush=True)
    write_manifest(exp_ids)
    print(f"=== v{VERSION} start {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(queue)} jobs ===", flush=True)
    active, failed, done = [], [], 0
    while queue or active or adopted:
        adopted = [(pid, w) for pid, w in adopted if Path(f"/proc/{pid}").exists()]
        load = sum(w for _, _, _, w in active) + sum(w for _, w in adopted)
        while queue and load + WEIGHT[queue[0][2]] <= CAPACITY:
            sd, s, a = queue.pop(0)
            p, f = launch(sd, s, a, exp_ids)
            active.append((p, f, f"seed{sd} {s}/{a}", WEIGHT[a]))
            load += WEIGHT[a]
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
    print(f"=== done {time.strftime('%Y-%m-%d %H:%M:%S')} — {done} completed, {len(failed)} failed ===", flush=True)
    if failed:
        print("FAILED:", failed, flush=True)
    write_report(exp_ids)


if __name__ == "__main__":
    main()
