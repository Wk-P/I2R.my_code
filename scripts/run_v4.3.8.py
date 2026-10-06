#!/usr/bin/env python3
"""v4.3.8: v4.3.7 + EXIT action for Mask.

Mask's action space is [ECU_1..ECU_N, EXIT]. EXIT is masked while any ECU is feasible and is
the only valid action when none is, so Mask never violates a constraint; EXIT ends the
episode with the failure reward -(1 - valid/M) and the agent has to learn the earlier
placements that avoid it. Reported for Mask: EXIT rate (its capacity / privacy violation
rates are 0 by construction). All other mechanisms are unchanged from v4.3.7.
--pilot trains only mask_ppo / mask_dqn (3 scenarios x seed 1 x 1M) and reuses the other
four pilot models of v4.3.7, whose training is identical.

v4.3.7 notes:

Every episode places all M services (--full-episode): where no ECU is feasible the agent's
choice is executed and counted as a capacity and/or privacy violation. Reported: capacity
violation rate, privacy violation rate (share of test instances with >= 1 such violation),
and AR / ILP AR / AR gap over the violation-free instances. No success rate, no dead ends.

Only change: the reward. --reward ar_pen: terminal AR in (0, 1] if all M services are
placed feasibly, else -(1 - valid/M) in [-1, 0), intermediate
steps 0, no scaling by M; and
--gamma 1 for every learner, so the return is the terminal reward
(Lagrangian additionally -lambda * sum_t c_t at the end, as before). Observation
(base), service order (descending demand), algorithms, other hyper-parameters,
5M steps, seeds 1-3, data, splits and evaluation are those of v4.3.1.6.
12 models x LT/EQ/GT x 3 seeds = 108 jobs.
The report compares with the same models on the base observation (v4.3.1.6, re-evaluated
in v4.3.1.7, legacy reward) and the greedy baseline; every model is evaluated deterministically,
once per test instance.

    nohup .venv/bin/python scripts/run_v4.3.8.py > scripts/logs/run_v4.3.8_driver.log 2>&1 &
    nohup .venv/bin/python scripts/run_v4.3.8.py --pilot > scripts/logs/run_v4.3.8_pilot_driver.log 2>&1 &
    nohup .venv/bin/python scripts/run_v4.3.8.py --resume >> scripts/logs/run_v4.3.8_driver.log 2>&1 &
    .venv/bin/python scripts/run_v4.3.8.py --report
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PY = str(ROOT / ".venv" / "bin" / "python")
VERSION = "4.3.8"
LOG_DIR = ROOT / "scripts" / "logs" / f"v{VERSION}"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = LOG_DIR / "manifest.json"
REPORT = ROOT / "paper_contents" / f"v{VERSION}" / "report.md"
BASELINE = ROOT / "paper_contents" / "v4.3.1" / "v4.3.1.7" / "summary.json"
RESULTS = ROOT / "results" / "unified"

CAPACITY = int(56 * 0.93)
SCENARIOS = ["lt", "eq", "gt"]
MECHS = ["", "lagrange", "mask", "repair"]
LEARNERS = ["ppo", "dqn", "ddqn"]
ALGOS = [f"{m}_{l}" if m else l for l in LEARNERS for m in MECHS]
WEIGHT = {a: (4 if a.endswith("ppo") else 2) for a in ALGOS}      # as in v4.3.1.6
SEEDS = [1, 2, 3]
STEPS = 5_000_000
REWARD, NORM, OBS, GAMMA = "ar_pen", "none", "base", 1.0
# --pilot (professor's comment 1, quick check): 6 models x 3 scenarios x seed 1 x 1M steps.
PILOT = "--pilot" in sys.argv
if PILOT:
    ALGOS = ["ppo", "lagrange_ppo", "mask_ppo", "repair_ppo", "mask_dqn", "repair_dqn"]
    SEEDS, STEPS = [1], 1_000_000
    REUSE = ROOT / "scripts" / "logs" / "v4.3.7_pilot" / "manifest.json"    # non-Mask models: unchanged
    LOG_DIR = ROOT / "scripts" / "logs" / f"v{VERSION}_pilot"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST = LOG_DIR / "manifest.json"
    REPORT = ROOT / "paper_contents" / f"v{VERSION}" / "pilot_report.md"
TRAIN_ALGOS = [a for a in ALGOS if a.startswith("mask")] if PILOT else ALGOS
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
                               "reward_norm": NORM, "obs": OBS, "gamma": GAMMA, "full_episode": True, "exit_action": True, "scenarios": SCENARIOS, "algos": ALGOS, "seeds": SEEDS,
                               "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST)


def launch(seed, scen, algo, exp_ids):
    exp_id = new_exp_id()
    log = open(LOG_DIR / f"seed{seed}_{scen}_{algo}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(seed),
           "PAPER_VERSION": VERSION}
    proc = subprocess.Popen([PY, "-u", "-m", "paper_rl.train", "--scen", scen, "--algo", algo, "--reward", REWARD,
                             "--reward-norm", NORM, "--obs", OBS, "--gamma", str(GAMMA), "--full-episode", "--exit-action", "--steps", str(STEPS), "--seed", str(seed)],
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
    return job, evaluate(model, test, mech, REWARD, lam, learner, OBS, full_episode=True,
                         exit_action=mech == "mask")


def stats(ev):
    """capacity / privacy violation rates; AR / ILP AR / AR gap over the violation-free instances."""
    ok = [e for e in ev if e["cap_v"] == 0 and e["conf_v"] == 0 and not e.get("exited", False)]
    ar = float(np.mean([e["ar"] for e in ok])) if ok else None
    ilp = float(np.mean([e["ar_star"] for e in ok])) if ok else None
    return {"cap": float(np.mean([e["cap_v"] > 0 for e in ev])), "conf": float(np.mean([e["conf_v"] > 0 for e in ev])),
            "ar": ar, "ilp_ar": ilp, "gap": None if ar is None else ilp - ar,
            "exit": float(np.mean([e.get("exited", False) for e in ev]))}


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
        env, ev = PlacementEnv(test, "mask", REWARD, full_episode=True), []
        for k in range(len(test)):
            env.use_instance(k)
            env.reset()
            done = False
            while not done:
                _, _, done, _, info = env.step(greedy_action(env))
            ev.append({"cap_v": info["capacity_violations"], "conf_v": info["conflict_violations"],
                       "ar": info["ar"], "ar_star": test[k]["ar_star"]})
        return stats(ev)

    def ms(vals, f):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        return f"{f(np.mean(vals))} ± {f(sd)}" + ("" if len(vals) == 3 else f" (n={len(vals)})")

    def row(label, mech, st):
        ex = ms([x["exit"] for x in st], pc) if mech == "Maskable" else "—"
        return "| " + " | ".join([label, mech, ms([x["cap"] for x in st], pc), ms([x["conf"] for x in st], pc), ex,
                                  ms([x["ar"] for x in st], f4), ms([x["ilp_ar"] for x in st], f4),
                                  ms([x["gap"] for x in st], f4)]) + " |"

    f4 = lambda v: f"{v:.4f}"
    pc = lambda v: f"{100 * v:.1f}%"
    lines = [f"# v{VERSION} Maskable 加 EXIT 动作（其余同 v4.3.7）", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 本版本改动：Maskable 的动作为 [ECU1, …, ECUn, EXIT]。有合法 ECU 时 EXIT 被屏蔽；一个合法 ECU 都没有时 EXIT 是唯一合法动作，"
             "选后回合结束，得失败奖励 −(1 − valid/M)。因此 Maskable 永不违约（两种违约率恒为 0），报 EXIT 率 = 选了 EXIT 的测试实例比例。其他机制与 v4.3.7 完全相同。",
             "- 唯一改动：奖励。终局 r = AR ∈ (0, 1]（M 个服务全部合法放置），否则 r = −(1 − valid/M) ∈ [−1, 0)，中间步 0，不乘 M；"
             "所有学习器 γ = 1（Lagrangian 照旧终局减 λΣc_t）；每回合放满 M 个服务，没有可行 ECU 时照样执行并记为 capacity / privacy 违约。其余与 v4.3.1.6 相同（原观测、需求降序、p = 0.6 数据、同一划分与其余超参、5M 步、3 种子）。",
             "- 违约率 = 至少出现一次该类违约的测试实例比例；AR、ILP AR 只在无违约（Maskable：未 EXIT）的测试实例上平均。「AR」= 本版本，生成报告时重新评估；「贪心」= `paper_rl/greedy.py`"
             "（可行 ECU 中先选已开的，再选需求/容量最大的；无可行 ECU 时强放到无 privacy 冲突、剩余容量最大的 ECU，记为违约）。均为每个测试实例 1 次确定性输出。",
             "- AR 与 ILP AR 都只在该方法无违约且未 EXIT 的测试实例上平均（同一批实例）；AR gap = ILP AR − AR。"
             "均值 ± 样本标准差（3 种子的测试集）。",
             f"- manifest：`{MANIFEST.relative_to(ROOT)}`"]
    if PILOT:
        lines.append("- **试跑**：6 个模型 × 种子 1 × 1M 步；只重训 Maskable PPO / DQN，其余 4 个模型取自 v4.3.7 试跑（训练完全相同）。")
    lines.append("")
    order = sorted(ALGOS, key=lambda a: (LEARNERS.index(split(a)[1]), MECHS.index(split(a)[0])))
    for s in SCENARIOS:
        lines += [f"## {s.upper()}", "",
                  "| 模型 | 约束处理 | capacity 违约率 | privacy 违约率 | EXIT 率 | AR | ILP AR | AR gap |",
                  "|---|---|---|---|---|---|---|---|",
                  row("贪心", "—", [greedy_stats(s, sd) for sd in SEEDS])]
        for a in order:
            mech, learner = split(a)
            lines.append(row(learner.upper(), MECH_LABEL[mech], [got[(s, a, sd)] for sd in SEEDS if (s, a, sd) in got]))
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
    if PILOT:
        old = json.loads(REUSE.read_text())["exp_ids"]
        exp_ids = {sd: {sc: {a: e for a, e in d2.items() if a not in TRAIN_ALGOS} for sc, d2 in d.items()}
                   for sd, d in old.items()}
    # PPO first: they are the longest jobs, so the DQN ones fill in the tail.
    queue = [(sd, s, a) for sd in SEEDS for a in TRAIN_ALGOS for s in SCENARIOS]
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
