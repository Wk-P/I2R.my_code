#!/usr/bin/env python3
"""v4.4.1: v4.3.8 on data with a random conflict density per instance.

Only change: the data. data/v4.4.1/ draws p ~ U[0, 1) per instance (ILP-infeasible draws are
redrawn together with p, so every instance is ILP-feasible); ILP AR is the Dinkelbach optimum.
Everything else is v4.3.8: fixed descending-demand order, the agent picks only the ECU, base
observation, reward ar_pen, gamma 1, full episode, Mask with EXIT, 40 envs x 512 steps,
batch 256. Since every instance is ILP-feasible, every EXIT is an instance the policy made
infeasible. --pilot: Mask PPO x 3 scenarios x seed 1 x 1M.
The report gives the overall table and the same metrics per conflict-density bin
(rho = conflict_ratio, measured fraction of conflicting service pairs).

    nohup .venv/bin/python scripts/run_v4.4.1.py --pilot > scripts/logs/run_v4.4.1_pilot_driver.log 2>&1 &
    .venv/bin/python scripts/run_v4.4.1.py --pilot --report
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["DATA_VERSION"] = "v4.4.1"          # before paper_rl.config is imported (also inherited by training)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PY = str(ROOT / ".venv" / "bin" / "python")
VERSION = "4.4.1"
LOG_DIR = ROOT / "scripts" / "logs" / f"v{VERSION}"
MANIFEST = LOG_DIR / "manifest.json"
REPORT = ROOT / "paper_contents" / f"v{VERSION}" / "report.md"
RESULTS = ROOT / "results" / "unified"

CAPACITY = int(56 * 0.93)
SCENARIOS = ["lt", "eq", "gt"]
ALGOS = ["mask_ppo"]
WEIGHT = {"mask_ppo": 4}
SEEDS = [1, 2, 3]
STEPS = 5_000_000
REWARD, NORM, OBS, GAMMA = "ar_pen", "none", "base", 1.0
PILOT = "--pilot" in sys.argv
if PILOT:
    SEEDS, STEPS = [1], 1_000_000
    LOG_DIR = ROOT / "scripts" / "logs" / f"v{VERSION}_pilot"
    MANIFEST = LOG_DIR / "manifest.json"
    REPORT = ROOT / "paper_contents" / f"v{VERSION}" / "pilot_report.md"
LOG_DIR.mkdir(parents=True, exist_ok=True)
BINS = [i / 10 for i in range(11)]


def new_exp_id():
    return subprocess.run([PY, "-c", "from shared.paths import new_exp_id; print(new_exp_id())"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def write_manifest(exp_ids):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": VERSION, "commit": commit, "data": "v4.4.1", "steps": STEPS, "reward": REWARD,
                               "reward_norm": NORM, "obs": OBS, "gamma": GAMMA, "full_episode": True, "exit_action": True,
                               "scenarios": SCENARIOS, "algos": ALGOS, "seeds": SEEDS, "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST)


def launch(seed, scen, algo, exp_ids):
    exp_id = new_exp_id()
    log = open(LOG_DIR / f"seed{seed}_{scen}_{algo}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(seed),
           "PAPER_VERSION": VERSION, "DATA_VERSION": "v4.4.1"}
    proc = subprocess.Popen([PY, "-u", "-m", "paper_rl.train", "--scen", scen, "--algo", algo, "--reward", REWARD,
                             "--reward-norm", NORM, "--obs", OBS, "--gamma", str(GAMMA), "--full-episode", "--exit-action",
                             "--steps", str(STEPS), "--seed", str(seed)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    exp_ids.setdefault(str(seed), {}).setdefault(scen, {})[algo] = exp_id
    write_manifest(exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched seed{seed} {scen}/{algo} exp_id={exp_id} pid={proc.pid}", flush=True)
    return proc, log


def eval_run(job):
    """Deterministic evaluation of one saved model, per test instance; adds each instance's rho."""
    import torch
    torch.set_num_threads(1)
    from paper_rl.train import evaluate, model_class, split_algo, split_instances
    scen, algo, seed, exp_id = job
    run_dir = RESULTS / scen / algo / exp_id
    if not (run_dir / "results.json").exists():
        return job, None
    mech, learner = split_algo(algo)
    model = model_class(learner, mech).load(str(next(run_dir.glob("model_*"))), device="cpu")
    _, _, test = split_instances(scen, seed)
    ev = evaluate(model, test, mech, REWARD, 0.0, learner, OBS, full_episode=True, exit_action=mech == "mask")
    for e, x in zip(ev, test):
        e["rho"] = x["conflict_ratio"]
    return job, ev


def stats(ev):
    """EXIT rate; AR / ILP AR / absolute and relative AR gap over the instances completed without EXIT."""
    ok = [e for e in ev if e["cap_v"] == 0 and e["conf_v"] == 0 and not e["exited"]]
    ar = float(np.mean([e["ar"] for e in ok])) if ok else None
    ilp = float(np.mean([e["ar_star"] for e in ok])) if ok else None
    return {"n": len(ev), "exit": float(np.mean([e["exited"] for e in ev])) if ev else None,
            "viol": float(np.mean([e["cap_v"] > 0 or e["conf_v"] > 0 for e in ev])) if ev else None,
            "ar": ar, "ilp_ar": ilp, "gap": None if ar is None else ilp - ar,
            "rel": None if ar is None else (ilp - ar) / ilp}


def write_report(exp_ids):
    from multiprocessing import Pool
    jobs = [(s, a, sd, exp_ids[str(sd)][s][a]) for sd in SEEDS for s in SCENARIOS for a in ALGOS
            if exp_ids.get(str(sd), {}).get(s, {}).get(a)]
    with Pool(16) as pool:
        evs = {(s, a, sd): ev for (s, a, sd, _), ev in pool.map(eval_run, jobs, chunksize=1) if ev}

    def ms(vals, f):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        return f"{f(np.mean(vals))}" + (f" ± {f(sd)}" if len(vals) > 1 else "")

    f4 = lambda v: f"{v:.4f}"
    pc = lambda v: f"{100 * v:.1f}%"

    def cells(st):
        return [ms([x["exit"] for x in st], pc), ms([x["ar"] for x in st], f4), ms([x["ilp_ar"] for x in st], f4),
                ms([x["gap"] for x in st], f4), ms([x["rel"] for x in st], pc)]

    bin_of = lambda r: min(int(r * 10), 9)
    label = lambda b: f"[{BINS[b]:.1f}, {BINS[b + 1]:.1f}" + ("]" if b == 9 else ")")
    lines = [f"# v{VERSION} 随机冲突密度数据（其余同 v4.3.8）", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 唯一改动：数据。`data/v4.4.1/` 每个实例的冲突密度 p ~ U[0,1)，ILP 不可行的实例连同 p 一起重抽，所以**每个实例都 ILP 可行**；"
             "ILP AR 为 Dinkelbach 最优值。其余与 v4.3.8 完全相同（需求降序、只选 ECU、原观测、ar_pen 奖励、γ = 1、不提前结束、Maskable 带 EXIT、"
             "40 环境 × 512 步、batch 256）。",
             "- 每个实例都 ILP 可行，所以 **EXIT 率 = 策略把一个本可完成的实例做成死局的比例**。Maskable 违约率恒为 0（已核对，表中不列）。",
             "- AR 与 ILP AR 只在未 EXIT 的测试实例上平均（同一批实例）；绝对 gap = ILP AR − AR，相对 gap = (ILP AR − AR) / ILP AR。",
             "- ρ = 实测冲突服务对比例（`conflict_ratio`）。每个测试实例 1 次确定性输出。",
             f"- manifest：`{MANIFEST.relative_to(ROOT)}`"]
    if PILOT:
        lines.append("- **试跑**：Maskable PPO × 3 场景 × 种子 1 × 1M 步。")
    else:
        lines.append("- 均值 ± 样本标准差（3 种子的测试集）。")
    viol = [st["viol"] for ev in evs.values() for st in [stats(ev)]]
    if any(v for v in viol if v):
        lines.append(f"- **注意：出现违约**（{max(viol):.1%}），与 Maskable 永不违约的设计不符，需检查。")
    lines.append("")
    for s in SCENARIOS:
        lines += [f"## {s.upper()}", "",
                  "| 模型 | 测试实例数 | EXIT 率 | AR | ILP AR | 绝对 gap | 相对 gap |", "|---|---|---|---|---|---|---|"]
        for a in ALGOS:
            st = [stats(evs[(s, a, sd)]) for sd in SEEDS if (s, a, sd) in evs]
            lines.append("| " + " | ".join(["Maskable PPO", str(sum(x["n"] for x in st))] + cells(st)) + " |")
        lines += ["", "按冲突密度 ρ 分档：", "",
                  "| 模型 | ρ 档 | 测试实例数 | EXIT 率 | AR | ILP AR | 绝对 gap | 相对 gap |", "|---|---|---|---|---|---|---|---|"]
        for a in ALGOS:
            for b in range(10):
                st = [stats([e for e in evs[(s, a, sd)] if bin_of(e["rho"]) == b]) for sd in SEEDS if (s, a, sd) in evs]
                st = [x for x in st if x["n"]]
                if not st:
                    continue
                lines.append("| " + " | ".join(["Maskable PPO", label(b), str(sum(x["n"] for x in st))] + cells(st)) + " |")
        lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"report -> {REPORT}", flush=True)


def main():
    if "--report" in sys.argv:
        write_report(json.loads(MANIFEST.read_text())["exp_ids"])
        return
    exp_ids = {}
    queue = [(sd, s, a) for sd in SEEDS for a in ALGOS for s in SCENARIOS]
    write_manifest(exp_ids)
    print(f"=== v{VERSION} start {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(queue)} jobs ===", flush=True)
    active, failed, done = [], [], 0
    while queue or active:
        load = sum(w for _, _, _, w in active)
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
        if queue or active:
            time.sleep(15)
    print(f"=== done {time.strftime('%Y-%m-%d %H:%M:%S')} — {done} completed, {len(failed)} failed ===", flush=True)
    if failed:
        print("FAILED:", failed, flush=True)
    write_report(exp_ids)


if __name__ == "__main__":
    main()
