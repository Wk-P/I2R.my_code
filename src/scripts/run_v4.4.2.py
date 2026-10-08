#!/usr/bin/env python3
"""v4.4.2: ILP-demonstration warm start for Mask PPO (BC -> PPO), otherwise v4.3.8.

Before PPO, the Mask PPO policy is fitted to the ILP optimum of every training instance replayed
in the environment's order (paper_rl/bc.py): multi-label cross-entropy (every legal empty ECU of
the same capacity counts when the expert opens an ECU) plus the value head fitted to the expert
return (= ILP AR, gamma = 1). Then the usual PPO training. Data v4.3.1.4 (p = 0.6), reward
ar_pen, gamma 1, full episode, EXIT, base observation, descending demand, ECU action, PPO
hyper-parameters: all as v4.3.8. --pilot: 3 scenarios x seed 1 x 1M; the report compares
scratch Mask PPO (v4.3.8 pilot), the BC warm start alone, and BC -> PPO, including the
ECU-opening diagnostic (scripts/diag_open_ecu.py).

    .venv/bin/python -m paper_rl.bc v4.3.1.4                  # once: ILP allocations
    nohup .venv/bin/python scripts/run_v4.4.2.py --pilot > scripts/logs/run_v4.4.2_pilot_driver.log 2>&1 &
    .venv/bin/python scripts/run_v4.4.2.py --pilot --report
"""
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
PY = str(ROOT / ".venv" / "bin" / "python")
VERSION = "4.4.2"
DATA = "v4.3.1.4"
LOG_DIR = ROOT / "scripts" / "logs" / f"v{VERSION}"
MANIFEST = LOG_DIR / "manifest.json"
REPORT = ROOT / "paper_contents" / f"v{VERSION}" / "report.md"
RESULTS = ROOT / "results" / "unified"
SCRATCH = ROOT / "scripts" / "logs" / "v4.3.8_pilot" / "manifest.json"     # scratch Mask PPO, same setting

CAPACITY = int(56 * 0.93)
SCENARIOS = ["lt", "eq", "gt"]
ALGO = "mask_ppo"
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


def new_exp_id():
    return subprocess.run([PY, "-c", "from shared.paths import new_exp_id; print(new_exp_id())"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def write_manifest(exp_ids):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": VERSION, "commit": commit, "data": DATA, "bc": True, "steps": STEPS,
                               "reward": REWARD, "reward_norm": NORM, "obs": OBS, "gamma": GAMMA, "full_episode": True,
                               "exit_action": True, "scenarios": SCENARIOS, "algos": [ALGO], "seeds": SEEDS,
                               "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST)


def launch(seed, scen, exp_ids):
    exp_id = new_exp_id()
    log = open(LOG_DIR / f"seed{seed}_{scen}_{ALGO}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(seed),
           "PAPER_VERSION": VERSION, "DATA_VERSION": DATA}
    proc = subprocess.Popen([PY, "-u", "-m", "paper_rl.train", "--scen", scen, "--algo", ALGO, "--reward", REWARD,
                             "--reward-norm", NORM, "--obs", OBS, "--gamma", str(GAMMA), "--full-episode", "--exit-action",
                             "--bc", "--steps", str(STEPS), "--seed", str(seed)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    exp_ids.setdefault(str(seed), {}).setdefault(scen, {})[ALGO] = exp_id
    write_manifest(exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched seed{seed} {scen} exp_id={exp_id} pid={proc.pid}", flush=True)
    return proc, log


def write_report(exp_ids):
    from diag_open_ecu import replay
    scratch = json.loads(SCRATCH.read_text())["exp_ids"]
    kinds = [("从零训练 Mask PPO（v4.3.8 试跑）", "model_*", scratch), ("只做 BC（ILP 示范）", "bc_only_*", exp_ids),
             ("BC → Mask PPO", "model_*", exp_ids)]
    jobs = [(lb, DATA, s, ALGO, ids[str(sd)][s][ALGO], pat) for lb, pat, ids in kinds for sd in SEEDS for s in SCENARIOS
            if ids.get(str(sd), {}).get(s, {}).get(ALGO)]
    with Pool(min(len(jobs), 24)) as pool:
        out = pool.map(replay, jobs)
    res = {}
    for job, opens, insts, n in out:
        lb, _, s = job[:3]
        exit_r = 1 - len(insts) / n
        ar = float(np.mean([x["ar"] for x in insts])) if insts else None
        ilp = float(np.mean([x["ilp"] for x in insts])) if insts else None
        res.setdefault((lb, s), []).append({
            "exit": exit_r, "ar": ar, "ilp": ilp, "gap": None if ar is None else ilp - ar,
            "rel": None if ar is None else (ilp - ar) / ilp,
            "act": float(np.mean([x["act"] for x in insts])) if insts else None,
            "ilp_act": float(np.mean([x["ilp_act"] for x in insts])) if insts else None,
            "forced": sum(o["kind"] == "forced" for o in opens) / n,
            "vol": sum(o["kind"] == "voluntary" for o in opens) / n})

    def ms(vals, f):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        return f(np.mean(vals)) + (f" ± {f(np.std(vals, ddof=1))}" if len(vals) > 1 else "")

    f4 = lambda v: f"{v:.4f}"
    f2 = lambda v: f"{v:.2f}"
    pc = lambda v: f"{100 * v:.1f}%"
    lines = [f"# v{VERSION} ILP 示范预训练（BC → Mask PPO，其余同 v4.3.8）", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 本版本改动：PPO 训练前，先用每个训练实例的 ILP 最优分配（Dinkelbach）按需求降序回放得到专家轨迹，做行为克隆："
             "策略用多标签交叉熵（专家开一个空 ECU 时，所有同容量、合法的空 ECU 都算正确标签），价值头拟合专家回报（γ = 1 时即 ILP AR）。"
             "之后照常 PPO 训练。数据 v4.3.1.4（p = 0.6）、奖励 ar_pen、γ = 1、不提前结束、EXIT、原观测、需求降序、只选 ECU、PPO 超参均与 v4.3.8 相同。",
             "- 三行：从零训练 = v4.3.8 试跑的 Mask PPO；只做 BC = 预训练后、PPO 之前的模型；BC → Mask PPO = 本版本最终模型。",
             "- AR、ILP AR 在未 EXIT 的同一批测试实例上平均；绝对 gap = ILP AR − AR，相对 gap = (ILP AR − AR) / ILP AR。Mask 违约率恒为 0。",
             "- 开启 ECU 数 / ILP 开启数在未 EXIT 的实例上平均；被逼开启、主动开启为每个测试实例的平均次数（定义见 `paper_contents/v4.4.1/open_diag.md`）。",
             f"- manifest：`{MANIFEST.relative_to(ROOT)}`"]
    lines.append("- **试跑**：Mask PPO × 3 场景 × 种子 1 × 1M 步。" if PILOT else "- 均值 ± 样本标准差（3 种子）。")
    lines.append("")
    for s in SCENARIOS:
        lines += [f"## {s.upper()}", "",
                  "| 模型 | EXIT 率 | AR | ILP AR | 绝对 gap | 相对 gap | 开启 ECU 数 | ILP 开启数 | 被逼开启/实例 | 主动开启/实例 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for lb, _, _ in kinds:
            st = res.get((lb, s), [])
            if st:
                lines.append("| " + " | ".join([lb, ms([x["exit"] for x in st], pc), ms([x["ar"] for x in st], f4),
                                                  ms([x["ilp"] for x in st], f4), ms([x["gap"] for x in st], f4),
                                                  ms([x["rel"] for x in st], pc), ms([x["act"] for x in st], f2),
                                                  ms([x["ilp_act"] for x in st], f2), ms([x["forced"] for x in st], f2),
                                                  ms([x["vol"] for x in st], f2)]) + " |")
        bc = [json.loads((RESULTS / s / ALGO / exp_ids[str(sd)][s][ALGO] / "results.json").read_text())["bc"]
              for sd in SEEDS if exp_ids.get(str(sd), {}).get(s, {}).get(ALGO)]
        if bc:
            lines += ["", "BC 预训练：" + "；".join(
                f"样本 {b['bc_samples']}，{b['bc_epochs']} 轮，训练集动作准确率 {pc(b['bc_train_acc'])}，"
                f"损失 {b['bc_loss_first']:.3f} → {b['bc_loss_last']:.3f}，价值 MSE {b['bc_value_mse']:.4f}，耗时 {b['bc_seconds']:.0f} s"
                for b in bc)]
        lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"report -> {REPORT}", flush=True)


def main():
    if "--report" in sys.argv:
        write_report(json.loads(MANIFEST.read_text())["exp_ids"])
        return
    exp_ids = {}
    queue = [(sd, s) for sd in SEEDS for s in SCENARIOS]
    write_manifest(exp_ids)
    print(f"=== v{VERSION} start {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(queue)} jobs ===", flush=True)
    active, failed, done = [], [], 0
    while queue or active:
        while queue and 4 * (len(active) + 1) <= CAPACITY:
            sd, s = queue.pop(0)
            p, f = launch(sd, s, exp_ids)
            active.append((p, f, f"seed{sd} {s}"))
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
        if queue or active:
            time.sleep(15)
    print(f"=== done {time.strftime('%Y-%m-%d %H:%M:%S')} — {done} completed, {len(failed)} failed ===", flush=True)
    if failed:
        print("FAILED:", failed, flush=True)
    write_report(exp_ids)


if __name__ == "__main__":
    main()
