#!/usr/bin/env python3
"""v4.4.7: service ordering ablation -- descending / ascending / random demand order (3 seeds x 3 scenarios x 1M, GPU).

Network and PPO = the final configuration (v4.4.5 baseline: structure-aware net without sigma_util,
lr 3e-4, ent 0.005, GAE lambda 0.95, ar_pen, gamma 1, EXIT, GPU, 1M). Only the placement order changes
(paper_rl.env.service_order): desc (as before; the v4.4.5 GPU baseline runs of seeds 1-3 are reused),
asc (stable ascending demand), random (one fixed permutation per instance, seeded by the instance content
and order_seed 0 -- not reshuffled during training; the same for every training seed).
18 new jobs = 2 orders x 3 seeds x 3 scenarios, one GPU per scenario.
Report: mean +- SD over seeds of relative gap, steps 1-5 regret / optimal-action rate, step-1 optimal
rate, EXIT, active ECUs - ILP, forced openings; every seed listed; each run evaluated on the test split
of its own seed, in its own order.

    nohup .venv/bin/python src/scripts/run_v4.4.7.py > logs/run_v4.4.7_driver.log 2>&1 &
    .venv/bin/python src/scripts/run_v4.4.7.py --report
"""
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))
PY = str(ROOT / ".venv" / "bin" / "python")
PY_GPU = str(ROOT / ".venv-gpu" / "bin" / "python")
VERSION = "4.4.7"
DATA = "v4.3.1.4"
LOG_DIR = ROOT / "logs" / f"v{VERSION}"
MANIFEST = LOG_DIR / "manifest.json"
BASE = {1: ROOT / "logs" / "v4.4.5_pilot" / "manifest.json", 2: ROOT / "logs" / "v4.4.5_seeds" / "manifest.json",
        3: ROOT / "logs" / "v4.4.5_seeds" / "manifest.json"}          # v4.4.5 GPU baseline, key mask_ppo_base
PREV = ROOT / "paper_contents" / "v4.4.5"                                  # its regret summaries
OUT = ROOT / "paper_contents" / f"v{VERSION}"
REPORT = OUT / "report.md"

SCENARIOS = ["lt", "eq", "gt"]
ALGO = "mask_ppo"
SEEDS, STEPS = [1, 2, 3], 1_000_000
REWARD, NORM, OBS, GAMMA, NET = "ar_pen", "none", "base", 1.0, "graph"
# manifest key -> (label, extra paper_rl.train args); keys match the pilot manifest
VARIANTS = {"mask_ppo_base": ("需求降序（现行）", None), "mask_ppo_asc": ("需求升序", ["--order", "asc"]),
            "mask_ppo_random": ("随机（每实例固定排列）", ["--order", "random", "--order-seed", "0"])}
NEWS = ["mask_ppo_asc", "mask_ppo_random"]
EARLY = 5
LOG_DIR.mkdir(parents=True, exist_ok=True)


def new_exp_id():
    return subprocess.run([PY, "-c", "from shared.paths import new_exp_id; print(new_exp_id())"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def write_manifest(exp_ids):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": VERSION, "commit": commit, "data": DATA, "net": NET, "obs": "raw", "steps": STEPS,
                               "reward": REWARD, "reward_norm": NORM, "gamma": GAMMA, "gae_lambda": 0.95,
                               "lr": 3e-4, "ent_coef": 0.005, "orders": {"mask_ppo_asc": "asc", "mask_ppo_random": "random"}, "order_seed": 0, "device": "cuda",
                               "full_episode": True, "exit_action": True, "scenarios": SCENARIOS, "algos": [ALGO],
                               "keys": {"mask_ppo_asc": "asc", "mask_ppo_random": "random"}, "seeds": SEEDS,
                               "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST)


def launch(seed, scen, key, exp_ids):
    exp_id = new_exp_id()
    log = open(LOG_DIR / f"seed{seed}_{scen}_{key}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(seed),
           "PAPER_VERSION": VERSION, "DATA_VERSION": DATA, "CUDA_VISIBLE_DEVICES": str(SCENARIOS.index(scen) % 3)}
    proc = subprocess.Popen([PY_GPU, "-u", "-m", "paper_rl.train", "--device", "cuda", "--scen", scen, "--algo", ALGO,
                             "--reward", REWARD, "--reward-norm", NORM, "--obs", OBS, "--gamma", str(GAMMA),
                             *VARIANTS[key][1], "--full-episode", "--exit-action", "--net", NET,
                             "--steps", str(STEPS), "--seed", str(seed)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    exp_ids.setdefault(str(seed), {}).setdefault(scen, {})[key] = exp_id
    write_manifest(exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched seed{seed} {scen} {key} exp_id={exp_id} pid={proc.pid}", flush=True)
    return proc, log


def regret(seed, key, manifest):
    """diag_regret summary of one (seed, variant) over the 3 scenarios; the baseline reuses v4.4.5's."""
    if key not in NEWS:
        md = PREV / ("regret_base.md" if seed == 1 else f"regret_s{seed}_base.md")
        return json.loads(md.with_suffix(".json").read_text())
    md = OUT / f"regret_s{seed}_{key.removeprefix('mask_ppo_')}.md"
    js = md.with_suffix(".json")
    if not js.exists():
        subprocess.run([PY, str(ROOT / "src" / "scripts" / "diag_regret.py"), "--out", str(md), "--seed", str(seed),
                        "--manifest", str(manifest.relative_to(ROOT)), "--key", key,
                        "--label", f"v{VERSION} 结构感知 Mask PPO {VARIANTS[key][0]}（种子 {seed}、1M、GPU）"],
                       cwd=ROOT, check=True)
    return json.loads(js.read_text())


def write_report(exp_ids):
    from diag_open_ecu import replay
    ids = {str(sd): {s: {"mask_ppo_base": json.loads(BASE[sd].read_text())["exp_ids"][str(sd)][s]["mask_ppo_base"],
                         **{k: exp_ids[str(sd)][s][k] for k in NEWS}} for s in SCENARIOS} for sd in SEEDS}
    seeds = SEEDS
    man = {sd: MANIFEST for sd in SEEDS}
    jobs = [(key, DATA, s, ALGO, ids[str(sd)][s][key], "model_*", sd) for sd in seeds for key in VARIANTS for s in SCENARIOS]
    with Pool(len(jobs)) as pool:
        out = pool.map(replay, jobs)
    res = {}
    for job, opens, insts, n in out:
        key, s, sd = job[0], job[2], job[6]
        ar, ilp = np.mean([x["ar"] for x in insts]), np.mean([x["ilp"] for x in insts])
        res[(key, s, sd)] = {"exit": 1 - len(insts) / n, "rel": (ilp - ar) / ilp,
                             "dact": np.mean([x["act"] - x["ilp_act"] for x in insts]),
                             "forced": sum(o["kind"] == "forced" for o in opens) / n}
    for sd in seeds:
        for key in VARIANTS:
            g = regret(sd, key, man[sd])
            for s in SCENARIOS:
                res[(key, s, sd)].update({"r15": sum(g[s]["regret_t"][:EARLY]), "p15": float(np.mean(g[s]["opt_rate_t"][:EARLY])),
                                          "p1": g[s]["opt_rate_t"][0]})

    def ms(vals, f):
        return f(np.mean(vals)) + (f" ± {f(np.std(vals, ddof=1))}" if len(vals) > 1 else "")
    pc = lambda v: f"{100 * v:.1f}%"
    f4 = lambda v: f"{v:.4f}"
    f2 = lambda v: f"{v:+.2f}"
    f2u = lambda v: f"{v:.2f}"
    cols = [("相对 gap", "rel", pc), (f"第 1–{EARLY} 步 regret", "r15", f4), (f"第 1–{EARLY} 步选中最优", "p15", pc),
            ("第 1 步选中最优", "p1", pc), ("EXIT 率", "exit", pc), ("开启 ECU 数 − ILP", "dact", f2), ("被逼开启/实例", "forced", f2u)]
    lines = [f"# v{VERSION} 服务顺序消融：需求降序 / 需求升序 / 随机", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 唯一改动：服务的放置顺序（`--order`，`src/paper_rl/env.service_order`）。降序 = 现行设置（稳定排序）；升序 = 需求从小到大（稳定排序）；"
             "随机 = 每个实例由其内容与 order_seed = 0 固定一个排列，训练中不重新打乱，三个训练种子用同一套排列，测试时也用同一排列。",
             "- 网络与 PPO 为 v4.4.5 定稿配置（结构感知网络，不含 v4.4.6 的 σ_util；lr 3e-4、熵系数 0.005、GAE λ = 0.95、ar_pen 奖励、γ = 1、EXIT、GPU、1M 步）。",
             f"- 降序 = v4.4.5 在 GPU 上训练的三种子基线（种子 1：`logs/v4.4.5_pilot`，种子 2、3：`logs/v4.4.5_seeds`），直接复用；"
             f"升序、随机为本次新训练，manifest：`{MANIFEST.relative_to(ROOT)}`。",
             "- 每个模型都在它自己训练种子的测试集上、按它自己的顺序评估；ILP AR 与顺序无关。相对 gap 先逐种子计算再取平均；表中为三种子均值 ± 样本标准差。",
             f"- 第 1–{EARLY} 步 regret 与选中最优的比例来自 `src/scripts/diag_regret.py`（各种子测试集前 400 个实例，按各自顺序）。"
             "注意：不同顺序下「第 t 步」放的是不同的服务，逐步指标只在同一顺序内可比，跨顺序以相对 gap 为准。", ""]
    lines += ["## 三种子汇总", "", "| 场景 | 模型 | " + " | ".join(c[0] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for s in SCENARIOS:
        for key, (lb, _) in VARIANTS.items():
            lines.append(f"| {s.upper()} | {lb} | " + " | ".join(ms([res[(key, s, sd)][c] for sd in seeds], f) for _, c, f in cols) + " |")
    lines += ["", "三场景平均（每个种子先对三场景取平均，再对种子求均值 ± SD）：", "",
              "| 模型 | 相对 gap | " + f"第 1–{EARLY} 步 regret | 第 1–{EARLY} 步选中最优 | EXIT 率 |", "|---|---|---|---|---|"]
    for key, (lb, _) in VARIANTS.items():
        avg = lambda c: [np.mean([res[(key, s, sd)][c] for s in SCENARIOS]) for sd in seeds]
        lines.append(f"| {lb} | {ms(avg('rel'), pc)} | {ms(avg('r15'), f4)} | {ms(avg('p15'), pc)} | {ms(avg('exit'), pc)} |")
    lines += ["", "## 逐种子", "", "| 场景 | 种子 | 模型 | " + " | ".join(c[0] for c in cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for s in SCENARIOS:
        for sd in seeds:
            for key, (lb, _) in VARIANTS.items():
                r = res[(key, s, sd)]
                lines.append(f"| {s.upper()} | {sd} | {lb} | " + " | ".join(f(r[c]) for _, c, f in cols) + " |")
    lines.append("")
    REPORT.write_text("\n".join(lines))
    print(f"report -> {REPORT}", flush=True)


def main():
    if "--report" in sys.argv:
        write_report(json.loads(MANIFEST.read_text())["exp_ids"])
        return
    exp_ids = {}
    write_manifest(exp_ids)
    print(f"=== v{VERSION} seeds {SEEDS} start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    active = [(*launch(sd, s, k, exp_ids), f"seed{sd} {s} {k}") for sd in SEEDS for k in NEWS for s in SCENARIOS]
    failed = []
    for p, f, label in active:
        ret = p.wait()
        f.close()
        print(f"[{time.strftime('%H:%M:%S')}] finished {label} ({'OK' if ret == 0 else f'EXIT={ret}'})", flush=True)
        if ret != 0:
            failed.append(label)
    print(f"=== done {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(failed)} failed ===", flush=True)
    if failed:
        print("FAILED:", failed, flush=True)
        return
    write_report(exp_ids)


if __name__ == "__main__":
    main()
