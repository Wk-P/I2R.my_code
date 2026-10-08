#!/usr/bin/env python3
"""v4.4.6: structure-aware Mask PPO + sigma_util in the global token (3 seeds x 3 scenarios x 1M, GPU).

Single change from the final configuration (v4.4.5 baseline: lr 3e-4, ent 0.005, GAE lambda 0.95,
graph net, GPU, 1M): --glob-std adds sigma_util = std of u_j / c_j over the active ECUs (x 2, in
[0, 1]) to the global token. The global token already carried AR_t and the unplaced share (M - t) / M
since v4.4.3, so this completes progress + objective + distribution quality without duplicating them.
Baseline = the v4.4.5 GPU baseline runs of seeds 1-3 (logs/v4.4.5_pilot seed 1, logs/v4.4.5_seeds
seeds 2-3), reused, not retrained. 9 jobs, one GPU per scenario.
Report: mean +- SD over seeds of relative gap, steps 1-5 regret / optimal-action rate, step-1
optimal rate, EXIT, active ECUs - ILP, forced openings; every seed listed; each run evaluated on the
test split of its own seed.

    nohup .venv/bin/python src/scripts/run_v4.4.6.py > logs/run_v4.4.6_driver.log 2>&1 &
    .venv/bin/python src/scripts/run_v4.4.6.py --report
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
VERSION = "4.4.6"
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
VARIANTS = {"mask_ppo_base": ("基线（v4.4.5 定稿配置）", None), "mask_ppo_std": ("+ σ_util（global token）", ["--glob-std"])}
NEW = "mask_ppo_std"
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
                               "lr": 3e-4, "ent_coef": 0.005, "glob_std": True, "device": "cuda",
                               "full_episode": True, "exit_action": True, "scenarios": SCENARIOS, "algos": [ALGO],
                               "keys": {NEW: "glob_std"}, "seeds": SEEDS,
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
    if key != NEW:
        md = PREV / ("regret_base.md" if seed == 1 else f"regret_s{seed}_base.md")
        return json.loads(md.with_suffix(".json").read_text())
    md = OUT / f"regret_s{seed}_std.md"
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
                         NEW: exp_ids[str(sd)][s][NEW]} for s in SCENARIOS} for sd in SEEDS}
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
    lines = [f"# v{VERSION} 显式全局特征：global token 加入 σ_util", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 唯一改动：global token 加入已开启 ECU 利用率的标准差 σ_util = sqrt(mean_j∈on (u_j/c_j − AR_t)²)，乘 2 归一化到 [0, 1]（`--glob-std`）。"
             "global token 自 v4.4.3 起已显式包含 AR_t 与未放置服务占比 (M − t)/M（`src/paper_rl/graph_net.build`），因此不重复加入；"
             "加入后 global token 同时描述任务进度、当前目标水平和资源分配的离散程度。",
             "- 其余与 v4.4.5 定稿配置完全相同：结构感知网络、lr 3e-4、熵系数 0.005、GAE λ = 0.95、ar_pen 奖励、γ = 1、EXIT、需求降序、GPU、1M 步。",
             f"- 基线 = v4.4.5 在 GPU 上训练的三种子基线（种子 1：`logs/v4.4.5_pilot`，种子 2、3：`logs/v4.4.5_seeds`），直接复用，没有重新训练；"
             f"新模型 manifest：`{MANIFEST.relative_to(ROOT)}`。",
             "- 每个模型都在它自己训练种子的测试集上评估；相对 gap 先逐种子计算再取平均；表中为三种子均值 ± 样本标准差。",
             f"- 第 1–{EARLY} 步 regret 与选中最优的比例来自 `src/scripts/diag_regret.py`（各种子测试集前 400 个实例）。"
             "参照噪声（v4.4.5）：同配置三种子的相对 gap SD 为 0.4–0.7pp。", ""]
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
    active = [(*launch(sd, s, NEW, exp_ids), f"seed{sd} {s} {NEW}") for sd in SEEDS for s in SCENARIOS]
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
