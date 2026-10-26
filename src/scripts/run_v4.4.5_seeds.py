#!/usr/bin/env python3
"""v4.4.5 multi-seed check: structure-aware Mask PPO, baseline (lr 3e-4) vs LR (lr 1e-4), seeds 2 and 3.

Same setting as the v4.4.5 pilot (GPU, 1M steps, ent 0.005, GAE lambda 0.95, everything else as
v4.4.3); seed 1 of both variants is taken from the pilot (logs/v4.4.5_pilot, GPU baseline rerun).
12 jobs = 2 seeds x 2 variants x 3 scenarios, one GPU per scenario (4 jobs per GPU).
Report: mean +- SD over seeds 1-3 of relative gap, steps 1-5 regret, steps 1-5 optimal-action rate,
step-1 optimal rate, EXIT, active ECUs - ILP, forced openings; every seed listed. Each run is
evaluated on the test split of its own training seed. Decision rules fixed before the results:
see the report header.

    nohup .venv/bin/python src/scripts/run_v4.4.5_seeds.py > logs/run_v4.4.5_seeds_driver.log 2>&1 &
    .venv/bin/python src/scripts/run_v4.4.5_seeds.py --report
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
VERSION = "4.4.5"
DATA = "v4.3.1.4"
LOG_DIR = ROOT / "logs" / f"v{VERSION}_seeds"
MANIFEST = LOG_DIR / "manifest.json"
PILOT = ROOT / "logs" / f"v{VERSION}_pilot" / "manifest.json"                # seed 1 of both variants
OUT = ROOT / "paper_contents" / f"v{VERSION}"
REPORT = OUT / "seeds_report.md"

SCENARIOS = ["lt", "eq", "gt"]
ALGO = "mask_ppo"
SEEDS, STEPS = [2, 3], 1_000_000
REWARD, NORM, OBS, GAMMA, NET = "ar_pen", "none", "base", 1.0, "graph"
# manifest key -> (label, extra paper_rl.train args); keys match the pilot manifest
VARIANTS = {"mask_ppo_base": ("基线（lr 3e-4）", []), "mask_ppo_lr": ("LR（lr 1e-4）", ["--lr", "1e-4"])}
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
                               "lr": {"mask_ppo_base": 3e-4, "mask_ppo_lr": 1e-4}, "ent_coef": 0.005, "device": "cuda",
                               "full_episode": True, "exit_action": True, "scenarios": SCENARIOS, "algos": [ALGO],
                               "keys": {"mask_ppo_base": "base", "mask_ppo_lr": "lr1e-4"}, "seeds": SEEDS,
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
    """diag_regret summary of one (seed, variant) over the 3 scenarios; seed 1 reuses the pilot's."""
    md = OUT / (f"regret_{key.removeprefix('mask_ppo_')}.md" if seed == 1 else f"regret_s{seed}_{key.removeprefix('mask_ppo_')}.md")
    js = md.with_suffix(".json")
    if not js.exists():
        subprocess.run([PY, str(ROOT / "src" / "scripts" / "diag_regret.py"), "--out", str(md), "--seed", str(seed),
                        "--manifest", str(manifest.relative_to(ROOT)), "--key", key,
                        "--label", f"v{VERSION} 结构感知 Mask PPO {VARIANTS[key][0]}（种子 {seed}、1M、GPU）"],
                       cwd=ROOT, check=True)
    return json.loads(js.read_text())


def write_report(exp_ids):
    from diag_open_ecu import replay
    ids = {**{"1": json.loads(PILOT.read_text())["exp_ids"]["1"]}, **exp_ids}
    seeds = sorted(int(s) for s in ids)
    man = {1: PILOT, **{s: MANIFEST for s in SEEDS}}
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
    lines = [f"# v{VERSION} 多种子确认：基线（lr 3e-4）vs LR（lr 1e-4）", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"- 种子 {', '.join(map(str, seeds))}；种子 1 取自 v{VERSION} 试跑（`{PILOT.relative_to(ROOT)}`，GPU 重跑的基线和 LR），"
             f"种子 2、3 为本次新训练（`{MANIFEST.relative_to(ROOT)}`）。设置与试跑完全相同：结构感知网络、GPU、1M 步、熵系数 0.005、GAE λ = 0.95，其余同 v4.4.3。",
             "- 每个模型都在它自己训练种子的测试集（20% 留出，按种子划分）上评估，所以不同种子的 ILP AR 不同；相对 gap 先逐种子计算再取平均。表中为三种子均值 ± 样本标准差。",
             f"- 第 1–{EARLY} 步 regret 与选中最优的比例来自 `src/scripts/diag_regret.py`（各种子测试集前 400 个实例）。其余定义同 `pilot_report.md`。",
             "- **事先定好的判断标准**（结果出来之前写下）：",
             "  1. EQ 在三种子下仍稳定出现约 10% → 7–8% 的 gap 下降，且前几步 regret 同步下降，才算 lr 1e-4 对 EQ 真有效；",
             "  2. 不按场景分别选学习率；lr 1e-4 三场景整体更好或至少不损害其他场景 → 采用 1e-4；只有 EQ 好、LT/GT 持平或变差 → 保留 3e-4；",
             "  3. EQ 的提升在种子 2、3 上不能复现 → 判定为单种子波动，保留 3e-4；",
             "  4. 这组结束后停止超参数搜索，不做 instance-wise baseline。", ""]
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
    active = [(*launch(sd, s, k, exp_ids), f"seed{sd} {s} {k}") for sd in SEEDS for k in VARIANTS for s in SCENARIOS]
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
