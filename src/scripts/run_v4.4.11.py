#!/usr/bin/env python3
"""v4.4.11: same-instance multi-trajectory Mask PPO with an instance-wise baseline (3 seeds x 3 scenarios x 1M, GPU).

Single change from the final configuration (v4.4.5 baseline): --group-k 4 (paper_rl/group_ppo.py).
The 40 environments form 10 groups of K = 4; each group runs 4 stochastic episodes of the current
policy on the same training instance, and every step of trajectory k gets A = G_k - mean_k G_k
(no critic / GAE in the advantage; the critic is still trained on G_k). ILP is not used in training.
Evaluation unchanged: one deterministic (argmax) rollout per test instance, no best-of-K.
Network (structure-aware, shared encoder, original ECU head), reward ar_pen, lr 3e-4, ent 0.005,
clip 0.1, batch 256, 10 epochs, gamma 1, demand descending: unchanged.
Baseline = the v4.4.5 GPU baseline runs of seeds 1-3, reused. 9 jobs, one GPU per scenario.

    nohup .venv/bin/python src/scripts/run_v4.4.11.py > logs/run_v4.4.11_driver.log 2>&1 &
    .venv/bin/python src/scripts/run_v4.4.11.py --report
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
VERSION = "4.4.11"
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
VARIANTS = {"mask_ppo_base": ("基线（critic + GAE 的 advantage）", None), "mask_ppo_group": ("同实例 4 条轨迹，实例内基线", None)}
NEW = "mask_ppo_group"
LABEL_EN = {"mask_ppo_base": "baseline (critic + GAE advantage)", "mask_ppo_group": "4 trajectories per instance, instance-wise baseline"}
RESULTS = ROOT / "results" / "unified"
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
                               "lr": 3e-4, "ent_coef": 0.005, "device": "cuda",
                               "full_episode": True, "exit_action": True, "scenarios": SCENARIOS, "algos": [ALGO],
                               "group_k": 4, "keys": {NEW: "group_k4"}, "seeds": SEEDS,
                               "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST)


def launch(seed, scen, key, exp_ids):
    exp_id = new_exp_id()
    log = open(LOG_DIR / f"seed{seed}_{scen}_{key}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(seed),
           "PAPER_VERSION": VERSION, "DATA_VERSION": DATA, "CUDA_VISIBLE_DEVICES": str(SCENARIOS.index(scen) % 3)}
    proc = subprocess.Popen([PY_GPU, "-u", "-m", "paper_rl.train", "--device", "cuda", "--scen", scen, "--algo", ALGO,
                             "--reward", REWARD, "--reward-norm", NORM, "--obs", OBS, "--gamma", str(GAMMA),
                             "--group-k", "4", "--full-episode", "--exit-action", "--net", NET,
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
    md = OUT / f"regret_s{seed}_group.md"
    js = md.with_suffix(".json")
    if not js.exists():
        subprocess.run([PY, str(ROOT / "src" / "scripts" / "diag_regret.py"), "--out", str(md), "--seed", str(seed),
                        "--manifest", str(manifest.relative_to(ROOT)), "--key", key,
                        "--label", f"v{VERSION} 结构感知 Mask PPO {VARIANTS[key][0]}（种子 {seed}、1M、GPU）",
                        "--label-en", f"v{VERSION} structure-aware Mask PPO, {LABEL_EN[key]} (seed {seed}, 1M, GPU)"],
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
    for lang in ("zh", "en"):
        lines = render(lang, res, seeds, ms, pc, f4, f2, f2u)
        out = REPORT if lang == "zh" else REPORT.with_suffix(".en.md")
        out.write_text("\n".join(lines))
        print(f"report -> {out}", flush=True)


def render(lang, res, seeds, ms, pc, f4, f2, f2u):
    """The report in Chinese (zh) or English (en); same data, same tables."""
    zh = lang == "zh"
    T = lambda z, e: z if zh else e
    lb = lambda key: VARIANTS[key][0] if zh else LABEL_EN[key]
    cols = [(T("相对 gap", "Relative gap"), "rel", pc), (T(f"第 1–{EARLY} 步 regret", f"Steps 1–{EARLY} regret"), "r15", f4),
            (T(f"第 1–{EARLY} 步选中最优", f"Steps 1–{EARLY} optimal-action rate"), "p15", pc),
            (T("第 1 步选中最优", "Step-1 optimal-action rate"), "p1", pc), (T("EXIT 率", "EXIT rate"), "exit", pc),
            (T("开启 ECU 数 − ILP", "Active ECUs − ILP"), "dact", f2), (T("被逼开启/实例", "Forced openings / instance"), "forced", f2u)]
    if zh:
        lines = [f"# v{VERSION} 同实例多轨迹 PPO（实例内基线）", "",
                 f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
                 "- 唯一改动：advantage 的来源（`--group-k 4`，`src/paper_rl/group_ppo.py`）。40 个环境分成 10 组、每组 K = 4；同组每次都在同一个训练实例上"
                 "用当前策略各采样一条完整轨迹，轨迹 k 的每一步 A = G_k − b_x，b_x = (1/4) Σ_k G_k（γ = 1，G_k 为该轨迹回报）。advantage 不再经过 critic 和 GAE；"
                 "critic 仍以 G_k 为目标训练（PPO 损失不变），但不参与 advantage。每轮只收完整的组，凑满 20480 条转移即更新，未完成的组丢弃（这些步数仍计入 1M 预算）。",
                 "- ILP 完全不参与训练。测试不变：每个测试实例一次确定性（argmax）rollout，不做 best-of-K。",
                 "- 校验：同组 4 条轨迹实例相同；逐条轨迹 adv = G_k − b_x、ret = G_k（误差 3e−8，float32）；每组 advantage 之和为 0；重算 log-prob 与采样时一致；动作全部合法。",
                 "- 其余全部不变：结构感知网络（共享编码器、原 ECU 打分头）、ar_pen 奖励、lr 3e-4、熵系数 0.005、clip 0.1、batch 256、10 epochs、γ = 1、EXIT、需求降序、GPU、1M 步。",
                 f"- 基线 = v4.4.5 在 GPU 上训练的三种子基线，直接复用；新模型 manifest：`{MANIFEST.relative_to(ROOT)}`。",
                 "- 每个模型都在它自己训练种子的测试集上评估；相对 gap 先逐种子计算再取平均；表中为三种子均值 ± 样本标准差。",
                 f"- 第 1–{EARLY} 步 regret 与选中最优的比例来自 `src/scripts/diag_regret.py`（各种子测试集前 400 个实例）。"
                 "噪声参照：同配置三种子 SD 0.4–0.7pp，同种子重跑可差 1.4–2.0pp；EQ 基线 10.2% 很可能偏高（其他四个变体均为 9.4–9.5%）。", ""]
    else:
        lines = [f"# v{VERSION} Same-instance multi-trajectory PPO (instance-wise baseline)", "",
                 f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 "- Single change: the source of the advantage (`--group-k 4`, `src/paper_rl/group_ppo.py`). The 40 environments form 10 groups of "
                 "K = 4; each group runs one complete trajectory per member with the current policy on the same training instance, and every step of "
                 "trajectory k gets A = G_k − b_x, b_x = (1/4) Σ_k G_k (γ = 1, G_k is the trajectory return). The advantage no longer goes through "
                 "the critic or GAE; the critic is still trained with target G_k (the PPO loss is unchanged) but does not enter the advantage. "
                 "A rollout keeps complete groups only and updates once 20480 transitions are collected; unfinished groups are discarded "
                 "(their steps still count towards the 1M budget).",
                 "- The ILP is not used in training at all. Evaluation is unchanged: one deterministic (argmax) rollout per test instance, no best-of-K.",
                 "- Checks: the 4 trajectories of a group share the instance; per trajectory adv = G_k − b_x and ret = G_k (error 3e−8, float32); "
                 "advantages sum to 0 within each group; recomputed log-probs match those at sampling; all actions legal.",
                 "- Everything else unchanged: structure-aware network (shared encoder, original ECU head), ar_pen reward, lr 3e-4, entropy "
                 "coefficient 0.005, clip 0.1, batch 256, 10 epochs, γ = 1, EXIT, descending demand, GPU, 1M steps.",
                 f"- Baseline = the v4.4.5 three-seed GPU baseline, reused; manifest of the new models: `{MANIFEST.relative_to(ROOT)}`.",
                 "- Every model is evaluated on the test split of its own training seed; the relative gap is computed per seed and then "
                 "averaged; tables show the mean ± sample std over three seeds.",
                 f"- Steps 1–{EARLY} regret and optimal-action rates come from `src/scripts/diag_regret.py` (first 400 instances of each seed's "
                 "test split). Noise reference: three-seed SD 0.4–0.7pp for the same configuration, and a rerun of the same seed can differ by "
                 "1.4–2.0pp; the EQ baseline of 10.2% is probably high (the other four variants all give 9.4–9.5%).", ""]
    lines += [T("## 三种子汇总", "## Three-seed summary"), "",
              T("| 场景 | 模型 | ", "| Scenario | Model | ") + " | ".join(c[0] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for s in SCENARIOS:
        for key in VARIANTS:
            lines.append(f"| {s.upper()} | {lb(key)} | " + " | ".join(ms([res[(key, s, sd)][c] for sd in seeds], f) for _, c, f in cols) + " |")
    lines += ["", T("三场景平均（每个种子先对三场景取平均，再对种子求均值 ± SD）：",
                    "Average over the three scenarios (per seed first, then mean ± SD over seeds):"), "",
              T("| 模型 | 相对 gap | ", "| Model | Relative gap | ")
              + T(f"第 1–{EARLY} 步 regret | 第 1–{EARLY} 步选中最优 | EXIT 率 |",
                  f"Steps 1–{EARLY} regret | Steps 1–{EARLY} optimal-action rate | EXIT rate |"), "|---|---|---|---|---|"]
    for key in VARIANTS:
        avg = lambda c: [np.mean([res[(key, s, sd)][c] for s in SCENARIOS]) for sd in seeds]
        lines.append(f"| {lb(key)} | {ms(avg('rel'), pc)} | {ms(avg('r15'), f4)} | {ms(avg('p15'), pc)} | {ms(avg('exit'), pc)} |")
    lines += ["", T("## 逐种子", "## Per seed"), "",
              T("| 场景 | 种子 | 模型 | ", "| Scenario | Seed | Model | ") + " | ".join(c[0] for c in cols) + " |",
              "|---|---|---|" + "---|" * len(cols)]
    for s in SCENARIOS:
        for sd in seeds:
            for key in VARIANTS:
                r = res[(key, s, sd)]
                lines.append(f"| {s.upper()} | {sd} | {lb(key)} | " + " | ".join(f(r[c]) for _, c, f in cols) + " |")
    lines.append("")
    return lines


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
