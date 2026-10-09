#!/usr/bin/env python3
"""v4.4.12: global-plan-conditioned Mask PPO (3 seeds x 3 scenarios x 1M, GPU).

Single change from the final configuration (v4.4.5 baseline): --plan. A separate plan encoder (same
structure as the main one, own parameters) reads the instance's initial state s_0 -- recovered exactly
from any s_t by clearing the assignment and setting t = 0 -- and forms a service x ECU plan matrix
P_ij = <W_s h_i(s_0), W_e h_j(s_0)> / sqrt(d), constant over the episode; the ECU logits become
P[i_t, j] + the usual per-step score (global plan + sequential correction). W_e starts at zero, so the
policy equals the baseline at initialisation. EXIT head, critic, mask, reward (ar_pen), MaskablePPO and
its hyper-parameters unchanged; no ILP, no BC, one deterministic rollout at test time.
Baseline = the v4.4.5 GPU baseline runs of seeds 1-3, reused. 9 jobs, one GPU per scenario.

    nohup .venv/bin/python src/scripts/run_v4.4.12.py > logs/run_v4.4.12_driver.log 2>&1 &
    .venv/bin/python src/scripts/run_v4.4.12.py --report
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
VERSION = "4.4.12"
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
VARIANTS = {"mask_ppo_base": ("基线（反应式策略）", None), "mask_ppo_plan": ("全局计划 + 逐步修正", None)}
NEW = "mask_ppo_plan"
REGRET_KEYS = [NEW]                                                        # regret computed here; the baseline reuses v4.4.5's
LABEL_EN = {"mask_ppo_base": "baseline (reactive policy)", "mask_ppo_plan": "global plan + sequential correction"}
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
                               "plan": True, "keys": {NEW: "plan"}, "seeds": SEEDS,
                               "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST)


def launch(seed, scen, key, exp_ids):
    exp_id = new_exp_id()
    log = open(LOG_DIR / f"seed{seed}_{scen}_{key}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(seed),
           "PAPER_VERSION": VERSION, "DATA_VERSION": DATA, "CUDA_VISIBLE_DEVICES": str(SCENARIOS.index(scen) % 3)}
    proc = subprocess.Popen([PY_GPU, "-u", "-m", "paper_rl.train", "--device", "cuda", "--scen", scen, "--algo", ALGO,
                             "--reward", REWARD, "--reward-norm", NORM, "--obs", OBS, "--gamma", str(GAMMA),
                             "--plan", "--full-episode", "--exit-action", "--net", NET,
                             "--steps", str(STEPS), "--seed", str(seed)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    exp_ids.setdefault(str(seed), {}).setdefault(scen, {})[key] = exp_id
    write_manifest(exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched seed{seed} {scen} {key} exp_id={exp_id} pid={proc.pid}", flush=True)
    return proc, log


def regret(seed, key, manifest, step=None):
    """diag_regret summary of one (seed, variant) over the 3 scenarios; the baseline reuses v4.4.5's."""
    if key != NEW:
        md = PREV / ("regret_base.md" if seed == 1 else f"regret_s{seed}_base.md")
        return json.loads(md.with_suffix(".json").read_text())
    md = OUT / f"regret_s{seed}_plan.md"
    js = md.with_suffix(".json")
    if not js.exists():
        if step:                                                # progress line, parsed by the panel
            print(f"[report] regret {step[0]}/{step[1]} | seed {seed} {key}", flush=True)
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
    print("[report] replay", flush=True)
    with Pool(len(jobs)) as pool:
        out = pool.map(replay, jobs)
    res = {}
    for job, opens, insts, n in out:
        key, s, sd = job[0], job[2], job[6]
        ar, ilp = np.mean([x["ar"] for x in insts]), np.mean([x["ilp"] for x in insts])
        res[(key, s, sd)] = {"exit": 1 - len(insts) / n, "rel": (ilp - ar) / ilp,
                             "dact": np.mean([x["act"] - x["ilp_act"] for x in insts]),
                             "forced": sum(o["kind"] == "forced" for o in opens) / n}
    todo = [(sd, key) for sd in seeds for key in REGRET_KEYS if not (OUT / f"regret_s{sd}_{key.removeprefix('mask_ppo_')}.json").exists()]
    i_todo = 0
    for sd in seeds:
        for key in VARIANTS:
            if (sd, key) in todo:
                i_todo += 1
            g = regret(sd, key, man[sd], (i_todo, len(todo)) if (sd, key) in todo else None)
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
        lines = [f"# v{VERSION} 全局计划条件策略（global plan + sequential correction）", "",
                 f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
                 "- 唯一改动：策略结构从反应式改为计划条件（`--plan`，`src/paper_rl/graph_net.PlanEncoder`）。一个独立的规划编码器（与主编码器结构相同、参数独立）"
                 "读取实例的初始状态 s_0（由任意 s_t 清空分配、t 置 0 精确还原），生成服务 × ECU 计划矩阵 P_ij = ⟨W_s h_i(s_0), W_e h_j(s_0)⟩ / √d，"
                 "整个 episode 内不变；每步 ECU logit = P[i_t, j] + 原有的逐步打分。W_e 零初始化，训练开始时 P ≡ 0、策略与基线完全相同。参数量 45 万 → 88 万（多出的全部是规划编码器）。",
                 "- 冒烟测试：零初始化时与普通网络 logits 完全相同；同一 episode 内 P 完全不变；随机 W_e 下 ECU 置换等变（误差 1.5e−6）；mask 与 EXIT 不变；策略采样无非法动作。",
                 "- 其余全部不变：EXIT 头、critic、ar_pen 奖励、MaskablePPO、lr 3e-4、熵系数 0.005、GAE λ = 0.95、γ = 1、需求降序、GPU、1M 步；"
                 "不用 ILP、不用 BC，测试只做一次确定性 rollout。",
                 f"- 基线 = v4.4.5 在 GPU 上训练的三种子基线，直接复用；新模型 manifest：`{MANIFEST.relative_to(ROOT)}`。",
                 "- 每个模型都在它自己训练种子的测试集上评估；相对 gap 先逐种子计算再取平均；表中为三种子均值 ± 样本标准差。",
                 f"- 第 1–{EARLY} 步 regret 与选中最优的比例来自 `src/scripts/diag_regret.py`（各种子测试集前 400 个实例）。"
                 "噪声参照：同配置三种子 SD 0.4–0.7pp，同种子重跑可差 1.4–2.0pp；EQ 基线 10.2% 很可能偏高（多个变体为 9.4–9.5%）。",
                 "- 判断顺序（事先定好）：先看第 1–5 步 regret 是否下降、选中最优是否上升，再看多开 ECU 是否减少，最后看 gap。", ""]
    else:
        lines = [f"# v{VERSION} Global-plan-conditioned policy (global plan + sequential correction)", "",
                 f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 "- Single change: the policy goes from reactive to plan-conditioned (`--plan`, `src/paper_rl/graph_net.PlanEncoder`). A separate plan "
                 "encoder (same structure as the main encoder, its own parameters) reads the instance's initial state s_0 (recovered exactly from any "
                 "s_t by clearing the assignment and setting t = 0) and forms a service × ECU plan matrix P_ij = ⟨W_s h_i(s_0), W_e h_j(s_0)⟩ / √d, "
                 "constant over the episode; the ECU logit at each step is P[i_t, j] + the usual per-step score. W_e is zero-initialised, so P ≡ 0 and "
                 "the policy equals the baseline at the start of training. Parameters 450k → 880k (all of the increase is the plan encoder).",
                 "- Smoke tests: identical logits to the plain network at zero initialisation; P exactly constant within an episode; ECU permutation "
                 "equivariance with a random W_e (error 1.5e−6); mask and EXIT unchanged; no illegal action when sampling the policy.",
                 "- Everything else unchanged: EXIT head, critic, ar_pen reward, MaskablePPO, lr 3e-4, entropy coefficient 0.005, GAE λ = 0.95, γ = 1, "
                 "descending demand, GPU, 1M steps; no ILP, no BC, one deterministic rollout at test time.",
                 f"- Baseline = the v4.4.5 three-seed GPU baseline, reused; manifest of the new models: `{MANIFEST.relative_to(ROOT)}`.",
                 "- Every model is evaluated on the test split of its own training seed; the relative gap is computed per seed and then averaged; "
                 "tables show the mean ± sample std over three seeds.",
                 f"- Steps 1–{EARLY} regret and optimal-action rates come from `src/scripts/diag_regret.py` (first 400 instances of each seed's test "
                 "split). Noise reference: three-seed SD 0.4–0.7pp for the same configuration, a rerun of the same seed can differ by 1.4–2.0pp; the "
                 "EQ baseline of 10.2% is probably high (several variants give 9.4–9.5%).",
                 "- Order of judgement (fixed in advance): first whether steps 1–5 regret falls and the optimal-action rate rises, then whether extra "
                 "ECUs fall, and only then the gap.", ""]
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
