#!/usr/bin/env python3
"""v4.4.16: continue the nine v4.4.15 models (50k instances, 5M) to 10M, checkpoint at 7.5M (GPU).

Nothing changes but the step budget: each v4.4.15 final model is loaded with its optimiser state and step
count (train.py --resume) and trained 5M more steps on the same 50k ILP-verified pool, same network,
reward (ar_pen), mask, descending order and MaskablePPO configuration. The learning curve joins
1M / 3M / 5M (v4.4.15 checkpoints) and 7.5M / 10M (this run), each with the test and training-set gap, to
apply the decision tree fixed in advance (see the report header). 9 jobs, one GPU per scenario; email
on completion / failure.

    nohup .venv/bin/python src/scripts/run_v4.4.16.py > logs/run_v4.4.16_driver.log 2>&1 &
    .venv/bin/python src/scripts/run_v4.4.16.py --report
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
VERSION = "4.4.16"
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
SEEDS, STEPS = [1, 2, 3], 5_000_000                                     # steps added to the 5M models
CKPTS, TOTAL = [7_500_000], 10_000_000
REWARD, NORM, OBS, GAMMA, NET = "ar_pen", "none", "base", 1.0, "graph"
# manifest key -> (label, extra paper_rl.train args); keys match the pilot manifest
VARIANTS = {"mask_ppo_base": ("基线（1600 实例，1M）", None), "mask_ppo_pool5m": ("5 万实例，5M（v4.4.15）", None),
            "mask_ppo_pool10m": ("5 万实例，10M", None)}
NEW = "mask_ppo_pool10m"
P15 = ROOT / "logs" / "v4.4.15" / "manifest.json"                              # v4.4.15: 50k instances, 5M (+ 1M / 3M ckpts)
POOL = "results/sup_diag/{scen}.json"
REGRET_KEYS = [NEW]                                                        # regret computed here; the baseline reuses v4.4.5's
LABEL_EN = {"mask_ppo_base": "baseline (1600 instances, 1M)", "mask_ppo_pool5m": "50k instances, 5M (v4.4.15)",
            "mask_ppo_pool10m": "50k instances, 10M"}
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
                               "train_pool": POOL, "checkpoints": CKPTS, "resume_from": str(P15.relative_to(ROOT)), "total_steps": TOTAL,
                               "keys": {NEW: "pool50k_10m"}, "seeds": SEEDS,
                               "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST)


def launch(seed, scen, key, exp_ids):
    exp_id = new_exp_id()
    src = json.loads(P15.read_text())["exp_ids"][str(seed)][scen]["mask_ppo_pool5m"]
    resume = next((RESULTS / scen / ALGO / src).glob("model_*")).relative_to(ROOT)
    log = open(LOG_DIR / f"seed{seed}_{scen}_{key}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(seed),
           "PAPER_VERSION": VERSION, "DATA_VERSION": DATA, "CUDA_VISIBLE_DEVICES": str(SCENARIOS.index(scen) % 3)}
    proc = subprocess.Popen([PY_GPU, "-u", "-m", "paper_rl.train", "--device", "cuda", "--scen", scen, "--algo", ALGO,
                             "--reward", REWARD, "--reward-norm", NORM, "--obs", OBS, "--gamma", str(GAMMA),
                             "--train-pool", POOL, "--resume", str(resume), "--checkpoints", ",".join(map(str, CKPTS)), "--full-episode", "--exit-action", "--net", NET,
                             "--steps", str(STEPS), "--seed", str(seed)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    exp_ids.setdefault(str(seed), {}).setdefault(scen, {})[key] = exp_id
    write_manifest(exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched seed{seed} {scen} {key} exp_id={exp_id} pid={proc.pid}", flush=True)
    return proc, log


def regret(seed, key, manifest, step=None):
    """diag_regret summary of one (seed, variant) over the 3 scenarios; the baseline reuses v4.4.5's."""
    if key == "mask_ppo_base":
        md = PREV / ("regret_base.md" if seed == 1 else f"regret_s{seed}_base.md")
        return json.loads(md.with_suffix(".json").read_text())
    if key == "mask_ppo_pool5m":                                # v4.4.15 (50k, 5M)
        return json.loads((ROOT / "paper_contents" / "v4.4.15" / f"regret_s{seed}_pool5m.json").read_text())
    md = OUT / f"regret_s{seed}_pool10m.md"
    js = md.with_suffix(".json")
    if not js.exists():
        if step:                                                # progress line, parsed by the panel
            print(f"[report] regret {step[0]}/{step[1]} | seed {seed} {key}", flush=True)
        subprocess.run([PY, str(ROOT / "src" / "scripts" / "diag_regret.py"), "--out", str(md), "--seed", str(seed),
                        "--manifest", str(manifest.relative_to(ROOT)), "--key", key,
                        "--label", f"v{VERSION} 结构感知 Mask PPO {VARIANTS[key][0]}（种子 {seed}、10M、GPU）",
                        "--label-en", f"v{VERSION} structure-aware Mask PPO, {LABEL_EN[key]} (seed {seed}, 10M, GPU)"],
                       cwd=ROOT, check=True)
    return json.loads(js.read_text())


def _gap(model, insts):
    from paper_rl.train import evaluate
    ev = evaluate(model, insts, "mask", "ar_pen", 0.0, "ppo", "raw", True, True)
    ok = [e for e in ev if not e["exited"]]
    ar, ilp = np.mean([e["ar"] for e in ok]), np.mean([e["ar_star"] for e in ok])
    return float((ilp - ar) / ilp), 1 - len(ok) / len(ev)


def ckpt_eval(job):
    """Test gap (the seed's 400 test instances) and training-set gap (fixed 1600-instance subset of the 50k pool)
    of one model file: pattern 'model_*' (final) or 'ckpt_<step>_*'."""
    import random as _r
    import torch
    torch.set_num_threads(1)
    from paper_rl.train import model_class, split_instances
    scen, seed, eid, pattern = job
    pool = json.loads((ROOT / POOL.format(scen=scen)).read_text())
    sub = [pool[m] for m in sorted(_r.Random(0).sample(range(len(pool)), 1600))]
    test = split_instances(scen, seed)[2]
    model = model_class("ppo", "mask").load(str(next((RESULTS / scen / ALGO / eid).glob(pattern))), device="cpu")
    (te, te_exit), (tr, _) = _gap(model, test), _gap(model, sub)
    return scen, seed, eid, pattern, te, tr, te_exit


def write_report(exp_ids):
    from diag_open_ecu import replay
    p15 = json.loads(P15.read_text())["exp_ids"]
    ids = {str(sd): {s: {"mask_ppo_base": json.loads(BASE[sd].read_text())["exp_ids"][str(sd)][s]["mask_ppo_base"],
                         "mask_ppo_pool5m": p15[str(sd)][s]["mask_ppo_pool5m"], NEW: exp_ids[str(sd)][s][NEW]}
                     for s in SCENARIOS} for sd in SEEDS}
    seeds = SEEDS
    man = {sd: MANIFEST for sd in SEEDS}
    jobs = [(key, DATA, s, ALGO, ids[str(sd)][s][key], "model_*", sd) for sd in seeds for key in VARIANTS for s in SCENARIOS]
    print("[report] replay", flush=True)
    with Pool(len(jobs)) as pool:
        out = pool.map(replay, jobs)
    res = {}
    for job, opens, insts, n in out:
        key, s, sd = job[0], job[2], job[6]
        ar, ilp = np.mean([e["ar"] for e in insts]), np.mean([e["ilp"] for e in insts])
        res[(key, s, sd)] = {"exit": 1 - len(insts) / n, "rel": (ilp - ar) / ilp,
                             "dact": np.mean([e["act"] - e["ilp_act"] for e in insts]),
                             "forced": sum(o["kind"] == "forced" for o in opens) / n}
    print("[report] training-set gap", flush=True)
    base_tr = {(r["scen"], r["seed"]): r["train"]["rel"]
               for r in json.loads((ROOT / "results" / "sup_diag" / "data_rl_train_test.json").read_text())}
    cjobs = [(s, sd, ids[str(sd)][s][NEW], pat) for sd in seeds for s in SCENARIOS for pat in [f"ckpt_{c}_*" for c in CKPTS] + ["model_*"]]
    cjobs += [(s, sd, ids[str(sd)][s]["mask_ppo_pool5m"], pat) for sd in seeds for s in SCENARIOS
              for pat in ["ckpt_1000000_*", "ckpt_3000000_*", "model_*"]]
    with Pool(min(len(cjobs), 36)) as pool:
        ce = pool.map(ckpt_eval, cjobs)
    curve = {}                                              # (scen, seed, step) -> (test gap, train gap, test EXIT)
    for s, sd, eid, pat, te, tr, ex in ce:
        new = eid == ids[str(sd)][s][NEW]
        step = int(pat.split("_")[1]) if pat.startswith("ckpt_") else (TOTAL if new else 5_000_000)
        curve[(s, sd, step)] = (te, tr, ex)
        if pat == "model_*":
            res[(NEW if new else "mask_ppo_pool5m", s, sd)]["train_rel"] = tr
    for sd in seeds:
        for s in SCENARIOS:
            res[("mask_ppo_base", s, sd)]["train_rel"] = base_tr[(s, sd)]
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
        lines = render(lang, res, seeds, ms, pc, f4, f2, f2u, curve)
        out = REPORT if lang == "zh" else REPORT.with_suffix(".en.md")
        out.write_text("\n".join(lines))
        print(f"report -> {out}", flush=True)


def render(lang, res, seeds, ms, pc, f4, f2, f2u, curve):
    """The report in Chinese (zh) or English (en); same data, same tables."""
    zh = lang == "zh"
    T = lambda z, e_: z if zh else e_
    lb = lambda key: VARIANTS[key][0] if zh else LABEL_EN[key]
    cols = [(T("测试集相对 gap", "Test relative gap"), "rel", pc), (T("训练集相对 gap", "Train relative gap"), "train_rel", pc), (T(f"第 1–{EARLY} 步 regret", f"Steps 1–{EARLY} regret"), "r15", f4),
            (T(f"第 1–{EARLY} 步选中最优", f"Steps 1–{EARLY} optimal-action rate"), "p15", pc),
            (T("第 1 步选中最优", "Step-1 optimal-action rate"), "p1", pc), (T("EXIT 率", "EXIT rate"), "exit", pc),
            (T("开启 ECU 数 − ILP", "Active ECUs − ILP"), "dact", f2), (T("被逼开启/实例", "Forced openings / instance"), "forced", f2u)]
    steps = [1_000_000, 3_000_000, 5_000_000] + CKPTS + [TOTAL]
    sl = lambda st: f"{st / 1e6:g}M"
    if zh:
        lines = [f"# v{VERSION} 5 万训练实例：续训到 10M（学习曲线 1M / 3M / 5M / 7.5M / 10M）", "",
                 f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
                 "- 唯一改动：训练步数。v4.4.15 的 9 个 5M 模型连同优化器状态与步数一起加载（`train.py --resume`），在同一个 5 万实例训练池上再训练 5M 步，7.5M 存 checkpoint；"
                 "网络、ar_pen 奖励、mask、需求降序、MaskablePPO 配置全部不变。学习曲线的 1M / 3M / 5M 来自 v4.4.15 的 checkpoint，7.5M / 10M 来自本次续训。",
                 "- 测试集 gap：各种子自己的 400 个测试实例；训练集 gap：5 万训练池中按固定种子抽取的 1600 个实例；均为确定性回放。",
                 "- **事先定好的判断**：10M 大致到 LT ≈ 8%、EQ ≈ 6%、GT ≈ 5.5–6% → 路线正确，继续训练有价值。决策树：gap 继续明显下降且训练集 ≈ 测试集 → 继续训练；"
                 "训练集下降而测试集不降 → 扩大训练实例（例如 5 万 → 20 万）；两者同时停止下降 → PPO / 模型进入瓶颈，再动算法；只有 LT 停而 EQ/GT 继续降 → 单独研究 LT。",
                 f"- 对照：v4.4.5 基线（1600 实例、1M）与 v4.4.15（5 万实例、5M）直接复用；新模型 manifest：`{MANIFEST.relative_to(ROOT)}`。",
                 f"- 第 1–{EARLY} 步 regret 与选中最优的比例来自 `src/scripts/diag_regret.py`（各种子测试集前 400 个实例，10M 模型）。", ""]
    else:
        lines = [f"# v{VERSION} 50k training instances: continued to 10M (learning curve 1M / 3M / 5M / 7.5M / 10M)", "",
                 f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 "- Single change: the step budget. The nine 5M models of v4.4.15 are loaded with their optimiser state and step count "
                 "(`train.py --resume`) and trained 5M more steps on the same 50k-instance training pool, with a checkpoint at 7.5M; network, "
                 "ar_pen reward, mask, descending demand and MaskablePPO configuration unchanged. The 1M / 3M / 5M points come from the v4.4.15 "
                 "checkpoints, 7.5M / 10M from this continuation.",
                 "- Test gap: each seed's own 400 test instances; training-set gap: a fixed 1600-instance subset of the 50k training pool; both "
                 "deterministic rollouts.",
                 "- **Criteria fixed in advance**: 10M roughly at LT ≈ 8%, EQ ≈ 6%, GT ≈ 5.5–6% → the route is right and more training pays. "
                 "Decision tree: the gap keeps falling clearly with train ≈ test → keep training; train falls but test does not → widen the "
                 "training instances (e.g. 50k → 200k); both stop falling → PPO / the model is the bottleneck, change the algorithm; only LT "
                 "stops while EQ/GT keep falling → study LT separately.",
                 f"- References: the v4.4.5 baseline (1600 instances, 1M) and v4.4.15 (50k instances, 5M), reused; manifest of the new models: "
                 f"`{MANIFEST.relative_to(ROOT)}`.",
                 f"- Steps 1–{EARLY} regret and optimal-action rates come from `src/scripts/diag_regret.py` (first 400 instances of each seed's test "
                 "split, 10M models).", ""]
    # learning curve
    lines += [T("## 学习曲线（1M / 3M / 5M：v4.4.15；7.5M / 10M：续训）", "## Learning curve (1M / 3M / 5M: v4.4.15; 7.5M / 10M: continued)"), "",
              T("| 场景 | 步数 | 测试集相对 gap | 训练集相对 gap | 测试 EXIT 率 | 逐种子测试 gap（1 / 2 / 3） | 逐种子训练集 gap（1 / 2 / 3） |",
                "| Scenario | Steps | Test relative gap | Train relative gap | Test EXIT rate | Test gap per seed (1 / 2 / 3) | Train gap per seed (1 / 2 / 3) |"),
              "|---|---|---|---|---|---|---|"]
    for s in SCENARIOS:
        for st in steps:
            c = [curve[(s, sd, st)] for sd in seeds if (s, sd, st) in curve]
            if c:
                lines.append(f"| {s.upper()} | {sl(st)} | {ms([e[0] for e in c], pc)} | {ms([e[1] for e in c], pc)} | {ms([e[2] for e in c], pc)} | "
                             + " / ".join(pc(e[0]) for e in c) + " | " + " / ".join(pc(e[1]) for e in c) + " |")
    avg = lambda st, k: [np.mean([curve[(s, sd, st)][k] for s in SCENARIOS]) for sd in seeds if all((s, sd, st) in curve for s in SCENARIOS)]
    lines += ["", T("三场景平均（每个种子先对三场景取平均，再对种子求均值 ± SD）：", "Average over the three scenarios (per seed first, then mean ± SD over seeds):"), "",
              T("| 步数 | 测试集相对 gap | 训练集相对 gap |", "| Steps | Test relative gap | Train relative gap |"), "|---|---|---|"]
    for st in steps:
        if avg(st, 0):
            lines.append(f"| {sl(st)} | {ms(avg(st, 0), pc)} | {ms(avg(st, 1), pc)} |")
    lines.append("")
    lines += [T("## 三种子汇总", "## Three-seed summary"), "",
              T("| 场景 | 模型 | ", "| Scenario | Model | ") + " | ".join(c[0] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for s in SCENARIOS:
        for key in VARIANTS:
            lines.append(f"| {s.upper()} | {lb(key)} | " + " | ".join(ms([res[(key, s, sd)][c] for sd in seeds], f) for _, c, f in cols) + " |")
    lines += ["", T("三场景平均（每个种子先对三场景取平均，再对种子求均值 ± SD）：",
                    "Average over the three scenarios (per seed first, then mean ± SD over seeds):"), "",
              T("| 模型 | 测试集相对 gap | ", "| Model | Test relative gap | ")
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
    from shared.report_mail import send_failure, send_report       # email (.env.email); never fails the run
    if failed:
        print("FAILED:", failed, flush=True)
        send_failure(f"v{VERSION}", failed, log_dir=LOG_DIR.relative_to(ROOT), manifest=MANIFEST.relative_to(ROOT))
        return
    write_report(exp_ids)
    send_report(f"v{VERSION}", REPORT, manifest=MANIFEST.relative_to(ROOT), driver_log=f"logs/run_v{VERSION}_driver.log")


if __name__ == "__main__":
    main()
