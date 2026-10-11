#!/usr/bin/env python3
"""v4.5.x: network (MLP vs structure-aware Transformer) x learner (Mask PPO / DQN / DDQN) x training instances,
each run 20M steps with checkpoints at 1 / 3 / 5 / 7.5 / 10 / 15M (3 seeds x 3 scenarios).

    4.5.1  MLP Mask PPO, 50k instances (CPU); compared with the Transformer Mask PPO curve of v4.4.15-17
    4.5.2  Mask-DQN / Mask-DDQN x {MLP (CPU), Transformer (GPU)}, 50k instances
    4.5.3  Mask-DQN / Mask-DDQN x {MLP, Transformer}, the 1600-instance split; the 50k half is v4.5.2

Everything else as the final configuration: reward ar_pen, gamma 1, full episode, EXIT, descending
demand; PPO / DQN hyper-parameters from paper_rl/config.py, the same for both networks. Report: learning
curve (test gap on each seed's 400 test instances, training-set gap on 1600 training instances) of every
variant and reference, and the 20M models with the per-step regret; Chinese + English, email at the end.

    nohup .venv/bin/python src/scripts/run_v4.5.py --version 4.5.1 > logs/run_v4.5.1_driver.log 2>&1 &
    .venv/bin/python src/scripts/run_v4.5.py --version 4.5.1 --report
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))
PY, PY_GPU = str(ROOT / ".venv" / "bin" / "python"), str(ROOT / ".venv-gpu" / "bin" / "python")
DATA, POOL = "v4.3.1.4", "results/sup_diag/{scen}.json"
RESULTS = ROOT / "results" / os.environ.get("RESULTS_SPACE", "unified")
SCENARIOS, SEEDS = ["lt", "eq", "gt"], [1, 2, 3]
STEPS = 20_000_000
CKPTS = [1_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000, 15_000_000]
SMOKE = os.environ.get("V45_SMOKE") == "1"          # end-to-end test: tiny budget, seed 1, results in RESULTS_SPACE
if SMOKE:
    SEEDS, STEPS, CKPTS = [1], 41_000, [20_480]
EARLY = 5

# variant -> (algo, net, 50k pool?, label zh, label en)
CONFIG = {
    "4.5.1": {"variants": {"ppo_mlp": ("mask_ppo", "mlp", True, "MLP Mask PPO（5 万实例）", "MLP Mask PPO (50k instances)")},
              "title": ("MLP vs Transformer：Mask PPO（5 万实例、20M）", "MLP vs Transformer: Mask PPO (50k instances, 20M)")},
    "4.5.2": {"variants": {"dqn_mlp": ("mask_dqn", "mlp", True, "MLP Mask-DQN（5 万）", "MLP Mask-DQN (50k)"),
                           "dqn_graph": ("mask_dqn", "graph", True, "Transformer Mask-DQN（5 万）", "Transformer Mask-DQN (50k)"),
                           "ddqn_mlp": ("mask_ddqn", "mlp", True, "MLP Mask-DDQN（5 万）", "MLP Mask-DDQN (50k)"),
                           "ddqn_graph": ("mask_ddqn", "graph", True, "Transformer Mask-DDQN（5 万）", "Transformer Mask-DDQN (50k)")},
              "title": ("MLP vs Transformer：Mask-DQN / Mask-DDQN（5 万实例、20M）", "MLP vs Transformer: Mask-DQN / Mask-DDQN (50k instances, 20M)")},
    "4.5.3": {"variants": {"dqn_mlp_1600": ("mask_dqn", "mlp", False, "MLP Mask-DQN（1600）", "MLP Mask-DQN (1600)"),
                           "dqn_graph_1600": ("mask_dqn", "graph", False, "Transformer Mask-DQN（1600）", "Transformer Mask-DQN (1600)"),
                           "ddqn_mlp_1600": ("mask_ddqn", "mlp", False, "MLP Mask-DDQN（1600）", "MLP Mask-DDQN (1600)"),
                           "ddqn_graph_1600": ("mask_ddqn", "graph", False, "Transformer Mask-DDQN（1600）", "Transformer Mask-DDQN (1600)")},
              "title": ("Mask-DQN / Mask-DDQN：训练实例数（1600 vs 5 万）× 步数", "Mask-DQN / Mask-DDQN: training instances (1600 vs 50k) x steps")},
}


def references(version):
    """(label zh, label en, algo, [(seed, scen, exp_id, pattern, step)]) of the curves to compare against."""
    refs = []
    if version == "4.5.1":                          # Transformer Mask PPO, 50k: v4.4.15 (1/3/5M), v4.4.16 (7.5/10M), v4.4.17 (15/20M)
        pts = []
        for ver, key, plan in (("v4.4.15", "mask_ppo_pool5m", [("ckpt_1000000_*", 1e6), ("ckpt_3000000_*", 3e6), ("model_*", 5e6)]),
                               ("v4.4.16", "mask_ppo_pool10m", [("ckpt_7500000_*", 7.5e6), ("model_*", 10e6)]),
                               ("v4.4.17", "mask_ppo_pool20m", [("ckpt_15000000_*", 15e6), ("model_*", 20e6)])):
            man = ROOT / "logs" / ver / "manifest.json"
            if not man.exists():
                continue
            ids = json.loads(man.read_text())["exp_ids"]
            pts += [(sd, s, ids[str(sd)][s][key], pat, int(st)) for sd in SEEDS for s in SCENARIOS for pat, st in plan
                    if ids.get(str(sd), {}).get(s, {}).get(key)]
        refs.append(("Transformer Mask PPO（5 万实例）", "Transformer Mask PPO (50k instances)", "mask_ppo", True, pts))
    if version == "4.5.3":                          # the 50k half: v4.5.2
        man = ROOT / "logs" / "v4.5.2" / "manifest.json"
        if man.exists():
            ids = json.loads(man.read_text())["exp_ids"]
            for key, (algo, net, _, lz, le) in CONFIG["4.5.2"]["variants"].items():
                pts = [(sd, s, ids[str(sd)][s][key], pat, st) for sd in SEEDS for s in SCENARIOS
                       for pat, st in [(f"ckpt_{c}_*", c) for c in CKPTS] + [("model_*", STEPS)] if ids.get(str(sd), {}).get(s, {}).get(key)]
                refs.append((lz, le, algo, True, pts))
    return refs


def new_exp_id():
    return subprocess.run([PY, "-c", "from shared.paths import new_exp_id; print(new_exp_id())"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def launch(V, seed, scen, key, exp_ids):
    algo, net, pool, *_ = V["cfg"]["variants"][key]
    exp_id = new_exp_id()
    log = open(V["log_dir"] / f"seed{seed}_{scen}_{key}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(seed),
           "PAPER_VERSION": V["version"], "DATA_VERSION": DATA}
    gpu = net == "graph"
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(SCENARIOS.index(scen) % 3)
    cmd = [PY_GPU if gpu else PY, "-u", "-m", "paper_rl.train", "--scen", scen, "--algo", algo, "--net", net,
           "--reward", "ar_pen", "--reward-norm", "none", "--obs", "base", "--gamma", "1.0", "--full-episode", "--exit-action",
           "--steps", str(STEPS), "--checkpoints", ",".join(map(str, CKPTS)), "--seed", str(seed)]
    cmd += ["--device", "cuda"] if gpu else []
    cmd += ["--train-pool", POOL] if pool else []
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    exp_ids.setdefault(str(seed), {}).setdefault(scen, {})[key] = exp_id
    write_manifest(V, exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched seed{seed} {scen} {key} exp_id={exp_id} pid={proc.pid}", flush=True)
    return proc, log


def write_manifest(V, exp_ids):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    vs = V["cfg"]["variants"]
    algos = sorted({v[0] for v in vs.values()})
    tmp = V["manifest"].with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": V["version"], "commit": commit, "data": DATA, "steps": STEPS, "checkpoints": CKPTS,
                               "reward": "ar_pen", "gamma": 1.0, "full_episode": True, "exit_action": True,
                               "scenarios": SCENARIOS, "algos": algos[:1] if len(algos) == 1 else algos,
                               "keys": {k: f"{v[1]}{'' if v[2] else ' · 1600'}" for k, v in vs.items()},
                               "variants": {k: {"algo": v[0], "net": v[1], "train_pool": POOL if v[2] else None} for k, v in vs.items()},
                               "seeds": SEEDS, "exp_ids": exp_ids}, indent=2))
    tmp.replace(V["manifest"])


# ── evaluation ──────────────────────────────────────────────────────────────
def _gap(model, insts, learner):
    from paper_rl.train import evaluate
    obs = "raw" if model.policy.__class__.__name__.startswith("Graph") else "base"
    ev = evaluate(model, insts, "mask", "ar_pen", 0.0, learner, obs, True, True)
    ok = [e for e in ev if not e["exited"]]
    ar, ilp = np.mean([e["ar"] for e in ok]), np.mean([e["ar_star"] for e in ok])
    return float((ilp - ar) / ilp), 1 - len(ok) / len(ev)


def ckpt_eval(job):
    """Test gap (own seed's 400 test instances), training-set gap (1600: a fixed subset of the 50k pool, or the
    1600-instance split itself) and test EXIT of one saved model."""
    import torch
    torch.set_num_threads(1)
    from paper_rl.train import model_class, split_algo, split_instances
    tag, algo, pool, sd, s, eid, pat, step = job
    files = list((RESULTS / s / algo / eid).glob(pat))
    if not files:
        return job, None
    mech, learner = split_algo(algo)
    _, train, test = split_instances(s, sd)
    if pool:
        p = json.loads((ROOT / POOL.format(scen=s)).read_text())
        train = [p[i] for i in sorted(random.Random(0).sample(range(len(p)), 1600))]
    model = model_class(learner, mech).load(str(files[0]), device="cpu")
    (te, ex), (tr, _) = _gap(model, test, learner), _gap(model, train, learner)
    return job, (te, tr, ex)


def regret(V, key, algo, seed):
    md = V["out"] / f"regret_s{seed}_{key}.md"
    js = md.with_suffix(".json")
    if not js.exists():
        lz, le = V["cfg"]["variants"][key][3:]
        subprocess.run([PY, str(ROOT / "src" / "scripts" / "diag_regret.py"), "--out", str(md), "--seed", str(seed), "--algo", algo,
                        *(["--n", "25", "--workers", "24"] if SMOKE else []),
                        "--manifest", str(V["manifest"].relative_to(ROOT)), "--key", key,
                        "--label", f"v{V['version']} {lz}（种子 {seed}、20M）", "--label-en", f"v{V['version']} {le} (seed {seed}, 20M)"],
                       cwd=ROOT, check=True)
    return json.loads(js.read_text())


def write_report(V, exp_ids):
    vs = V["cfg"]["variants"]
    series = [(lz, le, algo, pool, [(sd, s, exp_ids[str(sd)][s][k], pat, st) for sd in SEEDS for s in SCENARIOS
                                   for pat, st in [(f"ckpt_{c}_*", c) for c in CKPTS] + [("model_*", STEPS)]
                                   if exp_ids.get(str(sd), {}).get(s, {}).get(k)])
              for k, (algo, net, pool, lz, le) in vs.items()] + references(V["version"])
    jobs = [(i, algo, pool, *p) for i, (_, _, algo, pool, pts) in enumerate(series) for p in pts]
    print(f"[report] learning-curve evaluation ({len(jobs)} models)", flush=True)
    with Pool(min(len(jobs), 36)) as pool:
        out = pool.map(ckpt_eval, jobs)
    curve = {}                                       # (series idx, scen, seed, step) -> (test, train, exit)
    for job, r in out:
        if r is not None:
            curve[(job[0], job[4], job[3], job[7])] = r
    reg = {}
    todo = [(k, v[0], sd) for k, v in vs.items() for sd in SEEDS]
    for i, (k, algo, sd) in enumerate(todo, 1):
        print(f"[report] regret {i}/{len(todo)} | seed {sd} {k}", flush=True)
        g = regret(V, k, algo, sd)
        for s in SCENARIOS:
            reg[(k, s, sd)] = (sum(g[s]["regret_t"][:EARLY]), float(np.mean(g[s]["opt_rate_t"][:EARLY])))
    steps = CKPTS + [STEPS]
    ms = lambda v, f: (f(np.mean(v)) + (f" ± {f(np.std(v, ddof=1))}" if len(v) > 1 else "")) if v else "—"
    pc = lambda v: f"{100 * v:.1f}%"
    sl = lambda st: f"{st / 1e6:g}M"
    for lang in ("zh", "en"):
        zh = lang == "zh"
        T = lambda z, e: z if zh else e
        lab = lambda i: series[i][0] if zh else series[i][1]
        L = [f"# v{V['version']} {T(*V['cfg']['title'])}", "",
             T(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/run_v4.5.py --version {V['version']}`。",
               f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; script `src/scripts/run_v4.5.py --version {V['version']}`."),
             T("- 每个变体 3 种子 × 3 场景、各训练 20M 步，同一次训练在 1 / 3 / 5 / 7.5 / 10 / 15M 存 checkpoint；ar_pen 奖励、γ = 1、不提前结束、EXIT、需求降序；"
               "PPO / DQN 超参取自 `paper_rl/config.py`，MLP 与 Transformer 相同；Transformer 在 GPU、MLP 在 CPU 上训练。5 万实例 = 经 ILP 验证的实例池（`results/sup_diag/<场景>.json`）。",
               "- Every variant: 3 seeds × 3 scenarios, 20M steps each, checkpoints of the same run at 1 / 3 / 5 / 7.5 / 10 / 15M; ar_pen reward, "
               "γ = 1, full episode, EXIT, descending demand; PPO / DQN hyper-parameters from `paper_rl/config.py`, identical for MLP and "
               "Transformer; Transformer trained on GPU, MLP on CPU. 50k instances = the ILP-verified pool (`results/sup_diag/<scenario>.json`)."),
             T("- 测试集 gap：各种子自己的 400 个测试实例；训练集 gap：1600 个训练实例（5 万实例池中按固定种子抽取，或 1600 实例划分本身）；均为确定性回放；均值 ± 样本标准差。",
               "- Test gap: each seed's own 400 test instances; training-set gap: 1600 training instances (a fixed subset of the 50k pool, or the "
               "1600-instance split itself); deterministic rollouts; mean ± sample std."),
             T(f"- manifest：`{V['manifest'].relative_to(ROOT)}`。", f"- Manifest: `{V['manifest'].relative_to(ROOT)}`."), ""]
        L += [T("## 学习曲线：测试集相对 gap", "## Learning curve: test relative gap"), ""]
        for s in SCENARIOS:
            L += [f"### {s.upper()}", "", "| " + T("步数", "Steps") + " | " + " | ".join(lab(i) for i in range(len(series))) + " |",
                  "|---|" + "---|" * len(series)]
            for st in steps:
                L.append(f"| {sl(st)} | " + " | ".join(ms([curve[(i, s, sd, st)][0] for sd in SEEDS if (i, s, sd, st) in curve], pc)
                                                        for i in range(len(series))) + " |")
            L.append("")
        L += [T("## 学习曲线：三场景平均（每个种子先对三场景取平均）", "## Learning curve: mean over the three scenarios (per seed first)"), "",
              "| " + T("步数", "Steps") + " | " + " | ".join(lab(i) for i in range(len(series))) + " |", "|---|" + "---|" * len(series)]
        for st in steps:
            cells = []
            for i in range(len(series)):
                v = [np.mean([curve[(i, s, sd, st)][0] for s in SCENARIOS]) for sd in SEEDS if all((i, s, sd, st) in curve for s in SCENARIOS)]
                cells.append(ms(v, pc))
            L.append(f"| {sl(st)} | " + " | ".join(cells) + " |")
        L += ["", T("## 20M：测试集与训练集 gap、前几步决策", "## 20M: test and training-set gap, early decisions"), "",
              T(f"| 场景 | 模型 | 测试集相对 gap | 训练集相对 gap | 测试 EXIT 率 | 第 1–{EARLY} 步 regret | 第 1–{EARLY} 步选中最优 |",
                f"| Scenario | Model | Test relative gap | Train relative gap | Test EXIT rate | Steps 1–{EARLY} regret | Steps 1–{EARLY} optimal-action rate |"),
              "|---|---|---|---|---|---|---|"]
        keys = list(vs)
        for s in SCENARIOS:
            for i in range(len(series)):
                c = [curve[(i, s, sd, STEPS)] for sd in SEEDS if (i, s, sd, STEPS) in curve]
                r = [reg[(keys[i], s, sd)] for sd in SEEDS if i < len(keys) and (keys[i], s, sd) in reg]
                L.append(f"| {s.upper()} | {lab(i)} | {ms([x[0] for x in c], pc)} | {ms([x[1] for x in c], pc)} | {ms([x[2] for x in c], pc)} | "
                         f"{ms([x[0] for x in r], lambda v: f'{v:.4f}')} | {ms([x[1] for x in r], pc)} |")
        L += ["", T("## 逐种子测试集 gap（20M）", "## Test gap per seed (20M)"), "",
              T("| 场景 | 模型 | 种子 1 | 种子 2 | 种子 3 |", "| Scenario | Model | Seed 1 | Seed 2 | Seed 3 |"), "|---|---|---|---|---|"]
        for s in SCENARIOS:
            for i in range(len(series)):
                L.append(f"| {s.upper()} | {lab(i)} | " + " | ".join(pc(curve[(i, s, sd, STEPS)][0]) if (i, s, sd, STEPS) in curve else "—"
                                                                    for sd in SEEDS) + " |")
        L.append("")
        p = V["report"] if zh else V["report"].with_suffix(".en.md")
        p.write_text("\n".join(L))
        print(f"report -> {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=list(CONFIG))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    sfx = "_smoke" if SMOKE else ""
    V = {"version": a.version, "cfg": CONFIG[a.version], "log_dir": ROOT / "logs" / f"v{a.version}{sfx}",
         "out": ROOT / ("results" if SMOKE else "paper_contents") / (f"smoke_v{a.version}" if SMOKE else f"v{a.version}")}
    V["manifest"], V["report"] = V["log_dir"] / "manifest.json", V["out"] / "report.md"
    V["log_dir"].mkdir(parents=True, exist_ok=True)
    V["out"].mkdir(parents=True, exist_ok=True)
    if a.report:
        write_report(V, json.loads(V["manifest"].read_text())["exp_ids"])
        return
    exp_ids = {}
    write_manifest(V, exp_ids)
    print(f"=== v{a.version} start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    active = [(*launch(V, sd, s, k, exp_ids), f"seed{sd} {s} {k}") for sd in SEEDS for k in V["cfg"]["variants"] for s in SCENARIOS]
    failed = []
    for p, f, label in active:
        ret = p.wait()
        f.close()
        print(f"[{time.strftime('%H:%M:%S')}] finished {label} ({'OK' if ret == 0 else f'EXIT={ret}'})", flush=True)
        if ret != 0:
            failed.append(label)
    print(f"=== done {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(failed)} failed ===", flush=True)
    from shared.report_mail import send_failure, send_report
    if SMOKE:                                       # no email for the end-to-end test
        send_failure = send_report = lambda *a_, **k_: None
    if failed:
        send_failure(f"v{a.version}", failed, log_dir=V["log_dir"].relative_to(ROOT), manifest=V["manifest"].relative_to(ROOT))
        return
    write_report(V, exp_ids)
    send_report(f"v{a.version}", V["report"], manifest=V["manifest"].relative_to(ROOT), driver_log=f"logs/run_v{a.version}_driver.log")


if __name__ == "__main__":
    main()
