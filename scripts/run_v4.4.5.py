#!/usr/bin/env python3
"""v4.4.5: structure-aware Mask PPO, two single-variable pilots on the optimizer (3 scenarios x seed 1 x 1M).

    LR : learning rate 3e-4 -> 1e-4   (ent_coef 0.005)
    ENT: ent_coef 0.005 -> 0.001       (lr 3e-4)
Everything else frozen as the v4.4.3 pilot (GAE lambda 0.95; v4.4.4 lambda = 1 had no effect): data
v4.3.1.4 (p = 0.6), descending demand, ECU action, reward ar_pen, gamma 1, full episode, EXIT, graph
net (obs raw), 40 envs x 512 steps, batch 256, 10 epochs, clip 0.1, seed 1, 1M steps.
GPU (.venv-gpu, one GPU per scenario, ~3x faster than CPU): the baseline is retrained on GPU too
(key mask_ppo_base), so all variants share the device; the CPU v4.4.3 pilot stays in the report as
a reference for device / run-to-run noise.
The report compares v4.4.3 pilot (baseline), LR, ENT and the ILP-supervised reference: steps 1-5
regret and optimal-action rate first, then step-1 optimal rate, EXIT, active ECUs vs ILP, forced
openings, relative gap. Baseline / reference regret come from paper_contents/v4.4.4 (with regret_t).

    nohup .venv/bin/python scripts/run_v4.4.5.py > scripts/logs/run_v4.4.5_pilot_driver.log 2>&1 &
    .venv/bin/python scripts/run_v4.4.5.py --report
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
PY_GPU = str(ROOT / ".venv-gpu" / "bin" / "python")
VERSION = "4.4.5"
DATA = "v4.3.1.4"
LOG_DIR = ROOT / "scripts" / "logs" / f"v{VERSION}_pilot"
MANIFEST = LOG_DIR / "manifest.json"
OUT = ROOT / "paper_contents" / f"v{VERSION}"
REPORT = OUT / "pilot_report.md"
BASE = ROOT / "scripts" / "logs" / "v4.4.3_pilot" / "manifest.json"          # lambda 0.95, same everything else
REF = ROOT / "results" / "sup_diag"                                           # attn_<scen>_50000.pt (diag_arch.py)

SCENARIOS = ["lt", "eq", "gt"]
ALGO = "mask_ppo"
SEED, STEPS = 1, 1_000_000
REWARD, NORM, OBS, GAMMA, NET = "ar_pen", "none", "base", 1.0, "graph"
# manifest key -> (variant label shown on the panel, extra paper_rl.train args)
VARIANTS = {"mask_ppo_base": ("base", []), "mask_ppo_lr": ("lr1e-4", ["--lr", "1e-4"]), "mask_ppo_ent": ("ent0.001", ["--ent-coef", "0.001"])}
PREV = ROOT / "paper_contents" / "v4.4.4"                                    # regret of baseline / reference (regret_t)
EARLY = 5                                                                     # "early steps" = steps 1..5
LOG_DIR.mkdir(parents=True, exist_ok=True)


def new_exp_id():
    return subprocess.run([PY, "-c", "from shared.paths import new_exp_id; print(new_exp_id())"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def write_manifest(exp_ids):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": VERSION, "commit": commit, "data": DATA, "net": NET, "obs": "raw", "steps": STEPS,
                               "reward": REWARD, "reward_norm": NORM, "gamma": GAMMA, "gae_lambda": 0.95, "lr": {"mask_ppo_base": 3e-4, "mask_ppo_lr": 1e-4, "mask_ppo_ent": 3e-4},
                               "ent_coef": {"mask_ppo_base": 0.005, "mask_ppo_lr": 0.005, "mask_ppo_ent": 0.001}, "device": "cuda",
                               "full_episode": True, "exit_action": True, "scenarios": SCENARIOS, "algos": [ALGO],
                               "keys": {k: v[0] for k, v in VARIANTS.items()}, "seeds": [SEED], "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST)


def launch(scen, key, exp_ids):
    exp_id = new_exp_id()
    log = open(LOG_DIR / f"seed{SEED}_{scen}_{key}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(SEED),
           "PAPER_VERSION": VERSION, "DATA_VERSION": DATA, "CUDA_VISIBLE_DEVICES": str(SCENARIOS.index(scen) % 3)}
    proc = subprocess.Popen([PY_GPU, "-u", "-m", "paper_rl.train", "--device", "cuda", "--scen", scen, "--algo", ALGO, "--reward", REWARD,
                             "--reward-norm", NORM, "--obs", OBS, "--gamma", str(GAMMA), *VARIANTS[key][1],
                             "--full-episode", "--exit-action", "--net", NET, "--steps", str(STEPS), "--seed", str(SEED)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    exp_ids.setdefault(str(SEED), {}).setdefault(scen, {})[key] = exp_id
    write_manifest(exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched {scen} {key} exp_id={exp_id} pid={proc.pid}", flush=True)
    return proc, log


def regret_summary(out_md, args):
    """diag_regret.py summary; rerun if missing or written before regret_t existed (v4.4.4)."""
    js = Path(out_md).with_suffix(".json")
    if not js.exists() or "regret_t" not in next(iter(json.loads(js.read_text()).values())):
        subprocess.run([PY, str(ROOT / "scripts" / "diag_regret.py"), "--out", str(out_md), *args], cwd=ROOT, check=True)
    return json.loads(js.read_text())


def write_report(exp_ids):
    from diag_open_ecu import replay
    rel = lambda p: str(p.relative_to(ROOT))
    ref = {"1": {s: {ALGO: f"ref:{REF / f'attn_{s}_50000.pt'}"} for s in SCENARIOS}}
    # (label, {seed: {scen: {key: exp_id}}}, key, regret report, regret args)
    kinds = [("基线（lr 3e-4，ent 0.005，GPU 重跑）", exp_ids, "mask_ppo_base", OUT / "regret_base.md",
              ["--manifest", rel(MANIFEST), "--key", "mask_ppo_base", "--label", f"v{VERSION} 基线结构感知 Mask PPO（GPU 重跑，种子 1、1M）"]),
             ("LR（lr 1e-4，ent 0.005）", exp_ids, "mask_ppo_lr", OUT / "regret_lr.md",
              ["--manifest", rel(MANIFEST), "--key", "mask_ppo_lr", "--label", f"v{VERSION}-LR 结构感知 Mask PPO（lr 1e-4，种子 1、1M）"]),
             ("ENT（lr 3e-4，ent 0.001）", exp_ids, "mask_ppo_ent", OUT / "regret_ent.md",
              ["--manifest", rel(MANIFEST), "--key", "mask_ppo_ent", "--label", f"v{VERSION}-ENT 结构感知 Mask PPO（ent 0.001，种子 1、1M）"]),
             ("v4.4.3 基线（CPU，原试跑）", json.loads(BASE.read_text())["exp_ids"], ALGO, PREV / "regret_v4.4.3_pilot.md", []),
             ("ILP 监督参照", ref, ALGO, PREV / "regret_supervised_ref.md", [])]
    jobs = [(lb, DATA, s, ALGO, ids["1"][s][key]) for lb, ids, key, _, _ in kinds for s in SCENARIOS]
    with Pool(len(jobs)) as pool:
        out = pool.map(replay, jobs)
    reg = {lb: regret_summary(md, args) for lb, _, _, md, args in kinds}
    res, done = {}, {}
    for job, opens, insts, n in out:
        lb, s = job[0], job[2]
        done[(lb, s)] = {x["k"]: x for x in insts}
        ar, ilp = np.mean([x["ar"] for x in insts]), np.mean([x["ilp"] for x in insts])
        act, ilp_act = np.mean([x["act"] for x in insts]), np.mean([x["ilp_act"] for x in insts])
        res[(lb, s)] = {"exit": 1 - len(insts) / n, "ar": ar, "ilp": ilp, "gap": ilp - ar, "rel": (ilp - ar) / ilp,
                        "act": act, "dact": act - ilp_act, "forced": sum(o["kind"] == "forced" for o in opens) / n}

    f4 = lambda v: f"{v:.4f}"
    pc = lambda v: f"{100 * v:.1f}%"
    lines = [f"# v{VERSION} 结构感知 Mask PPO：学习率 / 熵系数单变量试跑", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 两个单变量实验，各只改一个量：LR = 学习率 3e-4 → 1e-4（熵系数 0.005 不变）；ENT = 熵系数 0.005 → 0.001（学习率 3e-4 不变）。",
             "- 其余全部与 v4.4.3 试跑相同：GAE λ = 0.95（v4.4.4 的 λ = 1 无效果）、结构感知网络（obs = raw）、数据 v4.3.1.4（p = 0.6）、需求降序、只选 ECU、"
             "ar_pen 奖励、γ = 1、不提前结束、EXIT、40 环境 × 512 步、batch 256、10 epochs、clip 0.1、种子 1、1M 步。",
             "- 设备：为提速改在 GPU 上训练（每个场景一块卡）。基线在 GPU 上重跑，三行同设备可比；v4.4.3 原试跑（CPU）一行保留，"
             "它与 GPU 重跑基线之间的差异可视为设备 / 单次运行的噪声幅度。",
             f"- 首要判断量：第 1–{EARLY} 步 regret（每个完成实例前 {EARLY} 步 regret 之和的平均，各步 regret 之和 = 绝对 gap）与第 1–{EARLY} 步选中最优的比例；"
             "其次是第 1 步选中最优、EXIT、开启 ECU 数 − ILP、被逼开启、相对 gap。",
             "- AR、ILP AR、开启 ECU 数在各自未 EXIT 的测试实例上平均；相对 gap = (ILP AR − AR) / ILP AR；Mask 违约率恒为 0。"
             "被逼开启 = 每个测试实例的平均次数（定义见 `paper_contents/v4.4.1/open_diag.md`）。",
             "- 逐步 regret 诊断：`scripts/diag_regret.py`，种子 1 测试集前 400 个实例，regret ≤ 1e-4 记为最优。基线与监督参照的诊断取自 `paper_contents/v4.4.4/`。",
             "- ILP 监督参照：同一网络用 5 万个实例的 ILP 最优动作监督训练（`scripts/diag_arch.py`），不是 RL，也不是上限，只作参照。",
             f"- manifest：`{rel(MANIFEST)}`；基线：`{rel(BASE)}`", ""]
    for s in SCENARIOS:
        lines += [f"## {s.upper()}", "",
                  f"| 模型 | EXIT 率 | AR | ILP AR | 相对 gap | 开启 ECU 数 − ILP | 被逼开启/实例 | 第 1 步选中最优 | 第 1–{EARLY} 步选中最优 "
                  f"| 第 1–{EARLY} 步 regret | 第 1–{EARLY} 步 regret 占 gap | 全部步选中最优 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for lb, *_ in kinds:
            r, g = res[(lb, s)], reg[lb][s]
            early = sum(g["regret_t"][:EARLY])
            lines.append("| " + " | ".join([lb, pc(r["exit"]), f4(r["ar"]), f4(r["ilp"]), pc(r["rel"]), f"{r['dact']:+.2f}",
                                              f"{r['forced']:.2f}", pc(g["opt_rate_t"][0]), pc(np.mean(g["opt_rate_t"][:EARLY])),
                                              f4(early), pc(sum(g["regret_share_t"][:EARLY])), pc(g["opt_rate"])]) + " |")
        common = set.intersection(*[set(done[(lb, s)]) for lb, *_ in kinds])
        lines += ["", f"共同完成实例上的相对 gap（三行都未 EXIT 的 {len(common)} 个实例）：", "", "| 模型 | AR | ILP AR | 相对 gap |", "|---|---|---|---|"]
        for lb, *_ in kinds:
            xs = [done[(lb, s)][k] for k in common]
            ar, il = np.mean([x["ar"] for x in xs]), np.mean([x["ilp"] for x in xs])
            lines.append(f"| {lb} | {f4(ar)} | {f4(il)} | {pc((il - ar) / il)} |")
        m = len(reg[kinds[0][0]][s]["opt_rate_t"])
        head = "| 模型 | " + " | ".join(f"第 {t + 1} 步" for t in range(m)) + " |"
        lines += ["", "选中最优动作的比例，按步号：", "", head, "|---|" + "---|" * m]
        lines += ["| " + lb + " | " + " | ".join(pc(v) for v in reg[lb][s]["opt_rate_t"]) + " |" for lb, *_ in kinds]
        lines += ["", "平均 regret，按步号：", "", head, "|---|" + "---|" * m]
        lines += ["| " + lb + " | " + " | ".join(f4(v) for v in reg[lb][s]["regret_t"]) + " |" for lb, *_ in kinds]
        lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"report -> {REPORT}", flush=True)


def main():
    if "--report" in sys.argv:
        write_report(json.loads(MANIFEST.read_text())["exp_ids"])
        return
    exp_ids = {}
    write_manifest(exp_ids)
    print(f"=== v{VERSION} pilot start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    active = [(*launch(s, k, exp_ids), f"{s} {k}") for k in VARIANTS for s in SCENARIOS]
    failed = []
    for p, f, s in active:
        ret = p.wait()
        f.close()
        print(f"[{time.strftime('%H:%M:%S')}] finished {s} ({'OK' if ret == 0 else f'EXIT={ret}'})", flush=True)
        if ret != 0:
            failed.append(s)
    print(f"=== done {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(failed)} failed ===", flush=True)
    if failed:
        print("FAILED:", failed, flush=True)
        return
    write_report(exp_ids)


if __name__ == "__main__":
    main()
