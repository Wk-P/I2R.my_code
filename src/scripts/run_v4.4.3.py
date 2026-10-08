#!/usr/bin/env python3
"""v4.4.3: Mask PPO with the structure-aware policy network (src/paper_rl/graph_policy.py), pure RL.

Single change from v4.3.8: the policy / value network. MLP 256-256 on the flat observation ->
relation-biased attention over ECU / service / global tokens (src/paper_rl/graph_net.py) on the raw
state (obs = raw, which carries the same information). Shared encoder; every ECU scored by the same
actor head (ECU-permutation equivariant), EXIT by the global token, V(s) from the global token +
token mean (permutation invariant). Everything else as v4.3.8: data v4.3.1.4 (p = 0.6), descending
demand, ECU action, reward ar_pen, gamma 1, full episode, EXIT, 40 envs x 512 steps, batch 256,
10 epochs, clip 0.1, ent 0.005, lr 3e-4. --pilot: 3 scenarios x seed 1 x 1M.
The report puts side by side: MLP Mask PPO (v4.3.8 pilot), structure-aware Mask PPO (this
version), and the oracle-supervised reference (same network trained on ILP actions,
src/scripts/diag_arch.py, 50k instances): relative gap, EXIT, active ECUs, forced openings, and the
optimal-action rates of the regret diagnostic (src/scripts/diag_regret.py).

    nohup .venv/bin/python src/scripts/run_v4.4.3.py --pilot > logs/run_v4.4.3_pilot_driver.log 2>&1 &
    .venv/bin/python src/scripts/run_v4.4.3.py --pilot --report
    nohup .venv/bin/python src/scripts/run_v4.4.3.py --seeds 1 > logs/run_v4.4.3_driver.log 2>&1 &     # 5M, seed 1
    nohup .venv/bin/python src/scripts/run_v4.4.3.py --seeds 2,3 >> logs/run_v4.4.3_driver.log 2>&1 &  # later: adds seeds
    .venv/bin/python src/scripts/run_v4.4.3.py --report
5M runs also train an MLP Mask PPO control with the same budget (manifest key mask_ppo_mlp).
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
VERSION = "4.4.3"
DATA = "v4.3.1.4"
LOG_DIR = ROOT / "logs" / f"v{VERSION}"
MANIFEST = LOG_DIR / "manifest.json"
REPORT = ROOT / "paper_contents" / f"v{VERSION}" / "report.md"
RESULTS = ROOT / "results" / "unified"
MLP = ROOT / "logs" / "v4.3.8_pilot" / "manifest.json"          # MLP Mask PPO, same setting
REF = ROOT / "results" / "sup_diag"                                           # attn_<scen>_50000.pt (diag_arch.py)

CAPACITY = int(56 * 0.93)
SCENARIOS = ["lt", "eq", "gt"]
ALGO = "mask_ppo"
SEEDS = [1, 2, 3]
STEPS = 5_000_000
REWARD, NORM, OBS, GAMMA = "ar_pen", "none", "base", 1.0
PILOT = "--pilot" in sys.argv
if "--seeds" in sys.argv:                              # e.g. --seeds 1  (later --seeds 2,3 adds to the manifest)
    SEEDS = [int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
# full run: the structure-aware model plus an MLP control with the same 5M budget (key mask_ppo_mlp)
KEYS = {"mask_ppo": "graph", "mask_ppo_mlp": "mlp"}
PILOT_MANIFEST = ROOT / "logs" / f"v{VERSION}_pilot" / "manifest.json"
GPU = "--gpu" in sys.argv          # structure-aware runs on CUDA (.venv-gpu), one GPU per scenario; MLP stays on CPU
PY_GPU = str(ROOT / ".venv-gpu" / "bin" / "python")
if PILOT:
    KEYS = {"mask_ppo": "graph"}
    SEEDS, STEPS = [1], 1_000_000
    LOG_DIR = ROOT / "logs" / f"v{VERSION}_pilot"
    MANIFEST = LOG_DIR / "manifest.json"
    REPORT = ROOT / "paper_contents" / f"v{VERSION}" / "pilot_report.md"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def new_exp_id():
    return subprocess.run([PY, "-c", "from shared.paths import new_exp_id; print(new_exp_id())"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def write_manifest(exp_ids):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": VERSION, "commit": commit, "data": DATA, "net": "graph", "obs": "raw", "steps": STEPS,
                               "reward": REWARD, "reward_norm": NORM, "gamma": GAMMA, "full_episode": True,
                               "exit_action": True, "scenarios": SCENARIOS, "algos": [ALGO], "keys": KEYS,
                               "seeds": sorted(int(k) for k in exp_ids) or SEEDS,
                               "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST)


def launch(seed, scen, key, exp_ids):
    exp_id = new_exp_id()
    log = open(LOG_DIR / f"seed{seed}_{scen}_{key}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(seed),
           "PAPER_VERSION": VERSION, "DATA_VERSION": DATA}
    gpu = GPU and KEYS[key] == "graph"
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(SCENARIOS.index(scen) % 3)
    proc = subprocess.Popen([PY_GPU if gpu else PY, "-u", "-m", "paper_rl.train", *(["--device", "cuda"] if gpu else []), "--scen", scen, "--algo", ALGO, "--reward", REWARD,
                             "--reward-norm", NORM, "--obs", OBS, "--gamma", str(GAMMA), "--full-episode", "--exit-action",
                             "--net", KEYS[key], "--steps", str(STEPS), "--seed", str(seed)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    exp_ids.setdefault(str(seed), {}).setdefault(scen, {})[key] = exp_id
    write_manifest(exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched seed{seed} {scen} {key} exp_id={exp_id} pid={proc.pid}", flush=True)
    return proc, log


def regret_summary(out_md, args):
    js = Path(out_md).with_suffix(".json")
    if not js.exists():
        subprocess.run([PY, str(ROOT / "src" / "scripts" / "diag_regret.py"), "--out", str(out_md), *args], cwd=ROOT, check=True)
    return json.loads(js.read_text())


def write_report(exp_ids):
    from diag_open_ecu import replay
    ref = {"1": {s: {ALGO: f"ref:{REF / f'attn_{s}_50000.pt'}"} for s in SCENARIOS}}
    rel = lambda p: str(p.relative_to(ROOT))
    # (label, {seed: {scen: {key: exp_id}}}, key, regret report, regret args)
    if PILOT:
        kinds = [("MLP Mask PPO（v4.3.8 试跑，1M）", json.loads(MLP.read_text())["exp_ids"], ALGO,
                  ROOT / "paper_contents" / "v4.4.2" / "regret_diag.md", []),
                 ("结构感知 Mask PPO（1M，纯 RL）", exp_ids, ALGO, REPORT.parent / "regret_graph_ppo_pilot.md",
                  ["--manifest", rel(MANIFEST), "--label", f"v{VERSION} 结构感知 Mask PPO（种子 1、1M）"])]
    else:
        kinds = [("MLP Mask PPO（5M，对照）", exp_ids, "mask_ppo_mlp", REPORT.parent / "regret_mlp_ppo_5m.md",
                  ["--manifest", rel(MANIFEST), "--key", "mask_ppo_mlp", "--label", "MLP Mask PPO（v4.3.8 设置，种子 1、5M）"]),
                 ("结构感知 Mask PPO（1M 试跑）", json.loads(PILOT_MANIFEST.read_text())["exp_ids"], ALGO,
                  REPORT.parent / "regret_graph_ppo_pilot.md",
                  ["--manifest", rel(PILOT_MANIFEST), "--label", f"v{VERSION} 结构感知 Mask PPO（种子 1、1M）"]),
                 ("结构感知 Mask PPO（5M，纯 RL）", exp_ids, ALGO, REPORT.parent / "regret_graph_ppo_5m.md",
                  ["--manifest", rel(MANIFEST), "--label", f"v{VERSION} 结构感知 Mask PPO（种子 1、5M）"])]
    kinds.append(("ILP 监督参照（同一网络，模仿 ILP 动作）", ref, ALGO, REPORT.parent / "regret_supervised_ref.md",
                  ["--ref", str(REF), "--label", "ILP 监督参照（结构感知网络模仿 ILP 动作，5 万实例）"]))
    jobs = [(lb, DATA, s, ALGO, ids[sd][s][key]) for lb, ids, key, _, _ in kinds for sd in sorted(ids) for s in SCENARIOS
            if ids[sd].get(s, {}).get(key)]
    with Pool(min(len(jobs), 24)) as pool:
        out = pool.map(replay, jobs)
    reg = {lb: regret_summary(md, args) for lb, ids, key, md, args in kinds if ids.get("1")}
    s1 = {(lb, sc): ids["1"][sc][key] for lb, ids, key, _, _ in kinds for sc in SCENARIOS
          if ids.get("1", {}).get(sc, {}).get(key)}
    seed1 = {}                                            # seed-1 completed instances, for the common subset
    for job, opens, insts, n in out:
        if s1.get((job[0], job[2])) == job[4]:
            seed1[(job[0], job[2])] = {x["k"]: x for x in insts}
    res = {}
    for job, opens, insts, n in out:
        lb, _, s = job[:3]
        ar = float(np.mean([x["ar"] for x in insts])) if insts else None
        ilp = float(np.mean([x["ilp"] for x in insts])) if insts else None
        res.setdefault((lb, s), []).append({
            "exit": 1 - len(insts) / n, "ar": ar, "ilp": ilp, "gap": None if ar is None else ilp - ar,
            "rel": None if ar is None else (ilp - ar) / ilp,
            "act": float(np.mean([x["act"] for x in insts])) if insts else None,
            "ilp_act": float(np.mean([x["ilp_act"] for x in insts])) if insts else None,
            "forced": sum(o["kind"] == "forced" for o in opens) / n})

    def ms(vals, f):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        return f(np.mean(vals)) + (f" ± {f(np.std(vals, ddof=1))}" if len(vals) > 1 else "")

    f4 = lambda v: f"{v:.4f}"
    f2 = lambda v: f"{v:.2f}"
    pc = lambda v: f"{100 * v:.1f}%"
    lines = [f"# v{VERSION} 结构感知策略网络（纯 RL，其余同 v4.3.8）", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 唯一改动：策略 / 价值网络。MLP 256-256（扁平观测）→ 关系偏置注意力网络（`src/paper_rl/graph_net.py`，ECU / 服务 / 全局各为 token，"
             "四种关系作为注意力偏置），输入原始状态（`obs = raw`，信息与原观测相同）。编码器共享；每台 ECU 用同一个 actor 打分头（ECU 编号置换等变），"
             "EXIT 由全局 token 打分，V(s) 由全局 token 与 token 平均得到（置换不变）。其余与 v4.3.8 完全相同：数据 v4.3.1.4（p = 0.6）、需求降序、只选 ECU、"
             "ar_pen 奖励、γ = 1、不提前结束、EXIT、40 环境 × 512 步、batch 256、10 epochs、clip 0.1、熵系数 0.005、学习率 3e-4。",
             "- 各行：MLP Mask PPO = v4.3.8 设置（试跑报告为 v4.3.8 试跑 1M；正式报告为同一 5M 预算重新训练的对照）；结构感知 Mask PPO = 本版本（纯 RL，从零训练）；"
             "ILP 监督参照 = 同一网络用 5 万个实例的 ILP 最优动作做监督训练（`src/scripts/diag_arch.py`），**不是 RL，也不是上限**，只作表示能力的参照。",
             "- AR、ILP AR 在未 EXIT 的同一批测试实例上平均；绝对 gap = ILP AR − AR，相对 gap = (ILP AR − AR) / ILP AR。Mask 违约率恒为 0。",
             "- 开启 ECU 数在未 EXIT 的实例上平均；被逼开启 = 每个测试实例的平均次数（定义见 `paper_contents/v4.4.1/open_diag.md`）。",
             "- 选中最优动作的比例来自逐步 regret 诊断（`src/scripts/diag_regret.py`，ILP 固定前缀求最优完成，regret ≤ 1e-4 记为最优，在完成的实例上统计）。",
             f"- manifest：`{MANIFEST.relative_to(ROOT)}`"]
    seeds = sorted(exp_ids)
    lines.append("- **试跑**：结构感知 Mask PPO × 3 场景 × 种子 1 × 1M 步。" if PILOT else
                 f"- 5M 步，种子 {', '.join(seeds)}" + ("（均值 ± 样本标准差）" if len(seeds) > 1 else "（单种子）") + "；regret 诊断只用种子 1。")
    lines.append("")
    for s in SCENARIOS:
        lines += [f"## {s.upper()}", "",
                  "| 模型 | EXIT 率 | AR | ILP AR | 绝对 gap | 相对 gap | 开启 ECU 数 | ILP 开启数 | 被逼开启/实例 | 第 1 步选中最优 | 全部步选中最优 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for lb, *_ in kinds:
            st = res.get((lb, s), [])
            if not st:
                continue
            rg = reg.get(lb, {}).get(s)
            lines.append("| " + " | ".join([lb, ms([x["exit"] for x in st], pc), ms([x["ar"] for x in st], f4),
                                              ms([x["ilp"] for x in st], f4), ms([x["gap"] for x in st], f4),
                                              ms([x["rel"] for x in st], pc), ms([x["act"] for x in st], f2),
                                              ms([x["ilp_act"] for x in st], f2), ms([x["forced"] for x in st], f2),
                                              pc(rg["opt_rate_t"][0]) if rg else "—", pc(rg["opt_rate"]) if rg else "—"]) + " |")
        common = set.intersection(*[set(seed1[(lb, s)]) for lb, *_ in kinds if (lb, s) in seed1])
        lines += ["", f"共同完成实例上的 gap（种子 1，所有行都未 EXIT 的 {len(common)} 个实例，消除各自 EXIT 造成的选择差异）：", "",
                  "| 模型 | AR | ILP AR | 绝对 gap | 相对 gap |", "|---|---|---|---|---|"]
        for lb, *_ in kinds:
            if (lb, s) in seed1 and common:
                xs = [seed1[(lb, s)][k] for k in common]
                ar, il = np.mean([x["ar"] for x in xs]), np.mean([x["ilp"] for x in xs])
                lines.append(f"| {lb} | {f4(ar)} | {f4(il)} | {f4(il - ar)} | {pc((il - ar) / il)} |")
        rows = [(lb, reg[lb][s]) for lb, *_ in kinds if lb in reg and s in reg[lb]]
        if rows:
            m = len(rows[0][1]["opt_rate_t"])
            lines += ["", "选中最优动作的比例，按步号：", "", "| 模型 | " + " | ".join(f"第 {t + 1} 步" for t in range(m)) + " |",
                      "|---|" + "---|" * m]
            lines += ["| " + lb + " | " + " | ".join(pc(v) for v in r["opt_rate_t"]) + " |" for lb, r in rows]
            lines += ["", "各步 regret 占该模型 gap 的比例：", "", "| 模型 | " + " | ".join(f"第 {t + 1} 步" for t in range(m)) + " |",
                      "|---|" + "---|" * m]
            lines += ["| " + lb + " | " + " | ".join(pc(v) for v in r["regret_share_t"]) + " |" for lb, r in rows]
        lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"report -> {REPORT}", flush=True)


def main():
    if "--report" in sys.argv:
        write_report(json.loads(MANIFEST.read_text())["exp_ids"])
        return
    exp_ids = json.loads(MANIFEST.read_text())["exp_ids"] if MANIFEST.exists() and not PILOT else {}
    queue = [(sd, s, k) for sd in SEEDS for k in KEYS for s in SCENARIOS
             if not exp_ids.get(str(sd), {}).get(s, {}).get(k)]
    write_manifest(exp_ids)
    print(f"=== v{VERSION} start {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(queue)} jobs ===", flush=True)
    active, failed, done = [], [], 0
    while queue or active:
        while queue and 4 * (len(active) + 1) <= CAPACITY:
            sd, s, k = queue.pop(0)
            p, f = launch(sd, s, k, exp_ids)
            active.append((p, f, f"seed{sd} {s} {k}"))
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
