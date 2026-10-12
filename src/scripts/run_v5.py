#!/usr/bin/env python3
"""v5.0.x campaigns (code version 5.0.0): structure-aware network on GPU (MLP on CPU), 20M steps with checkpoints
at 1 / 3 / 5 / 7.5 / 10 / 15M, 3 seeds x 3 scenarios, ar_pen, gamma 1, full episode, EXIT (Mask), descending demand;
PPO / DQN hyper-parameters from paper_rl/config.py, the same for both networks.

    5.0.1  Mask-DQN / Mask-DDQN x {MLP, Transformer}, the 1600-instance split (the v4.5.3 experiment, redone with the
           v5.0.0 code); the 50k half is v4.5.2
    5.0.2  Transformer x {unconstrained, Lagrangian, Repair} x {PPO, DQN, DDQN}, 50k instances; Maskable references:
           PPO v4.4.15-17, DQN / DDQN v4.5.2

Speed-ups that leave training unchanged (v5.0.0, checked by src/scripts/check_v5.py: training is bit-identical to the
v4.5 code on CPU and GPU): Mask DQN / DDQN build the state features once for Q-values and mask (1.20x total throughput
with 18 concurrent structure-aware Mask-DQN runs); at most `parallel` runs at a time, each GPU run on the GPU with the
fewest active runs. Optional, off by default because they did not help in that check: V5_NUMA=1 binds every run
(numactl) to the NUMA node of its GPU / the less loaded node (0.99x), V5_MPS=1 shares each GPU through NVIDIA MPS
with this campaign's own pipe directory (0.97x).
Report: learning curve (test gap / failure rate on each seed's 400 test instances, training-set gap on 1600 training
instances), the 20M models with violation rates and (Maskable only) the per-step regret; Chinese + English, email.

    nohup .venv/bin/python src/scripts/run_v5.py --version 5.0.1 > logs/run_v5.0.1_driver.log 2>&1 &
    .venv/bin/python src/scripts/run_v5.py --version 5.0.1 --report
    .venv/bin/python src/scripts/run_v5.py --version 5.0.1 --rerun 1 gt dqn_graph_1600
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
SMOKE = os.environ.get("V5_SMOKE") == "1"          # end-to-end test: tiny budget, seed 1, results in RESULTS_SPACE
if SMOKE:
    SEEDS, STEPS, CKPTS = [1], 41_000, [20_480]
EARLY = 5

# variant -> (algo, net, 50k pool?, label zh, label en)
REF_452 = {"dqn_mlp": ("mask_dqn", "mlp", True, "MLP Mask-DQN（5 万）", "MLP Mask-DQN (50k)"),
                           "dqn_graph": ("mask_dqn", "graph", True, "Transformer Mask-DQN（5 万）", "Transformer Mask-DQN (50k)"),
                           "ddqn_mlp": ("mask_ddqn", "mlp", True, "MLP Mask-DDQN（5 万）", "MLP Mask-DDQN (50k)"),
                           "ddqn_graph": ("mask_ddqn", "graph", True, "Transformer Mask-DDQN（5 万）", "Transformer Mask-DDQN (50k)")}   # v4.5.2: the 50k references

CONFIG = {
    "5.0.1": {"variants": {"dqn_mlp_1600": ("mask_dqn", "mlp", False, "MLP Mask-DQN（1600）", "MLP Mask-DQN (1600)"),
                           "dqn_graph_1600": ("mask_dqn", "graph", False, "Transformer Mask-DQN（1600）", "Transformer Mask-DQN (1600)"),
                           "ddqn_mlp_1600": ("mask_ddqn", "mlp", False, "MLP Mask-DDQN（1600）", "MLP Mask-DDQN (1600)"),
                           "ddqn_graph_1600": ("mask_ddqn", "graph", False, "Transformer Mask-DDQN（1600）", "Transformer Mask-DDQN (1600)")},
              "title": ("Mask-DQN / Mask-DDQN：训练实例数（1600 vs 5 万）× 步数", "Mask-DQN / Mask-DDQN: training instances (1600 vs 50k) x steps"),
              "parallel": 36, "gpu_round_robin": True},
    "5.0.2": {"variants": {
                           "ppo_none": ("ppo", "graph", True, "无约束 PPO", "Unconstrained PPO"),
                           "ppo_lagrange": ("lagrange_ppo", "graph", True, "Lagrangian PPO", "Lagrangian PPO"),
                           "ppo_repair": ("repair_ppo", "graph", True, "Repair PPO", "Repair PPO"),
                           "dqn_none": ("dqn", "graph", True, "无约束 DQN", "Unconstrained DQN"),
                           "dqn_lagrange": ("lagrange_dqn", "graph", True, "Lagrangian DQN", "Lagrangian DQN"),
                           "dqn_repair": ("repair_dqn", "graph", True, "Repair DQN", "Repair DQN"),
                           "ddqn_none": ("ddqn", "graph", True, "无约束 DDQN", "Unconstrained DDQN"),
                           "ddqn_lagrange": ("lagrange_ddqn", "graph", True, "Lagrangian DDQN", "Lagrangian DDQN"),
                           "ddqn_repair": ("repair_ddqn", "graph", True, "Repair DDQN", "Repair DDQN"),
                           },
              "title": ("结构感知网络：无约束 / Lagrangian / Repair × PPO / DQN / DDQN（5 万实例、20M）",
                        "Structure-aware network: unconstrained / Lagrangian / Repair x PPO / DQN / DDQN (50k instances, 20M)"),
              "parallel": 36, "gpu_round_robin": True, "time_bench": True},
}


def references(version):
    """(label zh, label en, algo, [(seed, scen, exp_id, pattern, step)]) of the curves to compare against."""
    refs = []
    if version == "5.0.2":                          # Maskable PPO (Transformer), 50k: v4.4.15 (1/3/5M), v4.4.16 (7.5/10M), v4.4.17 (15/20M)
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
        refs.append(("Maskable PPO（v4.4.15–17）", "Maskable PPO (v4.4.15-17)", "mask_ppo", True, pts))
    if version == "5.0.2":                          # Maskable DQN / DDQN with the structure-aware network: v4.5.2
        man = ROOT / "logs" / "v4.5.2" / "manifest.json"
        if man.exists():
            ids = json.loads(man.read_text())["exp_ids"]
            for key, lz, le in (("dqn_graph", "Maskable DQN（v4.5.2）", "Maskable DQN (v4.5.2)"),
                                ("ddqn_graph", "Maskable DDQN（v4.5.2）", "Maskable DDQN (v4.5.2)")):
                algo = REF_452["variants"][key][0]
                pts = [(sd, s, ids[str(sd)][s][key], pat, st) for sd in SEEDS for s in SCENARIOS
                       for pat, st in [(f"ckpt_{c}_*", c) for c in CKPTS] + [("model_*", STEPS)] if ids.get(str(sd), {}).get(s, {}).get(key)]
                refs.append((lz, le, algo, True, pts))
    if version == "5.0.1":                          # the 50k half: v4.5.2
        man = ROOT / "logs" / "v4.5.2" / "manifest.json"
        if man.exists():
            ids = json.loads(man.read_text())["exp_ids"]
            for key, (algo, net, _, lz, le) in REF_452["variants"].items():
                pts = [(sd, s, ids[str(sd)][s][key], pat, st) for sd in SEEDS for s in SCENARIOS
                       for pat, st in [(f"ckpt_{c}_*", c) for c in CKPTS] + [("model_*", STEPS)] if ids.get(str(sd), {}).get(s, {}).get(key)]
                refs.append((lz, le, algo, True, pts))
    return refs


# ── placement (v5.0.0): GPU, NUMA node, MPS -- none of it changes what a run computes ──────────────────
def gpu_nodes() -> dict[int, int]:
    """CUDA index (PCI bus order, as nvidia-smi) -> NUMA node of the GPU (0 if unknown)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"], capture_output=True, text=True).stdout
    nodes = {}
    for line in out.strip().splitlines():
        i, bus = [x.strip() for x in line.split(",")]
        try:
            n = int(Path(f"/sys/bus/pci/devices/{bus[4:].lower()}/numa_node").read_text())
        except (OSError, ValueError):
            n = 0
        nodes[int(i)] = max(n, 0)
    return nodes


def mps_env() -> dict:
    d = ROOT / "logs" / "mps"
    return {"CUDA_MPS_PIPE_DIRECTORY": str(d / "pipe"), "CUDA_MPS_LOG_DIRECTORY": str(d / "log")}


def mps_start() -> None:
    """Start this project's MPS control daemon (own pipe directory: only processes given mps_env() use it)."""
    env = mps_env()
    for v in env.values():
        Path(v).mkdir(parents=True, exist_ok=True)
    if (Path(env["CUDA_MPS_PIPE_DIRECTORY"]) / "control").exists() and subprocess.run(
            ["pgrep", "-u", str(os.getuid()), "-x", "nvidia-cuda-mps"], capture_output=True).returncode == 0:
        print("[mps] daemon already running", flush=True)
        return
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env={**os.environ, **env}, check=True)
    print(f"[mps] daemon started (pipe {env['CUDA_MPS_PIPE_DIRECTORY']})", flush=True)


def mps_stop() -> None:
    subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env={**os.environ, **mps_env()})
    print("[mps] daemon stopped", flush=True)


def place(V, gpu: bool) -> tuple[int | None, int | None]:
    """(GPU, NUMA node) of a new run: a GPU run goes to the GPU with the fewest active runs and is bound to that
    GPU's node; a CPU run to the node with the fewest active runs."""
    act = V["active_res"].values()
    if gpu:
        g = min(V["gpu_node"], key=lambda i: (sum(1 for x in act if x[0] == i), i))
        return g, V["gpu_node"][g]
    nodes = sorted(set(V["gpu_node"].values()) | {0})
    return None, min(nodes, key=lambda n: (sum(1 for x in act if x[1] == n), n))


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
    g, node = place(V, gpu)
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        if V["mps"]:
            env.update(mps_env())
    pre = ["numactl", f"--cpunodebind={node}", f"--preferred={node}"] if V["numa"] else []
    cmd = pre + [PY_GPU if gpu else PY, "-u", "-m", "paper_rl.train", "--scen", scen, "--algo", algo, "--net", net,
           "--reward", "ar_pen", "--reward-norm", "none", "--obs", "base", "--gamma", "1.0", "--full-episode", "--exit-action",
           "--steps", str(STEPS), "--checkpoints", ",".join(map(str, CKPTS)), "--seed", str(seed)]
    cmd += ["--device", "cuda"] if gpu else []
    cmd += ["--train-pool", POOL] if pool else []
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    V["active_res"][proc.pid] = (g, node)
    exp_ids.setdefault(str(seed), {}).setdefault(scen, {})[key] = exp_id
    write_manifest(V, exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched seed{seed} {scen} {key} exp_id={exp_id} pid={proc.pid}"
          f" | {'GPU ' + str(g) if gpu else 'CPU'}{' MPS' if gpu and V['mps'] else ''}{f' node {node}' if V['numa'] else ''}", flush=True)
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
def _gap(model, insts, learner, mech="mask", lam=0.0):
    """Relative gap on the violation-free instances (Mask: not EXITed -- Mask never violates), the share of
    instances that are not (EXIT / any violation), and the capacity / privacy violation rates (v5.0.0)."""
    from paper_rl.train import evaluate
    obs = "raw" if model.policy.__class__.__name__.startswith("Graph") else "base"
    ev = evaluate(model, insts, mech, "ar_pen", lam, learner, obs, True, True)
    ok = [e for e in ev if not e["exited"] and e["cap_v"] == 0 and e["conf_v"] == 0]
    ar, ilp = (np.mean([e["ar"] for e in ok]), np.mean([e["ar_star"] for e in ok])) if ok else (np.nan, np.nan)
    return (float((ilp - ar) / ilp), 1 - len(ok) / len(ev),
            float(np.mean([e["cap_v"] > 0 for e in ev])), float(np.mean([e["conf_v"] > 0 for e in ev])))


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
    lam = 0.0
    if mech == "lagrange":                          # v5.0.0: evaluate with the lambda the policy was trained at
        run = RESULTS / s / algo / eid
        lam = (json.loads((run / "ckpt_lambda.json").read_text())[str(step)] if pat.startswith("ckpt_")
               else json.loads((run / "results.json").read_text())["training"]["final_lambda"])
    (te, ex, cap, conf), (tr, *_) = _gap(model, test, learner, mech, lam), _gap(model, train, learner, mech, lam)
    return job, (te, tr, ex, cap, conf)


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


# ── per-instance time: every 20M model vs the ILP (v5.0.2) ────────────────────────────────────────────────
ILP_WORKERS, TIME_WORKERS, WARMUP = 12, 24, 5


def test_index(scen, seed):
    """Indices (into the scenario's instance file) of a seed's test split, as paper_rl.train.split_instances."""
    from paper_rl.data import load
    n = len(load(scen)["instances"])
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx[int(0.8 * n):]


def policy_time(job):
    """One 20M model on its seed's test split, one CPU thread: full deterministic episode (reset, mask,
    forward, step) and the forward passes alone, per instance."""
    import torch
    torch.set_num_threads(1)
    from paper_rl.env import PlacementEnv
    from paper_rl.train import model_class, split_algo, split_instances
    i, algo, sd, s, eid = job
    files = list((RESULTS / s / algo / eid).glob("model_*"))
    if not files:
        return job, None
    mech, learner = split_algo(algo)
    test = split_instances(s, sd)[2]
    model = model_class(learner, mech).load(str(files[0]), device="cpu")
    lam = json.loads((RESULTS / s / algo / eid / "results.json").read_text())["training"].get("final_lambda", 0.0) if mech == "lagrange" else 0.0
    obs_mode = "raw" if model.policy.__class__.__name__.startswith("Graph") else "base"
    env = PlacementEnv(test, mech, "ar_pen", lam=lam, obs_mode=obs_mode, full_episode=True, exit_action=True)
    masked_ppo = learner == "ppo" and mech == "mask"

    def episode(k):
        env.use_instance(k)
        obs, _ = env.reset()
        fwd, done = 0.0, False
        while not done:
            t = time.perf_counter()
            a, _ = model.predict(obs, deterministic=True, **({"action_masks": env.action_masks()} if masked_ppo else {}))
            fwd += time.perf_counter() - t
            obs, _, done, _, _ = env.step(int(a))
        return fwd
    for k in range(WARMUP):
        episode(k)
    rows = []
    for k in range(len(test)):
        t0 = time.perf_counter()
        fwd = episode(k)
        rows.append(((time.perf_counter() - t0) * 1000, fwd * 1000))
    return job, rows


def ilp_time(args):
    from paper_rl.data import load
    from shared.ilp_utils import solve_ilp_max_ar
    s, gi = args
    x = load(s)["instances"][gi]
    t0 = time.perf_counter()
    solve_ilp_max_ar(x["ECUs"], x["SVCs"], x["conflict_sets"])
    return s, gi, (time.perf_counter() - t0) * 1000


def time_bench(series):
    """{(series idx, scen): (model ms list, forward ms list, ILP ms list on the same instances)}; the ILP is solved
    once per instance of a scenario (union of the seeds' test splits), single-threaded CBC, model building included."""
    tidx = {(s, sd): test_index(s, sd) for s in SCENARIOS for sd in SEEDS}
    jobs = [(i, algo, sd, s, eid) for i, (_, _, algo, _, pts) in enumerate(series) for sd, s, eid, pat, st in pts if pat == "model_*"]
    print(f"[report] timing {len(jobs)} models (1 thread each)", flush=True)
    with Pool(min(len(jobs), TIME_WORKERS)) as pool:
        pol = pool.map(policy_time, jobs)
    todo = sorted({(s, gi) for (s, sd), ix in tidx.items() for gi in ix})
    print(f"[report] timing the ILP on {len(todo)} instances ({ILP_WORKERS} single-thread solvers)", flush=True)
    with Pool(ILP_WORKERS) as pool:
        ilp = {(s, gi): ms for s, gi, ms in pool.map(ilp_time, todo, chunksize=4)}
    out = {}
    for (i, algo, sd, s, eid), rows in pol:
        if rows is None:
            continue
        m, f, l = out.setdefault((i, s), ([], [], []))
        m += [r[0] for r in rows]
        f += [r[1] for r in rows]
        l += [ilp[(s, gi)] for gi in tidx[(s, sd)]]
    return out


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
    # per-step regret is defined over legal ECUs, i.e. for Maskable policies only (v5.0.0)
    todo = [(k, v[0], sd) for k, v in vs.items() for sd in SEEDS if v[0].startswith("mask_")]
    for i, (k, algo, sd) in enumerate(todo, 1):
        print(f"[report] regret {i}/{len(todo)} | seed {sd} {k}", flush=True)
        g = regret(V, k, algo, sd)
        for s in SCENARIOS:
            reg[(k, s, sd)] = (sum(g[s]["regret_t"][:EARLY]), float(np.mean(g[s]["opt_rate_t"][:EARLY])))
    times = time_bench(series) if V["cfg"].get("time_bench") else {}
    steps = CKPTS + [STEPS]
    ms = lambda v, f: (lambda v: (f(np.mean(v)) + (f" ± {f(np.std(v, ddof=1))}" if len(v) > 1 else "")) if v else "—")(
        [x for x in v if x == x])                    # NaN (no violation-free instance) is left out
    pc = lambda v: "—" if v != v else f"{100 * v:.1f}%"
    sl = lambda st: f"{st / 1e6:g}M"
    for lang in ("zh", "en"):
        zh = lang == "zh"
        T = lambda z, e: z if zh else e
        lab = lambda i: series[i][0] if zh else series[i][1]
        L = [f"# v{V['version']} {T(*V['cfg']['title'])}", "",
             T(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/run_v5.py --version {V['version']}`。",
               f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; script `src/scripts/run_v5.py --version {V['version']}`."),
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
        L += ["", T("## 学习曲线：测试集未通过率（EXIT 或任一违约）", "## Learning curve: test failure rate (EXIT or any violation)"), "",
              T("gap 只在未通过率之外的实例（无 EXIT、无违约）上计算；未通过率高时，gap 只反映容易的那部分实例。",
                "The gap is computed on the remaining instances (no EXIT, no violation); with a high failure rate it covers only the easier instances."), ""]
        for s in SCENARIOS:
            L += [f"### {s.upper()}", "", "| " + T("步数", "Steps") + " | " + " | ".join(lab(i) for i in range(len(series))) + " |",
                  "|---|" + "---|" * len(series)]
            for st in steps:
                L.append(f"| {sl(st)} | " + " | ".join(ms([curve[(i, s, sd, st)][2] for sd in SEEDS if (i, s, sd, st) in curve], pc)
                                                        for i in range(len(series))) + " |")
            L.append("")
        L += ["", T("## 20M：测试集与训练集 gap、违约、前几步决策", "## 20M: test and training-set gap, violations, early decisions"), "",
              T(f"| 场景 | 模型 | 测试集相对 gap | 训练集相对 gap | 测试未通过率（EXIT / 违约） | 容量违约率 | 隐私违约率 | 第 1–{EARLY} 步 regret | 第 1–{EARLY} 步选中最优 |",
                f"| Scenario | Model | Test relative gap | Train relative gap | Test failure rate (EXIT / violation) | Capacity violation rate | Privacy violation rate | Steps 1–{EARLY} regret | Steps 1–{EARLY} optimal-action rate |"),
              "|---|---|---|---|---|---|---|---|---|"]
        keys = list(vs)
        for s in SCENARIOS:
            for i in range(len(series)):
                c = [curve[(i, s, sd, STEPS)] for sd in SEEDS if (i, s, sd, STEPS) in curve]
                r = [reg[(keys[i], s, sd)] for sd in SEEDS if i < len(keys) and (keys[i], s, sd) in reg]
                L.append(f"| {s.upper()} | {lab(i)} | {ms([x[0] for x in c], pc)} | {ms([x[1] for x in c], pc)} | {ms([x[2] for x in c], pc)} | "
                         f"{ms([x[3] for x in c], pc)} | {ms([x[4] for x in c], pc)} | "
                         f"{ms([x[0] for x in r], lambda v: f'{v:.4f}')} | {ms([x[1] for x in r], pc)} |")
        L += ["", T("regret 按合法 ECU 之间的最优差距定义，只适用于 Maskable；其余机制可能执行违约动作，记为 —。",
                    "Regret is defined over the legal ECUs, i.e. for Maskable only; the other mechanisms may execute violating actions (—)."), ""]
        L += ["", T("## 逐种子测试集 gap（20M）", "## Test gap per seed (20M)"), "",
              T("| 场景 | 模型 | 种子 1 | 种子 2 | 种子 3 |", "| Scenario | Model | Seed 1 | Seed 2 | Seed 3 |"), "|---|---|---|---|---|"]
        for s in SCENARIOS:
            for i in range(len(series)):
                L.append(f"| {s.upper()} | {lab(i)} | " + " | ".join(pc(curve[(i, s, sd, STEPS)][0]) if (i, s, sd, STEPS) in curve else "—"
                                                                    for sd in SEEDS) + " |")
        L.append("")
        if times:
            q = lambda v, p_: float(np.percentile(v, p_))
            L += [T("## 求解时间：每个 20M 模型 vs ILP（逐实例）", "## Solve time: every 20M model vs the ILP (per instance)"), "",
                  T("单线程 CPU。模型：完整确定性 episode（reset、掩码、前向、step），另列仅前向；ILP：Dinkelbach + CBC 单线程，含建模，"
                    "每个场景对三个种子测试集的并集各求一次。加速比 = ILP 时间 / 模型时间，在同一批实例上计算（每个种子 400 个，三个种子合并）。",
                    "One CPU thread. Model: full deterministic episode (reset, mask, forward, step), and the forward passes alone; ILP: "
                    "Dinkelbach + single-thread CBC, model building included, solved once per instance of the union of the three seeds' test "
                    "splits. Speed-up = ILP time / model time on the same instances (400 per seed, three seeds pooled)."), ""]
            for s in SCENARIOS:
                L += [f"### {s.upper()}", "",
                      T("| 方法 | 平均 ms | 中位数 ms | P95 ms | 仅前向 平均 ms | 加速比（平均） | 加速比（中位数） |",
                        "| Method | Mean ms | Median ms | P95 ms | Forward only, mean ms | Speed-up (mean) | Speed-up (median) |"),
                      "|---|---|---|---|---|---|---|"]
                ilp_all = next((v[2] for (i, s2), v in times.items() if s2 == s), None)
                if ilp_all:
                    L.append(f"| ILP | {np.mean(ilp_all):.1f} | {q(ilp_all, 50):.1f} | {q(ilp_all, 95):.1f} | — | — | — |")
                for i in range(len(series)):
                    if (i, s) not in times:
                        continue
                    m, f, l = times[(i, s)]
                    L.append(f"| {lab(i)} | {np.mean(m):.1f} | {q(m, 50):.1f} | {q(m, 95):.1f} | {np.mean(f):.1f} | "
                             f"{np.mean(l) / np.mean(m):.1f}× | {q(l, 50) / q(m, 50):.1f}× |")
                L.append("")
        p = V["report"] if zh else V["report"].with_suffix(".en.md")
        p.write_text("\n".join(L))
        print(f"report -> {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=list(CONFIG))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rerun", nargs=3, metavar=("SEED", "SCEN", "KEY"), action="append",
                    help="re-train one failed run under a new exp_id, replacing it in the manifest; report + email at the end")
    a = ap.parse_args()
    sfx = "_smoke" if SMOKE else ""
    V = {"version": a.version, "cfg": CONFIG[a.version], "log_dir": ROOT / "logs" / f"v{a.version}{sfx}",
         "out": ROOT / ("results" if SMOKE else "paper_contents") / (f"smoke_v{a.version}" if SMOKE else f"v{a.version}")}
    V["manifest"], V["report"] = V["log_dir"] / "manifest.json", V["out"] / "report.md"
    V["active_res"], V["gpu_node"] = {}, gpu_nodes()
    V["numa"] = os.environ.get("V5_NUMA", "0") == "1"      # off by default: no gain in check_v5 speed (0.99x)
    V["mps"] = os.environ.get("V5_MPS", "0") == "1" and any(v[1] == "graph" for v in V["cfg"]["variants"].values())
    V["log_dir"].mkdir(parents=True, exist_ok=True)
    V["out"].mkdir(parents=True, exist_ok=True)
    if a.report:
        write_report(V, json.loads(V["manifest"].read_text())["exp_ids"])
        return
    if a.rerun:                                     # replace failed runs; the old exp_id stays only in its log file name
        exp_ids = json.loads(V["manifest"].read_text())["exp_ids"]
        runs = [(int(sd), s, k) for sd, s, k in a.rerun]
        for sd, s, k in runs:
            assert sd in SEEDS and s in SCENARIOS and k in V["cfg"]["variants"], (sd, s, k)
            print(f"=== v{a.version} rerun seed{sd} {s} {k} (was {exp_ids.get(str(sd), {}).get(s, {}).get(k)}) "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    else:
        exp_ids = {}
        write_manifest(V, exp_ids)
        print(f"=== v{a.version} start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
        runs = [(sd, s, k) for sd in SEEDS for k in V["cfg"]["variants"] for s in SCENARIOS]
        # v5.0.0: longest first (PPO, then DQN / DDQN) so that the slowest runs do not start last
        runs.sort(key=lambda r: V["cfg"]["variants"][r[2]][0].split("_")[-1] != "ppo")
    failed = []
    limit = int(os.environ.get("V5_PARALLEL", 0)) or V["cfg"].get("parallel") or len(runs)   # v5.0.0: at most `parallel` runs at a time
    if V["mps"]:
        mps_start()
    pending, active = list(runs), []
    while pending or active:
        while pending and len(active) < limit:
            sd, s, k = pending.pop(0)
            active.append((*launch(V, sd, s, k, exp_ids), f"seed{sd} {s} {k}"))
        running = []
        for p, f, label in active:
            if p.poll() is None:
                running.append((p, f, label))
                continue
            f.close()
            V["active_res"].pop(p.pid, None)
            print(f"[{time.strftime('%H:%M:%S')}] finished {label} ({'OK' if p.returncode == 0 else f'EXIT={p.returncode}'})", flush=True)
            if p.returncode != 0:
                failed.append(label)
        if len(running) == len(active):             # nothing finished: wait before polling again
            time.sleep(2 if SMOKE else 30)
        active = running
    if V["mps"]:
        mps_stop()
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
