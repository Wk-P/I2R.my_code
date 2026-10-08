#!/usr/bin/env python3
"""Diagnostic: can the policy network predict the ILP-optimal action from the observation?

Supervised oracle-action prediction, no RL. Labels: the ILP optimum (Dinkelbach) of every
instance replayed in the environment's order (descending demand); when the expert opens an
empty ECU, every legal empty ECU of the same capacity is also a label (paper_rl/bc.py).
Two observations, everything else equal:
  A  base      -- the observation of v4.3.8;
  B  conflict  -- base + conflict graph among unplaced services (v4.3.4; full graph at step 1).
Training sets of 2000 / 10000 / 50000 fresh p = 0.6 instances (same generator as v4.3.1.4,
other seeds); test set = the seed-1 test split of v4.3.1.4 (400 instances, the ones of the
regret diagnostic). Network = the PPO policy net (MLP 256-256 tanh, masked softmax).
Reported per step: train / test accuracy (argmax in the label set), and the learned policy
rolled out on the test set (EXIT rate, AR gap).

    .venv/bin/python scripts/diag_supervised.py gen     # instances + ILP allocations -> results/sup_diag/
    .venv/bin/python scripts/diag_supervised.py run     # 3 scenarios x 2 obs x 3 sizes -> paper_contents/v4.4.2/supervised_diag.md
"""
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "sup_diag"
REPORT = ROOT / "paper_contents" / "v4.4.2" / "supervised_diag.md"
SCENARIOS = ["lt", "eq", "gt"]
SIZES = [2000, 10000, 50000]
OBS = ["base", "conflict"]
P, DATA, SEED = 0.6, "v4.3.1.4", 1
STEPS, BATCH, LR = 25_000, 512, 1e-3
SCEN_NM = {"lt": (10, 15), "eq": (10, 10), "gt": (15, 10)}


# ── data ────────────────────────────────────────────────────────────────────
def _gen(args):
    from paper_rl.data import draw
    from shared.ilp_utils import solve_ilp_max_ar
    scen, seed = args
    n, m = SCEN_NM[scen]
    rng = random.Random(seed)
    while True:
        caps, reqs, sets = draw(n, m, P, rng)
        r = solve_ilp_max_ar(caps, reqs, sets)
        if r["status"] == "Optimal":
            alloc = [None] * m
            for j, svcs in r["allocation"].items():
                for i in svcs:
                    alloc[i] = int(j)
            return {"ECUs": caps, "SVCs": reqs, "conflict_sets": sets, "ar_star": round(float(r["avg_utilization"]), 6),
                    "alloc": alloc}


def gen():
    OUT.mkdir(parents=True, exist_ok=True)
    for scen in SCENARIOS:
        t0 = time.time()
        base = 700_000_000 + SCENARIOS.index(scen) * 10_000_000
        with Pool(48) as pool:
            inst = pool.map(_gen, [(scen, base + i) for i in range(max(SIZES))], chunksize=16)
        (OUT / f"{scen}.json").write_text(json.dumps(inst))
        print(f"{scen}: {len(inst)} instances, {(time.time() - t0) / 60:.1f} min", flush=True)


def test_set(scen):
    from paper_rl.bc import alloc_path
    from paper_rl.data import load
    inst = load(scen, DATA)["instances"]
    alloc = json.loads(alloc_path(scen, DATA).read_text())
    idx = list(range(len(inst)))
    random.Random(SEED).shuffle(idx)                   # same as paper_rl.train.split_instances
    return [{**inst[i], "alloc": alloc[i]} for i in idx[int(0.8 * len(inst)):]]


def collect(args):
    """Expert replay -> obs, mask, labels, step index (paper_rl.bc.expert_dataset + t)."""
    from paper_rl.env import PlacementEnv
    insts, obs_mode = args
    env = PlacementEnv(insts, "mask", "ar_pen", obs_mode=obs_mode, full_episode=True, exit_action=True)
    O, MK, LB, T = [], [], [], []
    for k, x in enumerate(insts):
        order = sorted(range(len(x["SVCs"])), key=lambda i: -x["SVCs"][i])
        env.use_instance(k)
        obs, _ = env.reset()
        for t in range(env.M):
            mask = env.action_masks()
            j = x["alloc"][order[t]]
            lab = np.zeros_like(mask)
            if env.hosted[j]:
                lab[j] = True
            else:
                for q in range(env.N):
                    lab[q] = mask[q] and not env.hosted[q] and env.cap[q] == env.cap[j]
            O.append(obs); MK.append(mask); LB.append(lab); T.append(t)
            obs, *_ = env.step(j)
    return np.array(O, np.float32), np.array(MK), np.array(LB), np.array(T)


def dataset(pool, insts, obs_mode):
    parts = pool.map(collect, [(insts[c:c + 500], obs_mode) for c in range(0, len(insts), 500)])
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


# ── model ───────────────────────────────────────────────────────────────────
def net(d_in, d_out):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(d_in, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, d_out))


def logp(model, o, mk):
    import torch
    return torch.log_softmax(model(o).masked_fill(~mk, -1e9), -1)


def acc_by_step(model, data, m):
    import torch
    o, mk, lb, t = (torch.as_tensor(x) for x in data)
    with torch.no_grad():
        hit = lb[torch.arange(len(o)), logp(model, o, mk).argmax(-1)].numpy()
    t = t.numpy()
    return float(hit.mean()), [float(hit[t == s].mean()) for s in range(m)]


def rollout(model, insts, obs_mode):
    import torch
    from paper_rl.env import PlacementEnv
    env = PlacementEnv(insts, "mask", "ar_pen", obs_mode=obs_mode, full_episode=True, exit_action=True)
    ok, n_exit = [], 0
    for k, x in enumerate(insts):
        env.use_instance(k)
        obs, _ = env.reset()
        done = False
        while not done:
            mk = torch.as_tensor(env.action_masks())[None]
            with torch.no_grad():
                a = int(logp(model, torch.as_tensor(obs)[None], mk).argmax(-1))
            obs, _, done, _, info = env.step(a)
        if info["exited"]:
            n_exit += 1
        else:
            ok.append((info["ar"], x["ar_star"]))
    ar, ilp = (float(np.mean([v[i] for v in ok])) for i in (0, 1))
    return {"exit": n_exit / len(insts), "ar": ar, "ilp": ilp, "gap": ilp - ar, "rel": (ilp - ar) / ilp}


def train_one(args):
    import torch
    torch.set_num_threads(2)
    scen, obs_mode, size = args
    torch.manual_seed(0)
    n, m = SCEN_NM[scen]
    inst = json.loads((OUT / f"{scen}.json").read_text())[:size]
    test = test_set(scen)
    with Pool(4) as pool:
        tr = dataset(pool, inst, obs_mode)
        te = dataset(pool, test, obs_mode)
    o, mk, lb = (torch.as_tensor(x) for x in tr[:3])
    model = net(o.shape[1], mk.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for _ in range(STEPS):
        b = torch.randint(len(o), (BATCH,), generator=g)
        loss = -torch.logsumexp(logp(model, o[b], mk[b]).masked_fill(~lb[b], -1e9), -1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    sub = np.random.default_rng(0).choice(len(tr[0]), min(len(tr[0]), 30_000), replace=False)
    tr_acc, tr_t = acc_by_step(model, tuple(x[sub] for x in tr), m)
    te_acc, te_t = acc_by_step(model, te, m)
    ro = rollout(model, test, obs_mode)
    print(f"{scen} {obs_mode} {size}: train {tr_acc:.3f} test {te_acc:.3f} rel gap {ro['rel']:.3f} "
          f"exit {ro['exit']:.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return {"scen": scen, "obs": obs_mode, "size": size, "samples": len(tr[0]), "train": tr_acc, "test": te_acc,
            "train_t": tr_t, "test_t": te_t, **ro}


def one(scen, obs_mode, size):
    r = train_one((scen, obs_mode, int(size)))
    (OUT / f"res_{scen}_{obs_mode}_{size}.json").write_text(json.dumps(r))


def run():
    """One subprocess per (scenario, observation, size): each opens its own small Pool."""
    import subprocess
    jobs = [(s, o, n) for s in SCENARIOS for o in OBS for n in SIZES]
    procs = [subprocess.Popen([sys.executable, __file__, "one", s, o, str(n)]) for s, o, n in jobs]
    for p in procs:
        p.wait()
    res = [json.loads((OUT / f"res_{s}_{o}_{n}.json").read_text()) for s, o, n in jobs
           if (OUT / f"res_{s}_{o}_{n}.json").exists()]
    (OUT / "results.json").write_text(json.dumps(res))
    write_report(res)


def write_report(res):
    pc = lambda v: f"{100 * v:.1f}%"
    regret = {"lt": [22.9, 19.7, 21.4], "eq": [28.8, 32.6, 33.6], "gt": [32.8, 26.5, 35.8]}   # regret_diag.md
    lines = ["# 监督诊断：网络能否从观测预测 ILP 最优动作（不训练 RL）", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `scripts/diag_supervised.py`。",
             "- 标签：每个实例的 ILP 最优分配（Dinkelbach）按需求降序回放；专家开一个空 ECU 时，同容量、合法的空 ECU 都算对（与 v4.4.2 BC 相同）。"
             "准确率 = 网络在合法动作中 argmax 落在标签集内的比例，按专家轨迹上的状态统计。",
             "- 观测 A = 原观测（v4.3.8）；B = 原观测 + 未放置服务之间的冲突图（v4.3.4 的 `obs = conflict`，第 1 步即完整冲突图）。其余完全相同。",
             f"- 训练集：新生成的 p = 0.6 实例（同一生成器、不同随机种子）前 2000 / 10000 / 50000 个；测试集：v4.3.1.4 种子 1 的测试集 400 个（与 regret 诊断相同）。"
             f"网络与 PPO 策略网络相同（MLP 256-256 tanh，带 mask 的 softmax），Adam、{STEPS} 步、batch {BATCH}、学习率 {LR} 余弦衰减。",
             "- 训练准确率在最多 3 万个训练样本上统计。「回放」= 用学到的网络在测试集上实际放置（取 argmax），报 EXIT 率与相对 gap（未 EXIT 实例，均值之比）。",
             "- 参照：从零训练的 Mask PPO（v4.3.8 试跑）在自己轨迹上选中最优动作的比例（regret 诊断），第 1/2/3 步见表头下方。"
             "注意两者口径不同：这里的标签只是 ILP 的一个最优解（加同容量空 ECU），regret 诊断里的「最优」是所有 regret ≤ 1e-4 的动作，后者略宽。", ""]
    for s in SCENARIOS:
        m = SCEN_NM[s][1]
        rows = [r for r in res if r["scen"] == s]
        lines += [f"## {s.upper()}", "",
                  f"参照（PPO 选中最优）：第 1/2/3 步 {regret[s][0]}% / {regret[s][1]}% / {regret[s][2]}%。", "",
                  "| 观测 | 训练实例 | 训练准确率 | 测试准确率 | 测试第 1 步 | 测试第 2 步 | 测试第 3 步 | 训练第 1 步 | 回放 EXIT 率 | 回放相对 gap |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: (OBS.index(r["obs"]), r["size"])):
            lines.append(f"| {'A 原观测' if r['obs'] == 'base' else 'B +冲突图'} | {r['size']} | {pc(r['train'])} | {pc(r['test'])} | "
                         f"{pc(r['test_t'][0])} | {pc(r['test_t'][1])} | {pc(r['test_t'][2])} | {pc(r['train_t'][0])} | "
                         f"{pc(r['exit'])} | {pc(r['rel'])} |")
        lines += ["", "测试准确率按步号：", "",
                  "| 观测 | 训练实例 | " + " | ".join(f"第 {t + 1} 步" for t in range(m)) + " |",
                  "|---|---|" + "---|" * m]
        for r in sorted(rows, key=lambda r: (OBS.index(r["obs"]), r["size"])):
            lines.append(f"| {'A' if r['obs'] == 'base' else 'B'} | {r['size']} | " + " | ".join(pc(v) for v in r["test_t"]) + " |")
        lines.append("")
    REPORT.write_text("\n".join(lines))
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "one":
        one(*sys.argv[2:5])
    else:
        {"gen": gen, "run": run, "report": lambda: write_report(json.loads((OUT / "results.json").read_text()))}[cmd]()
