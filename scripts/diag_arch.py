#!/usr/bin/env python3
"""Diagnostic: does a structure-aware network learn the ILP-optimal action where the MLP fails?

Same supervised oracle-action test as scripts/diag_supervised.py (same 50k p = 0.6 training
instances with ILP labels, same 400 test instances, same label sets), two new networks:
  attn  -- paper_rl/graph_net.GraphPolicyNet (tokens for ECUs / services / global, relation-
           biased attention, one shared ECU scoring head; ECU-permutation equivariant),
           trained on 10k and 50k instances;
  mlp3  -- a larger MLP (3 x 1024, tanh) on the base + conflict-graph observation, 50k (control:
           is the MLP just too small?).
Reported per step: train / test accuracy (argmax in the label set) and the learned policy rolled
out on the test set (EXIT rate, AR gap); the MLP 256-256 rows of supervised_diag.md for reference.

    .venv/bin/python scripts/diag_arch.py run   -> paper_contents/v4.4.3/arch_diag.md
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import diag_supervised as S  # noqa: E402

OUT = S.OUT
REPORT = ROOT / "paper_contents" / "v4.4.3" / "arch_diag.md"
SCENARIOS = S.SCENARIOS
JOBS = [(s, "attn", n) for s in SCENARIOS for n in (10000, 50000)] + [(s, "mlp3", 50000) for s in SCENARIOS]
STEPS, BATCH = 25_000, 256
ATTN_LR, MLP_LR, WARMUP = 5e-4, 1e-3, 1000


def raw(insts):
    """Instances (original index) -> placement-order tensors + expert ECU per step."""
    import torch
    caps, reqs, adj, exp = [], [], [], []
    for x in insts:
        M = len(x["SVCs"])
        order = sorted(range(M), key=lambda i: -x["SVCs"][i])                 # as PlacementEnv.reset
        pos = {o: n for n, o in enumerate(order)}
        a = np.zeros((M, M), dtype=bool)
        for s in x["conflict_sets"]:
            for u in s:
                for v in s:
                    if u != v:
                        a[pos[u], pos[v]] = True
        caps.append(x["ECUs"]); reqs.append([x["SVCs"][o] for o in order]); adj.append(a)
        exp.append([x["alloc"][o] for o in order])
    return (torch.tensor(caps, dtype=torch.float32), torch.tensor(reqs, dtype=torch.float32),
            torch.as_tensor(np.array(adj)), torch.tensor(exp))


def batch_states(d, idx, t):
    """Expert prefix states for instances idx at steps t -> net inputs + label sets."""
    import torch
    caps, reqs, adj, exp = (x[idx] for x in d)
    M = reqs.shape[1]
    assign = torch.where(torch.arange(M) < t.unsqueeze(1), exp, torch.full_like(exp, -1))
    return caps, reqs, adj, assign, t, exp[torch.arange(len(idx)), t]


def labels(caps, assign, mask, jstar):
    import torch
    N = caps.shape[1]
    B = len(jstar)
    active = (assign.unsqueeze(-1) == torch.arange(N)).any(1)
    one = torch.nn.functional.one_hot(jstar, N).bool()
    eq = mask[:, :N] & ~active & (caps == caps[torch.arange(B), jstar].unsqueeze(1))
    lab = torch.where(active[torch.arange(B), jstar].unsqueeze(1), one, one | eq)
    return torch.cat([lab, torch.zeros(B, 1, dtype=torch.bool)], 1)


def attn_eval(model, d, m):
    """Per-step accuracy on all expert states of d, in chunks."""
    import torch
    n = len(d[0])
    hits = np.zeros((n, m))
    with torch.no_grad():
        for t in range(m):
            for c in range(0, n, 2000):
                idx = torch.arange(c, min(c + 2000, n))
                caps, reqs, adj, assign, tt, js = batch_states(d, idx, torch.full((len(idx),), t))
                lg, _, mask, _ = model(caps, reqs, adj, assign, tt)
                lab = labels(caps, assign, mask, js)
                hits[c:c + len(idx), t] = lab[torch.arange(len(idx)), lg.argmax(-1)].numpy()
    return float(hits.mean()), [float(v) for v in hits.mean(0)]


def attn_rollout(model, d, ar_star):
    import torch
    caps, reqs, adj, _ = d
    B, M = reqs.shape
    assign = torch.full((B, M), -1)
    exited = torch.zeros(B, dtype=torch.bool)
    with torch.no_grad():
        for t in range(M):
            lg, _, mask, _ = model(caps, reqs, adj, assign, torch.full((B,), t))
            a = lg.argmax(-1)
            exited |= a == caps.shape[1]
            assign[:, t] = torch.where(exited, torch.full_like(a, -1), a)
        *_, ar = model(caps, reqs, adj, assign, torch.full((B,), M))
    ok = ~exited.numpy()
    ar, ils = ar.numpy()[ok], np.array(ar_star)[ok]
    return {"exit": float(exited.float().mean()), "ar": float(ar.mean()), "ilp": float(ils.mean()),
            "gap": float(ils.mean() - ar.mean()), "rel": float((ils.mean() - ar.mean()) / ils.mean())}


def train_attn(scen, size):
    import torch
    from paper_rl.graph_net import GraphPolicyNet
    torch.set_num_threads(4)
    torch.manual_seed(0)
    inst = json.loads((OUT / f"{scen}.json").read_text())[:size]
    test = S.test_set(scen)
    tr, te = raw(inst), raw(test)
    M = tr[1].shape[1]
    model = GraphPolicyNet()
    opt = torch.optim.AdamW(model.parameters(), lr=ATTN_LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + np.cos(np.pi * s / STEPS)))
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(STEPS):
        idx = torch.randint(size, (BATCH,), generator=g)
        t = torch.randint(M, (BATCH,), generator=g)
        caps, reqs, adj, assign, tt, js = batch_states(tr, idx, t)
        lg, _, mask, _ = model(caps, reqs, adj, assign, tt)
        lab = labels(caps, assign, mask, js)
        loss = -torch.logsumexp(torch.log_softmax(lg, -1).masked_fill(~lab, -1e9), -1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 5000 == 0:
            print(f"  {scen} attn {size} step {step} loss {float(loss):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    model.eval()
    sub = torch.as_tensor(np.random.default_rng(0).choice(size, min(size, 2000), replace=False))
    tr_acc, tr_t = attn_eval(model, tuple(x[sub] for x in tr), M)
    te_acc, te_t = attn_eval(model, te, M)
    ro = attn_rollout(model, te, [x["ar_star"] for x in test])
    torch.save(model.state_dict(), OUT / f"attn_{scen}_{size}.pt")
    return {"scen": scen, "net": "attn", "size": size, "train": tr_acc, "test": te_acc, "train_t": tr_t, "test_t": te_t,
            "minutes": (time.time() - t0) / 60, **ro}


def train_mlp3(scen, size):
    """The MLP test of diag_supervised.py with a 3 x 1024 network on obs base + conflict graph."""
    import torch.nn as nn
    S.net = lambda d_in, d_out: nn.Sequential(nn.Linear(d_in, 1024), nn.Tanh(), nn.Linear(1024, 1024), nn.Tanh(),
                                              nn.Linear(1024, 1024), nn.Tanh(), nn.Linear(1024, d_out))
    S.LR = MLP_LR
    t0 = time.time()
    r = S.train_one((scen, "conflict", size))
    return {**r, "net": "mlp3", "minutes": (time.time() - t0) / 60}


def one(scen, kind, size):
    r = (train_attn if kind == "attn" else train_mlp3)(scen, int(size))
    (OUT / f"arch_{scen}_{kind}_{size}.json").write_text(json.dumps(r))
    print(f"{scen} {kind} {size}: train {r['train']:.3f} test {r['test']:.3f} step1 {r['test_t'][0]:.3f} "
          f"rel gap {r['rel']:.3f} exit {r['exit']:.3f}", flush=True)


def run():
    procs = [subprocess.Popen([sys.executable, __file__, "one", s, k, str(n)]) for s, k, n in JOBS]
    for p in procs:
        p.wait()
    report()


def report():
    pc = lambda v: f"{100 * v:.1f}%"
    res = [json.loads(p.read_text()) for p in sorted(OUT.glob("arch_*.json"))]
    old = json.loads((OUT / "results.json").read_text())
    name = {"base": "MLP 256×2，原观测", "conflict": "MLP 256×2，+冲突图", "mlp3": "MLP 1024×3，+冲突图", "attn": "注意力网络（结构感知）"}
    lines = ["# 网络结构诊断：结构感知网络能否学会 ILP 最优动作（不训练 RL）", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `scripts/diag_arch.py`；网络 `paper_rl/graph_net.py`。",
             "- 与 `paper_contents/v4.4.2/supervised_diag.md` 完全相同的监督测试：同一批 p = 0.6 训练实例（带 ILP 最优标签）、同一 400 个测试实例、同一标签集（同容量的空 ECU 都算对）。",
             "- **注意力网络**：全局 / 每台 ECU / 每个服务各一个 token；服务间冲突、服务在哪台 ECU、服务与 ECU 上的服务冲突、服务放得进 ECU 剩余容量 四种关系"
             "作为可学习偏置加到注意力上；每台 ECU 用同一个打分头，EXIT 由全局 token 打分。输入只有原始状态（容量、需求、冲突图、已放置、当前步），"
             "特征在网络内部计算；已核对 2053 个状态的 mask / AR 与环境一致、ECU 换序时输出同步换序。3 层、d = 128、4 头，约 45 万参数。"
             f"AdamW、{STEPS} 步、batch {BATCH}、学习率 {ATTN_LR}（预热 + 余弦）。",
             f"- **MLP 1024×3**：对照，检验只是 MLP 太小；观测同 B（原观测 + 冲突图），{STEPS} 步、batch 512、学习率 {MLP_LR}。",
             "- 训练准确率：注意力网络在 2000 个训练实例上统计，MLP 在 3 万个训练样本上统计。回放 = 在测试集上取 argmax 实际放置。", ""]
    for s in SCENARIOS:
        m = S.SCEN_NM[s][1]
        rows = [({**r, "net": r["obs"]}) for r in old if r["scen"] == s and r["size"] == 50000]
        rows += sorted([r for r in res if r["scen"] == s], key=lambda r: (r["net"] != "mlp3", r["size"]))
        lines += [f"## {s.upper()}", "",
                  "| 网络 | 训练实例 | 训练准确率 | 测试准确率 | 测试第 1 步 | 测试第 2 步 | 测试第 3 步 | 训练第 1 步 | 回放 EXIT 率 | 回放相对 gap |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {name[r['net']]} | {r['size']} | {pc(r['train'])} | {pc(r['test'])} | {pc(r['test_t'][0])} | "
                         f"{pc(r['test_t'][1])} | {pc(r['test_t'][2])} | {pc(r['train_t'][0])} | {pc(r['exit'])} | {pc(r['rel'])} |")
        lines += ["", "测试准确率按步号：", "", "| 网络 | 训练实例 | " + " | ".join(f"第 {t + 1} 步" for t in range(m)) + " |",
                  "|---|---|" + "---|" * m]
        for r in rows:
            lines.append(f"| {name[r['net']]} | {r['size']} | " + " | ".join(pc(v) for v in r["test_t"]) + " |")
        lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    {"run": run, "report": report, "one": lambda: one(*sys.argv[2:5])}[sys.argv[1]]()
