#!/usr/bin/env python3
"""Diagnostic: is the RL gap a training-signal problem or a data (number of training instances) problem?

1. sup  -- the supervised reference of diag_arch.py (structure-aware network imitating ILP actions,
           25k steps x batch 256, same optimiser / schedule) trained on
             rl1600: exactly the 1600 training instances of RL seed 1 (v4.3.1.4 split, ILP allocations of
                     paper_rl.bc), i.e. the same instances as RL, different signal;
             fresh5000: the first 5000 fresh instances of the supervised pool (diag_supervised.py).
           Existing 10k / 50k points: results/sup_diag/arch_<scen>_attn_<n>.json. Test: the same 400
           seed-1 test instances, deterministic rollout.
2. rl   -- the v4.4.5 GPU baseline RL models (seeds 1-3), deterministic rollout on their own 1600
           training instances and on their 400 test instances: train gap vs test gap (memorisation?).

    .venv/bin/python src/scripts/diag_datasize.py run      # both, then the report
    .venv/bin/python src/scripts/diag_datasize.py report   -> paper_contents/v4.4.13/data_size_diag.md (+ .en.md)
"""
import json
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
import diag_arch as A          # noqa: E402
import diag_supervised as S    # noqa: E402

OUT = S.OUT
REPORT = ROOT / "paper_contents" / "v4.4.13" / "data_size_diag.md"
SCENARIOS = ["lt", "eq", "gt"]
SUP_JOBS = [(s, tag) for s in SCENARIOS for tag in ("rl1600", "fresh5000")]
THREADS = 8
BASE = {1: ROOT / "logs" / "v4.4.5_pilot" / "manifest.json", 2: ROOT / "logs" / "v4.4.5_seeds" / "manifest.json",
        3: ROOT / "logs" / "v4.4.5_seeds" / "manifest.json"}


def rl_train_set(scen, seed=1):
    """The RL training split (first 80% after the seeded shuffle) with ILP allocations."""
    from paper_rl.bc import alloc_path
    from paper_rl.data import load
    inst = load(scen, S.DATA)["instances"]
    alloc = json.loads(alloc_path(scen, S.DATA).read_text())
    idx = list(range(len(inst)))
    random.Random(seed).shuffle(idx)                   # same as paper_rl.train.split_instances
    return [{**inst[i], "alloc": alloc[i]} for i in idx[:int(0.8 * len(inst))]]


def train_sup(scen, tag):
    """diag_arch.train_attn with a given training set (same budget: 25k steps x 256)."""
    import torch
    from paper_rl.graph_net import GraphPolicyNet
    torch.set_num_threads(THREADS)
    torch.manual_seed(0)
    inst = rl_train_set(scen) if tag == "rl1600" else json.loads((OUT / f"{scen}.json").read_text())[:5000]
    size = len(inst)
    test = S.test_set(scen)
    tr, te = A.raw(inst), A.raw(test)
    M = tr[1].shape[1]
    model = GraphPolicyNet()
    opt = torch.optim.AdamW(model.parameters(), lr=A.ATTN_LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / A.WARMUP) * 0.5 * (1 + np.cos(np.pi * s / A.STEPS)))
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(A.STEPS):
        idx = torch.randint(size, (A.BATCH,), generator=g)
        t = torch.randint(M, (A.BATCH,), generator=g)
        caps, reqs, adj, assign, tt, js = A.batch_states(tr, idx, t)
        lg, _, mask, _ = model(caps, reqs, adj, assign, tt)
        lab = A.labels(caps, assign, mask, js)
        loss = -torch.logsumexp(torch.log_softmax(lg, -1).masked_fill(~lab, -1e9), -1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2500 == 0:
            print(f"[sup] {scen} {tag} step {step}/{A.STEPS} loss {float(loss.detach()):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    model.eval()
    tr_acc, _ = A.attn_eval(model, tr, M)
    te_acc, te_t = A.attn_eval(model, te, M)
    ro_te = A.attn_rollout(model, te, [x["ar_star"] for x in test])
    ro_tr = A.attn_rollout(model, tr, [x["ar_star"] for x in inst])
    torch.save(model.state_dict(), OUT / f"attn_{scen}_{tag}.pt")
    r = {"scen": scen, "tag": tag, "size": size, "train_acc": tr_acc, "test_acc": te_acc, "test_t": te_t,
         "test": ro_te, "train": ro_tr, "minutes": (time.time() - t0) / 60}
    (OUT / f"data_{scen}_{tag}.json").write_text(json.dumps(r))
    print(f"[sup] {scen} {tag}: train acc {tr_acc:.3f} test acc {te_acc:.3f} | test gap {ro_te['rel']:.4f} "
          f"train gap {ro_tr['rel']:.4f}", flush=True)


def rl_eval(job):
    """Deterministic rollout of one RL model on its train and test splits."""
    import torch
    torch.set_num_threads(1)
    from paper_rl.train import evaluate, model_class, split_instances
    scen, seed, eid = job
    _, train, test = split_instances(scen, seed)
    model = model_class("ppo", "mask").load(str(next((ROOT / "results" / "unified" / scen / "mask_ppo" / eid).glob("model_*"))),
                                            device="cpu")
    out = {}
    for name, insts in (("train", train), ("test", test)):
        ev = evaluate(model, insts, "mask", "ar_pen", 0.0, "ppo", "raw", True, True)
        ok = [e for e in ev if not e["exited"]]
        ar, ilp = np.mean([e["ar"] for e in ok]), np.mean([e["ar_star"] for e in ok])
        out[name] = {"n": len(ev), "exit": 1 - len(ok) / len(ev), "rel": float((ilp - ar) / ilp)}
    return scen, seed, eid, out


def run_rl():
    jobs = [(s, sd, json.loads(BASE[sd].read_text())["exp_ids"][str(sd)][s]["mask_ppo_base"]) for s in SCENARIOS for sd in (1, 2, 3)]
    with Pool(len(jobs)) as pool:
        res = pool.map(rl_eval, jobs)
    (OUT / "data_rl_train_test.json").write_text(json.dumps([{"scen": s, "seed": sd, "eid": e, **o} for s, sd, e, o in res]))
    print("[rl] done", flush=True)


def run():
    procs = [subprocess.Popen([sys.executable, __file__, "one", s, t]) for s, t in SUP_JOBS]
    run_rl()
    for p in procs:
        p.wait()
    report()


def report():
    pc = lambda v: f"{100 * v:.1f}%"
    sup = {}
    for s in SCENARIOS:
        for n in (10000, 50000):
            d = json.loads((OUT / f"arch_{s}_attn_{n}.json").read_text())
            sup[(s, n)] = {"src": "fresh", "train_acc": d["train"], "test_acc": d["test"], "rel": d["rel"], "exit": d["exit"], "train_rel": None}
        for tag in ("rl1600", "fresh5000"):
            f = OUT / f"data_{s}_{tag}.json"
            if f.exists():
                d = json.loads(f.read_text())
                sup[(s, d["size"])] = {"src": tag, "train_acc": d["train_acc"], "test_acc": d["test_acc"], "rel": d["test"]["rel"],
                                       "exit": d["test"]["exit"], "train_rel": d["train"]["rel"]}
    rl = json.loads((OUT / "data_rl_train_test.json").read_text()) if (OUT / "data_rl_train_test.json").exists() else []
    out = {}
    for lang in ("zh", "en"):
        zh = lang == "zh"
        T = lambda z, e: z if zh else e
        src = {"rl1600": T("RL 种子 1 的训练实例", "RL seed-1 training instances"), "fresh5000": T("新生成实例", "fresh instances"),
               "fresh": T("新生成实例", "fresh instances")}
        L = [T("# 数据量诊断：RL 的 gap 来自训练信号还是训练实例太少", "# Data-size diagnostic: is the RL gap due to the training signal or to too few training instances"), "",
             T(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/diag_datasize.py`。",
               f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; script `src/scripts/diag_datasize.py`."),
             T("- 监督参照：同一结构感知网络模仿 ILP 最优动作（`diag_arch.py` 的设置：25k 步 × batch 256、AdamW、学习率 5e-4、余弦调度），只改训练实例数。"
               "1600 = RL 种子 1 实际使用的 1600 个训练实例（同一批实例，只有训练信号不同）；5000 / 1 万 / 5 万 = 同一生成器新生成的实例。"
               "测试 = RL 种子 1 的 400 个测试实例，确定性回放；相对 gap 在未 EXIT 的实例上计算。",
               "- Supervised reference: the same structure-aware network imitating ILP-optimal actions (setting of `diag_arch.py`: 25k steps × "
               "batch 256, AdamW, lr 5e-4, cosine schedule); only the number of training instances changes. 1600 = exactly the 1600 training "
               "instances used by RL seed 1 (same instances, different training signal); 5000 / 10k / 50k = fresh instances from the same "
               "generator. Test = the 400 test instances of RL seed 1, deterministic rollout; relative gap over non-EXIT instances."),
             T("- RL：v4.4.5 在 GPU 上训练的三种子基线（1M 步），各自在自己的 1600 个训练实例和 400 个测试实例上做确定性回放。",
               "- RL: the v4.4.5 three-seed GPU baseline (1M steps), each rolled out deterministically on its own 1600 training instances and "
               "400 test instances."), "",
             T("## 1. 监督参照：训练实例数与测试 gap", "## 1. Supervised reference: number of training instances vs test gap"), "",
             T("| 场景 | 训练实例数 | 来源 | 训练准确率 | 测试准确率 | 测试相对 gap | 测试 EXIT 率 | 训练集相对 gap |",
               "| Scenario | Training instances | Source | Train accuracy | Test accuracy | Test relative gap | Test EXIT rate | Train-set relative gap |"),
             "|---|---|---|---|---|---|---|---|"]
        for s in SCENARIOS:
            for n in sorted(n for (sc, n) in sup if sc == s):
                d = sup[(s, n)]
                L.append(f"| {s.upper()} | {n} | {src[d['src']]} | {pc(d['train_acc'])} | {pc(d['test_acc'])} | {pc(d['rel'])} | {pc(d['exit'])} | "
                         f"{pc(d['train_rel']) if d['train_rel'] is not None else '—'} |")
        L += ["", T("## 2. RL：训练集与测试集上的 gap", "## 2. RL: gap on the training set vs the test set"), "",
              T("| 场景 | 种子 | 训练集相对 gap（1600） | 测试集相对 gap（400） | 训练集 EXIT | 测试集 EXIT |",
                "| Scenario | Seed | Train-set relative gap (1600) | Test-set relative gap (400) | Train EXIT | Test EXIT |"), "|---|---|---|---|---|---|"]
        for s in SCENARIOS:
            rs = sorted((r for r in rl if r["scen"] == s), key=lambda r: r["seed"])
            for r in rs:
                L.append(f"| {s.upper()} | {r['seed']} | {pc(r['train']['rel'])} | {pc(r['test']['rel'])} | {pc(r['train']['exit'])} | {pc(r['test']['exit'])} |")
            if rs:
                L.append(f"| {s.upper()} | {T('均值', 'mean')} | {pc(np.mean([r['train']['rel'] for r in rs]))} | {pc(np.mean([r['test']['rel'] for r in rs]))} | "
                         f"{pc(np.mean([r['train']['exit'] for r in rs]))} | {pc(np.mean([r['test']['exit'] for r in rs]))} |")
        L.append("")
        out[lang] = L
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(out["zh"]))
    REPORT.with_suffix(".en.md").write_text("\n".join(out["en"]))
    print(f"report -> {REPORT} (+ .en.md)", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "one":
        train_sup(sys.argv[2], sys.argv[3])
    else:
        {"run": run, "rl": lambda: (run_rl(), report()), "report": report}[cmd]()
