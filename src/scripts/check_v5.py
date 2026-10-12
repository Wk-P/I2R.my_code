#!/usr/bin/env python3
"""v5.0.0 acceptance checks: the speed-ups must not change training, and must make it faster.

    identical  Short runs with the same seed, once with the reference code (--old-src, default: the main checkout's
               src) and once with this checkout's src; training_curve.csv must match byte for byte and every model
               parameter must be equal. Runs on CPU (deterministic) and, separately, on GPU, where the reference
               code is also run twice to show how deterministic the GPU itself is.
    speed      K concurrent structure-aware runs (--algo mask_dqn: the v5.0.1 workload), steps/s per configuration
               (--configs): A old code | B new code, CUDA graphs off | C B + NUMA binding | D C + MPS | E new code
               (CUDA graphs on, the v5.0.0 default). Results (2026-10-12): Mask-DQN x 18: A 1.00 / B 1.20 / C 1.19 /
               D 1.16 / E 1.46; Mask-PPO x 9: A 1.00 / E 1.06 (logs/check_v5/speed_*.md).

    .venv/bin/python src/scripts/check_v5.py identical
    .venv/bin/python src/scripts/check_v5.py speed [--algo mask_dqn] [--k 18] [--steps 200000] [--configs A,B,E]
Results: logs/check_v5/{identical,speed}.md (+ .en.md)
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))
OUT = ROOT / "logs" / "check_v5"
PY, PY_GPU = str(ROOT / ".venv" / "bin" / "python"), str(ROOT / ".venv-gpu" / "bin" / "python")
OLD_SRC = "/home/soar009/github/my_code/src"
POOL = "results/sup_diag/{scen}.json"
COMMON = ["--reward", "ar_pen", "--reward-norm", "none", "--obs", "base", "--gamma", "1.0", "--full-episode", "--exit-action",
          "--seed", "1"]


def train(src: str, space: str, algo: str, net: str, steps: int, device: str, log: Path, extra_env=None, pre=(), wait=True):
    env = {**os.environ, "PYTHONPATH": src, "PYTHONUNBUFFERED": "1", "RESULTS_SPACE": space, "PAPER_VERSION": "5.0.0-check",
           "DATA_VERSION": "v4.3.1.4", **(extra_env or {})}
    env.pop("EXP_ID", None)
    cmd = [*pre, PY_GPU if device == "cuda" else PY, "-u", "-m", "paper_rl.train", "--scen", "eq", "--algo", algo, "--net", net,
           "--steps", str(steps), "--device", device, "--train-pool", POOL, *COMMON]
    log.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if wait:
        assert p.wait() == 0, f"training failed, see {log}"
    return p


def run_dir(log: Path) -> Path:
    m = re.search(r"saved -> (\S+)", log.read_text())
    assert m, f"no run directory in {log}"
    return Path(m[1])


def same_run(a: Path, b: Path) -> tuple[bool, str]:
    """(identical?, detail): training_curve.csv byte for byte, then every saved parameter."""
    import torch
    from paper_rl.train import model_class
    if (a / "training_curve.csv").read_bytes() != (b / "training_curve.csv").read_bytes():
        ra = list(csv.reader(open(a / "training_curve.csv")))
        rb = list(csv.reader(open(b / "training_curve.csv")))
        k = next((i for i, (x, y) in enumerate(zip(ra, rb)) if x != y), min(len(ra), len(rb)))
        return False, f"training_curve.csv differs from row {k} ({len(ra)} vs {len(rb)} rows)"
    res = json.loads((a / "results.json").read_text())
    from paper_rl.train import split_algo
    mech, learner = split_algo(res["algo"])
    ma = model_class(learner, mech).load(glob.glob(str(a / "model_*"))[0], device="cpu").policy.state_dict()
    mb = model_class(learner, mech).load(glob.glob(str(b / "model_*"))[0], device="cpu").policy.state_dict()
    diff = max((ma[k] - mb[k]).abs().max().item() for k in ma)
    return diff == 0.0, f"curve identical; max |parameter difference| = {diff:.3g}"


def identical(a):
    cases = [("mask_dqn", "graph", 20000), ("mask_ddqn", "graph", 20000), ("mask_ppo", "graph", 41000), ("mask_dqn", "mlp", 20000)]
    rows = []
    for dev in ("cpu", "cuda"):
        for algo, net, steps in cases:
            if dev == "cuda" and net == "mlp":
                continue
            tag = f"{dev}_{algo}_{net}"
            runs = {"old": train(a.old_src, "check_v5_old", algo, net, steps, dev, OUT / "identical" / f"{tag}_old.log", wait=False),
                    "new": train(str(ROOT / "src"), "check_v5_new", algo, net, steps, dev, OUT / "identical" / f"{tag}_new.log", wait=False)}
            if dev == "cuda":                       # GPU determinism of the reference code itself
                runs["old2"] = train(a.old_src, "check_v5_old2", algo, net, steps, dev, OUT / "identical" / f"{tag}_old2.log", wait=False)
            for k, p in runs.items():
                assert p.wait() == 0, f"{tag} {k} failed"
            d = {k: run_dir(OUT / "identical" / f"{tag}_{k}.log") for k in runs}
            ok, detail = same_run(d["old"], d["new"])
            ref = same_run(d["old"], d["old2"]) if "old2" in d else None
            rows.append((dev, algo, net, steps, ok, detail, ref))
            print(f"[identical] {tag}: {'IDENTICAL' if ok else 'DIFFERENT'} -- {detail}"
                  + (f" | old vs old: {'identical' if ref[0] else 'different'} ({ref[1]})" if ref else ""), flush=True)
    for lang in ("zh", "en"):
        zh = lang == "zh"
        L = ["# v5.0.0 训练一致性检查" if zh else "# v5.0.0 training-identity check", "",
             (f"同一种子、同样步数，参照代码 `{a.old_src}` 与本分支代码各训练一次，比较 training_curve.csv（逐字节）与全部模型参数。"
              "GPU 上参照代码再跑一次，用来判断 GPU 本身是否确定。" if zh else
              f"Same seed and steps, once with the reference code `{a.old_src}` and once with this branch; training_curve.csv "
              "(byte for byte) and every model parameter are compared. On GPU the reference code runs twice to show whether the "
              "GPU itself is deterministic."), "",
             "| 设备 | 算法 | 网络 | 步数 | 新旧一致 | 说明 | 参照代码两次是否一致 |" if zh else
             "| Device | Algorithm | Network | Steps | Old = new | Detail | Reference run twice identical |", "|---|---|---|---|---|---|---|"]
        for dev, algo, net, steps, ok, detail, ref in rows:
            yes, no = ("是", "否") if zh else ("yes", "no")
            L.append(f"| {dev} | {algo} | {net} | {steps:,} | {yes if ok else no} | {detail} | "
                     f"{'—' if ref is None else (yes if ref[0] else no) + ' (' + ref[1] + ')'} |")
        (OUT / ("identical.md" if zh else "identical.en.md")).write_text("\n".join(L) + "\n")
    print(f"-> {OUT / 'identical.md'}", flush=True)


def rate(log: Path, lo: int) -> float | None:
    """steps/s between the first [train] line at or after `lo` steps and the last one (cumulative averages ->
    elapsed times), i.e. without the start-up and the random warm-up."""
    rows = [(int(s.replace(",", "")), int(v.replace(",", ""))) for s, v in
            re.findall(r"\[train\] step=([\d,]+)/[\d,]+ .*?steps/s=([\d,]+)", log.read_text())]
    rows = [r for r in rows if r[1] > 0]
    first = next((r for r in rows if r[0] >= lo), None)
    if not first or rows[-1] == first:
        return None
    (s1, a1), (s2, a2) = first, rows[-1]
    return (s2 - s1) / (s2 / a2 - s1 / a1)


def speed(a):
    sys.path.insert(0, str(ROOT / "src" / "scripts"))
    import importlib
    v5 = importlib.import_module("run_v5")
    nodes = v5.gpu_nodes()
    gpus = sorted(nodes)
    new, off = str(ROOT / "src"), {"V5_CUDA_GRAPHS": "0"}
    configs = {"A": (a.old_src, False, False, {}), "B": (new, False, False, off), "C": (new, True, False, off),
               "D": (new, True, True, off), "E": (new, False, False, {})}
    rows = []
    for name in a.configs.split(","):
        src, numa, mps, extra = configs[name]
        if mps:
            v5.mps_start()
        procs = []
        for k in range(a.k):
            g = gpus[k % len(gpus)]
            env = {"CUDA_VISIBLE_DEVICES": str(g), **(v5.mps_env() if mps else {}), **extra}
            pre = ["numactl", f"--cpunodebind={nodes[g]}", f"--preferred={nodes[g]}"] if numa else []
            log = OUT / f"speed_{a.algo}" / f"{name}_{k}.log"
            procs.append((log, train(src, f"check_v5_speed_{name}", a.algo, "graph", a.steps, "cuda", log, env, pre, wait=False)))
        for log, p in procs:
            assert p.wait() == 0, f"{name}: {log} failed"
        if mps:
            v5.mps_stop()
        r = [rate(log, a.steps // 2) for log, _ in procs]
        r = [x for x in r if x]
        rows.append((name, numa, mps, extra, sum(r) / len(r), sum(r)))
        print(f"[speed] {name}: {sum(r) / len(r):.0f} steps/s per run, {sum(r):.0f} in total", flush=True)
    base = rows[0][5]
    for lang in ("zh", "en"):
        zh = lang == "zh"
        L = ["# v5.0.0 速度检查" if zh else "# v5.0.0 speed check", "",
             (f"{a.k} 个结构感知 {a.algo} 同时训练（轮流分到 {len(gpus)} 张 GPU），每个 {a.steps:,} 步；速度取后一半步数（去掉启动与随机预热）。"
              if zh else f"{a.k} concurrent structure-aware {a.algo} runs (round-robin over {len(gpus)} GPUs), {a.steps:,} steps each; "
              "speed over the second half of the steps (no start-up, no random warm-up)."), "",
             "| 配置 | 代码 | CUDA Graphs | NUMA 绑核 | MPS | 每个 run 步/秒 | 合计步/秒 | 相对 A |" if zh else
             "| Config | Code | CUDA graphs | NUMA binding | MPS | Steps/s per run | Total steps/s | vs A |", "|---|---|---|---|---|---|---|---|"]
        for name, numa, mps, extra, per, tot in rows:
            yes, no = ("是", "否") if zh else ("yes", "no")
            L.append(f"| {name} | {('参照' if zh else 'reference') if name == 'A' else 'v5.0.0'} | {no if name == 'A' or extra else yes} | "
                     f"{yes if numa else no} | {yes if mps else no} | "
                     f"{per:,.0f} | {tot:,.0f} | {tot / base:.2f}× |")
        (OUT / (f"speed_{a.algo}.md" if zh else f"speed_{a.algo}.en.md")).write_text("\n".join(L) + "\n")
    print(f"-> {OUT / f'speed_{a.algo}.md'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["identical", "speed"])
    ap.add_argument("--old-src", default=OLD_SRC)
    ap.add_argument("--k", type=int, default=18)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--algo", default="mask_dqn")
    ap.add_argument("--configs", default="A,B,E")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    identical(a) if a.what == "identical" else speed(a)


if __name__ == "__main__":
    main()
