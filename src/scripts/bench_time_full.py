#!/usr/bin/env python3
"""Per-instance wall time AND AR on the FULL fixed test set: ILP vs the 12 strategy x RL_algorithm variants.

Unlike bench_time_v6.py (first 200 instances, only means kept), every instance of data/testset_v6.0.4/{lt,eq,gt}.json
(5000 each) is timed separately and every per-instance value is stored, so the distributions can be plotted
(src/scripts/plot_time_full.py). No training; CPU only, one thread per worker.
  - RL : the seed-1 model of every summary cell (merge_2d_table.load, same models as the table); one environment per
         model, WARMUP warm-up episodes, then per instance: reset + (mask) + forward + step until done.
  - ILP: shared.ilp_utils.solve_ilp_max_ar (Dinkelbach, PuLP + CBC single thread), the whole call per instance
         (model building and every iteration); status, iterations and AR kept.

    .venv/bin/python src/scripts/bench_time_full.py [--workers 24] [--n 5000]
        -> paper_contents/ilp_vs_saferl/time_full.json   (log: logs/bench_time_full.log when run with nohup)
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))
OUT = ROOT / "paper_contents" / "ilp_vs_saferl"
TEST = ROOT / "data/testset_v6.0.4/{scen}.json"
WARMUP, SEED = 5, 1


def instances(scen, n):
    return json.loads(Path(str(TEST).format(scen=scen)).read_text())[:n]


def rl_time(job):
    row, n = job
    import torch
    torch.set_num_threads(1)
    from paper_rl.env import PlacementEnv
    from paper_rl.train import model_class
    d = Path(row["dir"])
    res = json.loads((d / "results.json").read_text())
    mech, learner = res["mechanism"], res["learner"]
    tr = res.get("training") or {}
    lam = tr.get("final_lambda", 0.0) if mech == "lagrange" else 0.0
    lm = tr.get("lambda_max", 50.0)
    lm = float("inf") if lm == "none" else float(lm)
    model = model_class(learner, mech).load(str(next(d.glob("model_*"))), device="cpu")
    test = instances(row["scen"], n)
    env = PlacementEnv(test, mech, res["reward_mode"], lam=lam, obs_mode=res["obs"], full_episode=res["full_episode"],
                       order=res["order"], order_seed=res["order_seed"], exit_action=res["exit_action"],
                       action_mode=res["action_mode"], lam_max=lm)
    masked = learner == "ppo" and mech == "mask"

    def episode(k):
        env.use_instance(k)
        obs, _ = env.reset()
        done = False
        while not done:
            a, _ = (model.predict(obs, deterministic=True, action_masks=env.action_masks()) if masked
                    else model.predict(obs, deterministic=True))
            obs, _, done, _, info = env.step(int(a))
        return info

    for k in range(min(WARMUP, len(test))):
        episode(k)
    ms, ar, ok = [], [], []
    for k in range(len(test)):
        t0 = time.perf_counter()
        info = episode(k)
        ms.append((time.perf_counter() - t0) * 1000)
        ar.append(float(info["ar"]))
        ok.append(bool(info["valid_placed"] == env.M and info["capacity_violations"] == 0 and info["conflict_violations"] == 0))
    return row["mech"], row["learner"], row["scen"], dict(ms=ms, ar=ar, feasible=ok)


def ilp_time(job):
    from shared.ilp_utils import solve_ilp_max_ar
    scen, k, n = job
    x = instances(scen, n)[k]
    t0 = time.perf_counter()
    r = solve_ilp_max_ar(x["ECUs"], x["SVCs"], x["conflict_sets"])
    return scen, k, (time.perf_counter() - t0) * 1000, r["iterations"], float(r["avg_utilization"]), str(r["status"])


def main():
    from merge_2d_table import load, SCENS
    t0 = time.time()
    rows = [r for r in load() if r["seed"] == SEED]
    assert len(rows) == 36, len(rows)
    n = ARGS.n
    sizes = {s: len(instances(s, n)) for s in SCENS}
    out = dict(generated_at=None, script="src/scripts/bench_time_full.py", testset="data/testset_v6.0.4", seed=SEED,
               n_inst=sizes, workers=ARGS.workers, ilp={}, rl={})
    with Pool(ARGS.workers) as pool:
        rl = pool.map_async(rl_time, [(r, n) for r in rows], chunksize=1)
        ilp = {s: [None] * sizes[s] for s in SCENS}
        done = 0
        for s, k, ms, it, ar, status in pool.imap_unordered(ilp_time, [(s, k, n) for s in SCENS for k in range(sizes[s])],
                                                             chunksize=4):
            ilp[s][k] = (ms, it, ar, status)
            done += 1
            if done % 1000 == 0:
                print(f"ILP {done}/{sum(sizes.values())}  {(time.time() - t0) / 60:.1f} min", flush=True)
        rl = rl.get()
    for s, v in ilp.items():
        out["ilp"][s] = dict(ms=[x[0] for x in v], iters=[x[1] for x in v], ar=[x[2] for x in v], status=[x[3] for x in v])
    for m, l, s, v in rl:
        out["rl"].setdefault(f"{m}|{l}", {})[s] = v
    out["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["minutes"] = (time.time() - t0) / 60
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "time_full.json").write_text(json.dumps(out))
    print(OUT / "time_full.json", f"({out['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n", type=int, default=5000, help="instances per scenario (first n of the test set)")
    ARGS = ap.parse_args()
    try:
        main()
    except Exception as e:  # report failures by mail like the other scripts
        try:
            from shared.report_mail import send_failure
            send_failure("bench_time_full", [repr(e)])
        except Exception:
            pass
        raise
