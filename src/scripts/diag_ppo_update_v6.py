#!/usr/bin/env python3
"""v6.0.3.1 diagnostic: how far does each PPO update move the v6.0.0 Mask PPO policy?

The 5M / 10M / 15M checkpoints and the 20M final model of every Mask PPO run (3 scenarios x seeds 1-3) are
trained on for a short window (--steps, default 245,760 = 12 PPO updates) with exactly the training settings
(same envs, training pool, reward, hyper-parameters, optimiser state), only logging SB3's per-update statistics:
approx_kl, clip_fraction, entropy, value loss, explained variance. Nothing is saved back; no model is changed.

    .venv/bin/python src/scripts/diag_ppo_update_v6.py   -> paper_contents/v6.0.3.1/ppo_update.{json,md,en.md}, mailed
"""
import argparse
import csv
import functools
import glob
import json
import statistics as st
import sys
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "paper_contents/v6.0.3.1"
POINTS = [5_000_000, 10_000_000, 15_000_000, 20_000_000]
SCENS = ["eq", "gt", "lt"]
KEYS = ["approx_kl", "clip_fraction", "entropy_loss", "value_loss", "explained_variance", "policy_gradient_loss"]


def runs():
    out = []
    for r in json.loads((ROOT / "logs/v6.0.0/manifest.json").read_text())["runs"]:
        if r["strategy"] == "maskable" and r["RL_algorithm"] == "PPO":
            d = Path(glob.glob(str(ROOT / f"results/unified/{r['scenario']}/{r['dir']}/{r['exp_id']}/results.json"))[0]).parent
            out.append(dict(dir=str(d), res=json.loads((d / "results.json").read_text())))
    return out


def probe(args):
    import torch
    torch.set_num_threads(3)
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from paper_rl import config as C
    from paper_rl.env import PlacementEnv
    from paper_rl.train import model_class
    run, point, steps, device = args
    res, d = run["res"], Path(run["dir"])
    train = json.loads((ROOT / res["train_pool"].format(scen=res["scenario"])).read_text())
    venv = DummyVecEnv([functools.partial(
        lambda s: Monitor(PlacementEnv(train, "mask", res["reward_mode"], lam=C.LAMBDA_INIT, rng_seed=s, obs_mode=res["obs"],
                                       full_episode=res["full_episode"], exit_action=res["exit_action"],
                                       action_mode=res["action_mode"], order=res["order"], order_seed=res["order_seed"])),
        res["seed"] * 1000 + 500 + k) for k in range(C.PPO_N_ENVS)])
    src = next(d.glob(f"ckpt_{point}_*.zip")) if point < 20_000_000 else next(d.glob("model_*"))
    model = model_class("ppo", "mask").load(str(src), env=venv, device=device)
    with tempfile.TemporaryDirectory() as tmp:
        model.set_logger(configure(tmp, ["csv"]))
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        rows = list(csv.DictReader(open(Path(tmp) / "progress.csv")))
    ups = [{k: float(r[f"train/{k}"]) for k in KEYS if r.get(f"train/{k}")} for r in rows]
    ups = [u for u in ups if "approx_kl" in u]
    return dict(scen=res["scenario"], seed=res["seed"], point=point, exp_id=res["exp_id"],
                clip_range=float(model.clip_range(1.0)), n_epochs=model.n_epochs, updates=ups,
                mean={k: st.mean(u[k] for u in ups) for k in KEYS if all(k in u for u in ups)})


def build(rows, lang, minutes, steps):
    zh = lang == "zh"
    T = (lambda z, e: z if zh else e)
    L = ["# v6.0.3.1 " + T("PPO 每次更新的幅度诊断", "PPO per-update step-size diagnostic"), ""]
    L.append(T(f"- 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/diag_ppo_update_v6.py`；耗时 {minutes:.1f} 分钟。",
               f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; script `src/scripts/diag_ppo_update_v6.py`; took {minutes:.1f} min."))
    L.append(T(f"- 对 v6.0.0 Mask PPO（三场景 × 种子 1–3）的 5M / 10M / 15M checkpoint 与 20M 最终模型，用与训练完全相同的设置"
               f"（训练池、奖励、超参、优化器状态）各续训 {steps:,} 步（{steps // 20480} 次 PPO 更新），只记录 SB3 每次更新的统计量；"
               "模型不保存、不改动原 run。",
               f"- The 5M / 10M / 15M checkpoints and the 20M final model of v6.0.0 Mask PPO (3 scenarios × seeds 1–3) each trained "
               f"on for {steps:,} steps ({steps // 20480} PPO updates) with exactly the training settings (pool, reward, hyper-parameters, "
               "optimiser state), only logging SB3's per-update statistics; nothing is saved, the original runs are untouched."))
    L.append(T(f"- clip_range = {rows[0]['clip_range']}，每次更新 {rows[0]['n_epochs']} 个 epoch。approx_kl = 新旧策略的近似 KL；"
               "clip_fraction = 被截断的样本比例；explained_variance = Critic 解释的回报方差（1 为完美，0 等于常数预测）。",
               f"- clip_range = {rows[0]['clip_range']}, {rows[0]['n_epochs']} epochs per update. approx_kl = approximate KL between old "
               "and new policy; clip_fraction = share of samples clipped; explained_variance = return variance explained by the critic "
               "(1 perfect, 0 = constant prediction)."))
    L.append(T("- 表中为三个种子的均值（每个种子先对其各次更新取平均）。", "- Means over the three seeds (each seed first averaged over its updates)."))
    L.append("")
    head = T("| 场景 | 训练进度 | approx_kl | clip_fraction | 熵（负的 entropy_loss） | value_loss | explained_variance |",
             "| Scen | Training point | approx_kl | clip_fraction | entropy (−entropy_loss) | value_loss | explained_variance |")
    L.append(head)
    L.append("|---" * 7 + "|")
    for s in SCENS:
        for p in POINTS:
            g = [r["mean"] for r in rows if r["scen"] == s and r["point"] == p]
            f = lambda k, fmt: format(st.mean(x[k] for x in g), fmt)
            L.append(f"| {s.upper()} | {p // 1_000_000}M | {f('approx_kl', '.5f')} | {float(f('clip_fraction', '.6f')):.3%} | "
                     f"{-float(f('entropy_loss', '.6f')):.4f} | {f('value_loss', '.6f')} | {f('explained_variance', '.4f')} |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12 * 20480)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-mail", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rs = runs()
    tasks = [(r, p, a.steps, f"cuda:{i % 3}") for i, (r, p) in enumerate((r, p) for r in rs for p in POINTS)]
    print(f"{len(tasks)} probes of {a.steps:,} steps", flush=True)
    rows = []
    with get_context("spawn").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(probe, tasks), 1):
            rows.append(r)
            print(f"[probe] {i}/{len(tasks)} {r['scen']} seed {r['seed']} {r['point'] // 1_000_000}M "
                  f"kl={r['mean']['approx_kl']:.5f} clip={r['mean']['clip_fraction']:.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
    minutes = (time.time() - t0) / 60
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ppo_update.json").write_text(json.dumps(rows, indent=1))
    (OUT / "ppo_update.md").write_text(build(rows, "zh", minutes, a.steps))
    (OUT / "ppo_update.en.md").write_text(build(rows, "en", minutes, a.steps))
    print(OUT / "ppo_update.md", f"({minutes:.1f} min)", flush=True)
    if not a.no_mail:
        from shared.report_mail import send_report
        send_report("v6.0.3.1", OUT / "ppo_update.md")


if __name__ == "__main__":
    from shared.batch_manifest import nontrain_batch      # docs/data_spec.md §6: listed by the v2 panel
    with nontrain_batch("v6.0.3.1", "diag", __file__,
                        {"zh": "v6.0.3.1：PPO 每次更新的幅度诊断（v6.0.0 Mask PPO 各 checkpoint 续训 3 次更新，记录 approx_kl / clip_fraction / explained_variance，不保存模型）",
                         "en": "v6.0.3.1: PPO update-size diagnosis (3 more updates from each v6.0.0 Mask PPO checkpoint; approx_kl / clip_fraction / explained_variance; no model saved)"},
                        depends_on=['v6.0.0'], report_dir="paper_contents/v6.0.3.1"):
        main()
