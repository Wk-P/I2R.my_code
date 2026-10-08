"""Train and evaluate one model (v4.3.1.3).

    python -m paper_rl.train --scen lt --algo mask_ppo --reward ar --steps 1000000 --seed 1

--gamma (v4.3.5) overrides the discount factor of every learner (ar_raw runs use 1).

--reward-norm m (v4.3.1.8) divides every reward the learner sees by M, so the
legacy terminal reward lies in [-1, 1] (succ_first: [-1, 3]); Monitor / training_curve.csv keep the
raw reward. Default none = v4.3.1.6 behaviour.

algo = <mechanism>_<learner> with mechanism in {mask, lagrange, repair} or the
bare learner for no constraint handling: ppo, mask_ppo, lagrange_ppo,
repair_ppo, dqn, mask_dqn, ..., repair_ddqn.

Output: results/<RESULTS_SPACE>/<scen>/<algo>/<exp_id>/
  model_<exp_id>_v<version>.zip, results.json, summary.csv,
  training_curve.csv, training_curve.png
Evaluation: one deterministic episode per test instance (20% held out,
split seeded by --seed).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import functools
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from paper_rl import config as C
from paper_rl.data import load
from paper_rl.env import REWARD_MODES, PlacementEnv, mask_slice

LEARNERS = ("ppo", "dqn", "ddqn")
MECH_PREFIX = {"mask": "mask", "lagrange": "lagrange", "repair": "repair"}
ALGOS = [f"{m}_{l}" if m else l for l in LEARNERS for m in ("", "mask", "lagrange", "repair")]
LABEL = {"": "", "mask": "Mask-", "lagrange": "Lagrange-", "repair": "Repair-"}


def split_algo(algo: str) -> tuple[str, str]:
    """'mask_ddqn' -> ('mask', 'ddqn'); 'ppo' -> ('none', 'ppo')"""
    if algo in LEARNERS:
        return "none", algo
    mech, _, learner = algo.partition("_")
    assert mech in MECH_PREFIX and learner in LEARNERS, algo
    return mech, learner


def algo_label(algo: str) -> str:
    mech, learner = split_algo(algo)
    return LABEL["" if mech == "none" else mech] + learner.upper()


# ── data ────────────────────────────────────────────────────────────────────
def split_instances(scen: str, seed: int):
    data = load(scen)
    inst = data["instances"]
    idx = list(range(len(inst)))
    random.Random(seed).shuffle(idx)
    n_train = int(C.TRAIN_FRACTION * len(inst))
    return data, [inst[i] for i in idx[:n_train]], [inst[i] for i in idx[n_train:]]


# ── learners ────────────────────────────────────────────────────────────────
def build_model(learner: str, mech: str, env, seed: int, n: int, gamma: float | None = None,
                exit_action: bool = False, m: int = 0, action_mode: str = "ecu", net: str = "mlp", device: str = "cpu",
                gae_lambda: float | None = None, lr: float | None = None, ent_coef: float | None = None,
                glob_std: bool = False):
    if learner == "ppo":
        kw = dict(policy="MlpPolicy", env=env, learning_rate=C.PPO_LR if lr is None else lr, n_steps=C.PPO_N_STEPS,
                  batch_size=C.PPO_BATCH_SIZE, n_epochs=C.PPO_N_EPOCHS, gamma=C.PPO_GAMMA if gamma is None else gamma,
                  gae_lambda=C.PPO_GAE_LAMBDA if gae_lambda is None else gae_lambda, clip_range=C.PPO_CLIP_RANGE, ent_coef=C.PPO_ENT_COEF if ent_coef is None else ent_coef,
                  policy_kwargs=dict(net_arch=C.PPO_NET_ARCH), device=device, verbose=0, seed=seed)
        if net == "graph":                              # v4.4.3: structure-aware policy, PPO settings unchanged
            assert mech == "mask" and action_mode == "ecu", "--net graph: Mask PPO, ECU action only"
            from paper_rl.graph_policy import GraphMaskablePolicy
            kw.update(policy=GraphMaskablePolicy, policy_kwargs=dict(n_ecu=n, n_svc=m, glob_std=glob_std))
        if mech == "mask":
            from sb3_contrib import MaskablePPO
            return MaskablePPO(**kw)
        from stable_baselines3 import PPO
        return PPO(**kw)
    from stable_baselines3 import DQN
    from shared.dqn_variants import DoubleDQN, MaskableDQN, MaskableDDQN
    kw = dict(policy="MlpPolicy", env=env, learning_rate=C.DQN_LR, buffer_size=C.DQN_BUFFER_SIZE,
              learning_starts=C.DQN_LEARNING_STARTS, batch_size=C.DQN_BATCH_SIZE, tau=C.DQN_TAU,
              gamma=C.DQN_GAMMA if gamma is None else gamma, train_freq=C.DQN_TRAIN_FREQ, gradient_steps=C.DQN_GRADIENT_STEPS,
              target_update_interval=C.DQN_TARGET_UPDATE, exploration_fraction=C.DQN_EXPLORATION_FRACTION,
              exploration_final_eps=C.DQN_EXPLORATION_FINAL_EPS,
              policy_kwargs=dict(net_arch=C.DQN_NET_ARCH), device="cpu", verbose=0, seed=seed)
    if mech == "mask":
        cls = MaskableDDQN if learner == "ddqn" else MaskableDQN
        return cls(mask_start=mask_slice(n, m, action_mode).start, exit_action=exit_action, **kw)
    return (DoubleDQN if learner == "ddqn" else DQN)(**kw)


def model_class(learner: str, mech: str):
    if learner == "ppo":
        if mech == "mask":
            from sb3_contrib import MaskablePPO
            return MaskablePPO
        from stable_baselines3 import PPO
        return PPO
    from stable_baselines3 import DQN
    from shared.dqn_variants import DoubleDQN, MaskableDQN, MaskableDDQN
    if mech == "mask":
        return MaskableDDQN if learner == "ddqn" else MaskableDQN
    return DoubleDQN if learner == "ddqn" else DQN


def scale_reward(env, scale: float):
    """Multiply the learner's reward by `scale` (outside Monitor, which logs the raw reward)."""
    import gymnasium as gym
    return gym.wrappers.TransformReward(env, lambda r: r * scale)


# ── training callback (logging + Lagrangian dual ascent) ────────────────────
def make_callback(mech: str, total_steps: int, outdir: Path, exp_id: str):
    from stable_baselines3.common.callbacks import BaseCallback
    from shared.paths import write_progress

    class Cb(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = []          # (timestep, reward, ar, success, valid, cap>0, conf>0, lambda)
            self.lam = C.LAMBDA_INIT
            self._viol = deque(maxlen=C.LAMBDA_UPDATE_WINDOW)
            self._n_ep = 0
            self._since = 0
            self._next = C.PROGRESS_LOG_EVERY_STEPS
            self._t0 = time.time()

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "episode" not in info:
                    continue
                success = info["valid_placed"] == self.training_env.get_attr("M", 0)[0]
                self.rows.append((self.num_timesteps, float(info["episode"]["r"]), info["ar"], int(success),
                                  info["valid_placed"], int(info["capacity_violations"] > 0),
                                  int(info["conflict_violations"] > 0), self.lam))
                if mech == "lagrange":
                    self._dual(info["viol_rate_ep"])
            if self.num_timesteps >= self._next:
                el = max(time.time() - self._t0, 1e-6)
                pct = min(100.0, 100.0 * self.num_timesteps / total_steps)
                last = self.rows[-500:]
                sr = np.mean([r[3] for r in last]) if last else 0.0
                lam = f" | lambda={self.lam:.3f}" if mech == "lagrange" else ""
                print(f"  [train] step={self.num_timesteps:,}/{total_steps:,} ({pct:5.1f}%) | "
                      f"eps={len(self.rows)} | success(last500)={sr:.3f} | steps/s={self.num_timesteps / el:,.0f}{lam}",
                      flush=True)
                write_progress(outdir, step=self.num_timesteps, total_steps=total_steps, pct=round(pct, 1),
                               episodes=len(self.rows), exp_id=exp_id)
                self._next += C.PROGRESS_LOG_EVERY_STEPS
            return True

        def _dual(self, viol_rate: float) -> None:
            self._n_ep += 1
            self._viol.append(viol_rate)
            if self._n_ep < C.LAMBDA_WARMUP_EPISODES:
                return
            self._since += 1
            if len(self._viol) == C.LAMBDA_UPDATE_WINDOW and self._since >= C.LAMBDA_UPDATE_WINDOW:
                self.lam = float(np.clip(self.lam + C.LAMBDA_LR * (np.mean(self._viol) - C.LAMBDA_TARGET),
                                         0.0, C.LAMBDA_MAX))
                self.training_env.env_method("set_lambda", self.lam)
                self._since = 0

    return Cb()


# ── evaluation ──────────────────────────────────────────────────────────────
def evaluate(model, test, mech: str, reward: str, lam: float, learner: str, obs_mode: str = "base",
             full_episode: bool = False, exit_action: bool = False, action_mode: str = "ecu",
             order: str = "desc", order_seed: int = 0) -> list[dict]:
    env = PlacementEnv(test, mech, reward, lam=lam, obs_mode=obs_mode, full_episode=full_episode, order=order, order_seed=order_seed,
                       exit_action=exit_action, action_mode=action_mode)
    out = []
    for k in range(len(test)):
        env.use_instance(k)
        obs, _ = env.reset()
        done = False
        while not done:
            if learner == "ppo" and mech == "mask":
                a, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            else:
                a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(int(a))
        out.append({"ar": info["ar"], "ar_star": info["ar_star"], "success": info["valid_placed"] == env.M,
                    "valid_placed": info["valid_placed"], "cap_v": info["capacity_violations"],
                    "conf_v": info["conflict_violations"], "dead_end": info["dead_end"], "exited": info["exited"],
                    "ecus_used": info["ecus_used"]})
    return out


# ── outputs ─────────────────────────────────────────────────────────────────
def plot_curve(rows, ar_star, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not rows:
        return
    a = np.array(rows, dtype=float)
    w = max(1, min(C.SMOOTH_W, len(a) // 10))
    sm = lambda y: np.convolve(y, np.ones(w) / w, mode="valid")
    ts = a[w - 1:, 0]
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    ax[0].plot(ts, sm(a[:, 1]), color="steelblue"); ax[0].set_ylabel("Episode reward")
    ax[1].plot(ts, sm(a[:, 2]), color="seagreen", label="AR")
    ax[1].axhline(ar_star, color="red", ls="--", label=f"ILP AR* = {ar_star:.4f}"); ax[1].set_ylabel("AR"); ax[1].legend()
    ax[2].plot(ts, sm(a[:, 3]), color="mediumseagreen"); ax[2].set_ylabel("success_rate"); ax[2].set_ylim(-0.02, 1.02)
    ax[3].plot(ts, sm(a[:, 5]), color="tomato", label="capacity violation")
    ax[3].plot(ts, sm(a[:, 6]), color="darkorange", label="privacy violation")
    ax[3].set_ylabel("violation rate"); ax[3].set_ylim(-0.02, 1.02); ax[3].legend(); ax[3].set_xlabel("training steps")
    for x in ax:
        x.grid(alpha=0.3)
    ax[0].set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", required=True, choices=["lt", "eq", "gt"])
    ap.add_argument("--algo", required=True, choices=ALGOS)
    ap.add_argument("--reward", default=os.environ.get("REWARD_MODE", "succ_first"),   # v4.3.1.9 default
                    choices=list(REWARD_MODES))
    ap.add_argument("--reward-norm", default=os.environ.get("REWARD_NORM", "none"), choices=["none", "m"])
    ap.add_argument("--obs", default=os.environ.get("OBS_MODE", "base"), choices=["base", "conflict", "feas", "raw"])   # v4.3.4 / v4.3.9
    ap.add_argument("--gamma", type=float, default=None,     # v4.3.5; default: PPO_GAMMA / DQN_GAMMA in config
                    help="discount factor for every learner (ar_raw uses 1, so the return is AR * 1{feasible})")
    ap.add_argument("--full-episode", action="store_true",      # v4.3.7: never stop early, always M steps
                    default=os.environ.get("FULL_EPISODE", "0") == "1")
    ap.add_argument("--action", default=os.environ.get("ACTION_MODE", "ecu"), choices=["ecu", "joint"],
                    help="v4.4.0: 'joint' = the agent picks (service, ECU); Mask + --full-episode only")
    ap.add_argument("--exit-action", action="store_true",       # Mask only: EXIT action, valid iff no ECU is feasible
                    default=os.environ.get("EXIT_ACTION", "0") == "1")
    ap.add_argument("--net", default=os.environ.get("POLICY_NET", "mlp"), choices=["mlp", "graph"],
                    help="v4.4.3: 'graph' = structure-aware policy (src/paper_rl/graph_policy.py); forces --obs raw")
    ap.add_argument("--device", default=os.environ.get("POLICY_DEVICE", "cpu"),
                    help="v4.4.3: torch device for PPO (cuda needs .venv-gpu); default cpu as before")
    ap.add_argument("--gae-lambda", type=float, default=None,   # v4.4.4; default: PPO_GAE_LAMBDA in config
                    help="GAE lambda for PPO (1 = Monte-Carlo return minus V(s))")
    ap.add_argument("--lr", type=float, default=None, help="v4.4.5: PPO learning rate (default PPO_LR in config)")
    ap.add_argument("--ent-coef", type=float, default=None, help="v4.4.5: PPO entropy coefficient (default PPO_ENT_COEF)")
    ap.add_argument("--order", default="desc", choices=["desc", "asc", "random"],
                    help="v4.4.7: service placement order (random = one fixed permutation per instance)")
    ap.add_argument("--order-seed", type=int, default=0, help="v4.4.7: seed of the random order")
    ap.add_argument("--glob-std", action="store_true",
                    help="v4.4.6: --net graph only; add the utilisation std of the active ECUs to the global token")
    ap.add_argument("--bc", action="store_true",                # v4.4.2: ILP-demonstration warm start (Mask PPO only)
                    default=os.environ.get("BC_WARMSTART", "0") == "1")
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=int(os.environ.get("TRAIN_SEED", "1")))
    a = ap.parse_args()

    import torch
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from shared.paths import VERSION, resolve_exp_id, results_dir

    if a.net == "graph":
        a.obs = "raw"
    mech, learner = split_algo(a.algo)
    torch.set_num_threads(C.PPO_TORCH_THREADS if learner == "ppo" else C.DQN_TORCH_THREADS)
    data, train, test = split_instances(a.scen, a.seed)
    n, m = data["N"], data["M"]
    outdir = results_dir(a.scen, a.algo)
    exp_id = resolve_exp_id(outdir)
    run_dir = outdir / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {a.scen.upper()} N={n} M={m} | {algo_label(a.algo)} | reward={a.reward} | "
          f"reward_norm={a.reward_norm} | obs={a.obs} | gamma={a.gamma} | gae_lambda={a.gae_lambda} | lr={a.lr} | ent_coef={a.ent_coef} | glob_std={a.glob_std} | order={a.order} | full_episode={a.full_episode} | steps={a.steps:,} | seed={a.seed} | exp_id={exp_id} ===", flush=True)

    n_envs = C.PPO_N_ENVS if learner == "ppo" else C.DQN_N_ENVS
    scale = 1.0 / m if a.reward_norm == "m" else 1.0
    venv = DummyVecEnv([functools.partial(
        lambda s: scale_reward(Monitor(PlacementEnv(train, mech, a.reward, lam=C.LAMBDA_INIT, rng_seed=s, obs_mode=a.obs,
                                             full_episode=a.full_episode, exit_action=a.exit_action,
                                             action_mode=a.action, order=a.order, order_seed=a.order_seed)), scale),
        a.seed * 1000 + k) for k in range(n_envs)])
    gamma = a.gamma if a.gamma is not None else (C.PPO_GAMMA if learner == "ppo" else C.DQN_GAMMA)
    model = build_model(learner, mech, venv, a.seed, n, gamma, a.exit_action, m, a.action, a.net, a.device, a.gae_lambda, a.lr, a.ent_coef, a.glob_std)
    bc_info = None
    if a.bc:                                            # v4.4.2: behaviour cloning on ILP optima, then PPO
        assert learner == "ppo" and mech == "mask" and a.action == "ecu", "--bc: Mask PPO, ECU action only"
        assert a.order == "desc", "--bc: demonstrations are built in descending order (paper_rl.bc)"
        from paper_rl.bc import alloc_path, expert_dataset, pretrain
        allocs_all = json.loads(alloc_path(a.scen).read_text())
        pos = {id(x): k for k, x in enumerate(data["instances"])}
        t_bc = time.time()
        demo = expert_dataset(dict(mechanism=mech, reward_mode=a.reward, obs_mode=a.obs, full_episode=a.full_episode,
                                   exit_action=a.exit_action), train, [allocs_all[pos[id(x)]] for x in train])
        bc_info = pretrain(model, demo, seed=a.seed)
        model.save(str(run_dir / f"bc_only_{exp_id}"))
        ev_bc = evaluate(model, test, mech, a.reward, 0.0, learner, a.obs, a.full_episode, a.exit_action, a.action)
        ok_bc = [e for e in ev_bc if not e["exited"] and e["cap_v"] == 0 and e["conf_v"] == 0]
        bc_info.update({"bc_seconds": round(time.time() - t_bc, 1),
                        "bc_only_exit_rate": round(float(np.mean([e["exited"] for e in ev_bc])), 6),
                        "bc_only_ar": round(float(np.mean([e["ar"] for e in ok_bc])), 6) if ok_bc else None,
                        "bc_only_ilp_ar": round(float(np.mean([e["ar_star"] for e in ok_bc])), 6) if ok_bc else None})
        print(f"  [bc] {bc_info}", flush=True)
    cb = make_callback(mech, a.steps, outdir, exp_id)
    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=cb)
    train_s = time.time() - t0
    lam = cb.lam if mech == "lagrange" else 0.0
    model.save(str(run_dir / f"model_{exp_id}_v{VERSION}-{a.reward}"))

    ev = evaluate(model, test, mech, a.reward, lam, learner, a.obs, a.full_episode, a.exit_action, a.action, a.order, a.order_seed)
    sr = float(np.mean([e["success"] for e in ev]))
    ars = np.array([e["ar"] for e in ev])
    succ_ratio = [e["ar"] / e["ar_star"] for e in ev if e["success"] and e["ar_star"] > 0]
    cap_rate = float(np.mean([e["cap_v"] > 0 for e in ev]))
    conf_rate = float(np.mean([e["conf_v"] > 0 for e in ev]))
    ilp_ar = float(np.mean([e["ar_star"] for e in ev]))
    ok = [e for e in ev if e["success"]]
    ar_ok = float(np.mean([e["ar"] for e in ok])) if ok else float("nan")
    ilp_ok = float(np.mean([e["ar_star"] for e in ok])) if ok else float("nan")
    print(f"  eval: violation-free={sr:.4f} | violation-free instances: AR={ar_ok:.4f} ILP AR={ilp_ok:.4f} "
          f"AR gap={ilp_ok - ar_ok:.4f} | cap viol={cap_rate:.4f} | privacy viol={conf_rate:.4f} | "
          f"EXIT={np.mean([e['exited'] for e in ev]):.4f} | "
          f"train {train_s / 60:.1f} min", flush=True)

    res = {
        "created_at": datetime.datetime.now().isoformat(), "exp_id": exp_id, "version": VERSION,
        "commit": git_commit(), "scenario": a.scen, "N": n, "M": m, "algo": a.algo,
        "mechanism": mech, "learner": learner, "reward_mode": a.reward, "reward_norm": a.reward_norm, "obs": a.obs, "gamma": gamma, "full_episode": a.full_episode, "exit_action": a.exit_action, "action_mode": a.action, "seed": a.seed,
        "net": a.net, "device": a.device, "bc": bc_info, "glob_std": a.glob_std, "order": a.order, "order_seed": a.order_seed,
        **({"gae_lambda": C.PPO_GAE_LAMBDA if a.gae_lambda is None else a.gae_lambda,
            "lr": C.PPO_LR if a.lr is None else a.lr,
            "ent_coef": C.PPO_ENT_COEF if a.ent_coef is None else a.ent_coef} if learner == "ppo" else {}),
        "train_count": len(train), "test_count": len(test),
        "data": {"conflict_pair_prob": data["conflict_pair_prob"], "k_sets": data["k_sets"]},
        "ilp": {"ar": round(ilp_ar, 6)},
        a.algo: {
            "success_rate": round(sr, 6),
            "ar_mean": round(float(ars.mean()), 6), "ar_std": round(float(ars.std()), 6),
            "ar_ratio_successful": round(float(np.mean(succ_ratio)), 6) if succ_ratio else None,
            "cap_viol_rate": round(cap_rate, 6), "conflict_viol_rate": round(conf_rate, 6),
            "cap_viol_total": int(sum(e["cap_v"] for e in ev)),
            "conflict_viol_total": int(sum(e["conf_v"] for e in ev)),
            "dead_end_rate": round(float(np.mean([e["dead_end"] for e in ev])), 6),
            "exit_rate": round(float(np.mean([e["exited"] for e in ev])), 6),
        },
        "training": {"total_steps": a.steps, "n_episodes": len(cb.rows), "train_seconds": round(train_s, 1),
                     "ar_last50": round(float(np.mean([r[2] for r in cb.rows[-50:]])), 6) if cb.rows else None,
                     **({"final_lambda": round(lam, 6)} if mech == "lagrange" else {})},
    }
    (run_dir / "results.json").write_text(json.dumps(res, indent=2))
    with open(run_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "ar_mean", "ar_std", "success_rate", "cap_viol_rate", "conflict_viol_rate",
                    "ar_ratio_successful", "dead_end_rate"])
        w.writerow(["ILP (Optimal)", round(ilp_ar, 6), 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        r = res[a.algo]
        w.writerow([algo_label(a.algo), r["ar_mean"], r["ar_std"], r["success_rate"], r["cap_viol_rate"],
                    r["conflict_viol_rate"], r["ar_ratio_successful"], r["dead_end_rate"]])
    with open(run_dir / "training_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestep", "episode_reward", "episode_ar", "episode_success", "episode_valid_placed",
                    "episode_cap_violated", "episode_conflict_violated", "lambda"])
        w.writerows(cb.rows)
    plot_curve(cb.rows, ilp_ar, run_dir / "training_curve.png",
               f"{algo_label(a.algo)} — {a.scen.upper()} — reward={a.reward} — {a.steps:,} steps")
    print(f"  saved -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
