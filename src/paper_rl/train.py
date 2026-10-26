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
    return data, [inst[m] for m in idx[:n_train]], [inst[m] for m in idx[n_train:]]


# ── learners ────────────────────────────────────────────────────────────────
def build_model(learner: str, mech: str, env, seed: int, n: int, gamma: float | None = None,
                exit_action: bool = False, m: int = 0, action_mode: str = "ecu", net: str = "mlp", device: str = "cpu",
                gae_lambda: float | None = None, lr: float | None = None, ent_coef: float | None = None,
                glob_std: bool = False, separate: bool = False, pair_head: bool = False, group_k: int = 0,
                plan: bool = False, plan_decay: bool = False, glob3: bool = False, eps_frac: float | None = None,
                n_epochs: int | None = None, target_kl: float | None = None, hp: dict | None = None):
    if learner == "ppo":
        kw = dict(policy="MlpPolicy", env=env, learning_rate=C.PPO_LR if lr is None else lr, n_steps=C.PPO_N_STEPS,
                  batch_size=C.PPO_BATCH_SIZE, n_epochs=C.PPO_N_EPOCHS if n_epochs is None else n_epochs, target_kl=target_kl, gamma=C.PPO_GAMMA if gamma is None else gamma,
                  gae_lambda=C.PPO_GAE_LAMBDA if gae_lambda is None else gae_lambda, clip_range=C.PPO_CLIP_RANGE, ent_coef=C.PPO_ENT_COEF if ent_coef is None else ent_coef,
                  policy_kwargs=dict(net_arch=C.PPO_NET_ARCH), device=device, verbose=0, seed=seed)
        kw.update(hp or {})                             # v6.1.0: --hp overrides any PPO constructor argument
        if net == "graph":                              # v4.4.3: structure-aware policy, PPO settings unchanged
            # v4.5.4: every mechanism; mask keeps the masked logits, the others use the unmasked scores
            assert action_mode == "ecu", "--net graph: ECU action only"
            from paper_rl.graph_policy import GraphMaskablePolicy
            kw.update(policy=GraphMaskablePolicy, policy_kwargs=dict(n_ecu=n, n_svc=m, glob_std=glob_std, separate=separate, pair_head=pair_head, plan=plan, plan_decay=plan_decay, glob3=glob3,
                                                                  lam_in=mech == "lagrange", mask_logits=mech == "mask",
                                                                  feas_in=mech != "none"))
        if mech == "mask" and group_k:                 # v4.4.11: same-instance groups, instance-wise baseline
            from paper_rl.group_ppo import GroupMaskablePPO
            return GroupMaskablePPO(group_k=group_k, group_seed=seed, **kw)
        if mech == "mask":
            from sb3_contrib import MaskablePPO
            return MaskablePPO(**kw)
        assert not group_k, "--group-k: Mask PPO only"
        from stable_baselines3 import PPO
        return PPO(**kw)
    from stable_baselines3 import DQN
    from shared.dqn_variants import DoubleDQN, MaskableDQN, MaskableDDQN
    kw = dict(policy="MlpPolicy", env=env, learning_rate=C.DQN_LR, buffer_size=C.DQN_BUFFER_SIZE,
              learning_starts=C.DQN_LEARNING_STARTS, batch_size=C.DQN_BATCH_SIZE, tau=C.DQN_TAU,
              gamma=C.DQN_GAMMA if gamma is None else gamma, train_freq=C.DQN_TRAIN_FREQ, gradient_steps=C.DQN_GRADIENT_STEPS,
              target_update_interval=C.DQN_TARGET_UPDATE,
              exploration_fraction=C.DQN_EXPLORATION_FRACTION if eps_frac is None else eps_frac,   # v5.0.3.3: --eps-steps
              exploration_final_eps=C.DQN_EXPLORATION_FINAL_EPS,
              policy_kwargs=dict(net_arch=C.DQN_NET_ARCH), device=device, verbose=0, seed=seed)
    kw.update(hp or {})                                 # v6.1.0: --hp overrides any DQN / DDQN constructor argument
    if net == "graph":                                  # v4.5.2: structure-aware Q-network, same DQN settings
        assert action_mode == "ecu", "--net graph with DQN / DDQN: ECU action only"
        from paper_rl.graph_dqn import GraphDQNPolicy
        kw.update(policy=GraphDQNPolicy, policy_kwargs=dict(n_ecu=n, n_svc=m, lam_in=mech == "lagrange", feas_in=mech != "none"))
        if mech == "mask":
            cls = MaskableDDQN if learner == "ddqn" else MaskableDQN
            return cls(mask_start=None, raw_dims=(n, m), exit_action=exit_action, **kw)
        return (DoubleDQN if learner == "ddqn" else DQN)(**kw)     # v4.5.4: unconstrained / Lagrange / Repair
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
def make_callback(mech: str, total_steps: int, outdir: Path, exp_id: str, start: int = 0, lam_max: float = C.LAMBDA_MAX,
                  lam_update: str = "linear", lam_lr: float = C.LAMBDA_LR, lam_cost: str = "behavior",
                  greedy_env=None, n_greedy: int = 100, lam_init: float = C.LAMBDA_INIT, lam_target: float = C.LAMBDA_TARGET,
                  lam_window: str = "episodes"):
    from stable_baselines3.common.callbacks import BaseCallback
    from shared.paths import write_progress

    class Cb(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = []          # (timestep, reward, ar, success, valid, cap>0, conf>0, lambda, exited (v5.0.0))
            self.lam = lam_init
            self._adam = None
            if lam_update == "adam":    # v5.0.3.7: OmniSafe PPOLag style, lambda a parameter, loss -lambda (J_C - d), Adam, lambda >= 0
                import torch
                self._lam_t = torch.nn.Parameter(torch.tensor(float(lam_init)))
                self._adam = torch.optim.Adam([self._lam_t], lr=lam_lr)
            self._roll = []         # v5.0.3.7: episode costs since the last update (lam_window "rollout")
            self._viol = deque(maxlen=C.LAMBDA_UPDATE_WINDOW)
            self._n_ep = 0
            self._since = 0
            self._next = start + C.PROGRESS_LOG_EVERY_STEPS   # v4.4.16: a resumed run starts at `start`
            self._t0 = time.time()
            self.dual_rows = []     # one per lambda update: (episode, timestep, J_C behaviour, J_C used, lambda after) (v5.0.3.3)

        def _on_training_start(self) -> None:
            if mech == "lagrange":
                self.training_env.env_method("set_lambda", self.lam)

        def _on_rollout_end(self) -> None:
            """v5.0.3.7 (lam_window "rollout"): one update per PPO rollout from its episodes' mean cost."""
            if mech == "lagrange" and lam_window == "rollout" and self._roll:
                n = float(np.mean(self._roll))
                self._step_lambda(n, n)
                self._roll = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "episode" not in info:
                    continue
                success = info["valid_placed"] == self.training_env.get_attr("M", 0)[0]
                self.rows.append((self.num_timesteps, float(info["episode"]["r"]), info["ar"], int(success),
                                  info["valid_placed"], int(info["capacity_violations"] > 0),
                                  int(info["conflict_violations"] > 0), self.lam, int(info["exited"])))
                if mech == "lagrange":
                    self._dual(info["cost_ep"])
            if self.num_timesteps >= self._next:
                el = max(time.time() - self._t0, 1e-6)
                pct = min(100.0, 100.0 * self.num_timesteps / total_steps)
                last = self.rows[-500:]
                sr = np.mean([r[3] for r in last]) if last else 0.0
                lam = f" | lambda={self.lam:.3f}" if mech == "lagrange" else ""
                print(f"  [train] step={self.num_timesteps:,}/{total_steps:,} ({pct:5.1f}%) | "
                      f"eps={len(self.rows)} | success(last500)={sr:.3f} | steps/s={(self.num_timesteps - start) / el:,.0f}{lam}",
                      flush=True)
                write_progress(outdir, step=self.num_timesteps, total_steps=total_steps, pct=round(pct, 1),
                               episodes=len(self.rows), exp_id=exp_id)
                self._next += C.PROGRESS_LOG_EVERY_STEPS
            return True

        def _dual(self, cost: float) -> None:
            """J_C = mean violations per episode (v5.0.3; was violations / M); lambda is updated every
            LAMBDA_UPDATE_WINDOW episodes from the first episode on (no warm-up since 2026-10-13)."""
            self._n_ep += 1
            if lam_window == "rollout":
                self._roll.append(cost)
                return
            self._viol.append(cost)
            self._since += 1
            if len(self._viol) == C.LAMBDA_UPDATE_WINDOW and self._since >= C.LAMBDA_UPDATE_WINDOW:
                j_beh = float(np.mean(self._viol))
                n = self._greedy_jc() if lam_cost == "greedy" else j_beh
                self._step_lambda(j_beh, n)
                self._since = 0

        def _step_lambda(self, j_beh: float, n: float) -> None:
            d = lam_target
            if self._adam is not None:
                self._adam.zero_grad()
                (-self._lam_t * (n - d)).backward()
                self._adam.step()
                with __import__("torch").no_grad():
                    self._lam_t.clamp_(0.0, lam_max)
                self.lam = float(self._lam_t.item())
            else:
                # linear: e = J_C - d (v5.0.3); normalized (v5.0.3.1): e = (J_C - d) / max(J_C, d) in (-1, 1), so a step
                # up and a step down have the same size lam_lr (with d ~ 0 the linear error can rise ~1000x faster than it falls)
                e = n - d if lam_update == "linear" else (n - d) / max(n, d, 1e-12)
                self.lam = float(np.clip(self.lam + lam_lr * e, 0.0, lam_max))
            self.training_env.env_method("set_lambda", self.lam)
            self.dual_rows.append((self._n_ep, self.num_timesteps, j_beh, n, self.lam))

        def _greedy_jc(self) -> float:
            """v5.0.3.3: J_C of the greedy (epsilon = 0) policy that is deployed and tested, not of the epsilon-greedy
            behaviour policy: n_greedy extra episodes on training instances at the current lambda; they do not enter the
            replay buffer and are not counted as training steps."""
            greedy_env.set_lambda(self.lam)
            cost = []
            for _ in range(n_greedy):
                obs, _ = greedy_env.reset()
                done = False
                while not done:
                    a, _ = self.model.predict(obs, deterministic=True)
                    obs, _, done, _, info = greedy_env.step(int(a))
                cost.append(info["cost_ep"])
            return float(np.mean(cost))

    return Cb()


def lam_max_arg(x: str) -> float:
    """--lambda-max: a number, or 'none' for no cap (v5.0.3.2; kept as inf in memory)."""
    return float("inf") if x.strip().lower() in ("none", "inf") else float(x)


def lam_max_json(v: float):
    """lambda_max as written to run.json / results.json / ckpt_lambda.json: "none" when there is no cap (JSON has no inf)."""
    return "none" if v == float("inf") else v


def checkpoint_callback(steps: list[int], run_dir: Path, exp_id: str, lam_fn=None, lam_max: float = C.LAMBDA_MAX):
    """v4.4.15: save the model the first time num_timesteps reaches each of `steps`.
    v4.5.4: with lam_fn (Lagrange), also record the dual variable at each checkpoint in ckpt_lambda.json,
    since the policy reads lambda from its observation and must be evaluated with the value it was trained at."""
    from stable_baselines3.common.callbacks import BaseCallback

    class Ckpt(BaseCallback):
        def __init__(self):
            super().__init__()
            self.todo = list(steps)

        def _on_step(self) -> bool:
            while self.todo and self.num_timesteps >= self.todo[0]:
                step = self.todo.pop(0)
                self.model.save(str(run_dir / f"ckpt_{step}_{exp_id}"))
                if lam_fn is not None:
                    path = run_dir / "ckpt_lambda.json"
                    lams = json.loads(path.read_text()) if path.exists() else {}
                    lams[str(step)] = round(float(lam_fn()), 6)
                    lams["lambda_max"] = lam_max_json(lam_max)   # v5.0.3: the observation's lambda scale ("none": no cap)
                    path.write_text(json.dumps(lams, indent=2))
            return True

    return Ckpt()


def val_callback(every: int, path: Path, eval_fn):
    """v6.1.0: every `every` steps evaluate on the test set (the tuner passes its validation set) and append
    {step, rel_gap, viol, exit} to `path` (JSON lines), which the Optuna driver reads for pruning."""
    from stable_baselines3.common.callbacks import BaseCallback

    class Val(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next = every

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next:
                self.next += every
                with open(path, "a") as f:
                    f.write(json.dumps({"step": int(self.num_timesteps), **val_summary(eval_fn(self.model))}) + "\n")
            return True

    return Val()


def val_summary(ev: list[dict]) -> dict:
    """v6.1.0: relative AR gap over the violation-free instances, violation rate (capacity or privacy), EXIT rate,
    fail = 1 - violation-free rate (violation or EXIT)."""
    ok = [e for e in ev if e["success"]]
    rel = float(np.mean([(e["ar_star"] - e["ar"]) / e["ar_star"] for e in ok])) if ok else 1.0
    return {"rel_gap": round(rel, 6), "viol": round(float(np.mean([e["cap_v"] > 0 or e["conf_v"] > 0 for e in ev])), 6),
            "exit": round(float(np.mean([e["exited"] for e in ev])), 6),
            "fail": round(1 - len(ok) / len(ev), 6)}


# ── evaluation ──────────────────────────────────────────────────────────────
def evaluate(model, test, mech: str, reward: str, lam: float, learner: str, obs_mode: str = "base",
             full_episode: bool = False, exit_action: bool = False, action_mode: str = "ecu",
             order: str = "desc", order_seed: int = 0, lam_max: float = C.LAMBDA_MAX) -> list[dict]:
    env = PlacementEnv(test, mech, reward, lam=lam, obs_mode=obs_mode, full_episode=full_episode, order=order, order_seed=order_seed,
                       exit_action=exit_action, action_mode=action_mode, lam_max=lam_max)
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
        out.append({"ar": info["ar"], "ar_obj": info["ar_obj"], "ar_star": info["ar_star"], "success": info["valid_placed"] == env.M,
                    "valid_placed": info["valid_placed"], "cap_v": info["capacity_violations"],
                    "conf_v": info["conflict_violations"], "dead_end": info["dead_end"], "exited": info["exited"],
                    "ecus_used": info["ecus_used"]})
    return out


# ── outputs ─────────────────────────────────────────────────────────────────
def curve_history(run_dir: Path) -> tuple[list, list[int]]:
    """v4.4.16: training-curve rows of the runs this run was resumed from (oldest first, each cut at the
    step where the next one took over) and the resume steps -- so a continued run plots its whole history."""
    rows, marks, d = [], [], run_dir
    chain = []
    while True:
        try:
            res = json.loads((d / "results.json").read_text())
        except (OSError, json.JSONDecodeError):
            break
        src = (res.get("resumed") or {}).get("from")
        if not src:
            break
        at = int(res["resumed"]["at_step"])
        d = (ROOT / src).parent
        chain.append((d, at))
    for d, at in reversed(chain):
        with open(d / "training_curve.csv") as f:
            prev = [tuple(float(x) for x in r) for r in list(csv.reader(f))[1:]]
        rows = [r for r in rows if r[0] <= prev[0][0]] if rows and prev else rows
        rows += [r for r in prev if r[0] <= at]
        marks.append(at)
    return rows, marks


def plot_curve(rows, ar_star, path: Path, title: str, history: tuple[list, list[int]] | None = None,
               x_max: float | None = None) -> None:
    """x_max: total training steps (resumed history included); the step axis is fixed to [0, x_max]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shared.ilp_utils import moving_avg
    if history and history[0]:                          # prepend the runs this one continues (v4.4.16)
        rows = list(history[0]) + list(rows)
    if not rows:
        return
    a = np.array([r[:8] for r in rows], dtype=float)    # v5.0.0 rows carry a 9th column (exited); history may have 8
    w = max(1, min(C.SMOOTH_W, len(a) // 10))
    sm = lambda y: moving_avg(y, w)[0]                  # trailing mean: the curve starts at the first episode
    ts = a[:, 0]
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    ax[0].plot(ts, sm(a[:, 1]), color="steelblue"); ax[0].set_ylabel("Episode reward")
    ax[1].plot(ts, sm(a[:, 2]), color="seagreen", label="AR")
    ax[1].axhline(ar_star, color="red", ls="--", label=f"ILP AR* = {ar_star:.4f}"); ax[1].set_ylabel("AR"); ax[1].legend()
    ax[2].plot(ts, sm(a[:, 3]), color="mediumseagreen"); ax[2].set_ylabel("success_rate"); ax[2].set_ylim(-0.02, 1.02)
    ax[3].plot(ts, sm(a[:, 5]), color="tomato", label="capacity violation")
    ax[3].plot(ts, sm(a[:, 6]), color="darkorange", label="privacy violation")
    ex = np.array([(r[0], r[8]) for r in rows if len(r) > 8], dtype=float)   # v5.0.0: EXIT (Maskable), rows that record it
    if len(ex) and ex[:, 1].any():
        ax[3].plot(ex[:, 0], moving_avg(ex[:, 1], w)[0], color="purple", label="EXIT")
    ax[3].set_ylabel("violation / EXIT rate"); ax[3].set_ylim(-0.02, 1.02); ax[3].legend(); ax[3].set_xlabel("training steps")
    for x in ax:
        x.grid(alpha=0.3)
        x.set_xlim(0, x_max if x_max else max(float(ts[-1]), 1.0))   # step axis: 0 .. total steps
        for mk in (history[1] if history else []):    # where a continued run took over
            x.axvline(mk, color="gray", ls=":", lw=1)
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
    ap.add_argument("--n-epochs", type=int, default=None, help="v6.0.0.2: PPO epochs per update (default PPO_N_EPOCHS)")
    ap.add_argument("--target-kl", type=float, default=None, help="v6.0.0.1: PPO early stop of an update once approx KL > 1.5 x this (default none)")
    ap.add_argument("--hp", default=None, help="v6.1.0: JSON dict of constructor arguments (overrides config and the flags above)")
    ap.add_argument("--val-every", type=int, default=0, help="v6.1.0: evaluate on the test set every this many steps -> val_curve.jsonl")
    ap.add_argument("--order", default="desc", choices=["desc", "asc", "random"],
                    help="v4.4.7: service placement order (random = one fixed permutation per instance)")
    ap.add_argument("--order-seed", type=int, default=0, help="v4.4.7: seed of the random order")
    ap.add_argument("--glob-std", action="store_true",
                    help="v4.4.6: --net graph only; add the utilisation std of the active ECUs to the global token")
    ap.add_argument("--separate-critic", action="store_true",
                    help="v4.4.8: --net graph only; actor and critic use separate encoders of the same structure")
    ap.add_argument("--pair-head", action="store_true",
                    help="v4.4.10: --net graph only; score ECU j by MLP([h_svc, h_ecu, h_svc*h_ecu, h_glob])")
    ap.add_argument("--plan", action="store_true",
                    help="v4.4.12: --net graph only; global plan matrix P = F(s_0), ECU logits P[i_t, j] + per-step score")
    ap.add_argument("--resume", default=None,
                    help="v4.4.16: path of a saved model to continue (weights, optimiser state, step count); --steps = steps to add")
    ap.add_argument("--checkpoints", default=None,
                    help="v4.4.15: comma-separated step counts at which to save ckpt_<step>_<exp_id>.zip (learning curve)")
    ap.add_argument("--train-pool", default=None,
                    help="v4.4.14: train on this ILP-verified instance pool (JSON list, e.g. results/sup_diag/<scen>.json, "
                         "{scen} is substituted) instead of the 1600-instance split; the test split is unchanged")
    ap.add_argument("--test-set", default=None,
                    help="v6.0.4: evaluate on this fixed ILP-verified test set (JSON list, e.g. data/testset_v6.0.4/{scen}.json, "
                         "{scen} is substituted) instead of the seed's 400-instance split")
    ap.add_argument("--glob3", action="store_true",
                    help="v4.4.13: --net graph only; global token input = exactly [M_rem / M, AR_t, sigma_util,t]")
    ap.add_argument("--plan-decay", action="store_true",
                    help="v4.4.13: with --plan, weight the plan term by beta_t = (M - t) / M")
    ap.add_argument("--group-k", type=int, default=0,
                    help="v4.4.11: Mask PPO; K trajectories per training instance, advantage G_k - mean_k G (0 = off)")
    ap.add_argument("--grad-diag", action="store_true",
                    help="v4.4.8: --net graph, shared encoder; log cos(grad L_pi, grad vf_coef*L_V) on the encoder per rollout")
    ap.add_argument("--bc", action="store_true",                # v4.4.2: ILP-demonstration warm start (Mask PPO only)
                    default=os.environ.get("BC_WARMSTART", "0") == "1")
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--lambda-max", type=lam_max_arg, default=C.LAMBDA_MAX,   # v5.0.3: upper bound of the Lagrange multiplier
                    help="Lagrange: clip of lambda, also its scale in the observation (lambda / lambda_max); "
                         "'none' = no cap (v5.0.3.2), the observation is then lambda / (1 + lambda)")
    ap.add_argument("--lambda-init", type=float, default=C.LAMBDA_INIT, help="v5.0.3.7: initial lambda")
    ap.add_argument("--lambda-target", type=float, default=C.LAMBDA_TARGET, help="v5.0.3.7: cost limit d")
    ap.add_argument("--lambda-window", choices=["episodes", "rollout"], default="episodes",
                    help="v5.0.3.7: update lambda every LAMBDA_UPDATE_WINDOW episodes, or once per PPO rollout")
    ap.add_argument("--lambda-update", choices=["linear", "normalized", "adam"], default="linear",   # v5.0.3.1
                    help="Lagrange dual step: linear lr*(J_C - d), or normalized lr*(J_C - d)/max(J_C, d)")
    ap.add_argument("--lambda-cost", choices=["behavior", "greedy"], default="behavior",   # v5.0.3.3
                    help="Lagrange: J_C from the training episodes (behaviour policy, with epsilon for DQN) or from extra "
                         "greedy (deterministic) episodes on training instances")
    ap.add_argument("--lambda-eval-episodes", type=int, default=100, help="v5.0.3.3: greedy episodes per lambda update")
    ap.add_argument("--eps-steps", type=int, default=None,   # v5.0.3.3
                    help="DQN / DDQN: epsilon falls from 1 to 0 over this many steps (default: DQN_EXPLORATION_FRACTION of --steps)")
    ap.add_argument("--lambda-lr", type=float, default=C.LAMBDA_LR, help="Lagrange dual step size eta (per update)")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("TRAIN_SEED", "1")))
    a = ap.parse_args()

    import torch
    from stable_baselines3.common.monitor import Monitor
    from paper_rl.vec_env import FastDummyVecEnv as DummyVecEnv    # v5.0.0: no per-step deepcopy of the infos
    from shared.paths import VERSION, resolve_exp_id, results_dir

    if a.net == "graph":
        a.obs = "raw"
        if a.device.startswith("cuda") and os.environ.get("V5_CUDA_GRAPHS", "1") == "1":   # v5.0.0, see graph_net.run_graphed
            import paper_rl.graph_net as _gn
            _gn.CUDA_GRAPHS = True
    mech, learner = split_algo(a.algo)
    torch.set_num_threads(C.PPO_TORCH_THREADS if learner == "ppo" else C.DQN_TORCH_THREADS)
    data, train, test = split_instances(a.scen, a.seed)
    if a.train_pool:                                    # v4.4.14: many unique (ILP-verified) training instances
        pool = json.loads((ROOT / a.train_pool.format(scen=a.scen)).read_text())
        assert all(x.get("ar_star", 0) > 0 for x in pool), "every training instance must have an ILP optimum"
        train = pool
    if a.test_set:                                      # v6.0.4: one fixed test set for every seed
        test = json.loads((ROOT / a.test_set.format(scen=a.scen)).read_text())
        assert all(x.get("ar_star", 0) > 0 for x in test), "every test instance must have an ILP optimum"
    n, m = data["N"], data["M"]
    outdir = results_dir(a.scen, a.algo)
    exp_id = resolve_exp_id(outdir)
    run_dir = outdir / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # v5.0.3: self-describing run (docs/data_spec.md §2); batch / key / role / group / params come from the driver
    (run_dir / "run.json").write_text(json.dumps({
        "schema": 1, "exp_id": exp_id, "batch": os.environ.get("RUN_BATCH"), "key": os.environ.get("RUN_KEY"),
        "role": os.environ.get("RUN_ROLE", "main"), "group": os.environ.get("RUN_GROUP") or None,
        "version": VERSION, "commit": git_commit(), "scenario": a.scen, "seed": a.seed, "algo": a.algo,
        "learner": learner, "mechanism": mech, "net": a.net, "steps": a.steps,
        "params": {**({"lambda_max": lam_max_json(a.lambda_max)} if mech == "lagrange" else {}),
                   **json.loads(os.environ.get("RUN_PARAMS") or "{}")},
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:]}, indent=2))
    print(f"=== {a.scen.upper()} N={n} M={m} | {algo_label(a.algo)} | reward={a.reward} | "
          f"reward_norm={a.reward_norm} | obs={a.obs} | gamma={a.gamma} | gae_lambda={a.gae_lambda} | lr={a.lr} | ent_coef={a.ent_coef} | n_epochs={a.n_epochs} | target_kl={a.target_kl} | glob_std={a.glob_std} | separate_critic={a.separate_critic} | pair_head={a.pair_head} | group_k={a.group_k} | train={len(train)} | plan={a.plan} | plan_decay={a.plan_decay} | glob3={a.glob3} | order={a.order} | full_episode={a.full_episode} | steps={a.steps:,} | seed={a.seed} | exp_id={exp_id} ===", flush=True)

    n_envs = C.PPO_N_ENVS if learner == "ppo" else C.DQN_N_ENVS
    scale = 1.0 / m if a.reward_norm == "m" else 1.0
    venv = DummyVecEnv([functools.partial(
        lambda s: scale_reward(Monitor(PlacementEnv(train, mech, a.reward, lam=C.LAMBDA_INIT, rng_seed=s, obs_mode=a.obs, lam_max=a.lambda_max,
                                             full_episode=a.full_episode, exit_action=a.exit_action,
                                             action_mode=a.action, order=a.order, order_seed=a.order_seed)), scale),
        a.seed * 1000 + k) for k in range(n_envs)])
    gamma = a.gamma if a.gamma is not None else (C.PPO_GAMMA if learner == "ppo" else C.DQN_GAMMA)
    resumed = None
    if a.resume:                                        # v4.4.16: continue a saved run (weights, optimiser, step count)
        src = Path(a.resume) if Path(a.resume).is_absolute() else ROOT / a.resume
        model = model_class(learner, mech).load(str(src), env=venv, device=a.device)
        resumed = {"from": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src), "at_step": int(model.num_timesteps)}
        print(f"  [resume] {resumed['from']} at step {resumed['at_step']:,}; +{a.steps:,} steps", flush=True)
    else:
        model = build_model(learner, mech, venv, a.seed, n, gamma, a.exit_action, m, a.action, a.net, a.device, a.gae_lambda, a.lr, a.ent_coef, a.glob_std, a.separate_critic, a.pair_head, a.group_k, a.plan, a.plan_decay, a.glob3,
                            None if a.eps_steps is None else min(1.0, a.eps_steps / a.steps), a.n_epochs, a.target_kl,
                            json.loads(a.hp) if a.hp else None)
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
    start = resumed["at_step"] if resumed else 0
    greedy_env = None
    if mech == "lagrange" and a.lambda_cost == "greedy":   # v5.0.3.3: same settings as the training envs, own instance stream
        greedy_env = PlacementEnv(train, mech, a.reward, lam=C.LAMBDA_INIT, rng_seed=a.seed * 1000 + 999, obs_mode=a.obs,
                                  lam_max=a.lambda_max, full_episode=a.full_episode, exit_action=a.exit_action,
                                  action_mode=a.action, order=a.order, order_seed=a.order_seed)
    cb = main_cb = make_callback(mech, start + a.steps, outdir, exp_id, start, a.lambda_max, a.lambda_update, a.lambda_lr,
                                 a.lambda_cost, greedy_env, a.lambda_eval_episodes,
                                 a.lambda_init, a.lambda_target, a.lambda_window)
    extra = []
    if a.grad_diag:                                     # v4.4.8: actor / critic gradient conflict on the encoder
        assert a.net == "graph" and not a.separate_critic, "--grad-diag: shared graph encoder only"
        from paper_rl.grad_diag import GradDiagCallback
        extra.append(GradDiagCallback(run_dir / "grad_cos.csv"))
    if a.checkpoints:                                   # v4.4.15: save the model at these step counts
        extra.append(checkpoint_callback(sorted(int(x) for x in a.checkpoints.split(",")), run_dir, exp_id,
                                         (lambda: main_cb.lam) if mech == "lagrange" else None, a.lambda_max))
    if a.val_every:                                     # v6.1.0: intermediate validation for the tuner
        extra.append(val_callback(a.val_every, run_dir / "val_curve.jsonl", lambda mdl: evaluate(
            mdl, test, mech, a.reward, main_cb.lam if mech == "lagrange" else 0.0, learner, a.obs, a.full_episode,
            a.exit_action, a.action, a.order, a.order_seed, a.lambda_max)))
    if extra:
        from stable_baselines3.common.callbacks import CallbackList
        cb = CallbackList([main_cb, *extra])
    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=cb, reset_num_timesteps=not resumed)
    train_s = time.time() - t0
    cb = main_cb
    lam = cb.lam if mech == "lagrange" else 0.0
    model.save(str(run_dir / f"model_{exp_id}_v{VERSION}-{a.reward}"))

    ev = evaluate(model, test, mech, a.reward, lam, learner, a.obs, a.full_episode, a.exit_action, a.action, a.order, a.order_seed,
                  a.lambda_max)
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
    ar_obj = float(np.mean([e["ar_obj"] for e in ev]))
    if mech == "none":                                  # v6.0.1: the unconstrained objective over all test instances
        print(f"  eval (unconstrained objective, all instances): AR_obj={ar_obj:.4f} ILP AR={ilp_ar:.4f} "
              f"| any violation={1 - sr:.4f}", flush=True)

    res = {
        "created_at": datetime.datetime.now().isoformat(), "exp_id": exp_id, "version": VERSION,
        "commit": git_commit(), "scenario": a.scen, "N": n, "M": m, "algo": a.algo,
        "mechanism": mech, "learner": learner, "reward_mode": a.reward, "reward_norm": a.reward_norm, "obs": a.obs, "gamma": gamma, "full_episode": a.full_episode, "exit_action": a.exit_action, "action_mode": a.action, "seed": a.seed,
        "net": a.net, "device": a.device, "bc": bc_info, "glob_std": a.glob_std, "separate_critic": a.separate_critic, "pair_head": a.pair_head, "group_k": a.group_k, "train_pool": a.train_pool, "test_set": a.test_set, "checkpoints": a.checkpoints, "resumed": resumed, "plan": a.plan, "plan_decay": a.plan_decay, "glob3": a.glob3, "order": a.order, "order_seed": a.order_seed,
        **({"gae_lambda": C.PPO_GAE_LAMBDA if a.gae_lambda is None else a.gae_lambda,
            "lr": C.PPO_LR if a.lr is None else a.lr,
            "ent_coef": C.PPO_ENT_COEF if a.ent_coef is None else a.ent_coef,
            "n_epochs": C.PPO_N_EPOCHS if a.n_epochs is None else a.n_epochs, "target_kl": a.target_kl} if learner == "ppo" else {}),
        "hp": json.loads(a.hp) if a.hp else None, "val": val_summary(ev),
        "train_count": len(train), "test_count": len(test),
        "data": {"conflict_pair_prob": data["conflict_pair_prob"], "k_sets": data["k_sets"]},
        "ilp": {"ar": round(ilp_ar, 6)},
        a.algo: {
            "success_rate": round(sr, 6),
            "ar_mean": round(float(ars.mean()), 6), "ar_std": round(float(ars.std()), 6),
            "ar_obj_mean": round(ar_obj, 6),
            "ar_ratio_successful": round(float(np.mean(succ_ratio)), 6) if succ_ratio else None,
            "cap_viol_rate": round(cap_rate, 6), "conflict_viol_rate": round(conf_rate, 6),
            "cap_viol_total": int(sum(e["cap_v"] for e in ev)),
            "conflict_viol_total": int(sum(e["conf_v"] for e in ev)),
            "dead_end_rate": round(float(np.mean([e["dead_end"] for e in ev])), 6),
            "exit_rate": round(float(np.mean([e["exited"] for e in ev])), 6),
        },
        "training": {"total_steps": start + a.steps, "steps_this_run": a.steps, "n_episodes": len(cb.rows), "train_seconds": round(train_s, 1),
                     "ar_last50": round(float(np.mean([r[2] for r in cb.rows[-50:]])), 6) if cb.rows else None,
                     **({"final_lambda": round(lam, 6), "lambda_max": lam_max_json(a.lambda_max), "lambda_lr": a.lambda_lr, "lambda_update": a.lambda_update,
                         "lambda_init": a.lambda_init, "lambda_target": a.lambda_target,
                         "lambda_window": "rollout" if a.lambda_window == "rollout" else C.LAMBDA_UPDATE_WINDOW, "lambda_warmup": 0,
                         "lambda_cost": a.lambda_cost, **({"lambda_eval_episodes": a.lambda_eval_episodes} if a.lambda_cost == "greedy" else {})}
                        if mech == "lagrange" else {}),
                     **({"eps_steps": a.eps_steps} if a.eps_steps is not None else {})},
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
                    "episode_cap_violated", "episode_conflict_violated", "lambda", "episode_exited"])
        w.writerows(cb.rows)
    if mech == "lagrange":                              # v5.0.3.3: one row per lambda update
        with open(run_dir / "lambda_updates.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "timestep", "jc_behavior", "jc_used", "lambda"])
            w.writerows(cb.dual_rows)
    plot_curve(cb.rows, ilp_ar, run_dir / "training_curve.png",
               f"{algo_label(a.algo)} — {a.scen.upper()} — reward={a.reward} — {start + a.steps:,} steps"
               + (f" (resumed at {start:,})" if resumed else ""), history=curve_history(run_dir) if resumed else None,
               x_max=start + a.steps)
    print(f"  saved -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
