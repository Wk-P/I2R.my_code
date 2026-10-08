"""src/shared/dqn_variant_runner.py — one pipeline for the v4.3.0 DQN variants:

    mask_dqn / mask_ddqn      — Mask-PPO environment (ppo_mask/env.py::P4Env)
                                + MaskableDQN / MaskableDDQN
    repair_dqn / repair_ddqn  — Repair-PPO environment (ppo_opt/env.py::P6Env)
                                + DQN / DoubleDQN
    lagrange_dqn / lagrange_ddqn — Lagrange-PPO environment (ppo_lagrangian/env.py::
                                LagrangeEnv) + DQN / DoubleDQN, with the same dual
                                ascent on lambda as Lagrange-PPO (v4.3.1.2)

Each variant reuses its PPO counterpart's environment unchanged (same
observation, reward and constraint mechanism), so Mask-DQN vs Mask-PPO,
Repair-DQN vs Repair-PPO and Lagrange-DQN vs Lagrange-PPO differ only in the
learning algorithm. Lagrange: lambda is updated every LAMBDA_UPDATE_WINDOW
episodes after LAMBDA_WARMUP_EPISODES by
lambda <- clip(lambda + LAMBDA_LR * (mean violation rate - LAMBDA_TARGET), 0, LAMBDA_MAX)
(same constants as <scen>/ppo_lagrangian/config.py) and frozen at its final
value for evaluation. Transitions already in the replay buffer keep the
reward computed under the lambda in force when they were collected.
P4Env's curriculum weight _ar_weight is left at its default 1.0, i.e. the
plain terminal reward M*(2*AR-1) -- the curriculum is a Mask-PPO training
callback, not part of the environment.

scenarios/<scen>/<variant>/run_all.py is a thin wrapper:

    import config as C
    from shared.dqn_variant_runner import make_runner
    R = make_runner(C, "mask", double=False)
    run_episodes, MODEL_CLS = R.run_episodes, R.MODEL_CLS
    if __name__ == "__main__":
        R.main()

Outputs match scenarios/*/dqn/run_all.py: results/<branch>/<scen>/<variant>/
<exp_id>/{model_*.zip, results.json, summary.csv, training_curve.csv,
training_curve.png, comparison.png}.

Violation columns count violations of the executed placement: for mask
variants only possible when a state has no valid ECU; for repair variants
always 0 (repair triggers are not recorded).
"""

from __future__ import annotations

import csv
import datetime
import functools
import json
import random
import sys
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["agg.path.chunksize"] = 10000
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import shared.timer_utils as timer_utils
from shared.dqn_variants import DoubleDQN, MaskableDQN, MaskableDDQN
from shared.ilp_utils import load_scenario, moving_avg, parse_args, resolve_device, solve_ilp_all_scenarios
from shared.paths import VERSION, resolve_exp_id, write_progress
from ilp.objects import ECU, SVC


def _episode_stats(info: dict, M: int, variant: str) -> dict:
    """Normalise P4Env / P6Env terminal info into one record."""
    valid_placed = int(info.get("valid_placed", 0))
    if variant == "repair":
        # The executed placement never violates; repair triggers are not reported.
        cap_v = conf_v = 0
        success = valid_placed == M
    else:
        cap_v = int(info.get("capacity_violations", info.get("cap_violations", 0)))
        conf_v = int(info.get("conflict_violations", 0))
        success = valid_placed == M and cap_v == 0 and conf_v == 0
    return {
        "ar": float(info.get("ar", 0.0)),
        "placed": int(info.get("services_placed", 0)),
        "valid_placed": valid_placed,
        "ecus_used": int(info.get("ecus_used", 0)),
        "cap_v": cap_v,
        "conf_v": conf_v,
        "success": bool(success),
    }


def make_runner(C, variant: str, double: bool) -> SimpleNamespace:
    assert variant in ("mask", "repair", "lagrange")
    algo = f"{variant}_{'ddqn' if double else 'dqn'}"
    label = {"mask_dqn": "Mask-DQN", "mask_ddqn": "Mask-DDQN",
             "repair_dqn": "Repair-DQN", "repair_ddqn": "Repair-DDQN",
             "lagrange_dqn": "Lagrange-DQN", "lagrange_ddqn": "Lagrange-DDQN"}[algo]

    if variant == "mask":
        from ppo_mask.env import P4Env as EnvCls
        model_cls = MaskableDDQN if double else MaskableDQN
        # P4Env observation: 7 scalars, then initial_cap, remaining,
        # conflict_flag, ecu_allowed_frac (N each), then valid_flag = action_masks().
        extra_kwargs = {"mask_start": 7 + 4 * C.N}
    elif variant == "repair":
        from ppo_opt.env import P6Env as EnvCls
        model_cls = DoubleDQN if double else DQN
        extra_kwargs = {}
    else:
        from ppo_lagrangian.env import LagrangeEnv as EnvCls
        model_cls = DoubleDQN if double else DQN
        extra_kwargs = {}

    # lambda for the Lagrange env: the dual variable during training, frozen
    # at its final value for evaluation
    state = {"lambda": getattr(C, "LAMBDA_INIT", 0.0)}

    def env_kwargs() -> dict:
        if variant != "lagrange":
            return {}
        return {"lambda_init": state["lambda"], "lambda_max": C.LAMBDA_MAX}

    def make_env(seed: int) -> Monitor:
        random.seed(seed)
        caps, reqs, _ = C.SCENARIOS[C.SCENARIO_IDX]
        ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
        services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
        return Monitor(EnvCls(ecus, services, scenarios=C.TRAIN_SCENARIOS, **env_kwargs()))

    # ── evaluation ───────────────────────────────────────────────────────────
    def run_episodes(ecus, services, policy_fn, n_samples: int = 1):
        """One episode per test scenario (n_samples kept for the shared eval
        tooling's signature; v4.3.0 always evaluates with n_samples=1)."""
        rows = []
        for scenario in C.TEST_SCENARIOS:
            caps, reqs, _ = scenario
            _ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
            _svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
            best = None
            for _ in range(max(1, n_samples)):
                env = EnvCls(_ecus, _svcs, scenarios=[scenario], **env_kwargs())
                obs, _ = env.reset()
                done, info = False, {}
                while not done:
                    obs, _, done, _, info = env.step(policy_fn(obs))
                st = _episode_stats(info, len(reqs), variant)
                key = (st["success"], st["valid_placed"], st["ar"])
                if best is None or key > best[0]:
                    best = (key, st)
                if st["success"]:
                    break
            rows.append(best[1])
        return {
            "ars": np.array([r["ar"] for r in rows]),
            "placed": np.array([r["placed"] for r in rows]),
            "valid_placed": np.array([r["valid_placed"] for r in rows]),
            "ecus_used": np.array([r["ecus_used"] for r in rows]),
            "cap_viols": np.array([r["cap_v"] for r in rows]),
            "conflict_viols": np.array([r["conf_v"] for r in rows]),
            "viols": np.array([int(r["cap_v"] + r["conf_v"] > 0) for r in rows]),
            "success": np.array([r["success"] for r in rows]),
        }

    # ── training ─────────────────────────────────────────────────────────────
    class Callback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rewards, self.ars, self.placed, self.valid_placed = [], [], [], []
            self.cap, self.conf, self.success, self.ts = [], [], [], []
            self._next_progress = C.PROGRESS_LOG_EVERY_STEPS
            self._t0 = 0.0
            self.lambdas = []
            if variant == "lagrange":
                from collections import deque
                self._viol_window = deque(maxlen=C.LAMBDA_UPDATE_WINDOW)
                self._n_ep = 0
                self._since_update = 0

        def _on_training_start(self) -> None:
            self._t0 = time.time()

        def _dual_step(self, viol_rate: float) -> None:
            self._n_ep += 1
            self._viol_window.append(viol_rate)
            if self._n_ep >= C.LAMBDA_WARMUP_EPISODES:
                self._since_update += 1
            if (self._n_ep >= C.LAMBDA_WARMUP_EPISODES
                    and len(self._viol_window) == C.LAMBDA_UPDATE_WINDOW
                    and self._since_update >= C.LAMBDA_UPDATE_WINDOW):
                lam = state["lambda"] + C.LAMBDA_LR * (float(np.mean(self._viol_window)) - C.LAMBDA_TARGET)
                state["lambda"] = float(np.clip(lam, 0.0, C.LAMBDA_MAX))
                self.training_env.env_method("set_lambda", state["lambda"])
                self._since_update = 0
            self.lambdas.append(state["lambda"])

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    st = _episode_stats(info, C.M, variant)
                    self.rewards.append(float(info["episode"]["r"]))
                    self.ars.append(st["ar"])
                    self.placed.append(st["placed"])
                    self.valid_placed.append(st["valid_placed"])
                    self.cap.append(int(st["cap_v"] > 0))
                    self.conf.append(int(st["conf_v"] > 0))
                    self.success.append(st["success"])
                    self.ts.append(self.num_timesteps)
                    if variant == "lagrange":
                        self._dual_step(float(info.get("viol_rate_ep", 0.0)))
            if self.num_timesteps >= self._next_progress:
                elapsed = max(time.time() - self._t0, 1e-6)
                pct = min(100.0, self.num_timesteps * 100.0 / C.TOTAL_STEPS)
                sps = self.num_timesteps / elapsed
                lam = f" | lambda={state['lambda']:.4f}" if variant == "lagrange" else ""
                print(f"  [train] step={self.num_timesteps:,}/{C.TOTAL_STEPS:,} "
                      f"({pct:5.1f}%) | eps={len(self.rewards)} | steps/s={sps:,.0f}{lam}")
                write_progress(C.OUTDIR, step=self.num_timesteps, total_steps=C.TOTAL_STEPS,
                               pct=round(pct, 1), episodes=len(self.rewards),
                               steps_per_sec=round(sps), exp_id=C.EXP_ID)
                self._next_progress += C.PROGRESS_LOG_EVERY_STEPS
            return True

    def train(device: str):
        import torch
        torch.set_num_threads(C.TORCH_NUM_THREADS)
        n_envs = max(1, int(C.N_ENVS))
        env = DummyVecEnv([functools.partial(make_env, C.SEED + i) for i in range(n_envs)])
        print(f"  Using DummyVecEnv: n_envs={n_envs}")
        cb = Callback()
        model = model_cls(
            policy="MlpPolicy", env=env,
            learning_rate=C.DQN_LR, buffer_size=C.DQN_BUFFER_SIZE,
            learning_starts=C.DQN_LEARNING_STARTS, batch_size=C.DQN_BATCH_SIZE,
            tau=C.DQN_TAU, gamma=C.DQN_GAMMA, train_freq=C.DQN_TRAIN_FREQ,
            gradient_steps=C.DQN_GRADIENT_STEPS, target_update_interval=C.DQN_TARGET_UPDATE,
            exploration_fraction=C.DQN_EXPLORATION_FRACTION,
            exploration_final_eps=C.DQN_EXPLORATION_FINAL_EPS,
            policy_kwargs=dict(net_arch=C.DQN_NET_ARCH),
            device=device, verbose=0, seed=C.SEED, **extra_kwargs,
        )
        t0 = time.time()
        model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
        env.close()
        n = len(cb.rewards)
        print(f"  Training done  {time.time() - t0:.1f}s | {n} eps "
              f"| reward(last50)={np.mean(cb.rewards[-50:]):.4f} "
              f"| AR(last50)={np.mean(cb.ars[-50:]):.4f} "
              f"| success(last50)={np.mean(cb.success[-50:]):.2%}")
        return model, cb

    # ── outputs ──────────────────────────────────────────────────────────────
    viol_word = "violation"

    def save_training_curve_csv(cb, outdir):
        path = outdir / "training_curve.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestep", "episode_ar", "episode_success", "episode_valid_placed",
                        "episode_placed", "episode_reward"])
            for i in range(len(cb.ts)):
                w.writerow([cb.ts[i], round(cb.ars[i], 6), int(cb.success[i]),
                            cb.valid_placed[i], cb.placed[i], round(cb.rewards[i], 6)])
        print(f"  Saved -> {path}")

    def plot_training_curve(cb, ilp_ar, outdir, scenario_name):
        fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
        ts = np.array(cb.ts)
        series = [
            (cb.rewards, "steelblue", "Episode Reward", "episode reward"),
            (cb.ars, "seagreen", "Episode AR", f"{label} AR"),
            (np.array(cb.cap, dtype=float), "tomato", f"Cap {viol_word} rate", f"cap {viol_word} rate"),
            (np.array(cb.conf, dtype=float), "darkorange", f"Privacy {viol_word} rate",
             f"privacy (conflict) {viol_word} rate"),
        ]
        for ax, (y, color, ylabel, name) in zip(axes, series):
            sm, off = moving_avg(y, C.SMOOTH_W)
            ax.plot(ts, y, color=color, alpha=0.2, linewidth=0.8)
            ax.plot(ts[off:off + len(sm)], sm, color=color, linewidth=2,
                    label=f"{name} (smoothed w={C.SMOOTH_W})")
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(alpha=0.3)
        axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        axes[1].axhline(ilp_ar, color="red", linestyle="--", linewidth=1.5,
                        label=f"ILP Optimal  AR={ilp_ar:.4f}")
        axes[1].set_ylim(0.0, 1.05)
        for ax in axes[2:]:
            ax.set_ylim(-0.02, 1.02)
        for ax in axes:
            ax.legend(fontsize=9)
        axes[0].set_title(f"Training Metrics — {label} — {scenario_name}  ({C.TOTAL_STEPS:,} steps)",
                          fontsize=12)
        axes[-1].set_xlabel("Training steps", fontsize=11)
        plt.tight_layout()
        path = outdir / "training_curve.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved -> {path}")

    def plot_comparison(ilp_ar, res, cap_rate, conf_rate, outdir, scenario_name):
        success_rate = float(np.mean(res["success"]))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle(f"ILP vs {label} - {scenario_name}", fontsize=13, fontweight="bold")
        means = [ilp_ar, float(np.mean(res["ars"]))]
        bars = ax1.bar(["ILP\n(Optimal)", label], means, yerr=[0.0, float(np.std(res["ars"]))],
                       capsize=5, color=["#e74c3c", "#2ecc71"], alpha=0.8, ecolor="black")
        for bar, v in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.4f}",
                     ha="center", fontsize=10, fontweight="bold")
        ax1.set_ylim(0, 1.1)
        ax1.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
        ax1.set_title("Test AR: Algorithm vs ILP", fontsize=11)
        ax1.grid(axis="y", alpha=0.3)
        vals = [success_rate, cap_rate, conf_rate]
        bars = ax2.bar(["Success\nRate", f"Cap\n{viol_word}", f"Privacy\n{viol_word}"], vals,
                       color=["#2ecc71", "#e67e22", "#c0392b"], alpha=0.8)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2%}",
                     ha="center", fontsize=10, fontweight="bold")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Rate", fontsize=11)
        ax2.set_title("Test Success Rate & Violation Rates", fontsize=11)
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = outdir / "comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved -> {path}")

    # ── main ─────────────────────────────────────────────────────────────────
    @timer_utils.timer
    def main():
        args = parse_args()
        if args.total_timesteps is not None:
            C.TOTAL_STEPS = int(args.total_timesteps)
            print(f"[override] TOTAL_STEPS={C.TOTAL_STEPS:,}")
        exp_id = resolve_exp_id(C.OUTDIR)
        C.EXP_ID = exp_id
        base_dir = C.OUTDIR / exp_id
        base_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}\n  {algo} run_all.py — {label}\n"
              f"  Config : {C.YAML_CONFIG.name}  |  train={len(C.TRAIN_SCENARIOS)}/test={len(C.TEST_SCENARIOS)}\n"
              f"{'=' * 60}\n")
        device = resolve_device(C.DEVICE)
        ecus, services, sc_name, prototype_name = load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
        N, M = len(ecus), len(services)

        print(f"\n[1/3] Solving ILP for {len(C.TEST_SCENARIOS)} test scenarios ...")
        ilp_ar, ilp_per_sc = solve_ilp_all_scenarios(C.YAML_CONFIG, C.TEST_SCENARIOS, C.OUTDIR)
        print(f"  ILP mean AR: {ilp_ar:.4f}")

        print(f"\n[2/3] {label} training ({C.TOTAL_STEPS:,} steps) ...")
        model, cb = train(device)
        model_path = base_dir / f"model_{exp_id}_v{VERSION}"
        model.save(str(model_path))
        print(f"  Model saved -> {model_path}.zip")

        print(f"\n[3/3] {label} evaluation ({len(C.TEST_SCENARIOS)} episodes, deterministic) ...")

        def policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        res = run_episodes(ecus, services, policy, n_samples=getattr(C, "EVAL_BEST_OF_N", 1))
        success_rate = float(np.mean(res["success"]))
        cap_rate = float(np.mean(res["cap_viols"] > 0))
        conf_rate = float(np.mean(res["conflict_viols"] > 0))
        print(f"  AR mean={np.mean(res['ars']):.4f} std={np.std(res['ars']):.4f} | "
              f"success={success_rate:.2%} | cap {viol_word}={cap_rate:.2%} | "
              f"conflict {viol_word}={conf_rate:.2%}")

        log = {
            "created_at": datetime.datetime.now().isoformat(),
            "scenario": sc_name,
            "prototype_scenario": prototype_name,
            "scenario_count": len(C.SCENARIOS),
            "train_count": len(C.TRAIN_SCENARIOS),
            "test_count": len(C.TEST_SCENARIOS),
            "N": N, "M": M,
            "ilp": {
                "ar": round(ilp_ar, 6),
                "ar_per_scenario": [round(r["avg_utilization"], 6) for r in ilp_per_sc],
                "violations": 0,
            },
            algo: {
                "ar_mean": round(float(np.mean(res["ars"])), 6),
                "ar_std": round(float(np.std(res["ars"])), 6),
                "placed_mean": round(float(np.mean(res["placed"])), 2),
                "viol_rate": round(float(np.mean(res["viols"])), 4),
                "success_rate": round(success_rate, 6),
                "cap_viol_rate": round(cap_rate, 6),
                "conflict_viol_rate": round(conf_rate, 6),
                "cap_viol_total": int(np.sum(res["cap_viols"])),
                "conflict_viol_total": int(np.sum(res["conflict_viols"])),
            },
            "training": {
                "total_steps": C.TOTAL_STEPS,
                "n_episodes": len(cb.rewards),
                "ar_last50": round(float(np.mean(cb.ars[-50:])), 6),
                "reward_last50": round(float(np.mean(cb.rewards[-50:])), 6),
                "success_last50": round(float(np.mean(cb.success[-50:])), 4),
                **({"final_lambda": round(state["lambda"], 6), "eval_lambda": round(state["lambda"], 6)}
                   if variant == "lagrange" else {}),
            },
        }
        with open(base_dir / "results.json", "w") as f:
            json.dump(log, f, indent=2)
        print(f"  JSON saved -> {base_dir / 'results.json'}")

        with open(base_dir / "summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "ar_mean", "ar_std", "placed_mean", "valid_placed_mean", "ecus_used_mean",
                        "success_rate", "cap_viol_rate", "conflict_viol_rate", "cap_viol_total",
                        "conflict_viol_total"])
            w.writerow(["ILP (Optimal)", round(ilp_ar, 6), 0.0, M, M, C.N, 1.0, 0.0, 0.0, 0, 0])
            w.writerow([label,
                        round(float(np.mean(res["ars"])), 6), round(float(np.std(res["ars"])), 6),
                        round(float(np.mean(res["placed"])), 2), round(float(np.mean(res["valid_placed"])), 2),
                        round(float(np.mean(res["ecus_used"])), 2), round(success_rate, 4),
                        round(cap_rate, 4), round(conf_rate, 4),
                        int(np.sum(res["cap_viols"])), int(np.sum(res["conflict_viols"]))])
        print(f"  CSV  saved -> {base_dir / 'summary.csv'}")

        save_training_curve_csv(cb, base_dir)
        plot_training_curve(cb, ilp_ar, base_dir, sc_name)
        plot_comparison(ilp_ar, res, cap_rate, conf_rate, base_dir, sc_name)
        print(f"\nAll done! Output dir: {base_dir}\n")

    return SimpleNamespace(algo=algo, label=label, EnvCls=EnvCls, MODEL_CLS=model_cls,
                           run_episodes=run_episodes, train=train, main=main)
