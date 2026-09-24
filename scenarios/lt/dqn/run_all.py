"""
run_all.py — One-shot DQN full pipeline:

  1. Load scenario from YAML
  2. Solve with ILP (PuLP)             -> ilp_ar (optimal upper bound)
  3. Evaluate random policy (no mask)  -> random_ars + violations
  4. Train DQN (no action masking)     -> training curve
  5. Evaluate trained DQN              -> dqn_ars + violations
  6. Produce plots:
    - comparison.png     — AR box plot + violation rate bar (3-way)
    - training_curve.png — AR & violation rate during training

DQN Design: NO action masking.
    - Constraint violations terminate the episode with an unfinished-services penalty.
    - Valid assignments earn their exact utilisation contribution n_i / e_j.

Run:
    python dqn/run_all.py
"""

import datetime
import csv
import argparse
import functools
import sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["agg.path.chunksize"] = 10000
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))  # project root for shared

import shared.timer_utils as timer_utils

import torch
import yaml
import pulp
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import config as C
from dqn.env import DQNEnv
from ilp.objects import ECU, SVC
from shared.ilp_utils import parse_args, resolve_device, moving_avg, solve_ilp, solve_ilp_all_scenarios, load_scenario
from shared.paths import VERSION, resolve_exp_id, write_progress


def _make_dqn_env(seed: int) -> Monitor:
    import random
    random.seed(seed)
    caps, reqs, _ = C.SCENARIOS[C.SCENARIO_IDX]
    ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return Monitor(DQNEnv(ecus, services, scenarios=C.TRAIN_SCENARIOS))


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 & 5 — Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episodes(ecus, services, policy_fn, n_samples: int = 1):
    """policy_fn(obs) -> int   (no mask)

    Run up to n_samples independent episodes per scenario in C.TEST_SCENARIOS
    and keep the best attempt (success first, then most services validly
    placed, then highest AR) -- same best-of-N re-roll pattern as
    ppo_mask/run_all.py::run_episodes(), added here for evaluation-protocol
    parity across all 6 algorithms. Requires policy_fn to sample with a
    non-zero exploration rate (deterministic=False AND model.exploration_rate
    set above 0 at the call site) -- SB3's DQN.predict(deterministic=False)
    only injects randomness with probability self.exploration_rate, which is
    annealed to DQN_EXPLORATION_FINAL_EPS=0.0 by the end of training, so
    without overriding it the "stochastic" call is still bit-for-bit
    deterministic and re-rolling wastes every extra sample.
    """
    ars, placed_list, viol_list, cap_viol_list, conflict_viol_list = [], [], [], [], []
    valid_placed_list, ecus_used_list, success_list, attempts_list = [], [], [], []
    for scenario in C.TEST_SCENARIOS:
        caps, reqs, cs = scenario
        M_sc = len(reqs)
        _ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
        _svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]

        best = None  # (success, valid_placed, ar, cap_v, conflict_v, placed, ecus_used)
        used_attempts = 0
        for attempt in range(max(1, n_samples)):
            env = DQNEnv(_ecus, _svcs, scenarios=[scenario])
            obs, _ = env.reset()
            done = False
            info = {}
            while not done:
                obs, _, done, _, info = env.step(policy_fn(obs))
            placed = info.get("services_placed", 0)
            valid_placed = int(info.get("valid_placed", placed))
            ar = info.get("ar", 0.0)
            ecus_used = int(info.get("ecus_used", 0))
            cap_v = int(info.get("capacity_violations", 0))
            conflict_v = int(info.get("conflict_violations", 0))
            success = bool(valid_placed == M_sc and cap_v == 0 and conflict_v == 0)
            used_attempts = attempt + 1
            candidate = (success, valid_placed, ar, cap_v, conflict_v, placed, ecus_used)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
            if success:
                break  # found a fully valid placement -- no need to re-roll further

        success, valid_placed, ar, cap_v, conflict_v, placed, ecus_used = best
        ars.append(ar)
        placed_list.append(placed)
        valid_placed_list.append(valid_placed)
        ecus_used_list.append(ecus_used)
        viol_list.append(1 if (cap_v + conflict_v) > 0 else 0)
        cap_viol_list.append(cap_v)
        conflict_viol_list.append(conflict_v)
        success_list.append(success)
        attempts_list.append(used_attempts)
    return {
        "ars":           np.array(ars),
        "placed":        np.array(placed_list),
        "viols":         np.array(viol_list),
        "cap_viols":     np.array(cap_viol_list),
        "conflict_viols": np.array(conflict_viol_list),
        "valid_placed": np.array(valid_placed_list),
        "ecus_used":    np.array(ecus_used_list),
        "success":      np.array(success_list),
        "attempts":     np.array(attempts_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Training
# ══════════════════════════════════════════════════════════════════════════════

class DQNCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards:       list[float] = []
        self.episode_ars:           list[float] = []
        self.episode_placed:        list[int]   = []
        self.episode_violated:      list[int]   = []
        self.episode_cap_violated:  list[int]   = []
        self.episode_conf_violated: list[int]   = []
        self.episode_valid_placed:  list[int]   = []
        self.episode_success:       list[bool]  = []
        self.timesteps_at_ep:       list[int]   = []
        self._next_progress_step = C.PROGRESS_LOG_EVERY_STEPS
        self._t_start = 0.0

    def _on_training_start(self) -> None:
        self._t_start = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_ars.append(float(info.get("ar", 0.0)))
                self.episode_placed.append(int(info.get("services_placed", 0)))
                cap_viols  = int(info.get("capacity_violations", 0))
                conf_viols = int(info.get("conflict_violations", 0))
                self.episode_violated.append(1 if (cap_viols + conf_viols) > 0 else 0)
                self.episode_cap_violated.append(1 if cap_viols > 0 else 0)
                self.episode_conf_violated.append(1 if conf_viols > 0 else 0)
                valid_placed = int(info.get("valid_placed", 0))
                self.episode_valid_placed.append(valid_placed)
                self.episode_success.append(bool(
                    valid_placed == C.M and cap_viols == 0 and conf_viols == 0
                ))
                self.timesteps_at_ep.append(self.num_timesteps)

        if self.num_timesteps >= self._next_progress_step:
            elapsed = max(time.time() - self._t_start, 1e-6)
            pct = min(100.0, self.num_timesteps * 100.0 / C.TOTAL_STEPS)
            eps = len(self.episode_rewards)
            sps = self.num_timesteps / elapsed
            print(
                f"  [train] step={self.num_timesteps:,}/{C.TOTAL_STEPS:,} "
                f"({pct:5.1f}%) | eps={eps} | steps/s={sps:,.0f}"
            )
            write_progress(
                C.OUTDIR,
                step=self.num_timesteps, total_steps=C.TOTAL_STEPS, pct=round(pct, 1),
                episodes=eps, steps_per_sec=round(sps), exp_id=C.EXP_ID,
            )
            self._next_progress_step += C.PROGRESS_LOG_EVERY_STEPS
        return True


def train_dqn(ecus, services, device: str):
    import torch as _torch
    _torch.set_num_threads(C.TORCH_NUM_THREADS)
    sys.stdout.flush()
    n_envs = max(1, int(C.N_ENVS))
    env = DummyVecEnv(
        [functools.partial(_make_dqn_env, C.SEED + i) for i in range(n_envs)],
    )
    print(f"  Using DummyVecEnv: n_envs={n_envs}")
    cb = DQNCallback()
    model = DQN(
        policy                 = "MlpPolicy",
        env                    = env,
        learning_rate          = C.DQN_LR,
        buffer_size            = C.DQN_BUFFER_SIZE,
        learning_starts        = C.DQN_LEARNING_STARTS,
        batch_size             = C.DQN_BATCH_SIZE,
        tau                    = C.DQN_TAU,
        gamma                  = C.DQN_GAMMA,
        train_freq             = C.DQN_TRAIN_FREQ,
        gradient_steps         = C.DQN_GRADIENT_STEPS,
        target_update_interval = C.DQN_TARGET_UPDATE,
        exploration_fraction   = C.DQN_EXPLORATION_FRACTION,
        exploration_final_eps  = C.DQN_EXPLORATION_FINAL_EPS,
        policy_kwargs          = dict(net_arch=C.DQN_NET_ARCH),
        device                 = device,
        verbose                = 0,
        seed                   = C.SEED,
    )
    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
    elapsed = time.time() - t0
    env.close()

    n_ep     = len(cb.episode_rewards)
    last50_r = np.mean(cb.episode_rewards[-50:]) if n_ep >= 50 else np.mean(cb.episode_rewards)
    last50_ar = np.mean(cb.episode_ars[-50:]) if n_ep >= 50 else np.mean(cb.episode_ars)
    last50_v = np.mean(cb.episode_violated[-50:]) if n_ep >= 50 else np.mean(cb.episode_violated)
    print(f"  Training done  {elapsed:.1f}s | {n_ep} eps "
            f"| reward(last50)={last50_r:.4f} | AR(last50)={last50_ar:.4f}"
            f" | viol_rate(last50)={last50_v:.2%}")
    return model, cb


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def save_training_curve_csv(cb, outdir):
    """Dump the raw per-episode training trajectory (not just the PNG) so the
    curve's SHAPE can be checked quantitatively later."""
    path = outdir / "training_curve.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "episode_ar", "episode_success", "episode_valid_placed", "episode_placed"])
        for i in range(len(cb.timesteps_at_ep)):
            writer.writerow([
                cb.timesteps_at_ep[i],
                round(cb.episode_ars[i], 6),
                int(cb.episode_success[i]),
                cb.episode_valid_placed[i],
                cb.episode_placed[i],
            ])
    print(f"  Saved -> {path}")


def plot_training_curve(cb, ilp_ar, outdir, scenario_name):
    # 2026-09-11: 4-panel layout (reward / AR / CapViolation / PrivacyViolation)
    # standardized across all 6 algorithms -- see ppo_mask/run_all.py's
    # plot_training_curve() comment for rationale.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
    ts = np.array(cb.timesteps_at_ep)

    sm_r, off_r = moving_avg(cb.episode_rewards, C.SMOOTH_W)
    ax1.plot(ts, cb.episode_rewards, color="steelblue", alpha=0.2, linewidth=0.8)
    ax1.plot(ts[off_r:off_r+len(sm_r)], sm_r, color="steelblue", linewidth=2,
             label=f"episode reward (smoothed w={C.SMOOTH_W})")
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax1.set_ylabel("Episode Reward", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_title(f"Training Metrics — {scenario_name}  ({C.TOTAL_STEPS:,} steps)", fontsize=12)
    ax1.grid(alpha=0.3)

    sm, off = moving_avg(cb.episode_ars, C.SMOOTH_W)
    ax2.plot(ts, cb.episode_ars, color="seagreen", alpha=0.2, linewidth=0.8)
    ax2.plot(ts[off:off+len(sm)], sm, color="seagreen", linewidth=2,
             label=f"DQN AR (smoothed w={C.SMOOTH_W})")
    ax2.axhline(ilp_ar, color="red", linestyle="--", linewidth=1.5,
                label=f"ILP Optimal  AR={ilp_ar:.4f}")
    ax2.set_ylabel("Episode AR", fontsize=11)
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    cap_rate  = np.array(cb.episode_cap_violated,  dtype=float)
    conf_rate = np.array(cb.episode_conf_violated, dtype=float)
    sm_cap, off_cap = moving_avg(cap_rate, C.SMOOTH_W)
    ax3.plot(ts, cap_rate, color="tomato", alpha=0.2, linewidth=0.8)
    ax3.plot(ts[off_cap:off_cap+len(sm_cap)], sm_cap, color="tomato", linewidth=2,
             label=f"cap violation rate (smoothed w={C.SMOOTH_W})")
    ax3.set_ylabel("Cap Violation Rate", fontsize=11)
    ax3.set_ylim(-0.02, 1.02)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    sm_conf, off_conf = moving_avg(conf_rate, C.SMOOTH_W)
    ax4.plot(ts, conf_rate, color="darkorange", alpha=0.2, linewidth=0.8)
    ax4.plot(ts[off_conf:off_conf+len(sm_conf)], sm_conf, color="darkorange", linewidth=2,
             label=f"privacy (conflict) violation rate (smoothed w={C.SMOOTH_W})")
    ax4.set_ylabel("Privacy Violation Rate", fontsize=11)
    ax4.set_xlabel("Training steps", fontsize=11)
    ax4.set_ylim(-0.02, 1.02)
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    path = outdir / "training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_comparison(ilp_ar, dqn_res, cap_viol_rate, conflict_viol_rate, outdir, scenario_name):
    # 2026-09-11: 2-panel layout (AR vs ILP bar chart / success+viol-rate bar
    # chart) standardized across all 6 algorithms -- see ppo_mask/run_all.py's
    # plot_comparison() comment for rationale.
    algo_label = "DQN\n(no mask)"
    success_rate = float(np.mean(dqn_res["success"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"ILP vs DQN - {scenario_name}", fontsize=13, fontweight="bold")

    labels1 = ["ILP\n(Optimal)", algo_label]
    ar_means = [ilp_ar, float(np.mean(dqn_res["ars"]))]
    ar_stds  = [0.0, float(np.std(dqn_res["ars"]))]
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax1.bar(labels1, ar_means, yerr=ar_stds, capsize=5, color=colors, alpha=0.8, ecolor="black")
    for bar, v in zip(bars, ar_means):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                  f"{v:.4f}", ha="center", fontsize=10, fontweight="bold", color="black")
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax1.set_title("Test AR: Algorithm vs ILP", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    labels2 = ["Success\nRate", "Cap Viol\nRate", "Privacy Viol\nRate"]
    vals2 = [success_rate, cap_viol_rate, conflict_viol_rate]
    colors2 = ["#2ecc71", "#e67e22", "#c0392b"]
    bars2 = ax2.bar(labels2, vals2, color=colors2, alpha=0.8)
    for bar, v in zip(bars2, vals2):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                  f"{v:.2%}", ha="center", fontsize=10, fontweight="bold", color="black")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Rate", fontsize=11)
    ax2.set_title("Test Success Rate & Violation Rates", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = outdir / "comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

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

    print(f"\n{'='*60}")
    print(f"  DQN run_all.py  \u2014  RL WITHOUT action masking")
    print(f"  Violation \u2192 reward=-1, episode terminates immediately.")
    print(f"  Config : {C.YAML_CONFIG.name}  |  train={len(C.TRAIN_SCENARIOS)}/test={len(C.TEST_SCENARIOS)}  |  prototype idx={C.SCENARIO_IDX}")
    print(f"{'='*60}\n")
    device = resolve_device(C.DEVICE)

    # 1. Load scenario
    ecus, services, sc_name, prototype_name = load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
    N, M = len(ecus), len(services)

    # 2. ILP (all scenarios)
    print(f"\n[1/4] Solving ILP for {len(C.TEST_SCENARIOS)} test scenarios ...")
    ilp_ar, ilp_per_sc = solve_ilp_all_scenarios(C.YAML_CONFIG, C.TEST_SCENARIOS, C.OUTDIR)
    print(f"  ILP mean AR across {len(C.TEST_SCENARIOS)} test scenarios: {ilp_ar:.4f}")

    # 4. DQN training
    print(f"\n[3/4] DQN training ({C.TOTAL_STEPS:,} steps) ...")
    model, cb = train_dqn(ecus, services, device)
    model_path = base_dir / f"model_{exp_id}_v{VERSION}"
    model.save(str(model_path))
    print(f"  Model saved -> {model_path}.zip")

    # 5. DQN evaluation
    print(f"\n[4/4] DQN evaluation ({len(C.TEST_SCENARIOS)} episodes, deterministic) ...")
    # 2026-09-11: reverted to a single deterministic pass (EVAL_BEST_OF_N=1) --
    # see lt/ppo/run_all.py's comment for why best-of-N was dropped. No
    # longer overriding model.exploration_rate -- with n_samples=1 there's
    # nothing to diversify across, so the eval should just be the model's
    # actual greedy policy.
    def dqn_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    dqn_res = run_episodes(ecus, services, dqn_policy, n_samples=C.EVAL_BEST_OF_N)
    print(f"  DQN AR  mean={np.mean(dqn_res['ars']):.4f}  "
          f"std={np.std(dqn_res['ars']):.4f}")
    print(f"  Placed/ep  mean={np.mean(dqn_res['placed']):.1f}/{M}")
    print(f"  Eval viol rate  {np.mean(dqn_res['viols']):.2%}")

    dqn_train_v = float(np.mean(dqn_res["viols"]))
    dqn_train_v_std = float(np.std(dqn_res["viols"]))
    success_rate = float(np.mean(dqn_res["success"]))
    cap_viol_rate = float(np.mean(dqn_res["cap_viols"] > 0))
    conflict_viol_rate = float(np.mean(dqn_res["conflict_viols"] > 0))

    # Summary
    print(f"\n{'='*68}")
    print(f"  {'Method':<24} {'AR (mean+/-std)':<22} {'Placed':<10} {'Viol%'}")
    print(f"  {'-'*24} {'-'*22} {'-'*10} {'-'*6}")
    print(f"  {'ILP (Optimal)':<24} {ilp_ar:.4f} +/- 0.0000     {M}/{M:<6} 0%")
    print(f"  {'DQN (no mask)':<24} "
          f"{np.mean(dqn_res['ars']):.4f} +/- {np.std(dqn_res['ars']):.4f}   "
            f"  {np.mean(dqn_res['placed']):.1f}/{M:<2}   {dqn_train_v:.0%}")
    print(f"{'='*68}\n")

    # Save JSON
    log = {
        "created_at": datetime.datetime.now().isoformat(),
        "scenario": sc_name,
        "prototype_scenario": prototype_name,
        "scenario_count": len(C.SCENARIOS),
        "train_count": len(C.TRAIN_SCENARIOS),
        "test_count": len(C.TEST_SCENARIOS),
        "N": N,
        "M": M,
        "ilp": {
            "ar": round(ilp_ar, 6),
            "ar_per_scenario": [round(r["avg_utilization"], 6) for r in ilp_per_sc],
            "violations": 0,
        },
        "dqn": {
            "ar_mean":            round(float(np.mean(dqn_res["ars"])), 6),
            "ar_std":             round(float(np.std(dqn_res["ars"])), 6),
            "placed_mean":        round(float(np.mean(dqn_res["placed"])), 2),
            "viol_rate":          round(float(dqn_train_v), 4),
            "success_rate":       round(success_rate, 6),
            "cap_viol_rate":      round(cap_viol_rate, 6),
            "conflict_viol_rate": round(conflict_viol_rate, 6),
            "cap_viol_total":     int(np.sum(dqn_res["cap_viols"])),
            "conflict_viol_total": int(np.sum(dqn_res["conflict_viols"])),
        },
        "training": {
            "total_steps":      C.TOTAL_STEPS,
            "n_episodes":       len(cb.episode_rewards),
            "ar_last50":        round(float(np.mean(cb.episode_ars[-50:])), 6),
            "reward_last50":    round(float(np.mean(cb.episode_rewards[-50:])), 6),
            "viol_rate_last50": round(float(np.mean(cb.episode_violated[-50:])), 4),
        },
    }
    log_path = base_dir / "results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  JSON saved -> {log_path}")

    # Save CSV summary
    csv_path = base_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ar_mean", "ar_std", "placed_mean", "valid_placed_mean", "ecus_used_mean", "success_rate", "cap_viol_rate", "conflict_viol_rate", "cap_viol_total", "conflict_viol_total"])
        writer.writerow(["ILP (Optimal)", round(ilp_ar, 6), 0.0, M, M, C.N, 1.0, 0.0, 0.0, 0, 0])
        writer.writerow([
            "DQN (no mask)",
            round(float(np.mean(dqn_res["ars"])), 6),
            round(float(np.std(dqn_res["ars"])), 6),
            round(float(np.mean(dqn_res["placed"])), 2),
            round(float(np.mean(dqn_res["valid_placed"])), 2),
            round(float(np.mean(dqn_res["ecus_used"])), 2),
            round(success_rate, 4),
            round(cap_viol_rate, 4),
            round(conflict_viol_rate, 4),
            int(np.sum(dqn_res["cap_viols"])),
            int(np.sum(dqn_res["conflict_viols"])),
        ])
    print(f"  CSV  saved -> {csv_path}")

    save_training_curve_csv(cb, base_dir)
    plot_training_curve(cb, ilp_ar, base_dir, sc_name)
    plot_comparison(ilp_ar, dqn_res, cap_viol_rate, conflict_viol_rate, base_dir, sc_name)

    print("\nAll done! Output files:")
    print(f"  {base_dir}/training_curve.png")
    print(f"  {base_dir}/comparison.png")
    print(f"  {base_dir}/results.json")
    print(f"  {base_dir}/summary.csv\n")


if __name__ == "__main__":
    main()
