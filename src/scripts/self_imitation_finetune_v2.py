"""self_imitation_finetune_v2.py — Expert Iteration v2: BC nudge + short RL
recovery, to fix the collapse seen in v1 (src/scripts/self_imitation_finetune.py).

v1 result (see version/v2.1.0.md): pure supervised BC on self-sampled
successful trajectories pushed N=1 (deterministic) success_rate DOWN
(0.425 -> 0.30 over 3 rounds) even though N=8 improved (0.525 -> 0.625).
Diagnosis: pure -log_prob BC pulls the action distribution tightly onto a
narrow set of ~155 successful trajectories (BC train acc climbed to 97%+),
collapsing the policy's exploration/generalization — that hurts single-shot
performance on scenarios whose successful path looks different from the
sampled ones, even though it doesn't hurt best-of-N (which just needs any
path in the (now narrower but still present) distribution to hit).

Standard Expert Iteration doesn't stop at the BC step: it always follows the
imitation nudge with continued RL optimization against the real reward, so
the reward signal can correct any distribution collapse the BC step caused.
This version adds that missing step:

    sample (N=64) -> filter successes -> light BC nudge (few epochs, low lr)
    -> short model.learn() against real reward (RL_STEPS) -> evaluate -> repeat

Usage:
    TRAIN_SEED=42 python src/scripts/self_imitation_finetune_v2.py \
        results/paper-verfication/lt/ppo_mask/0e1a36bb_bc/model_0e1a36bb_bc_v1.2.4 \
        --rounds 6 --n-samples 1024 --bc-epochs 3 --bc-lr 1e-4 --rl-steps 200000

n-samples defaults to 1024, matching the best-of-N ceiling established in
version/v2.0.0.md (N=1024 -> 95% success_rate on the test set) — sampling at
N=64 during data collection meant ~5/160 train scenarios never produced a
successful trajectory at all and got silently skipped, and evaluation only
checked N=1/N=8 instead of comparing against the N=1024 ceiling we already
know is achievable. Both are fixed here.
"""
import argparse
import datetime
import functools
import json
import sys
import time
from pathlib import Path

# Line-buffer stdout even when redirected to a file/log (nohup) — the
# default block-buffering meant earlier runs of this script wrote nothing
# to their log for minutes despite the process actively computing (verified
# via /proc/<pid>/fdinfo showing pos:0 while CPU sat at >2000%). Must happen
# before any print() call.
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scenarios" / "lt" / "ppo_mask"))

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import config as C
from shared.paths import VERSION, resolve_exp_id, results_dir, write_progress
from shared.bc_pretrain import pretrain_actor_critic
from ppo_mask.env import P4Env
import run_all as RA

# Own progress file, separate from lt/ppo_mask/.progress.json (the real
# training run's) — recognized by app/backend/main.py's _match_scenario_algo
# as scenario="lt", algo="ppo_mask_selfimit", so it shows up on the
# dashboard's live-progress panel as its own row instead of overwriting or
# colliding with the canonical lt/ppo_mask training display.
PROGRESS_DIR = results_dir("lt", "ppo_mask_selfimit")


class _SelfImitRLCallback(BaseCallback):
    """Local stand-in for RA.P4Callback: records episode ARs AND writes live
    progress, but to PROGRESS_DIR instead of C.OUTDIR (= lt/ppo_mask/) — see
    module docstring for why that separation matters."""
    def __init__(self, round_idx: int, total_rounds: int, rl_steps: int, exp_id: str):
        super().__init__()
        self.episode_ars: list[float] = []
        self.round_idx = round_idx
        self.total_rounds = total_rounds
        self.rl_steps = rl_steps
        self.exp_id = exp_id
        self._next_tick = getattr(C, "PROGRESS_LOG_EVERY_STEPS", 50_000)
        self._tick_every = self._next_tick
        self._t_start = 0.0
        self._base_timesteps = None

    def _on_training_start(self) -> None:
        self._t_start = time.time()
        self._base_timesteps = self.model.num_timesteps

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_ars.append(float(info.get("ar", 0.0)))
        step_in_round = self.model.num_timesteps - self._base_timesteps
        if step_in_round >= self._next_tick or step_in_round >= self.rl_steps:
            elapsed = max(time.time() - self._t_start, 1e-6)
            round_pct = min(100.0, step_in_round * 100.0 / self.rl_steps)
            overall_pct = min(100.0, ((self.round_idx - 1) + round_pct / 100.0) / self.total_rounds * 100.0)
            write_progress(
                PROGRESS_DIR,
                phase="rl_recovery", round=self.round_idx, total_rounds=self.total_rounds,
                step=step_in_round, total_steps=self.rl_steps, pct=round(overall_pct, 1),
                episodes=len(self.episode_ars), steps_per_sec=round(step_in_round / elapsed),
                exp_id=self.exp_id,
            )
            self._next_tick += self._tick_every
        return True


def collect_self_imitation_data(model, scenarios, n_samples):
    obs_list, mask_list, act_list = [], [], []
    n_success = 0
    for sc in scenarios:
        caps, reqs, cs = sc[0], sc[1], sc[2] if len(sc) > 2 else []
        M_sc = len(reqs)
        ecus = [RA.ECU(f"ECU{i}", c) for i, c in enumerate(caps)]
        svcs = [RA.SVC(f"SVC{i}", r) for i, r in enumerate(reqs)]
        for _attempt in range(n_samples):
            env = P4Env(ecus, svcs, scenarios=[sc])
            obs, _ = env.reset()
            ep_obs, ep_mask, ep_act = [], [], []
            done = False
            info = {}
            while not done:
                mask = env.action_masks()
                if not np.any(mask):
                    break
                action, _ = model.predict(obs, deterministic=False, action_masks=mask)
                action = int(action)
                ep_obs.append(obs)
                ep_mask.append(mask)
                ep_act.append(action)
                obs, _, done, _, info = env.step(action)
            if int(info.get("valid_placed", 0)) == M_sc:
                obs_list.extend(ep_obs)
                mask_list.extend(ep_mask)
                act_list.extend(ep_act)
                n_success += 1
                break
    print(f"  [self-imitation] {n_success}/{len(scenarios)} train scenarios contributed a "
          f"successful trajectory ({len(act_list)} transitions)")
    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(mask_list, dtype=bool),
        np.asarray(act_list, dtype=np.int64),
    )


def evaluate(model, label_n_pairs, progress_ctx=None):
    """progress_ctx, if given, is (phase, round, total_rounds, exp_id) — used
    to write a tick to PROGRESS_DIR before each label (N=1/N=8/N=1024) so the
    dashboard shows *something* moving during a slow eval instead of one
    static blob for the whole phase. The N=1024 pass is by far the slowest
    (avg ~90 attempts/scenario, see version/v2.0.0.md) and still won't have
    sub-progress within itself — this only ticks between labels, not within
    one — but that's a big improvement over nothing."""
    def ppo_policy(obs, mask):
        action, _ = model.predict(obs, deterministic=False, action_masks=mask)
        return int(action)
    def ppo_policy_det(obs, mask):
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        return int(action)
    out = {}
    for i, (label, det, n) in enumerate(label_n_pairs):
        if progress_ctx:
            phase, round_idx, total_rounds, exp_id = progress_ctx
            write_progress(PROGRESS_DIR, phase=phase, round=round_idx, total_rounds=total_rounds,
                            eval_label=label, eval_step=i + 1, eval_total=len(label_n_pairs),
                            pct=round(round_idx / max(total_rounds, 1) * 100, 1), exp_id=exp_id)
        pol = ppo_policy_det if det else ppo_policy
        res = RA.run_episodes(None, None, pol, n_samples=n)
        ars = res["ars"][res["success"]]
        ar_mean = float(np.mean(ars)) if len(ars) else float("nan")
        ar_std = float(np.std(ars)) if len(ars) else float("nan")
        success_rate = float(np.mean(res["success"]))
        print(f"  {label:<22} success_rate={success_rate:.4f}  ar_mean={ar_mean:.4f}")
        out[label] = {"success_rate": success_rate, "ar_mean": ar_mean, "ar_std": ar_std}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=1024)
    ap.add_argument("--bc-epochs", type=int, default=3)
    ap.add_argument("--bc-lr", type=float, default=1e-4)
    ap.add_argument("--rl-steps", type=int, default=200_000)
    args = ap.parse_args()

    print(f"TRAIN_SEED={C.SEED}  train_scenarios={len(C.TRAIN_SCENARIOS)}  "
          f"test_scenarios={len(C.TEST_SCENARIOS)}  model={args.model_path}")

    model = MaskablePPO.load(args.model_path)
    # MaskablePPO.load() doesn't restore an env; attach one so model.learn()
    # can run further on-policy optimization (the RL-recovery step below).
    n_envs = max(1, int(C.N_ENVS))
    env = DummyVecEnv([functools.partial(RA._make_p4_env, C.SEED + i) for i in range(n_envs)])
    model.set_env(env)

    exp_id = resolve_exp_id(C.OUTDIR)
    write_progress(PROGRESS_DIR, phase="baseline_eval", round=0, total_rounds=args.rounds,
                    pct=0.0, exp_id=exp_id)

    print("\n[baseline] eval before any self-imitation ...")
    baseline_metrics = evaluate(model, [("N=1 (deterministic)", True, 1), ("N=8", False, 8),
                                         ("N=1024", False, 1024)],
                                 progress_ctx=("baseline_eval", 0, args.rounds, exp_id))

    out_dir = C.OUTDIR / f"{exp_id}_selfimit_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    round_log = []

    for r in range(1, args.rounds + 1):
        write_progress(PROGRESS_DIR, phase="sampling", round=r, total_rounds=args.rounds,
                        pct=round((r - 1) / args.rounds * 100, 1), exp_id=exp_id)
        print(f"\n=== round {r}/{args.rounds}: sampling (N={args.n_samples}) ===")
        obs_arr, mask_arr, act_arr = collect_self_imitation_data(model, C.TRAIN_SCENARIOS, args.n_samples)
        if len(act_arr) == 0:
            print("  no successful trajectories collected -- stopping.")
            break

        write_progress(PROGRESS_DIR, phase="bc_nudge", round=r, total_rounds=args.rounds,
                        pct=round((r - 1) / args.rounds * 100, 1), exp_id=exp_id)
        print(f"  light BC nudge ({args.bc_epochs} epochs, lr={args.bc_lr}) ...")
        pretrain_actor_critic(model, obs_arr, act_arr, mask_arr=mask_arr,
                               epochs=args.bc_epochs, lr=args.bc_lr)

        print(f"  RL recovery: model.learn({args.rl_steps:,} steps) against real reward ...")
        cb = _SelfImitRLCallback(round_idx=r, total_rounds=args.rounds,
                                  rl_steps=args.rl_steps, exp_id=exp_id)
        model.learn(total_timesteps=args.rl_steps, callback=cb, reset_num_timesteps=False)
        if cb.episode_ars:
            last50 = np.mean(cb.episode_ars[-50:])
            print(f"  RL recovery done | {len(cb.episode_ars)} eps | AR(last50)={last50:.4f}")

        model_path = out_dir / f"model_{exp_id}_selfimit_v2_round{r}_v{VERSION}"
        model.save(str(model_path))
        print(f"  saved -> {model_path}.zip")

        print(f"  eval after round {r} ...")
        round_metrics = evaluate(model, [("N=1 (deterministic)", True, 1), ("N=8", False, 8),
                                          ("N=1024", False, 1024)],
                                  progress_ctx=("eval", r, args.rounds, exp_id))
        round_log.append({
            "round": r,
            "train_scenarios_with_success": None,  # filled below from collect() n_success if needed
            "n_transitions": len(act_arr),
            "rl_steps": args.rl_steps,
            "metrics": round_metrics,
        })

    env.close()

    # ── write results.json under a distinct pseudo-algo dir so it shows up
    # on the dashboard as its own row without overwriting lt/ppo_mask's
    # canonical training-run summary (see version/v2.1.0.md "记录" ask).
    ilp_cache_path = results_dir(C.OUTDIR.parent.name, "ilp") / "cf.json"
    ilp_ar = None
    try:
        ilp_cache = json.loads(ilp_cache_path.read_text())
        ars = [r_["avg_utilization"] for r_ in ilp_cache["results"] if r_.get("status") == "Optimal"]
        ilp_ar = float(np.mean(ars)) if ars else None
    except (OSError, json.JSONDecodeError, KeyError):
        pass

    selfimit_dir = results_dir(C.OUTDIR.parent.name, "ppo_mask_selfimit") / exp_id
    selfimit_dir.mkdir(parents=True, exist_ok=True)
    final = round_log[-1]["metrics"] if round_log else baseline_metrics
    results_payload = {
        "created_at": datetime.datetime.now().isoformat(),
        "exp_id": exp_id,
        "scenario": "All 200 Scenarios",
        "train_count": len(C.TRAIN_SCENARIOS),
        "test_count": len(C.TEST_SCENARIOS),
        "N": C.N if hasattr(C, "N") else None,
        "M": C.M if hasattr(C, "M") else None,
        "ilp": {"ar": ilp_ar},
        "maskable_ppo_selfimit": {
            "base_model": str(args.model_path),
            "rounds": args.rounds,
            "n_samples": args.n_samples,
            "bc_epochs": args.bc_epochs,
            "bc_lr": args.bc_lr,
            "rl_steps_per_round": args.rl_steps,
            "baseline": baseline_metrics,
            "round_log": round_log,
            "ar_mean": final.get("N=1024", {}).get("ar_mean"),
            "ar_std": final.get("N=1024", {}).get("ar_std"),
            "success_rate": final.get("N=1024", {}).get("success_rate"),
            "success_rate_n8": final.get("N=8", {}).get("success_rate"),
            "success_rate_n1": final.get("N=1 (deterministic)", {}).get("success_rate"),
        },
    }
    (selfimit_dir / "results.json").write_text(json.dumps(results_payload, indent=2))
    print(f"\n  results.json written -> {selfimit_dir / 'results.json'}")


if __name__ == "__main__":
    main()
