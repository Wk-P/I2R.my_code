"""self_imitation_finetune.py — Expert Iteration / Self-Imitation Learning
prototype for lt/ppo_mask.

v2.0.0 showed that best-of-N eval-time resampling closes most of the gap to
100% success_rate (36/40 solved within N=1024, 2 more recovered by N=8192).
That means the *already-trained* stochastic policy already contains the
successful behavior somewhere in its sampling distribution for almost every
scenario — it just doesn't pick it on the first (N=1, deterministic) try.

This script converts that eval-time compute (many rollouts, keep the best)
into training-time compute: it samples the current policy N times per TRAIN
scenario, keeps whichever rollouts fully succeeded (all M placed, zero
violations), and re-uses src/shared/bc_pretrain.py's supervised BC loss
(`pretrain_actor_critic`) to fine-tune the policy toward its own successful
trajectories — the same mechanism the codebase already uses for ILP-expert
BC warm-start, just fed self-generated data instead of ILP's.

Two scenarios have no successful rollout at any N tried so far (see
version/v2.0.0.md) — those contribute nothing here since there's no
successful trajectory to imitate; this only helps the "needs many tries but
CAN succeed" majority, not the genuinely-stuck minority.

Usage:
    TRAIN_SEED=42 python src/scripts/self_imitation_finetune.py \
        results/paper-verfication/lt/ppo_mask/0e1a36bb_bc/model_0e1a36bb_bc_v1.2.4 \
        --rounds 3 --n-samples 64 --bc-epochs 10
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scenarios" / "lt" / "ppo_mask"))

import numpy as np
from sb3_contrib import MaskablePPO

import config as C
from shared.paths import VERSION, resolve_exp_id
from shared.bc_pretrain import pretrain_actor_critic
from ppo_mask.env import P4Env
import run_all as RA


def collect_self_imitation_data(model, scenarios, n_samples):
    """Sample the current stochastic policy n_samples times per scenario,
    keep the full (obs, mask, action) trajectory of the first fully
    successful rollout (or the best-valid_placed one if none succeed —
    excluded from training data, only used for the success-rate stat)."""
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
            valid_placed = int(info.get("valid_placed", 0))
            if valid_placed == M_sc:
                obs_list.extend(ep_obs)
                mask_list.extend(ep_mask)
                act_list.extend(ep_act)
                n_success += 1
                break  # one successful trajectory per scenario is enough

    print(f"  [self-imitation] {n_success}/{len(scenarios)} train scenarios contributed a "
          f"successful trajectory ({len(act_list)} transitions)")
    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(mask_list, dtype=bool),
        np.asarray(act_list, dtype=np.int64),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--bc-epochs", type=int, default=10)
    ap.add_argument("--bc-lr", type=float, default=3e-4)
    args = ap.parse_args()

    print(f"TRAIN_SEED={C.SEED}  train_scenarios={len(C.TRAIN_SCENARIOS)}  "
          f"test_scenarios={len(C.TEST_SCENARIOS)}  model={args.model_path}")

    model = MaskablePPO.load(args.model_path)

    def ppo_policy(obs, mask):
        action, _ = model.predict(obs, deterministic=False, action_masks=mask)
        return int(action)

    def ppo_policy_det(obs, mask):
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        return int(action)

    ecus_dummy, services_dummy = None, None  # RA.run_episodes only reads C.TEST_SCENARIOS

    print("\n[baseline] eval before any self-imitation ...")
    for label, pol, n in [("N=1 (deterministic)", ppo_policy_det, 1), ("N=8", ppo_policy, 8)]:
        res = RA.run_episodes(ecus_dummy, services_dummy, pol, n_samples=n)
        print(f"  {label:<22} success_rate={np.mean(res['success']):.4f}")

    exp_id = resolve_exp_id(C.OUTDIR)
    out_dir = C.OUTDIR / f"{exp_id}_selfimit"
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in range(1, args.rounds + 1):
        print(f"\n=== round {r}/{args.rounds}: sampling self-imitation data (N={args.n_samples}) ===")
        obs_arr, mask_arr, act_arr = collect_self_imitation_data(model, C.TRAIN_SCENARIOS, args.n_samples)
        if len(act_arr) == 0:
            print("  no successful trajectories collected -- stopping.")
            break
        print(f"  fine-tuning ({args.bc_epochs} epochs, lr={args.bc_lr}) ...")
        pretrain_actor_critic(model, obs_arr, act_arr, mask_arr=mask_arr,
                               epochs=args.bc_epochs, lr=args.bc_lr)

        model_path = out_dir / f"model_{exp_id}_selfimit_round{r}_v{VERSION}"
        model.save(str(model_path))
        print(f"  saved -> {model_path}.zip")

        print(f"  eval after round {r} ...")
        for label, pol, n in [("N=1 (deterministic)", ppo_policy_det, 1), ("N=8", ppo_policy, 8)]:
            res = RA.run_episodes(ecus_dummy, services_dummy, pol, n_samples=n)
            ars = res["ars"][res["success"]]
            ar_mean = float(np.mean(ars)) if len(ars) else float("nan")
            print(f"  {label:<22} success_rate={np.mean(res['success']):.4f}  ar_mean={ar_mean:.4f}")


if __name__ == "__main__":
    main()
