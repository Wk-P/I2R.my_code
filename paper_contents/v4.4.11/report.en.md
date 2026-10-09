# v4.4.11 Same-instance multi-trajectory PPO (instance-wise baseline)

- Generated 2026-10-09 15:27:26
- Single change: the source of the advantage (`--group-k 4`, `src/paper_rl/group_ppo.py`). The 40 environments form 10 groups of K = 4; each group runs one complete trajectory per member with the current policy on the same training instance, and every step of trajectory k gets A = G_k − b_x, b_x = (1/4) Σ_k G_k (γ = 1, G_k is the trajectory return). The advantage no longer goes through the critic or GAE; the critic is still trained with target G_k (the PPO loss is unchanged) but does not enter the advantage. A rollout keeps complete groups only and updates once 20480 transitions are collected; unfinished groups are discarded (their steps still count towards the 1M budget).
- The ILP is not used in training at all. Evaluation is unchanged: one deterministic (argmax) rollout per test instance, no best-of-K.
- Checks: the 4 trajectories of a group share the instance; per trajectory adv = G_k − b_x and ret = G_k (error 3e−8, float32); advantages sum to 0 within each group; recomputed log-probs match those at sampling; all actions legal.
- Everything else unchanged: structure-aware network (shared encoder, original ECU head), ar_pen reward, lr 3e-4, entropy coefficient 0.005, clip 0.1, batch 256, 10 epochs, γ = 1, EXIT, descending demand, GPU, 1M steps.
- Baseline = the v4.4.5 three-seed GPU baseline, reused; manifest of the new models: `logs/v4.4.11/manifest.json`.
- Every model is evaluated on the test split of its own training seed; the relative gap is computed per seed and then averaged; tables show the mean ± sample std over three seeds.
- Steps 1–5 regret and optimal-action rates come from `src/scripts/diag_regret.py` (first 400 instances of each seed's test split). Noise reference: three-seed SD 0.4–0.7pp for the same configuration, and a rerun of the same seed can differ by 1.4–2.0pp; the EQ baseline of 10.2% is probably high (the other four variants all give 9.4–9.5%).

## Three-seed summary

| Scenario | Model | Relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | Step-1 optimal-action rate | EXIT rate | Active ECUs − ILP | Forced openings / instance |
|---|---|---|---|---|---|---|---|---|
| LT | baseline (critic + GAE advantage) | 11.5% ± 0.4% | 0.0533 ± 0.0029 | 32.8% ± 0.9% | 25.1% ± 2.0% | 3.5% ± 1.6% | +0.78 ± +0.06 | 5.60 ± 0.14 |
| LT | 4 trajectories per instance, instance-wise baseline | 12.3% ± 0.5% | 0.0530 ± 0.0014 | 33.8% ± 1.3% | 25.5% ± 0.7% | 4.2% ± 1.3% | +0.85 ± +0.08 | 5.83 ± 0.25 |
| EQ | baseline (critic + GAE advantage) | 10.2% ± 0.4% | 0.0683 ± 0.0024 | 48.3% ± 1.2% | 38.5% ± 1.8% | 0.1% ± 0.1% | +0.83 ± +0.05 | 4.85 ± 0.11 |
| EQ | 4 trajectories per instance, instance-wise baseline | 10.1% ± 0.8% | 0.0669 ± 0.0070 | 50.9% ± 0.1% | 39.6% ± 2.1% | 0.1% ± 0.1% | +0.92 ± +0.13 | 5.12 ± 0.20 |
| GT | baseline (critic + GAE advantage) | 8.7% ± 0.7% | 0.0590 ± 0.0052 | 50.3% ± 2.3% | 44.1% ± 1.2% | 0.0% ± 0.0% | +1.02 ± +0.04 | 5.41 ± 0.15 |
| GT | 4 trajectories per instance, instance-wise baseline | 10.6% ± 0.6% | 0.0736 ± 0.0050 | 47.2% ± 1.5% | 42.8% ± 0.9% | 0.0% ± 0.0% | +1.38 ± +0.04 | 5.88 ± 0.07 |

Average over the three scenarios (per seed first, then mean ± SD over seeds):

| Model | Relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | EXIT rate |
|---|---|---|---|---|
| baseline (critic + GAE advantage) | 10.1% ± 0.3% | 0.0602 ± 0.0025 | 43.8% ± 0.9% | 1.2% ± 0.5% |
| 4 trajectories per instance, instance-wise baseline | 11.0% ± 0.6% | 0.0645 ± 0.0044 | 44.0% ± 0.7% | 1.4% ± 0.5% |

## Per seed

| Scenario | Seed | Model | Relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | Step-1 optimal-action rate | EXIT rate | Active ECUs − ILP | Forced openings / instance |
|---|---|---|---|---|---|---|---|---|---|
| LT | 1 | baseline (critic + GAE advantage) | 11.5% | 0.0546 | 33.5% | 27.4% | 5.2% | +0.78 | 5.66 |
| LT | 1 | 4 trajectories per instance, instance-wise baseline | 12.2% | 0.0534 | 35.3% | 25.7% | 3.7% | +0.94 | 5.83 |
| LT | 2 | baseline (critic + GAE advantage) | 11.1% | 0.0499 | 33.2% | 23.7% | 3.0% | +0.72 | 5.44 |
| LT | 2 | 4 trajectories per instance, instance-wise baseline | 12.8% | 0.0541 | 32.9% | 24.7% | 5.8% | +0.80 | 5.58 |
| LT | 3 | baseline (critic + GAE advantage) | 11.9% | 0.0552 | 31.8% | 24.3% | 2.2% | +0.84 | 5.70 |
| LT | 3 | 4 trajectories per instance, instance-wise baseline | 11.9% | 0.0513 | 33.1% | 26.1% | 3.2% | +0.81 | 6.08 |
| EQ | 1 | baseline (critic + GAE advantage) | 10.4% | 0.0707 | 47.8% | 36.5% | 0.0% | +0.86 | 4.85 |
| EQ | 1 | 4 trajectories per instance, instance-wise baseline | 10.6% | 0.0689 | 50.8% | 37.8% | 0.0% | +0.98 | 5.13 |
| EQ | 2 | baseline (critic + GAE advantage) | 9.7% | 0.0659 | 47.5% | 39.3% | 0.2% | +0.77 | 4.75 |
| EQ | 2 | 4 trajectories per instance, instance-wise baseline | 10.6% | 0.0727 | 51.0% | 41.9% | 0.2% | +1.00 | 5.31 |
| EQ | 3 | baseline (critic + GAE advantage) | 10.4% | 0.0682 | 49.7% | 39.8% | 0.0% | +0.86 | 4.97 |
| EQ | 3 | 4 trajectories per instance, instance-wise baseline | 9.1% | 0.0592 | 50.8% | 39.2% | 0.0% | +0.76 | 4.91 |
| GT | 1 | baseline (critic + GAE advantage) | 9.3% | 0.0637 | 51.0% | 43.5% | 0.0% | +0.99 | 5.28 |
| GT | 1 | 4 trajectories per instance, instance-wise baseline | 11.2% | 0.0772 | 47.9% | 43.8% | 0.0% | +1.43 | 5.93 |
| GT | 2 | baseline (critic + GAE advantage) | 8.9% | 0.0599 | 47.7% | 43.2% | 0.0% | +1.01 | 5.39 |
| GT | 2 | 4 trajectories per instance, instance-wise baseline | 10.5% | 0.0758 | 45.5% | 42.0% | 0.0% | +1.36 | 5.79 |
| GT | 3 | baseline (critic + GAE advantage) | 8.0% | 0.0534 | 52.2% | 45.5% | 0.0% | +1.07 | 5.57 |
| GT | 3 | 4 trajectories per instance, instance-wise baseline | 10.0% | 0.0679 | 48.2% | 42.5% | 0.0% | +1.35 | 5.91 |
