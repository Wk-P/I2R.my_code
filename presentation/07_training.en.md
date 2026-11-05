# 7. Training

## Overall plan

- 12 variants × 3 scenarios × 3 seeds (1, 2, 3); each run trains for **20M steps**.
- Each episode draws a random instance from the 50,000 training instances.
- Checkpoints at 5M, 10M and 15M; the final model at 20M.
- Software: torch 2.5.1 (GPU), stable-baselines3 2.7.1, sb3-contrib 2.7.1, gymnasium 1.2.3, Python 3.12.

## Main hyperparameters

| | PPO | DQN / DDQN |
|---|---|---|
| Parallel envs | 40 | 12 |
| Learning rate | 3e-4 | 1e-3 |
| Batch | 20,480 steps, minibatches of 256 | 64 drawn from a 100k replay buffer |
| Passes / frequency | 6 for Mask, 10 for others | 1 update per 48 new transitions |
| Other | γ=1, GAE 0.95, clip 0.1, entropy 0.005 | γ=1, ε 1→0 over the first 2M steps, target sync about every 492 steps |

## How λ self-adjusts (Lagrangian only)

- λ is a parameter updated by Adam (lr 0.035, initial 0.001).
- Rule: if recent episodes' mean violation count J_C is above the target 0.001 → λ rises; below → λ falls; never below 0, no upper cap.
- Frequency: PPO once per batch (20,480 steps); DQN/DDQN once per 1,000 episodes.
- A new λ applies from the next episode; it is fixed within an episode.

## Callbacks: what training does "on the side"

Callbacks are hooks SB3 calls during training. We wrote two (`src/paper_rl/train.py`):

**1. Main callback (logging + λ updates)**
- After each episode, one row: step, total reward, AR, finished without violation, legal placements, over capacity, conflict, λ, EXIT → saved as `training_curve.csv` (the panel's curves come from here).
- Lagrangian: updates λ by the rule above and logs every update to `lambda_updates.csv`.
- Every 100k steps prints progress and writes `.progress.json` for the panel.

**2. Checkpoint callback**
- Saves the model at 5M / 10M / 15M; for Lagrangian also records the current λ (evaluating a checkpoint needs the same λ).

## Evaluation after training

- The final model runs **deterministically** (no sampling) on the 5,000 test instances of each scenario.
- AR is compared with ILP AR to get the gap; the violation rate is counted too.
