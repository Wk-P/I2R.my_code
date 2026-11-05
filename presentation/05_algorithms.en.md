# 5. Three RL algorithms: PPO, DQN, DDQN

All three use the **same network** (file 6). They differ in what the output scores mean and how they are learned.

## PPO: learns the probability of picking each ECU

- The network scores each ECU → softmax → probabilities. Training samples from them (exploration); testing takes the most likely.
- It also outputs **V(s)**: how many points the current situation is worth, used to judge whether a step went better or worse than expected.
- Learning: play a batch with the current policy (40 envs × 512 steps = 20,480 steps), then learn from it several times (6 for Mask, 10 for others), with small changes only (clip 0.1) for stability.
- **On-policy**: the batch is discarded after use.
- Library: stable-baselines3 `PPO`; Mask uses sb3-contrib `MaskablePPO`.

## DQN: learns how many points each ECU is worth (Q-value)

- The network gives each ECU a **Q-value** = "if the service goes here, the expected final score". Pick the largest.
- Exploration: ε-greedy, 100% random at first, linearly down to 0 over the first 2M steps.
- Learning: experience goes into a **replay buffer** (100k); random batches of 64 are drawn; the target is Q(s,a) ≈ reward + max next-step Q.
- A "target network" (an old copy, synced about every 500 steps) computes the target so the network does not chase itself.
- **Off-policy**: old experience is reused.
- Library: stable-baselines3 `DQN`; Mask uses our own `MaskableDQN`.

## DDQN = Double DQN

- DQN tends to **overestimate** Q (picking the max of noisy values is biased upward).
- Double DQN's fix: the **online network picks** the next action, the **target network scores** it. Separating the two reduces overestimation.
- Otherwise identical to DQN. Code: our `DoubleDQN` / `MaskableDDQN` (`src/shared/dqn_variants.py`), which only change `train()`.

## Comparison

| | PPO | DQN | DDQN |
|---|---|---|---|
| Output means | Probabilities (+ value V) | Q-values | Q-values |
| Exploration | Sampling | ε-greedy (to 0 within 2M steps) | Same as DQN |
| Data | Used once | Replay buffer | Same as DQN |
| Parallel envs | 40 | 12 | 12 |
| Learning rate | 3e-4 | 1e-3 | 1e-3 |
| At test time | Most likely | Largest Q | Largest Q |

## How Mask is implemented per algorithm

- PPO: the environment supplies the legal-action mask each step; it is stored and used during learning.
- DQN/DDQN: the network computes legal ECUs from the observation (same rules as the environment); random exploration and the max are both restricted to legal actions.
