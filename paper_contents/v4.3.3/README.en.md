# v4.3.3 — Inference-time best-of-K sampling

New minor version (tag `v4.3.3`). No retraining, evaluation only.

## 1. Motivation

Under every reward variant of v4.3.1.x, the RL AR gap (ILP AR − AR) is 0.15~0.17, equal to or slightly larger than the greedy baseline (`paper_rl/greedy.py`). This version is a diagnosis: **does the PPO policy's distribution contain solutions better than its deterministic output, and better than greedy?**
- Clearly beats greedy: the model "knows" better solutions; more training / another network makes sense, and the paper can report a time-for-quality curve.
- Barely improves even with 64 samples: the policy itself is stuck at greedy level; changing the network alone is likely not enough.

## 2. Method

- Models: the 4 PPO models of v4.3.1.6 × 3 scenarios × seeds 1–3 (5M); control: randomized greedy.
- 64 solutions per test instance: the first is the deterministic output / plain greedy, the rest are sampled from the policy (Maskable-PPO with masks) / pick a random feasible ECU with probability 0.2 at each step. K = 1, 4, 16, 64 take the first K.
- **Fairness**: selection uses only what the solution itself gives (success first, then highest AR); the optimal AR is used only for scoring; greedy gets the same K; time is reported as K times, on the same scale as the ILP.
- No DQN models: Q-values have no natural sampling distribution.
- Script: `scripts/eval_v4.3.3.py`; outputs `raw.jsonl`, `report.md`.

## 3. Results

See `report.md` (generated when finished).
