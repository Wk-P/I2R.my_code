# v4.3.6 — Unified reward: no M, positive on success, negative on failure (professor's comment 1-①)

New minor version (tag `v4.3.6`).

## 1. Rationale

Professor's comment (item 1 of `paper_contents/v4.3.4.0pv/readme.md`): remove M and unify the reward function. In legacy the reward term ranges over [−M, M] and the penalty over [−M, 0], which do not match; a penalty should be negative and a reward positive.

| Version | Success | Failure (violation / dead end) | γ |
|---|---|---|---|
| legacy (v4.3.1.6) | M(2AR − 1) ∈ [−M, M] | −M(1 − valid/M) ∈ [−M, 0) | 0.99 |
| ar_raw (v4.3.5) | AR ∈ (0, 1] | 0 | 1 |
| **ar_pen (this version)** | **AR ∈ (0, 1]** | **−(1 − valid/M) ∈ [−1, 0)** | **1** |

Intermediate rewards are 0. Lagrangian still subtracts $\lambda\sum_t c_t$ at the end.

## 2. Changes (only variable: the reward)

- `paper_rl/env.py`: new `reward_mode="ar_pen"`.
- `scripts/run_v4.3.6.py`: full experiment (12 models × 3 scenarios × 3 seeds × 5M); `--pilot` is a pilot (the four PPO variants + Mask-DQN + Repair-DQN × 3 scenarios × seed 1 × 1M, 18 jobs).
- Everything else as in v4.3.1.6: base observation, descending-demand order, a dead end terminates the episode (professor's comment 1-② "no early stop, always run M steps" is left to the next version), other hyper-parameters, data and splits, evaluation.

## 3. Self-check

3 scenarios × 4 mechanisms × 150 random episodes (1800, 756 successful): all intermediate rewards are 0; terminal rewards match the table above (Lagrangian additionally −λΣc_t) with 0 mismatches; successful episodes get (0, 1], failed ones [−1, 0).

## 4. Results

- Pilot: `pilot_report.md` (generated when finished), compared with the first 1M steps of the v4.3.1.6 seed-1 training curves.
