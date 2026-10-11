# v4.4.16 50k training instances: continued to 10M (learning curve 1M / 3M / 5M / 7.5M / 10M)

- Generated 2026-10-11 05:57:24
- Single change: the step budget. The nine 5M models of v4.4.15 are loaded with their optimiser state and step count (`train.py --resume`) and trained 5M more steps on the same 50k-instance training pool, with a checkpoint at 7.5M; network, ar_pen reward, mask, descending demand and MaskablePPO configuration unchanged. The 1M / 3M / 5M points come from the v4.4.15 checkpoints, 7.5M / 10M from this continuation.
- Test gap: each seed's own 400 test instances; training-set gap: a fixed 1600-instance subset of the 50k training pool; both deterministic rollouts.
- **Criteria fixed in advance**: 10M roughly at LT ≈ 8%, EQ ≈ 6%, GT ≈ 5.5–6% → the route is right and more training pays. Decision tree: the gap keeps falling clearly with train ≈ test → keep training; train falls but test does not → widen the training instances (e.g. 50k → 200k); both stop falling → PPO / the model is the bottleneck, change the algorithm; only LT stops while EQ/GT keep falling → study LT separately.
- References: the v4.4.5 baseline (1600 instances, 1M) and v4.4.15 (50k instances, 5M), reused; manifest of the new models: `logs/v4.4.16/manifest.json`.
- Steps 1–5 regret and optimal-action rates come from `src/scripts/diag_regret.py` (first 400 instances of each seed's test split, 10M models).

## Learning curve (1M / 3M / 5M: v4.4.15; 7.5M / 10M: continued)

| Scenario | Steps | Test relative gap | Train relative gap | Test EXIT rate | Test gap per seed (1 / 2 / 3) | Train gap per seed (1 / 2 / 3) |
|---|---|---|---|---|---|---|
| LT | 1M | 10.4% ± 0.1% | 10.4% ± 0.5% | 3.3% ± 0.8% | 10.4% / 10.5% / 10.2% | 10.1% / 11.0% / 10.2% |
| LT | 3M | 9.1% ± 0.2% | 9.0% ± 0.3% | 3.3% ± 1.3% | 9.4% / 8.9% / 9.0% | 9.2% / 8.9% / 8.7% |
| LT | 5M | 8.7% ± 0.2% | 8.5% ± 0.2% | 2.9% ± 1.0% | 8.8% / 8.5% / 8.7% | 8.7% / 8.3% / 8.5% |
| LT | 7.5M | 8.2% ± 0.2% | 8.1% ± 0.1% | 2.4% ± 0.1% | 8.5% / 8.3% / 8.0% | 8.2% / 8.1% / 8.0% |
| LT | 10M | 7.6% ± 0.3% | 7.7% ± 0.3% | 1.6% ± 0.6% | 7.9% / 7.6% / 7.3% | 8.0% / 7.6% / 7.5% |
| EQ | 1M | 8.6% ± 0.5% | 8.6% ± 0.6% | 0.0% ± 0.0% | 9.1% / 8.3% / 8.3% | 9.2% / 8.0% / 8.4% |
| EQ | 3M | 7.6% ± 0.5% | 7.5% ± 0.3% | 0.0% ± 0.0% | 7.5% / 7.1% / 8.0% | 7.6% / 7.1% / 7.7% |
| EQ | 5M | 6.8% ± 0.1% | 6.7% ± 0.5% | 0.1% ± 0.1% | 6.8% / 6.8% / 7.0% | 6.9% / 6.1% / 7.1% |
| EQ | 7.5M | 6.0% ± 0.7% | 6.1% ± 0.6% | 0.1% ± 0.1% | 6.4% / 5.2% / 6.5% | 6.2% / 5.5% / 6.7% |
| EQ | 10M | 5.9% ± 0.5% | 5.9% ± 0.5% | 0.1% ± 0.1% | 6.3% / 5.3% / 6.1% | 5.8% / 5.5% / 6.5% |
| GT | 1M | 8.9% ± 0.9% | 8.8% ± 1.3% | 0.0% ± 0.0% | 8.5% / 8.3% / 9.9% | 8.0% / 8.1% / 10.4% |
| GT | 3M | 7.4% ± 0.8% | 7.2% ± 1.1% | 0.0% ± 0.0% | 7.0% / 6.9% / 8.4% | 6.5% / 6.6% / 8.5% |
| GT | 5M | 6.6% ± 0.3% | 6.4% ± 0.6% | 0.0% ± 0.0% | 6.6% / 6.4% / 6.9% | 6.1% / 6.0% / 7.2% |
| GT | 7.5M | 5.8% ± 0.2% | 5.7% ± 0.5% | 0.0% ± 0.0% | 5.7% / 5.7% / 6.0% | 5.3% / 5.5% / 6.3% |
| GT | 10M | 5.4% ± 0.3% | 5.4% ± 0.4% | 0.0% ± 0.0% | 5.2% / 5.2% / 5.7% | 5.2% / 5.2% / 5.9% |

Average over the three scenarios (per seed first, then mean ± SD over seeds):

| Steps | Test relative gap | Train relative gap |
|---|---|---|
| 1M | 9.3% ± 0.2% | 9.3% ± 0.3% |
| 3M | 8.0% ± 0.4% | 7.9% ± 0.4% |
| 5M | 7.4% ± 0.2% | 7.2% ± 0.4% |
| 7.5M | 6.7% ± 0.2% | 6.6% ± 0.3% |
| 10M | 6.3% ± 0.2% | 6.3% ± 0.3% |

## Three-seed summary

| Scenario | Model | Test relative gap | Train relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | Step-1 optimal-action rate | EXIT rate | Active ECUs − ILP | Forced openings / instance |
|---|---|---|---|---|---|---|---|---|---|
| LT | baseline (1600 instances, 1M) | 11.5% ± 0.4% | 11.1% ± 0.2% | 0.0533 ± 0.0029 | 32.8% ± 0.9% | 25.1% ± 2.0% | 3.5% ± 1.6% | +0.78 ± +0.06 | 5.60 ± 0.14 |
| LT | 50k instances, 5M (v4.4.15) | 8.7% ± 0.2% | 8.5% ± 0.2% | 0.0376 ± 0.0010 | 42.2% ± 0.3% | 35.5% ± 0.2% | 2.9% ± 1.0% | +0.57 ± +0.03 | 5.30 ± 0.04 |
| LT | 50k instances, 10M | 7.6% ± 0.3% | 7.7% ± 0.3% | 0.0349 ± 0.0012 | 44.2% ± 1.6% | 37.9% ± 2.9% | 1.6% ± 0.6% | +0.49 ± +0.04 | 5.29 ± 0.22 |
| EQ | baseline (1600 instances, 1M) | 10.2% ± 0.4% | 7.8% ± 0.2% | 0.0683 ± 0.0024 | 48.3% ± 1.2% | 38.5% ± 1.8% | 0.1% ± 0.1% | +0.83 ± +0.05 | 4.85 ± 0.11 |
| EQ | 50k instances, 5M (v4.4.15) | 6.8% ± 0.1% | 6.7% ± 0.5% | 0.0467 ± 0.0019 | 54.2% ± 1.7% | 45.0% ± 3.6% | 0.1% ± 0.1% | +0.45 ± +0.03 | 4.35 ± 0.06 |
| EQ | 50k instances, 10M | 5.9% ± 0.5% | 5.9% ± 0.5% | 0.0417 ± 0.0038 | 57.3% ± 3.2% | 47.7% ± 5.8% | 0.1% ± 0.1% | +0.40 ± +0.07 | 4.39 ± 0.02 |
| GT | baseline (1600 instances, 1M) | 8.7% ± 0.7% | 7.0% ± 0.1% | 0.0590 ± 0.0052 | 50.3% ± 2.3% | 44.1% ± 1.2% | 0.0% ± 0.0% | +1.02 ± +0.04 | 5.41 ± 0.15 |
| GT | 50k instances, 5M (v4.4.15) | 6.6% ± 0.3% | 6.4% ± 0.6% | 0.0484 ± 0.0005 | 52.1% ± 2.8% | 45.9% ± 3.6% | 0.0% ± 0.0% | +0.67 ± +0.02 | 4.98 ± 0.07 |
| GT | 50k instances, 10M | 5.4% ± 0.3% | 5.4% ± 0.4% | 0.0397 ± 0.0015 | 56.9% ± 3.0% | 48.9% ± 4.1% | 0.0% ± 0.0% | +0.52 ± +0.04 | 4.79 ± 0.05 |

Average over the three scenarios (per seed first, then mean ± SD over seeds):

| Model | Test relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | EXIT rate |
|---|---|---|---|---|
| baseline (1600 instances, 1M) | 10.1% ± 0.3% | 0.0602 ± 0.0025 | 43.8% ± 0.9% | 1.2% ± 0.5% |
| 50k instances, 5M (v4.4.15) | 7.4% ± 0.2% | 0.0442 ± 0.0009 | 49.5% ± 0.7% | 1.0% ± 0.4% |
| 50k instances, 10M | 6.3% ± 0.2% | 0.0388 ± 0.0018 | 52.8% ± 1.4% | 0.6% ± 0.2% |

## Per seed

| Scenario | Seed | Model | Test relative gap | Train relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | Step-1 optimal-action rate | EXIT rate | Active ECUs − ILP | Forced openings / instance |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | 1 | baseline (1600 instances, 1M) | 11.5% | 11.2% | 0.0546 | 33.5% | 27.4% | 5.2% | +0.78 | 5.66 |
| LT | 1 | 50k instances, 5M (v4.4.15) | 8.8% | 8.7% | 0.0387 | 41.9% | 35.7% | 2.7% | +0.58 | 5.32 |
| LT | 1 | 50k instances, 10M | 7.9% | 8.0% | 0.0363 | 43.7% | 37.9% | 2.2% | +0.54 | 5.37 |
| LT | 2 | baseline (1600 instances, 1M) | 11.1% | 10.9% | 0.0499 | 33.2% | 23.7% | 3.0% | +0.72 | 5.44 |
| LT | 2 | 50k instances, 5M (v4.4.15) | 8.5% | 8.3% | 0.0372 | 42.4% | 35.4% | 4.0% | +0.54 | 5.25 |
| LT | 2 | 50k instances, 10M | 7.6% | 7.6% | 0.0340 | 43.0% | 35.1% | 1.0% | +0.45 | 5.04 |
| LT | 3 | baseline (1600 instances, 1M) | 11.9% | 11.1% | 0.0552 | 31.8% | 24.3% | 2.2% | +0.84 | 5.70 |
| LT | 3 | 50k instances, 5M (v4.4.15) | 8.7% | 8.5% | 0.0368 | 42.5% | 35.5% | 2.0% | +0.59 | 5.33 |
| LT | 3 | 50k instances, 10M | 7.3% | 7.5% | 0.0344 | 46.0% | 40.9% | 1.5% | +0.49 | 5.46 |
| EQ | 1 | baseline (1600 instances, 1M) | 10.4% | 7.9% | 0.0707 | 47.8% | 36.5% | 0.0% | +0.86 | 4.85 |
| EQ | 1 | 50k instances, 5M (v4.4.15) | 6.8% | 6.9% | 0.0477 | 52.9% | 44.5% | 0.0% | +0.48 | 4.40 |
| EQ | 1 | 50k instances, 10M | 6.3% | 5.8% | 0.0442 | 54.7% | 44.5% | 0.0% | +0.45 | 4.41 |
| EQ | 2 | baseline (1600 instances, 1M) | 9.7% | 7.6% | 0.0659 | 47.5% | 39.3% | 0.2% | +0.77 | 4.75 |
| EQ | 2 | 50k instances, 5M (v4.4.15) | 6.8% | 6.1% | 0.0445 | 56.1% | 48.9% | 0.2% | +0.44 | 4.38 |
| EQ | 2 | 50k instances, 10M | 5.3% | 5.5% | 0.0373 | 60.8% | 54.4% | 0.2% | +0.32 | 4.38 |
| EQ | 3 | baseline (1600 instances, 1M) | 10.4% | 8.0% | 0.0682 | 49.7% | 39.8% | 0.0% | +0.86 | 4.97 |
| EQ | 3 | 50k instances, 5M (v4.4.15) | 7.0% | 7.1% | 0.0479 | 53.5% | 41.8% | 0.0% | +0.42 | 4.28 |
| EQ | 3 | 50k instances, 10M | 6.1% | 6.5% | 0.0435 | 56.4% | 44.2% | 0.0% | +0.42 | 4.39 |
| GT | 1 | baseline (1600 instances, 1M) | 9.3% | 7.1% | 0.0637 | 51.0% | 43.5% | 0.0% | +0.99 | 5.28 |
| GT | 1 | 50k instances, 5M (v4.4.15) | 6.6% | 6.1% | 0.0485 | 55.3% | 50.0% | 0.0% | +0.68 | 5.04 |
| GT | 1 | 50k instances, 10M | 5.2% | 5.2% | 0.0390 | 57.6% | 48.5% | 0.0% | +0.49 | 4.76 |
| GT | 2 | baseline (1600 instances, 1M) | 8.9% | 7.2% | 0.0599 | 47.7% | 43.2% | 0.0% | +1.01 | 5.39 |
| GT | 2 | 50k instances, 5M (v4.4.15) | 6.4% | 6.0% | 0.0479 | 50.6% | 43.5% | 0.0% | +0.65 | 4.90 |
| GT | 2 | 50k instances, 10M | 5.2% | 5.2% | 0.0387 | 59.6% | 53.2% | 0.0% | +0.49 | 4.76 |
| GT | 3 | baseline (1600 instances, 1M) | 8.0% | 6.9% | 0.0534 | 52.2% | 45.5% | 0.0% | +1.07 | 5.57 |
| GT | 3 | 50k instances, 5M (v4.4.15) | 6.9% | 7.2% | 0.0488 | 50.3% | 44.2% | 0.0% | +0.69 | 4.98 |
| GT | 3 | 50k instances, 10M | 5.7% | 5.9% | 0.0414 | 53.7% | 45.0% | 0.0% | +0.57 | 4.86 |
