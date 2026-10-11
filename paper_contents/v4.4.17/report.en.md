# v4.4.17 50k training instances: continued to 20M (learning curve 1M → 20M)

- Generated 2026-10-11 19:38:45
- Single change: the step budget. The nine 10M models of v4.4.16 are loaded with their optimiser state and step count (`train.py --resume`) and trained 10M more steps on the same 50k-instance training pool, with a checkpoint at 15M; network, ar_pen reward, mask, descending demand and MaskablePPO configuration unchanged. Learning curve: 1M / 3M / 5M from v4.4.15, 7.5M / 10M from v4.4.16, 15M / 20M from this continuation.
- Test gap: each seed's own 400 test instances; training-set gap: a fixed 1600-instance subset of the 50k training pool; both deterministic rollouts.
- **Criteria fixed in advance**: whether GT is stably ≤ 5% around 15M, whether EQ's 6.0 → 5.9 is a temporary plateau or saturation, whether LT keeps its rate, whether train and test stay together, whether early regret keeps falling. 20M at about LT ≈ 6%, EQ ≈ 5%, GT < 5% → the target is very close and only LT needs targeted work; still falling clearly at 20M → continue to 25M / 30M; hardly any change from 15M to 20M → reconsider the algorithm.
- References: the v4.4.5 baseline (1600 instances, 1M) and v4.4.16 (50k instances, 10M), reused; manifest of the new models: `logs/v4.4.17/manifest.json`.
- Steps 1–5 regret and optimal-action rates come from `src/scripts/diag_regret.py` (first 400 instances of each seed's test split, 20M models).

## Learning curve (1M / 3M / 5M: v4.4.15; 7.5M / 10M: v4.4.16; 15M / 20M: this continuation)

| Scenario | Steps | Test relative gap | Train relative gap | Test EXIT rate | Test gap per seed (1 / 2 / 3) | Train gap per seed (1 / 2 / 3) |
|---|---|---|---|---|---|---|
| LT | 1M | 10.4% ± 0.1% | 10.4% ± 0.5% | 3.3% ± 0.8% | 10.4% / 10.5% / 10.2% | 10.1% / 11.0% / 10.2% |
| LT | 3M | 9.1% ± 0.2% | 9.0% ± 0.3% | 3.3% ± 1.3% | 9.4% / 8.9% / 9.0% | 9.2% / 8.9% / 8.7% |
| LT | 5M | 8.7% ± 0.2% | 8.5% ± 0.2% | 2.9% ± 1.0% | 8.8% / 8.5% / 8.7% | 8.7% / 8.3% / 8.5% |
| LT | 7.5M | 8.2% ± 0.2% | 8.1% ± 0.1% | 2.4% ± 0.1% | 8.5% / 8.3% / 8.0% | 8.2% / 8.1% / 8.0% |
| LT | 10M | 7.6% ± 0.3% | 7.7% ± 0.3% | 1.6% ± 0.6% | 7.9% / 7.6% / 7.3% | 8.0% / 7.6% / 7.5% |
| LT | 15M | 7.4% ± 0.2% | 7.2% ± 0.2% | 1.7% ± 0.0% | 7.5% / 7.5% / 7.2% | 7.4% / 7.1% / 7.0% |
| LT | 20M | 7.0% ± 0.3% | 6.7% ± 0.1% | 1.8% ± 0.3% | 6.8% / 7.3% / 6.8% | 6.6% / 6.8% / 6.8% |
| EQ | 1M | 8.6% ± 0.5% | 8.6% ± 0.6% | 0.0% ± 0.0% | 9.1% / 8.3% / 8.3% | 9.2% / 8.0% / 8.4% |
| EQ | 3M | 7.6% ± 0.5% | 7.5% ± 0.3% | 0.0% ± 0.0% | 7.5% / 7.1% / 8.0% | 7.6% / 7.1% / 7.7% |
| EQ | 5M | 6.8% ± 0.1% | 6.7% ± 0.5% | 0.1% ± 0.1% | 6.8% / 6.8% / 7.0% | 6.9% / 6.1% / 7.1% |
| EQ | 7.5M | 6.0% ± 0.7% | 6.1% ± 0.6% | 0.1% ± 0.1% | 6.4% / 5.2% / 6.5% | 6.2% / 5.5% / 6.7% |
| EQ | 10M | 5.9% ± 0.5% | 5.9% ± 0.5% | 0.1% ± 0.1% | 6.3% / 5.3% / 6.1% | 5.8% / 5.5% / 6.5% |
| EQ | 15M | 5.3% ± 0.5% | 5.2% ± 0.3% | 0.1% ± 0.1% | 5.5% / 4.7% / 5.6% | 5.4% / 4.9% / 5.4% |
| EQ | 20M | 4.9% ± 0.3% | 5.0% ± 0.3% | 0.1% ± 0.1% | 5.1% / 4.6% / 5.0% | 5.0% / 4.6% / 5.2% |
| GT | 1M | 8.9% ± 0.9% | 8.8% ± 1.3% | 0.0% ± 0.0% | 8.5% / 8.3% / 9.9% | 8.0% / 8.1% / 10.4% |
| GT | 3M | 7.4% ± 0.8% | 7.2% ± 1.1% | 0.0% ± 0.0% | 7.0% / 6.9% / 8.4% | 6.5% / 6.6% / 8.5% |
| GT | 5M | 6.6% ± 0.3% | 6.4% ± 0.6% | 0.0% ± 0.0% | 6.6% / 6.4% / 6.9% | 6.1% / 6.0% / 7.2% |
| GT | 7.5M | 5.8% ± 0.2% | 5.7% ± 0.5% | 0.0% ± 0.0% | 5.7% / 5.7% / 6.0% | 5.3% / 5.5% / 6.3% |
| GT | 10M | 5.4% ± 0.3% | 5.4% ± 0.4% | 0.0% ± 0.0% | 5.2% / 5.2% / 5.7% | 5.2% / 5.2% / 5.9% |
| GT | 15M | 5.0% ± 0.3% | 4.7% ± 0.5% | 0.0% ± 0.0% | 4.8% / 4.8% / 5.4% | 4.4% / 4.5% / 5.2% |
| GT | 20M | 4.6% ± 0.2% | 4.2% ± 0.4% | 0.0% ± 0.0% | 4.5% / 4.5% / 4.9% | 4.0% / 3.9% / 4.7% |

Average over the three scenarios (per seed first, then mean ± SD over seeds):

| Steps | Test relative gap | Train relative gap |
|---|---|---|
| 1M | 9.3% ± 0.2% | 9.3% ± 0.3% |
| 3M | 8.0% ± 0.4% | 7.9% ± 0.4% |
| 5M | 7.4% ± 0.2% | 7.2% ± 0.4% |
| 7.5M | 6.7% ± 0.2% | 6.6% ± 0.3% |
| 10M | 6.3% ± 0.2% | 6.3% ± 0.3% |
| 15M | 5.9% ± 0.2% | 5.7% ± 0.2% |
| 20M | 5.5% ± 0.0% | 5.3% ± 0.2% |

## Three-seed summary

| Scenario | Model | Test relative gap | Train relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | Step-1 optimal-action rate | EXIT rate | Active ECUs − ILP | Forced openings / instance |
|---|---|---|---|---|---|---|---|---|---|
| LT | baseline (1600 instances, 1M) | 11.5% ± 0.4% | 11.1% ± 0.2% | 0.0533 ± 0.0029 | 32.8% ± 0.9% | 25.1% ± 2.0% | 3.5% ± 1.6% | +0.78 ± +0.06 | 5.60 ± 0.14 |
| LT | 50k instances, 10M (v4.4.16) | 7.6% ± 0.3% | 7.7% ± 0.3% | 0.0349 ± 0.0012 | 44.2% ± 1.6% | 37.9% ± 2.9% | 1.6% ± 0.6% | +0.49 ± +0.04 | 5.29 ± 0.22 |
| LT | 50k instances, 20M | 7.0% ± 0.3% | 6.7% ± 0.1% | 0.0338 ± 0.0008 | 45.9% ± 0.3% | 41.2% ± 2.1% | 1.8% ± 0.3% | +0.45 ± +0.07 | 5.28 ± 0.15 |
| EQ | baseline (1600 instances, 1M) | 10.2% ± 0.4% | 7.8% ± 0.2% | 0.0683 ± 0.0024 | 48.3% ± 1.2% | 38.5% ± 1.8% | 0.1% ± 0.1% | +0.83 ± +0.05 | 4.85 ± 0.11 |
| EQ | 50k instances, 10M (v4.4.16) | 5.9% ± 0.5% | 5.9% ± 0.5% | 0.0417 ± 0.0038 | 57.3% ± 3.2% | 47.7% ± 5.8% | 0.1% ± 0.1% | +0.40 ± +0.07 | 4.39 ± 0.02 |
| EQ | 50k instances, 20M | 4.9% ± 0.3% | 5.0% ± 0.3% | 0.0365 ± 0.0022 | 59.5% ± 2.9% | 47.9% ± 7.1% | 0.1% ± 0.1% | +0.26 ± +0.03 | 4.27 ± 0.06 |
| GT | baseline (1600 instances, 1M) | 8.7% ± 0.7% | 7.0% ± 0.1% | 0.0590 ± 0.0052 | 50.3% ± 2.3% | 44.1% ± 1.2% | 0.0% ± 0.0% | +1.02 ± +0.04 | 5.41 ± 0.15 |
| GT | 50k instances, 10M (v4.4.16) | 5.4% ± 0.3% | 5.4% ± 0.4% | 0.0397 ± 0.0015 | 56.9% ± 3.0% | 48.9% ± 4.1% | 0.0% ± 0.0% | +0.52 ± +0.04 | 4.79 ± 0.05 |
| GT | 50k instances, 20M | 4.6% ± 0.2% | 4.2% ± 0.4% | 0.0362 ± 0.0012 | 61.6% ± 0.6% | 54.2% ± 1.0% | 0.0% ± 0.0% | +0.38 ± +0.05 | 4.74 ± 0.08 |

Average over the three scenarios (per seed first, then mean ± SD over seeds):

| Model | Test relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | EXIT rate |
|---|---|---|---|---|
| baseline (1600 instances, 1M) | 10.1% ± 0.3% | 0.0602 ± 0.0025 | 43.8% ± 0.9% | 1.2% ± 0.5% |
| 50k instances, 10M (v4.4.16) | 6.3% ± 0.2% | 0.0388 ± 0.0018 | 52.8% ± 1.4% | 0.6% ± 0.2% |
| 50k instances, 20M | 5.5% ± 0.0% | 0.0355 ± 0.0010 | 55.7% ± 0.9% | 0.6% ± 0.1% |

## Per seed

| Scenario | Seed | Model | Test relative gap | Train relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | Step-1 optimal-action rate | EXIT rate | Active ECUs − ILP | Forced openings / instance |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | 1 | baseline (1600 instances, 1M) | 11.5% | 11.2% | 0.0546 | 33.5% | 27.4% | 5.2% | +0.78 | 5.66 |
| LT | 1 | 50k instances, 10M (v4.4.16) | 7.9% | 8.0% | 0.0363 | 43.7% | 37.9% | 2.2% | +0.54 | 5.37 |
| LT | 1 | 50k instances, 20M | 6.8% | 6.6% | 0.0328 | 46.1% | 43.6% | 2.0% | +0.38 | 5.17 |
| LT | 2 | baseline (1600 instances, 1M) | 11.1% | 10.9% | 0.0499 | 33.2% | 23.7% | 3.0% | +0.72 | 5.44 |
| LT | 2 | 50k instances, 10M (v4.4.16) | 7.6% | 7.6% | 0.0340 | 43.0% | 35.1% | 1.0% | +0.45 | 5.04 |
| LT | 2 | 50k instances, 20M | 7.3% | 6.8% | 0.0342 | 46.1% | 40.3% | 2.0% | +0.46 | 5.23 |
| LT | 3 | baseline (1600 instances, 1M) | 11.9% | 11.1% | 0.0552 | 31.8% | 24.3% | 2.2% | +0.84 | 5.70 |
| LT | 3 | 50k instances, 10M (v4.4.16) | 7.3% | 7.5% | 0.0344 | 46.0% | 40.9% | 1.5% | +0.49 | 5.46 |
| LT | 3 | 50k instances, 20M | 6.8% | 6.8% | 0.0343 | 45.6% | 39.6% | 1.5% | +0.51 | 5.45 |
| EQ | 1 | baseline (1600 instances, 1M) | 10.4% | 7.9% | 0.0707 | 47.8% | 36.5% | 0.0% | +0.86 | 4.85 |
| EQ | 1 | 50k instances, 10M (v4.4.16) | 6.3% | 5.8% | 0.0442 | 54.7% | 44.5% | 0.0% | +0.45 | 4.41 |
| EQ | 1 | 50k instances, 20M | 5.1% | 5.0% | 0.0376 | 57.5% | 41.8% | 0.0% | +0.29 | 4.25 |
| EQ | 2 | baseline (1600 instances, 1M) | 9.7% | 7.6% | 0.0659 | 47.5% | 39.3% | 0.2% | +0.77 | 4.75 |
| EQ | 2 | 50k instances, 10M (v4.4.16) | 5.3% | 5.5% | 0.0373 | 60.8% | 54.4% | 0.2% | +0.32 | 4.38 |
| EQ | 2 | 50k instances, 20M | 4.6% | 4.6% | 0.0339 | 62.9% | 55.6% | 0.2% | +0.25 | 4.33 |
| EQ | 3 | baseline (1600 instances, 1M) | 10.4% | 8.0% | 0.0682 | 49.7% | 39.8% | 0.0% | +0.86 | 4.97 |
| EQ | 3 | 50k instances, 10M (v4.4.16) | 6.1% | 6.5% | 0.0435 | 56.4% | 44.2% | 0.0% | +0.42 | 4.39 |
| EQ | 3 | 50k instances, 20M | 5.0% | 5.2% | 0.0380 | 58.1% | 46.2% | 0.0% | +0.24 | 4.22 |
| GT | 1 | baseline (1600 instances, 1M) | 9.3% | 7.1% | 0.0637 | 51.0% | 43.5% | 0.0% | +0.99 | 5.28 |
| GT | 1 | 50k instances, 10M (v4.4.16) | 5.2% | 5.2% | 0.0390 | 57.6% | 48.5% | 0.0% | +0.49 | 4.76 |
| GT | 1 | 50k instances, 20M | 4.5% | 4.0% | 0.0353 | 61.6% | 54.8% | 0.0% | +0.35 | 4.68 |
| GT | 2 | baseline (1600 instances, 1M) | 8.9% | 7.2% | 0.0599 | 47.7% | 43.2% | 0.0% | +1.01 | 5.39 |
| GT | 2 | 50k instances, 10M (v4.4.16) | 5.2% | 5.2% | 0.0387 | 59.6% | 53.2% | 0.0% | +0.49 | 4.76 |
| GT | 2 | 50k instances, 20M | 4.5% | 3.9% | 0.0357 | 61.1% | 53.0% | 0.0% | +0.34 | 4.70 |
| GT | 3 | baseline (1600 instances, 1M) | 8.0% | 6.9% | 0.0534 | 52.2% | 45.5% | 0.0% | +1.07 | 5.57 |
| GT | 3 | 50k instances, 10M (v4.4.16) | 5.7% | 5.9% | 0.0414 | 53.7% | 45.0% | 0.0% | +0.57 | 4.86 |
| GT | 3 | 50k instances, 20M | 4.9% | 4.7% | 0.0375 | 62.2% | 54.8% | 0.0% | +0.44 | 4.83 |
