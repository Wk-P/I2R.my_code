# v4.4.15 Learning curve with 50k training instances (1M / 3M / 5M)

- Generated 2026-10-10 18:03:47
- Single change (from v4.4.14): the step budget 1M → 5M; the same run saves checkpoints at 1M and 3M (`train.py --checkpoints`), so the curve follows one training trajectory. Training pool (50k ILP-verified instances), test split, network, ar_pen reward, mask, descending demand and MaskablePPO configuration unchanged.
- Test gap: each seed's own 400 test instances; training-set gap: a fixed 1600-instance subset of the 50k training pool; both deterministic rollouts, relative gap over non-EXIT instances.
- **Criteria fixed in advance**: at 5M roughly LT ≤ 8%, EQ ≤ 6%, GT ≤ 6% → "more data + longer training" is close to the target range; only something like 9.3% → 8.8% → 8.6% → with data no longer limiting, PPO itself is the main bottleneck again.
- References: the v4.4.5 baseline (1600 instances, 1M) and v4.4.14 (50k instances, 1M), reused; manifest of the new models: `logs/v4.4.15/manifest.json`.
- Steps 1–5 regret and optimal-action rates come from `src/scripts/diag_regret.py` (first 400 instances of each seed's test split, 5M models). Noise reference: three-seed SD 0.4–0.7pp for the same configuration, a rerun of the same seed can differ by 1.4–2.0pp.

## Learning curve (1M / 3M / 5M of the same run)

| Scenario | Steps | Test relative gap | Train relative gap | Test EXIT rate | Test gap per seed (1 / 2 / 3) | Train gap per seed (1 / 2 / 3) |
|---|---|---|---|---|---|---|
| LT | 1M | 10.4% ± 0.1% | 10.4% ± 0.5% | 3.3% ± 0.8% | 10.4% / 10.5% / 10.2% | 10.1% / 11.0% / 10.2% |
| LT | 3M | 9.1% ± 0.2% | 9.0% ± 0.3% | 3.3% ± 1.3% | 9.4% / 8.9% / 9.0% | 9.2% / 8.9% / 8.7% |
| LT | 5M | 8.7% ± 0.2% | 8.5% ± 0.2% | 2.9% ± 1.0% | 8.8% / 8.5% / 8.7% | 8.7% / 8.3% / 8.5% |
| EQ | 1M | 8.6% ± 0.5% | 8.6% ± 0.6% | 0.0% ± 0.0% | 9.1% / 8.3% / 8.3% | 9.2% / 8.0% / 8.4% |
| EQ | 3M | 7.6% ± 0.5% | 7.5% ± 0.3% | 0.0% ± 0.0% | 7.5% / 7.1% / 8.0% | 7.6% / 7.1% / 7.7% |
| EQ | 5M | 6.8% ± 0.1% | 6.7% ± 0.5% | 0.1% ± 0.1% | 6.8% / 6.8% / 7.0% | 6.9% / 6.1% / 7.1% |
| GT | 1M | 8.9% ± 0.9% | 8.8% ± 1.3% | 0.0% ± 0.0% | 8.5% / 8.3% / 9.9% | 8.0% / 8.1% / 10.4% |
| GT | 3M | 7.4% ± 0.8% | 7.2% ± 1.1% | 0.0% ± 0.0% | 7.0% / 6.9% / 8.4% | 6.5% / 6.6% / 8.5% |
| GT | 5M | 6.6% ± 0.3% | 6.4% ± 0.6% | 0.0% ± 0.0% | 6.6% / 6.4% / 6.9% | 6.1% / 6.0% / 7.2% |

Average over the three scenarios (per seed first, then mean ± SD over seeds):

| Steps | Test relative gap | Train relative gap |
|---|---|---|
| 1M | 9.3% ± 0.2% | 9.3% ± 0.3% |
| 3M | 8.0% ± 0.4% | 7.9% ± 0.4% |
| 5M | 7.4% ± 0.2% | 7.2% ± 0.4% |

## Three-seed summary

| Scenario | Model | Test relative gap | Train relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | Step-1 optimal-action rate | EXIT rate | Active ECUs − ILP | Forced openings / instance |
|---|---|---|---|---|---|---|---|---|---|
| LT | baseline (1600 instances, 1M) | 11.5% ± 0.4% | 11.1% ± 0.2% | 0.0533 ± 0.0029 | 32.8% ± 0.9% | 25.1% ± 2.0% | 3.5% ± 1.6% | +0.78 ± +0.06 | 5.60 ± 0.14 |
| LT | 50k instances, 1M (v4.4.14) | 10.4% ± 0.2% | 10.3% ± 0.3% | 0.0470 ± 0.0024 | 35.9% ± 3.4% | 26.6% ± 3.4% | 3.2% ± 0.6% | +0.66 ± +0.02 | 5.40 ± 0.38 |
| LT | 50k instances, 5M | 8.7% ± 0.2% | 8.5% ± 0.2% | 0.0376 ± 0.0010 | 42.2% ± 0.3% | 35.5% ± 0.2% | 2.9% ± 1.0% | +0.57 ± +0.03 | 5.30 ± 0.04 |
| EQ | baseline (1600 instances, 1M) | 10.2% ± 0.4% | 7.8% ± 0.2% | 0.0683 ± 0.0024 | 48.3% ± 1.2% | 38.5% ± 1.8% | 0.1% ± 0.1% | +0.83 ± +0.05 | 4.85 ± 0.11 |
| EQ | 50k instances, 1M (v4.4.14) | 8.4% ± 0.8% | 8.5% ± 0.5% | 0.0561 ± 0.0059 | 49.9% ± 2.5% | 40.9% ± 4.5% | 0.1% ± 0.1% | +0.63 ± +0.07 | 4.59 ± 0.21 |
| EQ | 50k instances, 5M | 6.8% ± 0.1% | 6.7% ± 0.5% | 0.0467 ± 0.0019 | 54.2% ± 1.7% | 45.0% ± 3.6% | 0.1% ± 0.1% | +0.45 ± +0.03 | 4.35 ± 0.06 |
| GT | baseline (1600 instances, 1M) | 8.7% ± 0.7% | 7.0% ± 0.1% | 0.0590 ± 0.0052 | 50.3% ± 2.3% | 44.1% ± 1.2% | 0.0% ± 0.0% | +1.02 ± +0.04 | 5.41 ± 0.15 |
| GT | 50k instances, 1M (v4.4.14) | 9.1% ± 1.2% | 8.9% ± 1.4% | 0.0646 ± 0.0089 | 48.2% ± 0.6% | 42.7% ± 1.3% | 0.0% ± 0.0% | +1.11 ± +0.22 | 5.51 ± 0.28 |
| GT | 50k instances, 5M | 6.6% ± 0.3% | 6.4% ± 0.6% | 0.0484 ± 0.0005 | 52.1% ± 2.8% | 45.9% ± 3.6% | 0.0% ± 0.0% | +0.67 ± +0.02 | 4.98 ± 0.07 |

Average over the three scenarios (per seed first, then mean ± SD over seeds):

| Model | Test relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | EXIT rate |
|---|---|---|---|---|
| baseline (1600 instances, 1M) | 10.1% ± 0.3% | 0.0602 ± 0.0025 | 43.8% ± 0.9% | 1.2% ± 0.5% |
| 50k instances, 1M (v4.4.14) | 9.3% ± 0.3% | 0.0559 ± 0.0020 | 44.7% ± 0.4% | 1.1% ± 0.2% |
| 50k instances, 5M | 7.4% ± 0.2% | 0.0442 ± 0.0009 | 49.5% ± 0.7% | 1.0% ± 0.4% |

## Per seed

| Scenario | Seed | Model | Test relative gap | Train relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | Step-1 optimal-action rate | EXIT rate | Active ECUs − ILP | Forced openings / instance |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | 1 | baseline (1600 instances, 1M) | 11.5% | 11.2% | 0.0546 | 33.5% | 27.4% | 5.2% | +0.78 | 5.66 |
| LT | 1 | 50k instances, 1M (v4.4.14) | 10.3% | 10.2% | 0.0443 | 39.6% | 30.5% | 3.2% | +0.65 | 5.79 |
| LT | 1 | 50k instances, 5M | 8.8% | 8.7% | 0.0387 | 41.9% | 35.7% | 2.7% | +0.58 | 5.32 |
| LT | 2 | baseline (1600 instances, 1M) | 11.1% | 10.9% | 0.0499 | 33.2% | 23.7% | 3.0% | +0.72 | 5.44 |
| LT | 2 | 50k instances, 1M (v4.4.14) | 10.3% | 10.6% | 0.0488 | 33.0% | 24.4% | 3.7% | +0.65 | 5.02 |
| LT | 2 | 50k instances, 5M | 8.5% | 8.3% | 0.0372 | 42.4% | 35.4% | 4.0% | +0.54 | 5.25 |
| LT | 3 | baseline (1600 instances, 1M) | 11.9% | 11.1% | 0.0552 | 31.8% | 24.3% | 2.2% | +0.84 | 5.70 |
| LT | 3 | 50k instances, 1M (v4.4.14) | 10.7% | 10.0% | 0.0480 | 35.2% | 24.9% | 2.5% | +0.68 | 5.39 |
| LT | 3 | 50k instances, 5M | 8.7% | 8.5% | 0.0368 | 42.5% | 35.5% | 2.0% | +0.59 | 5.33 |
| EQ | 1 | baseline (1600 instances, 1M) | 10.4% | 7.9% | 0.0707 | 47.8% | 36.5% | 0.0% | +0.86 | 4.85 |
| EQ | 1 | 50k instances, 1M (v4.4.14) | 9.3% | 9.0% | 0.0628 | 47.1% | 36.0% | 0.0% | +0.68 | 4.72 |
| EQ | 1 | 50k instances, 5M | 6.8% | 6.9% | 0.0477 | 52.9% | 44.5% | 0.0% | +0.48 | 4.40 |
| EQ | 2 | baseline (1600 instances, 1M) | 9.7% | 7.6% | 0.0659 | 47.5% | 39.3% | 0.2% | +0.77 | 4.75 |
| EQ | 2 | 50k instances, 1M (v4.4.14) | 8.3% | 8.1% | 0.0537 | 52.0% | 44.9% | 0.2% | +0.67 | 4.70 |
| EQ | 2 | 50k instances, 5M | 6.8% | 6.1% | 0.0445 | 56.1% | 48.9% | 0.2% | +0.44 | 4.38 |
| EQ | 3 | baseline (1600 instances, 1M) | 10.4% | 8.0% | 0.0682 | 49.7% | 39.8% | 0.0% | +0.86 | 4.97 |
| EQ | 3 | 50k instances, 1M (v4.4.14) | 7.7% | 8.3% | 0.0517 | 50.7% | 41.8% | 0.0% | +0.55 | 4.34 |
| EQ | 3 | 50k instances, 5M | 7.0% | 7.1% | 0.0479 | 53.5% | 41.8% | 0.0% | +0.42 | 4.28 |
| GT | 1 | baseline (1600 instances, 1M) | 9.3% | 7.1% | 0.0637 | 51.0% | 43.5% | 0.0% | +0.99 | 5.28 |
| GT | 1 | 50k instances, 1M (v4.4.14) | 8.4% | 7.9% | 0.0588 | 48.9% | 43.8% | 0.0% | +0.95 | 5.28 |
| GT | 1 | 50k instances, 5M | 6.6% | 6.1% | 0.0485 | 55.3% | 50.0% | 0.0% | +0.68 | 5.04 |
| GT | 2 | baseline (1600 instances, 1M) | 8.9% | 7.2% | 0.0599 | 47.7% | 43.2% | 0.0% | +1.01 | 5.39 |
| GT | 2 | 50k instances, 1M (v4.4.14) | 8.3% | 8.3% | 0.0602 | 48.2% | 43.0% | 0.0% | +1.02 | 5.45 |
| GT | 2 | 50k instances, 5M | 6.4% | 6.0% | 0.0479 | 50.6% | 43.5% | 0.0% | +0.65 | 4.90 |
| GT | 3 | baseline (1600 instances, 1M) | 8.0% | 6.9% | 0.0534 | 52.2% | 45.5% | 0.0% | +1.07 | 5.57 |
| GT | 3 | 50k instances, 1M (v4.4.14) | 10.4% | 10.5% | 0.0748 | 47.8% | 41.2% | 0.0% | +1.36 | 5.82 |
| GT | 3 | 50k instances, 5M | 6.9% | 7.2% | 0.0488 | 50.3% | 44.2% | 0.0% | +0.69 | 4.98 |
