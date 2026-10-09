# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-09 21:23:55; script `src/scripts/diag_regret.py`; took 9.8 min.
- Model: v4.4.12 structure-aware Mask PPO, global plan + sequential correction (seed 2, 1M, GPU), replayed deterministically on the first 400 instances of the seed-2 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

384 completed instances, 16 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1259 |
| PPO regret per step: mean / median | 0.0084 / 0.0000 |
| Share of steps where PPO picks an optimal action | 52.8% |
| Legal actions / optimal actions per step (mean) | 5.10 / 1.25 |
| Share of steps with a unique optimal action | 79.3% |
| Steps with regret per instance (mean) | 7.08 |
| Share of the gap from steps 1–5 | 38.7% |
| Share of the gap from steps 6–10 | 31.3% |
| Share of the gap from steps 11–15 | 30.1% |
| Regret at steps followed by a forced opening: share of regret | 66.9% |
| Regret at steps that are themselves forced openings: share of regret | 28.2% |
| Steps from a regret step to the next forced opening (mean) | 2.43 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 5.69 / 6 (16 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0099 | 7.9% | 76.3% | 7.22 | 1.54 | 23.7% |
| 2 | 0.0098 | 7.8% | 66.1% | 6.76 | 1.45 | 33.9% |
| 3 | 0.0101 | 8.1% | 70.3% | 6.43 | 1.34 | 29.7% |
| 4 | 0.0094 | 7.5% | 65.4% | 6.13 | 1.36 | 34.6% |
| 5 | 0.0093 | 7.4% | 64.6% | 5.92 | 1.33 | 35.4% |
| 6 | 0.0097 | 7.7% | 58.3% | 5.65 | 1.26 | 41.7% |
| 7 | 0.0074 | 5.9% | 54.4% | 5.41 | 1.25 | 45.6% |
| 8 | 0.0076 | 6.0% | 51.0% | 5.22 | 1.26 | 49.0% |
| 9 | 0.0072 | 5.7% | 47.9% | 4.93 | 1.17 | 52.1% |
| 10 | 0.0075 | 5.9% | 38.0% | 4.61 | 1.19 | 62.0% |
| 11 | 0.0056 | 4.5% | 34.9% | 4.22 | 1.17 | 65.1% |
| 12 | 0.0036 | 2.9% | 26.3% | 3.94 | 1.16 | 73.7% |
| 13 | 0.0023 | 1.8% | 16.7% | 3.54 | 1.12 | 83.3% |
| 14 | 0.0028 | 2.2% | 11.5% | 3.40 | 1.12 | 88.5% |
| 15 | 0.0236 | 18.7% | 25.8% | 3.20 | 1.05 | 74.2% |

## EQ (M = 10)

399 completed instances, 1 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0822 |
| PPO regret per step: mean / median | 0.0082 / 0.0000 |
| Share of steps where PPO picks an optimal action | 66.3% |
| Legal actions / optimal actions per step (mean) | 6.02 / 1.23 |
| Share of steps with a unique optimal action | 80.0% |
| Steps with regret per instance (mean) | 3.37 |
| Share of the gap from steps 1–3 | 53.9% |
| Share of the gap from steps 4–6 | 35.7% |
| Share of the gap from steps 7–10 | 10.3% |
| Regret at steps followed by a forced opening: share of regret | 87.8% |
| Regret at steps that are themselves forced openings: share of regret | 57.4% |
| Steps from a regret step to the next forced opening (mean) | 1.41 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 1.00 / 1 (1 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0157 | 19.0% | 60.4% | 7.34 | 1.39 | 39.6% |
| 2 | 0.0154 | 18.7% | 57.4% | 6.95 | 1.36 | 42.6% |
| 3 | 0.0133 | 16.1% | 49.4% | 6.62 | 1.33 | 50.6% |
| 4 | 0.0124 | 15.1% | 44.4% | 6.40 | 1.33 | 55.6% |
| 5 | 0.0087 | 10.6% | 40.1% | 6.12 | 1.27 | 59.9% |
| 6 | 0.0083 | 10.1% | 34.1% | 5.82 | 1.20 | 65.9% |
| 7 | 0.0054 | 6.5% | 24.6% | 5.59 | 1.20 | 75.4% |
| 8 | 0.0025 | 3.0% | 19.3% | 5.33 | 1.13 | 80.7% |
| 9 | 0.0006 | 0.8% | 7.0% | 5.07 | 1.09 | 93.0% |
| 10 | 0.0000 | 0.0% | 0.5% | 4.91 | 1.04 | 99.5% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0715 |
| PPO regret per step: mean / median | 0.0071 / 0.0000 |
| Share of steps where PPO picks an optimal action | 66.8% |
| Legal actions / optimal actions per step (mean) | 10.31 / 1.33 |
| Share of steps with a unique optimal action | 73.5% |
| Steps with regret per instance (mean) | 3.32 |
| Share of the gap from steps 1–3 | 49.2% |
| Share of the gap from steps 4–6 | 40.0% |
| Share of the gap from steps 7–10 | 10.8% |
| Regret at steps followed by a forced opening: share of regret | 87.8% |
| Regret at steps that are themselves forced openings: share of regret | 66.0% |
| Steps from a regret step to the next forced opening (mean) | 1.24 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0122 | 17.1% | 53.5% | 10.93 | 1.54 | 46.5% |
| 2 | 0.0118 | 16.5% | 56.8% | 10.74 | 1.49 | 43.2% |
| 3 | 0.0111 | 15.6% | 48.0% | 10.65 | 1.45 | 52.0% |
| 4 | 0.0115 | 16.1% | 48.2% | 10.58 | 1.44 | 51.7% |
| 5 | 0.0104 | 14.6% | 42.5% | 10.52 | 1.40 | 57.5% |
| 6 | 0.0066 | 9.3% | 32.2% | 10.31 | 1.33 | 67.8% |
| 7 | 0.0045 | 6.3% | 27.0% | 10.10 | 1.29 | 73.0% |
| 8 | 0.0024 | 3.3% | 14.5% | 9.89 | 1.20 | 85.5% |
| 9 | 0.0008 | 1.2% | 8.2% | 9.79 | 1.13 | 91.8% |
| 10 | 0.0000 | 0.0% | 0.8% | 9.55 | 1.05 | 99.2% |
