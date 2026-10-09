# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-09 15:16:29; script `src/scripts/diag_regret.py`; took 9.4 min.
- Model: model 'mask_ppo_group' of logs/v4.4.11/manifest.json, seed 2, replayed deterministically on the first 400 instances of the seed-2 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

377 completed instances, 23 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 2.3e-04 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1132 |
| PPO regret per step: mean / median | 0.0075 / 0.0000 |
| Share of steps where PPO picks an optimal action | 55.1% |
| Legal actions / optimal actions per step (mean) | 5.04 / 1.27 |
| Share of steps with a unique optimal action | 77.6% |
| Steps with regret per instance (mean) | 6.73 |
| Share of the gap from steps 1–5 | 47.8% |
| Share of the gap from steps 6–10 | 39.2% |
| Share of the gap from steps 11–15 | 13.0% |
| Regret at steps followed by a forced opening: share of regret | 84.5% |
| Regret at steps that are themselves forced openings: share of regret | 39.2% |
| Steps from a regret step to the next forced opening (mean) | 2.15 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 5.43 / 7 (23 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0113 | 10.0% | 75.3% | 7.25 | 1.54 | 24.7% |
| 2 | 0.0122 | 10.8% | 68.2% | 6.77 | 1.48 | 31.8% |
| 3 | 0.0107 | 9.5% | 67.1% | 6.38 | 1.36 | 32.9% |
| 4 | 0.0090 | 7.9% | 58.6% | 6.00 | 1.39 | 41.4% |
| 5 | 0.0109 | 9.6% | 66.0% | 5.70 | 1.33 | 34.0% |
| 6 | 0.0100 | 8.9% | 57.3% | 5.51 | 1.32 | 42.7% |
| 7 | 0.0105 | 9.2% | 47.7% | 5.29 | 1.27 | 52.3% |
| 8 | 0.0079 | 6.9% | 43.2% | 5.09 | 1.28 | 56.8% |
| 9 | 0.0090 | 8.0% | 44.6% | 4.85 | 1.24 | 55.4% |
| 10 | 0.0070 | 6.2% | 41.4% | 4.51 | 1.20 | 58.6% |
| 11 | 0.0061 | 5.4% | 34.7% | 4.25 | 1.21 | 65.3% |
| 12 | 0.0046 | 4.1% | 29.4% | 3.89 | 1.19 | 70.6% |
| 13 | 0.0027 | 2.4% | 22.3% | 3.58 | 1.13 | 77.7% |
| 14 | 0.0013 | 1.2% | 15.4% | 3.41 | 1.11 | 84.6% |
| 15 | 0.0000 | 0.0% | 1.9% | 3.14 | 1.06 | 98.1% |

## EQ (M = 10)

399 completed instances, 1 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0911 |
| PPO regret per step: mean / median | 0.0091 / 0.0000 |
| Share of steps where PPO picks an optimal action | 66.6% |
| Legal actions / optimal actions per step (mean) | 5.85 / 1.25 |
| Share of steps with a unique optimal action | 79.0% |
| Steps with regret per instance (mean) | 3.34 |
| Share of the gap from steps 1–3 | 55.8% |
| Share of the gap from steps 4–6 | 32.4% |
| Share of the gap from steps 7–10 | 11.8% |
| Regret at steps followed by a forced opening: share of regret | 92.2% |
| Regret at steps that are themselves forced openings: share of regret | 59.2% |
| Steps from a regret step to the next forced opening (mean) | 1.33 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 2.00 / 2 (1 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0184 | 20.2% | 58.1% | 7.34 | 1.39 | 41.9% |
| 2 | 0.0187 | 20.5% | 56.6% | 6.91 | 1.39 | 43.4% |
| 3 | 0.0138 | 15.1% | 49.4% | 6.52 | 1.32 | 50.6% |
| 4 | 0.0131 | 14.3% | 45.9% | 6.25 | 1.32 | 54.1% |
| 5 | 0.0088 | 9.6% | 34.8% | 5.90 | 1.29 | 65.2% |
| 6 | 0.0077 | 8.5% | 32.3% | 5.57 | 1.22 | 67.7% |
| 7 | 0.0056 | 6.1% | 23.6% | 5.39 | 1.19 | 76.4% |
| 8 | 0.0027 | 3.0% | 16.5% | 5.06 | 1.16 | 83.5% |
| 9 | 0.0021 | 2.3% | 11.8% | 4.89 | 1.13 | 88.2% |
| 10 | 0.0004 | 0.4% | 4.8% | 4.66 | 1.06 | 95.2% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0957 |
| PPO regret per step: mean / median | 0.0096 / 0.0000 |
| Share of steps where PPO picks an optimal action | 62.7% |
| Legal actions / optimal actions per step (mean) | 10.10 / 1.35 |
| Share of steps with a unique optimal action | 71.8% |
| Steps with regret per instance (mean) | 3.73 |
| Share of the gap from steps 1–3 | 55.8% |
| Share of the gap from steps 4–6 | 32.1% |
| Share of the gap from steps 7–10 | 12.1% |
| Regret at steps followed by a forced opening: share of regret | 94.1% |
| Regret at steps that are themselves forced openings: share of regret | 66.1% |
| Steps from a regret step to the next forced opening (mean) | 1.27 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0187 | 19.6% | 58.0% | 10.93 | 1.54 | 42.0% |
| 2 | 0.0176 | 18.3% | 61.5% | 10.73 | 1.52 | 38.5% |
| 3 | 0.0171 | 17.9% | 59.2% | 10.62 | 1.47 | 40.8% |
| 4 | 0.0115 | 12.1% | 49.8% | 10.47 | 1.46 | 50.2% |
| 5 | 0.0108 | 11.3% | 43.8% | 10.31 | 1.47 | 56.2% |
| 6 | 0.0084 | 8.7% | 39.2% | 10.05 | 1.32 | 60.8% |
| 7 | 0.0078 | 8.1% | 30.5% | 9.81 | 1.26 | 69.5% |
| 8 | 0.0021 | 2.2% | 15.5% | 9.57 | 1.23 | 84.5% |
| 9 | 0.0016 | 1.7% | 13.5% | 9.39 | 1.17 | 86.5% |
| 10 | 0.0001 | 0.1% | 2.0% | 9.15 | 1.09 | 98.0% |
