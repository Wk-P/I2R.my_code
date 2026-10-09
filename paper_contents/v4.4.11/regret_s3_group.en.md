# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-09 15:26:32; script `src/scripts/diag_regret.py`; took 10.0 min.
- Model: model 'mask_ppo_group' of logs/v4.4.11/manifest.json, seed 3, replayed deterministically on the first 400 instances of the seed-3 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

387 completed instances, 13 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 7.4e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1056 |
| PPO regret per step: mean / median | 0.0070 / 0.0000 |
| Share of steps where PPO picks an optimal action | 55.1% |
| Legal actions / optimal actions per step (mean) | 4.96 / 1.26 |
| Share of steps with a unique optimal action | 78.8% |
| Steps with regret per instance (mean) | 6.74 |
| Share of the gap from steps 1–5 | 48.6% |
| Share of the gap from steps 6–10 | 40.5% |
| Share of the gap from steps 11–15 | 10.9% |
| Regret at steps followed by a forced opening: share of regret | 94.2% |
| Regret at steps that are themselves forced openings: share of regret | 40.7% |
| Steps from a regret step to the next forced opening (mean) | 2.05 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 5.15 / 5 (13 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0115 | 10.9% | 73.9% | 7.42 | 1.46 | 26.1% |
| 2 | 0.0100 | 9.5% | 72.4% | 6.83 | 1.48 | 27.6% |
| 3 | 0.0092 | 8.8% | 63.3% | 6.36 | 1.45 | 36.7% |
| 4 | 0.0098 | 9.3% | 63.6% | 5.95 | 1.36 | 36.4% |
| 5 | 0.0108 | 10.2% | 61.2% | 5.61 | 1.33 | 38.8% |
| 6 | 0.0087 | 8.3% | 57.6% | 5.44 | 1.30 | 42.4% |
| 7 | 0.0110 | 10.4% | 56.3% | 5.20 | 1.29 | 43.7% |
| 8 | 0.0080 | 7.6% | 48.8% | 4.94 | 1.29 | 51.2% |
| 9 | 0.0081 | 7.7% | 46.0% | 4.64 | 1.20 | 54.0% |
| 10 | 0.0069 | 6.6% | 40.8% | 4.41 | 1.21 | 59.2% |
| 11 | 0.0054 | 5.1% | 30.0% | 4.16 | 1.16 | 70.0% |
| 12 | 0.0037 | 3.5% | 26.4% | 3.68 | 1.12 | 73.6% |
| 13 | 0.0019 | 1.8% | 20.7% | 3.50 | 1.11 | 79.3% |
| 14 | 0.0005 | 0.4% | 10.1% | 3.17 | 1.13 | 89.9% |
| 15 | 0.0001 | 0.1% | 3.1% | 3.06 | 1.04 | 96.9% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0782 |
| PPO regret per step: mean / median | 0.0078 / 0.0000 |
| Share of steps where PPO picks an optimal action | 67.0% |
| Legal actions / optimal actions per step (mean) | 6.00 / 1.23 |
| Share of steps with a unique optimal action | 79.8% |
| Steps with regret per instance (mean) | 3.30 |
| Share of the gap from steps 1–3 | 49.0% |
| Share of the gap from steps 4–6 | 38.8% |
| Share of the gap from steps 7–10 | 12.2% |
| Regret at steps followed by a forced opening: share of regret | 91.5% |
| Regret at steps that are themselves forced openings: share of regret | 55.8% |
| Steps from a regret step to the next forced opening (mean) | 1.46 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0143 | 18.3% | 60.8% | 7.41 | 1.43 | 39.2% |
| 2 | 0.0125 | 15.9% | 59.8% | 6.99 | 1.35 | 40.2% |
| 3 | 0.0116 | 14.8% | 48.8% | 6.58 | 1.31 | 51.2% |
| 4 | 0.0118 | 15.1% | 41.2% | 6.28 | 1.27 | 58.8% |
| 5 | 0.0090 | 11.5% | 35.2% | 6.04 | 1.24 | 64.8% |
| 6 | 0.0095 | 12.1% | 31.5% | 5.87 | 1.22 | 68.5% |
| 7 | 0.0042 | 5.4% | 24.2% | 5.57 | 1.17 | 75.8% |
| 8 | 0.0042 | 5.4% | 17.5% | 5.40 | 1.15 | 82.5% |
| 9 | 0.0006 | 0.8% | 7.0% | 5.09 | 1.14 | 93.0% |
| 10 | 0.0005 | 0.7% | 3.8% | 4.80 | 1.07 | 96.2% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0907 |
| PPO regret per step: mean / median | 0.0091 / 0.0000 |
| Share of steps where PPO picks an optimal action | 64.1% |
| Legal actions / optimal actions per step (mean) | 10.05 / 1.35 |
| Share of steps with a unique optimal action | 73.0% |
| Steps with regret per instance (mean) | 3.59 |
| Share of the gap from steps 1–3 | 48.6% |
| Share of the gap from steps 4–6 | 36.5% |
| Share of the gap from steps 7–10 | 14.9% |
| Regret at steps followed by a forced opening: share of regret | 95.1% |
| Regret at steps that are themselves forced openings: share of regret | 68.1% |
| Steps from a regret step to the next forced opening (mean) | 1.24 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0160 | 17.6% | 57.5% | 10.91 | 1.51 | 42.5% |
| 2 | 0.0146 | 16.1% | 56.0% | 10.76 | 1.56 | 44.0% |
| 3 | 0.0136 | 14.9% | 57.5% | 10.51 | 1.42 | 42.5% |
| 4 | 0.0135 | 14.9% | 47.2% | 10.44 | 1.47 | 52.8% |
| 5 | 0.0103 | 11.4% | 40.5% | 10.15 | 1.46 | 59.5% |
| 6 | 0.0093 | 10.3% | 35.0% | 9.96 | 1.33 | 65.0% |
| 7 | 0.0078 | 8.6% | 28.2% | 9.79 | 1.30 | 71.8% |
| 8 | 0.0036 | 4.0% | 19.5% | 9.54 | 1.18 | 80.5% |
| 9 | 0.0018 | 2.0% | 12.0% | 9.30 | 1.17 | 88.0% |
| 10 | 0.0003 | 0.3% | 5.5% | 9.11 | 1.09 | 94.5% |
