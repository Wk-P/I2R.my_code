# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-10 13:28:14; script `src/scripts/diag_regret.py`; took 10.2 min.
- Model: v4.4.14 structure-aware Mask PPO, 50k training instances (seed 3, 1M, GPU), replayed deterministically on the first 400 instances of the seed-3 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

390 completed instances, 10 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 7.4e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0942 |
| PPO regret per step: mean / median | 0.0063 / 0.0000 |
| Share of steps where PPO picks an optimal action | 58.2% |
| Legal actions / optimal actions per step (mean) | 5.00 / 1.24 |
| Share of steps with a unique optimal action | 80.3% |
| Steps with regret per instance (mean) | 6.28 |
| Share of the gap from steps 1–5 | 51.0% |
| Share of the gap from steps 6–10 | 37.8% |
| Share of the gap from steps 11–15 | 11.2% |
| Regret at steps followed by a forced opening: share of regret | 85.1% |
| Regret at steps that are themselves forced openings: share of regret | 39.9% |
| Steps from a regret step to the next forced opening (mean) | 2.07 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 4.40 / 4 (10 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0092 | 9.8% | 75.1% | 7.42 | 1.46 | 24.9% |
| 2 | 0.0096 | 10.2% | 70.5% | 6.90 | 1.45 | 29.5% |
| 3 | 0.0088 | 9.4% | 60.3% | 6.43 | 1.41 | 39.7% |
| 4 | 0.0100 | 10.6% | 59.7% | 6.03 | 1.33 | 40.3% |
| 5 | 0.0104 | 11.0% | 58.2% | 5.67 | 1.32 | 41.8% |
| 6 | 0.0076 | 8.1% | 50.8% | 5.45 | 1.27 | 49.2% |
| 7 | 0.0090 | 9.6% | 46.2% | 5.14 | 1.28 | 53.8% |
| 8 | 0.0068 | 7.3% | 45.6% | 4.87 | 1.25 | 54.4% |
| 9 | 0.0078 | 8.3% | 37.4% | 4.62 | 1.18 | 62.6% |
| 10 | 0.0044 | 4.7% | 31.3% | 4.46 | 1.19 | 68.7% |
| 11 | 0.0055 | 5.8% | 32.8% | 4.18 | 1.12 | 67.2% |
| 12 | 0.0031 | 3.3% | 27.7% | 3.78 | 1.11 | 72.3% |
| 13 | 0.0014 | 1.5% | 18.5% | 3.59 | 1.09 | 81.5% |
| 14 | 0.0004 | 0.4% | 9.0% | 3.37 | 1.12 | 91.0% |
| 15 | 0.0001 | 0.2% | 4.6% | 3.18 | 1.04 | 95.4% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0661 |
| PPO regret per step: mean / median | 0.0066 / 0.0000 |
| Share of steps where PPO picks an optimal action | 68.2% |
| Legal actions / optimal actions per step (mean) | 6.13 / 1.23 |
| Share of steps with a unique optimal action | 80.5% |
| Steps with regret per instance (mean) | 3.18 |
| Share of the gap from steps 1–3 | 49.8% |
| Share of the gap from steps 4–6 | 38.0% |
| Share of the gap from steps 7–10 | 12.2% |
| Regret at steps followed by a forced opening: share of regret | 85.5% |
| Regret at steps that are themselves forced openings: share of regret | 53.6% |
| Steps from a regret step to the next forced opening (mean) | 1.50 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0112 | 16.9% | 58.2% | 7.41 | 1.43 | 41.8% |
| 2 | 0.0114 | 17.3% | 57.8% | 7.01 | 1.36 | 42.2% |
| 3 | 0.0103 | 15.6% | 48.0% | 6.66 | 1.33 | 52.0% |
| 4 | 0.0098 | 14.8% | 44.5% | 6.38 | 1.28 | 55.5% |
| 5 | 0.0090 | 13.6% | 38.2% | 6.22 | 1.24 | 61.8% |
| 6 | 0.0064 | 9.6% | 29.8% | 6.03 | 1.20 | 70.2% |
| 7 | 0.0041 | 6.3% | 18.5% | 5.74 | 1.13 | 81.5% |
| 8 | 0.0025 | 3.7% | 14.5% | 5.53 | 1.14 | 85.5% |
| 9 | 0.0015 | 2.2% | 8.2% | 5.27 | 1.10 | 91.8% |
| 10 | 0.0000 | 0.0% | 0.2% | 5.03 | 1.05 | 99.8% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0947 |
| PPO regret per step: mean / median | 0.0095 / 0.0000 |
| Share of steps where PPO picks an optimal action | 63.6% |
| Legal actions / optimal actions per step (mean) | 10.06 / 1.36 |
| Share of steps with a unique optimal action | 72.6% |
| Steps with regret per instance (mean) | 3.63 |
| Share of the gap from steps 1–3 | 52.0% |
| Share of the gap from steps 4–6 | 34.6% |
| Share of the gap from steps 7–10 | 13.4% |
| Regret at steps followed by a forced opening: share of regret | 93.5% |
| Regret at steps that are themselves forced openings: share of regret | 66.1% |
| Steps from a regret step to the next forced opening (mean) | 1.25 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0173 | 18.3% | 58.8% | 10.91 | 1.51 | 41.2% |
| 2 | 0.0169 | 17.8% | 57.8% | 10.76 | 1.53 | 42.2% |
| 3 | 0.0151 | 15.9% | 56.0% | 10.51 | 1.46 | 44.0% |
| 4 | 0.0139 | 14.7% | 46.0% | 10.46 | 1.51 | 54.0% |
| 5 | 0.0116 | 12.2% | 42.8% | 10.23 | 1.45 | 57.2% |
| 6 | 0.0072 | 7.6% | 36.5% | 10.01 | 1.35 | 63.5% |
| 7 | 0.0075 | 7.9% | 31.2% | 9.87 | 1.29 | 68.8% |
| 8 | 0.0041 | 4.3% | 23.0% | 9.55 | 1.20 | 77.0% |
| 9 | 0.0011 | 1.2% | 10.2% | 9.24 | 1.18 | 89.8% |
| 10 | 0.0000 | 0.0% | 1.2% | 9.02 | 1.09 | 98.8% |
