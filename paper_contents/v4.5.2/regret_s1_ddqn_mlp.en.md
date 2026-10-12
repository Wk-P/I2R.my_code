# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 21:27:08; script `src/scripts/diag_regret.py`; took 9.0 min.
- Model: v4.5.2 MLP Mask-DDQN (50k) (seed 1, 20M), replayed deterministically on the first 400 instances of the seed-1 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

351 completed instances, 49 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 7.3e-04 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1980 |
| PPO regret per step: mean / median | 0.0132 / 0.0037 |
| Share of steps where PPO picks an optimal action | 36.8% |
| Legal actions / optimal actions per step (mean) | 4.94 / 1.24 |
| Share of steps with a unique optimal action | 80.0% |
| Steps with regret per instance (mean) | 9.48 |
| Share of the gap from steps 1–5 | 41.3% |
| Share of the gap from steps 6–10 | 39.2% |
| Share of the gap from steps 11–15 | 19.5% |
| Regret at steps followed by a forced opening: share of regret | 90.2% |
| Regret at steps that are themselves forced openings: share of regret | 50.0% |
| Steps from a regret step to the next forced opening (mean) | 1.63 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 6.63 / 6 (49 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0141 | 7.1% | 78.3% | 7.45 | 1.54 | 21.7% |
| 2 | 0.0167 | 8.4% | 76.4% | 6.96 | 1.44 | 23.6% |
| 3 | 0.0172 | 8.7% | 78.1% | 6.51 | 1.34 | 21.9% |
| 4 | 0.0168 | 8.5% | 74.1% | 6.12 | 1.37 | 25.9% |
| 5 | 0.0170 | 8.6% | 74.6% | 5.74 | 1.30 | 25.4% |
| 6 | 0.0164 | 8.3% | 72.6% | 5.48 | 1.23 | 27.4% |
| 7 | 0.0171 | 8.6% | 69.5% | 5.20 | 1.26 | 30.5% |
| 8 | 0.0175 | 8.9% | 69.5% | 4.92 | 1.22 | 30.5% |
| 9 | 0.0143 | 7.2% | 64.4% | 4.55 | 1.18 | 35.6% |
| 10 | 0.0123 | 6.2% | 61.3% | 4.28 | 1.18 | 38.7% |
| 11 | 0.0112 | 5.7% | 55.3% | 3.87 | 1.15 | 44.7% |
| 12 | 0.0106 | 5.4% | 54.7% | 3.46 | 1.14 | 45.3% |
| 13 | 0.0074 | 3.7% | 46.4% | 3.41 | 1.12 | 53.6% |
| 14 | 0.0034 | 1.7% | 36.8% | 3.19 | 1.10 | 63.2% |
| 15 | 0.0059 | 3.0% | 36.5% | 2.99 | 1.03 | 63.5% |

## EQ (M = 10)

399 completed instances, 1 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1649 |
| PPO regret per step: mean / median | 0.0165 / 0.0030 |
| Share of steps where PPO picks an optimal action | 41.9% |
| Legal actions / optimal actions per step (mean) | 6.04 / 1.23 |
| Share of steps with a unique optimal action | 80.5% |
| Steps with regret per instance (mean) | 5.81 |
| Share of the gap from steps 1–3 | 46.1% |
| Share of the gap from steps 4–6 | 33.9% |
| Share of the gap from steps 7–10 | 20.0% |
| Regret at steps followed by a forced opening: share of regret | 93.7% |
| Regret at steps that are themselves forced openings: share of regret | 57.4% |
| Steps from a regret step to the next forced opening (mean) | 1.35 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 3.00 / 3 (1 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0230 | 13.9% | 79.4% | 7.40 | 1.42 | 20.6% |
| 2 | 0.0259 | 15.7% | 80.5% | 7.02 | 1.36 | 19.5% |
| 3 | 0.0272 | 16.5% | 75.4% | 6.80 | 1.30 | 24.6% |
| 4 | 0.0205 | 12.4% | 62.4% | 6.51 | 1.32 | 37.6% |
| 5 | 0.0184 | 11.2% | 60.4% | 6.23 | 1.29 | 39.6% |
| 6 | 0.0170 | 10.3% | 57.1% | 5.93 | 1.17 | 42.9% |
| 7 | 0.0149 | 9.0% | 53.9% | 5.62 | 1.17 | 46.1% |
| 8 | 0.0089 | 5.4% | 45.4% | 5.34 | 1.16 | 54.6% |
| 9 | 0.0062 | 3.7% | 36.8% | 4.89 | 1.10 | 63.2% |
| 10 | 0.0030 | 1.8% | 30.1% | 4.70 | 1.05 | 69.9% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.2045 |
| PPO regret per step: mean / median | 0.0205 / 0.0093 |
| Share of steps where PPO picks an optimal action | 33.8% |
| Legal actions / optimal actions per step (mean) | 10.54 / 1.33 |
| Share of steps with a unique optimal action | 74.4% |
| Steps with regret per instance (mean) | 6.62 |
| Share of the gap from steps 1–3 | 42.3% |
| Share of the gap from steps 4–6 | 36.4% |
| Share of the gap from steps 7–10 | 21.3% |
| Regret at steps followed by a forced opening: share of regret | 91.5% |
| Regret at steps that are themselves forced openings: share of regret | 62.3% |
| Steps from a regret step to the next forced opening (mean) | 1.32 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0283 | 13.8% | 85.0% | 11.00 | 1.57 | 15.0% |
| 2 | 0.0300 | 14.6% | 88.0% | 10.88 | 1.44 | 12.0% |
| 3 | 0.0281 | 13.8% | 77.0% | 10.99 | 1.46 | 23.0% |
| 4 | 0.0288 | 14.1% | 73.0% | 11.04 | 1.39 | 27.0% |
| 5 | 0.0241 | 11.8% | 69.2% | 10.94 | 1.36 | 30.8% |
| 6 | 0.0216 | 10.5% | 66.5% | 10.71 | 1.31 | 33.5% |
| 7 | 0.0179 | 8.8% | 61.0% | 10.46 | 1.30 | 39.0% |
| 8 | 0.0123 | 6.0% | 50.2% | 10.10 | 1.22 | 49.8% |
| 9 | 0.0081 | 4.0% | 44.0% | 9.73 | 1.19 | 56.0% |
| 10 | 0.0052 | 2.6% | 47.8% | 9.52 | 1.07 | 52.2% |
