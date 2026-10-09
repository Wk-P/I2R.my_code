# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-09 23:20:26; script `src/scripts/diag_regret.py`; took 9.4 min.
- Model: v4.4.13 structure-aware Mask PPO, professor's 3-dim state summary (seed 2, 1M, GPU), replayed deterministically on the first 400 instances of the seed-2 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

383 completed instances, 17 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 2.3e-04 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1075 |
| PPO regret per step: mean / median | 0.0072 / 0.0000 |
| Share of steps where PPO picks an optimal action | 59.1% |
| Legal actions / optimal actions per step (mean) | 4.79 / 1.27 |
| Share of steps with a unique optimal action | 77.8% |
| Steps with regret per instance (mean) | 6.13 |
| Share of the gap from steps 1–5 | 49.1% |
| Share of the gap from steps 6–10 | 40.0% |
| Share of the gap from steps 11–15 | 10.9% |
| Regret at steps followed by a forced opening: share of regret | 94.4% |
| Regret at steps that are themselves forced openings: share of regret | 45.9% |
| Steps from a regret step to the next forced opening (mean) | 1.79 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 4.65 / 4 (17 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0112 | 10.4% | 70.2% | 7.23 | 1.54 | 29.8% |
| 2 | 0.0115 | 10.7% | 61.9% | 6.66 | 1.47 | 38.1% |
| 3 | 0.0110 | 10.3% | 63.7% | 6.20 | 1.37 | 36.3% |
| 4 | 0.0095 | 8.8% | 55.9% | 5.81 | 1.34 | 44.1% |
| 5 | 0.0096 | 8.9% | 57.7% | 5.47 | 1.27 | 42.3% |
| 6 | 0.0093 | 8.6% | 55.9% | 5.20 | 1.31 | 44.1% |
| 7 | 0.0100 | 9.3% | 48.3% | 4.99 | 1.27 | 51.7% |
| 8 | 0.0077 | 7.2% | 43.9% | 4.80 | 1.26 | 56.1% |
| 9 | 0.0076 | 7.0% | 43.9% | 4.51 | 1.22 | 56.1% |
| 10 | 0.0084 | 7.8% | 37.1% | 4.22 | 1.25 | 62.9% |
| 11 | 0.0059 | 5.5% | 30.3% | 3.95 | 1.20 | 69.7% |
| 12 | 0.0040 | 3.7% | 24.3% | 3.66 | 1.15 | 75.7% |
| 13 | 0.0016 | 1.4% | 13.1% | 3.24 | 1.17 | 86.9% |
| 14 | 0.0003 | 0.2% | 5.7% | 3.07 | 1.13 | 94.3% |
| 15 | 0.0000 | 0.0% | 1.0% | 2.82 | 1.06 | 99.0% |

## EQ (M = 10)

399 completed instances, 1 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0860 |
| PPO regret per step: mean / median | 0.0086 / 0.0000 |
| Share of steps where PPO picks an optimal action | 62.7% |
| Legal actions / optimal actions per step (mean) | 6.06 / 1.25 |
| Share of steps with a unique optimal action | 79.4% |
| Steps with regret per instance (mean) | 3.73 |
| Share of the gap from steps 1–3 | 51.0% |
| Share of the gap from steps 4–6 | 35.0% |
| Share of the gap from steps 7–10 | 14.0% |
| Regret at steps followed by a forced opening: share of regret | 89.1% |
| Regret at steps that are themselves forced openings: share of regret | 51.1% |
| Steps from a regret step to the next forced opening (mean) | 1.57 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 3.00 / 3 (1 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0157 | 18.3% | 64.2% | 7.34 | 1.39 | 35.8% |
| 2 | 0.0156 | 18.1% | 58.9% | 6.96 | 1.37 | 41.1% |
| 3 | 0.0126 | 14.6% | 54.4% | 6.66 | 1.33 | 45.6% |
| 4 | 0.0125 | 14.5% | 55.1% | 6.49 | 1.33 | 44.9% |
| 5 | 0.0094 | 10.9% | 42.9% | 6.21 | 1.24 | 57.1% |
| 6 | 0.0083 | 9.6% | 34.6% | 5.90 | 1.22 | 65.4% |
| 7 | 0.0059 | 6.8% | 28.6% | 5.70 | 1.22 | 71.4% |
| 8 | 0.0055 | 6.4% | 24.6% | 5.40 | 1.17 | 75.4% |
| 9 | 0.0006 | 0.7% | 8.3% | 5.09 | 1.14 | 91.7% |
| 10 | 0.0000 | 0.0% | 1.3% | 4.85 | 1.07 | 98.7% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0951 |
| PPO regret per step: mean / median | 0.0095 / 0.0000 |
| Share of steps where PPO picks an optimal action | 61.9% |
| Legal actions / optimal actions per step (mean) | 10.24 / 1.35 |
| Share of steps with a unique optimal action | 71.5% |
| Steps with regret per instance (mean) | 3.81 |
| Share of the gap from steps 1–3 | 54.2% |
| Share of the gap from steps 4–6 | 33.7% |
| Share of the gap from steps 7–10 | 12.1% |
| Regret at steps followed by a forced opening: share of regret | 92.8% |
| Regret at steps that are themselves forced openings: share of regret | 62.2% |
| Steps from a regret step to the next forced opening (mean) | 1.27 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0183 | 19.2% | 59.5% | 10.93 | 1.54 | 40.5% |
| 2 | 0.0159 | 16.8% | 62.5% | 10.73 | 1.47 | 37.5% |
| 3 | 0.0173 | 18.2% | 58.5% | 10.62 | 1.48 | 41.5% |
| 4 | 0.0109 | 11.4% | 46.2% | 10.50 | 1.46 | 53.8% |
| 5 | 0.0112 | 11.8% | 48.0% | 10.41 | 1.44 | 52.0% |
| 6 | 0.0099 | 10.4% | 42.5% | 10.21 | 1.34 | 57.5% |
| 7 | 0.0061 | 6.4% | 33.5% | 10.09 | 1.31 | 66.5% |
| 8 | 0.0032 | 3.4% | 18.0% | 9.86 | 1.24 | 82.0% |
| 9 | 0.0017 | 1.7% | 10.5% | 9.69 | 1.20 | 89.5% |
| 10 | 0.0006 | 0.6% | 1.5% | 9.41 | 1.07 | 98.5% |
