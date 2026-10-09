# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-09 15:07:03; script `src/scripts/diag_regret.py`; took 9.8 min.
- Model: model 'mask_ppo_group' of logs/v4.4.11/manifest.json, seed 1, replayed deterministically on the first 400 instances of the seed-1 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

385 completed instances, 15 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1072 |
| PPO regret per step: mean / median | 0.0071 / 0.0000 |
| Share of steps where PPO picks an optimal action | 57.0% |
| Legal actions / optimal actions per step (mean) | 4.90 / 1.27 |
| Share of steps with a unique optimal action | 78.3% |
| Steps with regret per instance (mean) | 6.45 |
| Share of the gap from steps 1–5 | 49.8% |
| Share of the gap from steps 6–10 | 40.7% |
| Share of the gap from steps 11–15 | 9.5% |
| Regret at steps followed by a forced opening: share of regret | 87.7% |
| Regret at steps that are themselves forced openings: share of regret | 42.5% |
| Steps from a regret step to the next forced opening (mean) | 2.10 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 6.47 / 6 (15 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0111 | 10.4% | 74.3% | 7.41 | 1.56 | 25.7% |
| 2 | 0.0109 | 10.2% | 69.9% | 6.87 | 1.48 | 30.1% |
| 3 | 0.0102 | 9.5% | 65.5% | 6.38 | 1.39 | 34.5% |
| 4 | 0.0104 | 9.7% | 59.2% | 5.94 | 1.40 | 40.8% |
| 5 | 0.0108 | 10.1% | 54.5% | 5.57 | 1.37 | 45.5% |
| 6 | 0.0105 | 9.8% | 54.5% | 5.35 | 1.28 | 45.5% |
| 7 | 0.0088 | 8.2% | 50.4% | 5.15 | 1.27 | 49.6% |
| 8 | 0.0104 | 9.7% | 53.0% | 4.89 | 1.22 | 47.0% |
| 9 | 0.0079 | 7.4% | 40.8% | 4.59 | 1.21 | 59.2% |
| 10 | 0.0061 | 5.7% | 37.9% | 4.29 | 1.18 | 62.1% |
| 11 | 0.0054 | 5.1% | 26.0% | 3.89 | 1.17 | 74.0% |
| 12 | 0.0029 | 2.7% | 22.1% | 3.57 | 1.15 | 77.9% |
| 13 | 0.0009 | 0.9% | 17.1% | 3.43 | 1.16 | 82.9% |
| 14 | 0.0005 | 0.4% | 11.9% | 3.27 | 1.12 | 88.1% |
| 15 | 0.0004 | 0.4% | 8.3% | 2.98 | 1.05 | 91.7% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0900 |
| PPO regret per step: mean / median | 0.0090 / 0.0000 |
| Share of steps where PPO picks an optimal action | 65.3% |
| Legal actions / optimal actions per step (mean) | 5.87 / 1.26 |
| Share of steps with a unique optimal action | 78.0% |
| Steps with regret per instance (mean) | 3.46 |
| Share of the gap from steps 1–3 | 54.5% |
| Share of the gap from steps 4–6 | 31.8% |
| Share of the gap from steps 7–10 | 13.7% |
| Regret at steps followed by a forced opening: share of regret | 91.2% |
| Regret at steps that are themselves forced openings: share of regret | 57.5% |
| Steps from a regret step to the next forced opening (mean) | 1.31 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0184 | 20.4% | 62.3% | 7.39 | 1.42 | 37.8% |
| 2 | 0.0148 | 16.5% | 56.2% | 6.89 | 1.46 | 43.8% |
| 3 | 0.0159 | 17.6% | 50.7% | 6.51 | 1.37 | 49.2% |
| 4 | 0.0118 | 13.1% | 44.0% | 6.14 | 1.38 | 56.0% |
| 5 | 0.0081 | 9.0% | 33.0% | 5.83 | 1.30 | 67.0% |
| 6 | 0.0087 | 9.6% | 32.8% | 5.59 | 1.20 | 67.2% |
| 7 | 0.0057 | 6.3% | 21.8% | 5.37 | 1.20 | 78.2% |
| 8 | 0.0043 | 4.8% | 16.8% | 5.21 | 1.12 | 83.2% |
| 9 | 0.0015 | 1.7% | 17.2% | 4.93 | 1.12 | 82.8% |
| 10 | 0.0008 | 0.9% | 11.8% | 4.79 | 1.07 | 88.2% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1022 |
| PPO regret per step: mean / median | 0.0102 / 0.0000 |
| Share of steps where PPO picks an optimal action | 63.8% |
| Legal actions / optimal actions per step (mean) | 10.12 / 1.34 |
| Share of steps with a unique optimal action | 73.2% |
| Steps with regret per instance (mean) | 3.62 |
| Share of the gap from steps 1–3 | 51.3% |
| Share of the gap from steps 4–6 | 34.2% |
| Share of the gap from steps 7–10 | 14.5% |
| Regret at steps followed by a forced opening: share of regret | 94.8% |
| Regret at steps that are themselves forced openings: share of regret | 64.7% |
| Steps from a regret step to the next forced opening (mean) | 1.20 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0191 | 18.7% | 56.2% | 11.00 | 1.57 | 43.8% |
| 2 | 0.0188 | 18.4% | 59.0% | 10.79 | 1.51 | 41.0% |
| 3 | 0.0145 | 14.2% | 55.8% | 10.71 | 1.45 | 44.2% |
| 4 | 0.0135 | 13.2% | 48.8% | 10.57 | 1.45 | 51.2% |
| 5 | 0.0114 | 11.1% | 41.0% | 10.29 | 1.40 | 59.0% |
| 6 | 0.0101 | 9.9% | 36.0% | 9.99 | 1.30 | 64.0% |
| 7 | 0.0074 | 7.2% | 27.5% | 9.75 | 1.28 | 72.5% |
| 8 | 0.0044 | 4.3% | 19.2% | 9.54 | 1.23 | 80.8% |
| 9 | 0.0024 | 2.4% | 14.5% | 9.38 | 1.15 | 85.5% |
| 10 | 0.0006 | 0.6% | 4.0% | 9.16 | 1.09 | 96.0% |
