# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-11 19:19:26; script `src/scripts/diag_regret.py`; took 9.9 min.
- Model: v4.4.17 structure-aware Mask PPO, 50k instances, 20M (seed 1, 20M, GPU), replayed deterministically on the first 400 instances of the seed-1 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

392 completed instances, 8 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 1.5e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0597 |
| PPO regret per step: mean / median | 0.0040 / 0.0000 |
| Share of steps where PPO picks an optimal action | 67.4% |
| Legal actions / optimal actions per step (mean) | 5.00 / 1.24 |
| Share of steps with a unique optimal action | 80.2% |
| Steps with regret per instance (mean) | 4.88 |
| Share of the gap from steps 1–5 | 55.0% |
| Share of the gap from steps 6–10 | 38.8% |
| Share of the gap from steps 11–15 | 6.1% |
| Regret at steps followed by a forced opening: share of regret | 79.1% |
| Regret at steps that are themselves forced openings: share of regret | 44.3% |
| Steps from a regret step to the next forced opening (mean) | 1.95 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 2.38 / 2 (8 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0070 | 11.7% | 56.4% | 7.39 | 1.55 | 43.6% |
| 2 | 0.0064 | 10.6% | 58.9% | 6.84 | 1.47 | 41.1% |
| 3 | 0.0065 | 10.9% | 54.8% | 6.40 | 1.40 | 45.2% |
| 4 | 0.0067 | 11.2% | 53.6% | 6.00 | 1.33 | 46.4% |
| 5 | 0.0063 | 10.6% | 45.9% | 5.63 | 1.27 | 54.1% |
| 6 | 0.0054 | 9.0% | 41.6% | 5.36 | 1.28 | 58.4% |
| 7 | 0.0060 | 10.1% | 39.3% | 5.19 | 1.24 | 60.7% |
| 8 | 0.0053 | 8.9% | 34.2% | 4.95 | 1.26 | 65.8% |
| 9 | 0.0042 | 7.0% | 36.7% | 4.65 | 1.19 | 63.3% |
| 10 | 0.0022 | 3.7% | 24.2% | 4.38 | 1.19 | 75.8% |
| 11 | 0.0019 | 3.2% | 16.3% | 3.96 | 1.11 | 83.7% |
| 12 | 0.0012 | 2.1% | 12.0% | 3.76 | 1.12 | 88.0% |
| 13 | 0.0004 | 0.7% | 9.2% | 3.71 | 1.10 | 90.8% |
| 14 | 0.0001 | 0.2% | 3.3% | 3.51 | 1.09 | 96.7% |
| 15 | 0.0000 | 0.0% | 1.8% | 3.26 | 1.04 | 98.2% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0438 |
| PPO regret per step: mean / median | 0.0044 / 0.0000 |
| Share of steps where PPO picks an optimal action | 74.5% |
| Legal actions / optimal actions per step (mean) | 6.17 / 1.25 |
| Share of steps with a unique optimal action | 80.0% |
| Steps with regret per instance (mean) | 2.56 |
| Share of the gap from steps 1–3 | 55.5% |
| Share of the gap from steps 4–6 | 39.5% |
| Share of the gap from steps 7–10 | 4.9% |
| Regret at steps followed by a forced opening: share of regret | 87.3% |
| Regret at steps that are themselves forced openings: share of regret | 51.8% |
| Steps from a regret step to the next forced opening (mean) | 1.48 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0089 | 20.2% | 58.2% | 7.39 | 1.42 | 41.8% |
| 2 | 0.0083 | 18.9% | 48.5% | 6.96 | 1.39 | 51.5% |
| 3 | 0.0072 | 16.4% | 38.2% | 6.71 | 1.35 | 61.8% |
| 4 | 0.0076 | 17.4% | 37.0% | 6.46 | 1.32 | 63.0% |
| 5 | 0.0057 | 13.0% | 30.8% | 6.25 | 1.28 | 69.2% |
| 6 | 0.0040 | 9.2% | 20.2% | 6.00 | 1.24 | 79.8% |
| 7 | 0.0019 | 4.3% | 13.0% | 5.75 | 1.19 | 87.0% |
| 8 | 0.0002 | 0.6% | 6.2% | 5.63 | 1.15 | 93.8% |
| 9 | 0.0000 | 0.1% | 2.2% | 5.35 | 1.07 | 97.8% |
| 10 | 0.0000 | 0.0% | 1.0% | 5.22 | 1.07 | 99.0% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0412 |
| PPO regret per step: mean / median | 0.0041 / 0.0000 |
| Share of steps where PPO picks an optimal action | 76.5% |
| Legal actions / optimal actions per step (mean) | 10.57 / 1.31 |
| Share of steps with a unique optimal action | 76.6% |
| Steps with regret per instance (mean) | 2.35 |
| Share of the gap from steps 1–3 | 58.2% |
| Share of the gap from steps 4–6 | 36.1% |
| Share of the gap from steps 7–10 | 5.7% |
| Regret at steps followed by a forced opening: share of regret | 91.2% |
| Regret at steps that are themselves forced openings: share of regret | 64.9% |
| Steps from a regret step to the next forced opening (mean) | 1.41 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0077 | 18.6% | 45.2% | 11.00 | 1.57 | 54.8% |
| 2 | 0.0073 | 17.8% | 41.8% | 10.85 | 1.53 | 58.2% |
| 3 | 0.0090 | 21.9% | 41.0% | 10.86 | 1.44 | 59.0% |
| 4 | 0.0073 | 17.6% | 40.0% | 10.85 | 1.39 | 60.0% |
| 5 | 0.0040 | 9.8% | 24.0% | 10.74 | 1.34 | 76.0% |
| 6 | 0.0036 | 8.6% | 20.8% | 10.57 | 1.27 | 79.2% |
| 7 | 0.0013 | 3.1% | 10.5% | 10.43 | 1.25 | 89.5% |
| 8 | 0.0009 | 2.3% | 7.5% | 10.28 | 1.17 | 92.5% |
| 9 | 0.0001 | 0.1% | 2.2% | 10.15 | 1.08 | 97.8% |
| 10 | 0.0001 | 0.2% | 2.0% | 9.94 | 1.03 | 98.0% |
