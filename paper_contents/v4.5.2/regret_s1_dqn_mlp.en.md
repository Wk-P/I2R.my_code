# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 20:33:54; script `src/scripts/diag_regret.py`; took 8.9 min.
- Model: v4.5.2 MLP Mask-DQN (50k) (seed 1, 20M), replayed deterministically on the first 400 instances of the seed-1 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

335 completed instances, 65 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1942 |
| PPO regret per step: mean / median | 0.0129 / 0.0032 |
| Share of steps where PPO picks an optimal action | 38.4% |
| Legal actions / optimal actions per step (mean) | 4.84 / 1.23 |
| Share of steps with a unique optimal action | 80.5% |
| Steps with regret per instance (mean) | 9.24 |
| Share of the gap from steps 1–5 | 45.1% |
| Share of the gap from steps 6–10 | 38.0% |
| Share of the gap from steps 11–15 | 16.9% |
| Regret at steps followed by a forced opening: share of regret | 92.1% |
| Regret at steps that are themselves forced openings: share of regret | 50.8% |
| Steps from a regret step to the next forced opening (mean) | 1.61 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 6.88 / 7 (65 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0167 | 8.6% | 80.6% | 7.48 | 1.54 | 19.4% |
| 2 | 0.0159 | 8.2% | 78.2% | 6.98 | 1.42 | 21.8% |
| 3 | 0.0176 | 9.1% | 76.1% | 6.51 | 1.35 | 23.9% |
| 4 | 0.0184 | 9.5% | 74.9% | 6.08 | 1.35 | 25.1% |
| 5 | 0.0189 | 9.7% | 70.7% | 5.70 | 1.32 | 29.3% |
| 6 | 0.0167 | 8.6% | 71.0% | 5.38 | 1.27 | 29.0% |
| 7 | 0.0169 | 8.7% | 68.7% | 5.07 | 1.26 | 31.3% |
| 8 | 0.0156 | 8.1% | 65.4% | 4.80 | 1.23 | 34.6% |
| 9 | 0.0127 | 6.6% | 62.1% | 4.43 | 1.16 | 37.9% |
| 10 | 0.0119 | 6.1% | 60.9% | 4.14 | 1.14 | 39.1% |
| 11 | 0.0094 | 4.8% | 54.9% | 3.67 | 1.13 | 45.1% |
| 12 | 0.0082 | 4.2% | 45.4% | 3.36 | 1.10 | 54.6% |
| 13 | 0.0069 | 3.5% | 41.2% | 3.26 | 1.12 | 58.8% |
| 14 | 0.0052 | 2.7% | 39.1% | 3.05 | 1.08 | 60.9% |
| 15 | 0.0033 | 1.7% | 34.9% | 2.76 | 1.02 | 65.1% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1583 |
| PPO regret per step: mean / median | 0.0158 / 0.0020 |
| Share of steps where PPO picks an optimal action | 44.2% |
| Legal actions / optimal actions per step (mean) | 5.93 / 1.24 |
| Share of steps with a unique optimal action | 79.7% |
| Steps with regret per instance (mean) | 5.58 |
| Share of the gap from steps 1–3 | 46.3% |
| Share of the gap from steps 4–6 | 34.6% |
| Share of the gap from steps 7–10 | 19.0% |
| Regret at steps followed by a forced opening: share of regret | 94.0% |
| Regret at steps that are themselves forced openings: share of regret | 54.5% |
| Steps from a regret step to the next forced opening (mean) | 1.33 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0242 | 15.3% | 78.2% | 7.39 | 1.42 | 21.8% |
| 2 | 0.0235 | 14.8% | 73.0% | 6.95 | 1.41 | 27.0% |
| 3 | 0.0257 | 16.2% | 68.5% | 6.68 | 1.27 | 31.5% |
| 4 | 0.0213 | 13.4% | 68.0% | 6.39 | 1.31 | 32.0% |
| 5 | 0.0175 | 11.1% | 59.5% | 6.06 | 1.33 | 40.5% |
| 6 | 0.0160 | 10.1% | 56.8% | 5.77 | 1.20 | 43.2% |
| 7 | 0.0114 | 7.2% | 45.0% | 5.49 | 1.18 | 55.0% |
| 8 | 0.0093 | 5.9% | 41.2% | 5.17 | 1.15 | 58.8% |
| 9 | 0.0062 | 3.9% | 34.2% | 4.79 | 1.12 | 65.8% |
| 10 | 0.0033 | 2.1% | 33.2% | 4.62 | 1.04 | 66.8% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.2111 |
| PPO regret per step: mean / median | 0.0211 / 0.0108 |
| Share of steps where PPO picks an optimal action | 32.8% |
| Legal actions / optimal actions per step (mean) | 10.61 / 1.33 |
| Share of steps with a unique optimal action | 74.4% |
| Steps with regret per instance (mean) | 6.72 |
| Share of the gap from steps 1–3 | 40.6% |
| Share of the gap from steps 4–6 | 36.9% |
| Share of the gap from steps 7–10 | 22.5% |
| Regret at steps followed by a forced opening: share of regret | 90.9% |
| Regret at steps that are themselves forced openings: share of regret | 62.9% |
| Steps from a regret step to the next forced opening (mean) | 1.36 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0270 | 12.8% | 84.8% | 11.00 | 1.57 | 15.2% |
| 2 | 0.0310 | 14.7% | 84.8% | 10.88 | 1.47 | 15.2% |
| 3 | 0.0277 | 13.1% | 76.2% | 10.98 | 1.43 | 23.8% |
| 4 | 0.0296 | 14.0% | 76.8% | 11.07 | 1.32 | 23.2% |
| 5 | 0.0266 | 12.6% | 70.5% | 10.96 | 1.36 | 29.5% |
| 6 | 0.0218 | 10.3% | 62.0% | 10.79 | 1.31 | 38.0% |
| 7 | 0.0157 | 7.4% | 59.0% | 10.53 | 1.28 | 41.0% |
| 8 | 0.0153 | 7.2% | 58.5% | 10.24 | 1.24 | 41.5% |
| 9 | 0.0102 | 4.8% | 51.5% | 9.90 | 1.21 | 48.5% |
| 10 | 0.0062 | 2.9% | 48.5% | 9.71 | 1.09 | 51.5% |
