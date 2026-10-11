# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-11 19:38:45; script `src/scripts/diag_regret.py`; took 9.9 min.
- Model: v4.4.17 structure-aware Mask PPO, 50k instances, 20M (seed 3, 20M, GPU), replayed deterministically on the first 400 instances of the seed-3 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

394 completed instances, 6 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 7.4e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0598 |
| PPO regret per step: mean / median | 0.0040 / 0.0000 |
| Share of steps where PPO picks an optimal action | 68.2% |
| Legal actions / optimal actions per step (mean) | 4.87 / 1.23 |
| Share of steps with a unique optimal action | 81.5% |
| Steps with regret per instance (mean) | 4.78 |
| Share of the gap from steps 1–5 | 57.4% |
| Share of the gap from steps 6–10 | 35.8% |
| Share of the gap from steps 11–15 | 6.8% |
| Regret at steps followed by a forced opening: share of regret | 85.4% |
| Regret at steps that are themselves forced openings: share of regret | 46.3% |
| Steps from a regret step to the next forced opening (mean) | 1.90 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 4.67 / 4 (6 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0062 | 10.4% | 60.4% | 7.41 | 1.46 | 39.6% |
| 2 | 0.0079 | 13.3% | 59.1% | 6.83 | 1.44 | 40.9% |
| 3 | 0.0072 | 12.0% | 54.6% | 6.31 | 1.42 | 45.4% |
| 4 | 0.0066 | 11.1% | 49.0% | 5.85 | 1.36 | 51.0% |
| 5 | 0.0064 | 10.7% | 48.7% | 5.51 | 1.30 | 51.3% |
| 6 | 0.0048 | 8.0% | 39.3% | 5.27 | 1.27 | 60.7% |
| 7 | 0.0052 | 8.7% | 37.3% | 4.96 | 1.24 | 62.7% |
| 8 | 0.0051 | 8.5% | 33.2% | 4.68 | 1.24 | 66.8% |
| 9 | 0.0035 | 5.8% | 28.9% | 4.41 | 1.15 | 71.1% |
| 10 | 0.0029 | 4.8% | 23.1% | 4.19 | 1.17 | 76.9% |
| 11 | 0.0023 | 3.8% | 19.5% | 3.87 | 1.12 | 80.5% |
| 12 | 0.0010 | 1.8% | 12.9% | 3.67 | 1.10 | 87.1% |
| 13 | 0.0006 | 0.9% | 6.9% | 3.48 | 1.05 | 93.1% |
| 14 | 0.0001 | 0.2% | 3.8% | 3.32 | 1.09 | 96.2% |
| 15 | 0.0000 | 0.1% | 0.8% | 3.25 | 1.03 | 99.2% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0432 |
| PPO regret per step: mean / median | 0.0043 / 0.0000 |
| Share of steps where PPO picks an optimal action | 74.7% |
| Legal actions / optimal actions per step (mean) | 6.20 / 1.21 |
| Share of steps with a unique optimal action | 82.4% |
| Steps with regret per instance (mean) | 2.53 |
| Share of the gap from steps 1–3 | 56.4% |
| Share of the gap from steps 4–6 | 38.5% |
| Share of the gap from steps 7–10 | 5.1% |
| Regret at steps followed by a forced opening: share of regret | 87.7% |
| Regret at steps that are themselves forced openings: share of regret | 54.6% |
| Steps from a regret step to the next forced opening (mean) | 1.43 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0087 | 20.1% | 53.8% | 7.41 | 1.43 | 46.2% |
| 2 | 0.0076 | 17.7% | 45.8% | 7.06 | 1.34 | 54.2% |
| 3 | 0.0080 | 18.6% | 41.5% | 6.72 | 1.34 | 58.5% |
| 4 | 0.0070 | 16.2% | 38.2% | 6.46 | 1.26 | 61.8% |
| 5 | 0.0066 | 15.2% | 30.0% | 6.26 | 1.20 | 70.0% |
| 6 | 0.0031 | 7.1% | 20.5% | 6.02 | 1.21 | 79.5% |
| 7 | 0.0013 | 2.9% | 12.2% | 5.82 | 1.12 | 87.8% |
| 8 | 0.0008 | 1.8% | 4.5% | 5.60 | 1.11 | 95.5% |
| 9 | 0.0001 | 0.3% | 3.5% | 5.39 | 1.09 | 96.5% |
| 10 | 0.0000 | 0.1% | 2.8% | 5.24 | 1.04 | 97.2% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.2e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0442 |
| PPO regret per step: mean / median | 0.0044 / 0.0000 |
| Share of steps where PPO picks an optimal action | 76.7% |
| Legal actions / optimal actions per step (mean) | 10.44 / 1.31 |
| Share of steps with a unique optimal action | 76.2% |
| Steps with regret per instance (mean) | 2.33 |
| Share of the gap from steps 1–3 | 55.9% |
| Share of the gap from steps 4–6 | 36.8% |
| Share of the gap from steps 7–10 | 7.3% |
| Regret at steps followed by a forced opening: share of regret | 89.4% |
| Regret at steps that are themselves forced openings: share of regret | 67.8% |
| Steps from a regret step to the next forced opening (mean) | 1.31 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0081 | 18.4% | 45.2% | 10.91 | 1.51 | 54.8% |
| 2 | 0.0080 | 18.1% | 40.5% | 10.78 | 1.55 | 59.5% |
| 3 | 0.0086 | 19.4% | 39.5% | 10.62 | 1.43 | 60.5% |
| 4 | 0.0073 | 16.6% | 36.8% | 10.70 | 1.41 | 63.2% |
| 5 | 0.0055 | 12.4% | 27.0% | 10.61 | 1.36 | 73.0% |
| 6 | 0.0034 | 7.8% | 20.0% | 10.48 | 1.27 | 80.0% |
| 7 | 0.0022 | 5.0% | 12.5% | 10.35 | 1.26 | 87.5% |
| 8 | 0.0008 | 1.7% | 7.5% | 10.12 | 1.15 | 92.5% |
| 9 | 0.0003 | 0.6% | 3.2% | 9.98 | 1.11 | 96.8% |
| 10 | 0.0000 | 0.0% | 0.8% | 9.82 | 1.05 | 99.2% |
