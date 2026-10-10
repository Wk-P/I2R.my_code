# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-10 17:44:10; script `src/scripts/diag_regret.py`; took 9.9 min.
- Model: v4.4.15 structure-aware Mask PPO, 50k instances, 5M (seed 1, 5M, GPU), replayed deterministically on the first 400 instances of the seed-1 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

389 completed instances, 11 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0777 |
| PPO regret per step: mean / median | 0.0052 / 0.0000 |
| Share of steps where PPO picks an optimal action | 61.9% |
| Legal actions / optimal actions per step (mean) | 5.00 / 1.25 |
| Share of steps with a unique optimal action | 79.5% |
| Steps with regret per instance (mean) | 5.71 |
| Share of the gap from steps 1–5 | 49.8% |
| Share of the gap from steps 6–10 | 41.2% |
| Share of the gap from steps 11–15 | 9.0% |
| Regret at steps followed by a forced opening: share of regret | 84.4% |
| Regret at steps that are themselves forced openings: share of regret | 44.9% |
| Steps from a regret step to the next forced opening (mean) | 2.07 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 3.64 / 3 (11 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0076 | 9.8% | 64.3% | 7.40 | 1.55 | 35.7% |
| 2 | 0.0079 | 10.2% | 62.7% | 6.89 | 1.46 | 37.3% |
| 3 | 0.0075 | 9.6% | 57.1% | 6.45 | 1.37 | 42.9% |
| 4 | 0.0073 | 9.5% | 54.2% | 6.02 | 1.35 | 45.8% |
| 5 | 0.0084 | 10.8% | 52.4% | 5.68 | 1.32 | 47.6% |
| 6 | 0.0072 | 9.3% | 47.6% | 5.39 | 1.26 | 52.4% |
| 7 | 0.0071 | 9.1% | 44.0% | 5.19 | 1.28 | 56.0% |
| 8 | 0.0065 | 8.4% | 42.7% | 4.95 | 1.25 | 57.3% |
| 9 | 0.0061 | 7.9% | 37.8% | 4.68 | 1.19 | 62.2% |
| 10 | 0.0051 | 6.5% | 33.2% | 4.39 | 1.15 | 66.8% |
| 11 | 0.0036 | 4.6% | 26.7% | 4.02 | 1.12 | 73.3% |
| 12 | 0.0019 | 2.5% | 19.8% | 3.81 | 1.12 | 80.2% |
| 13 | 0.0010 | 1.3% | 13.4% | 3.64 | 1.12 | 86.6% |
| 14 | 0.0004 | 0.5% | 9.3% | 3.38 | 1.12 | 90.7% |
| 15 | 0.0001 | 0.1% | 6.4% | 3.14 | 1.06 | 93.6% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0577 |
| PPO regret per step: mean / median | 0.0058 / 0.0000 |
| Share of steps where PPO picks an optimal action | 69.8% |
| Legal actions / optimal actions per step (mean) | 6.12 / 1.25 |
| Share of steps with a unique optimal action | 79.8% |
| Steps with regret per instance (mean) | 3.02 |
| Share of the gap from steps 1–3 | 55.9% |
| Share of the gap from steps 4–6 | 35.7% |
| Share of the gap from steps 7–10 | 8.4% |
| Regret at steps followed by a forced opening: share of regret | 87.7% |
| Regret at steps that are themselves forced openings: share of regret | 56.9% |
| Steps from a regret step to the next forced opening (mean) | 1.49 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0096 | 16.7% | 55.5% | 7.39 | 1.42 | 44.5% |
| 2 | 0.0120 | 20.8% | 52.5% | 6.94 | 1.43 | 47.5% |
| 3 | 0.0106 | 18.4% | 48.2% | 6.63 | 1.31 | 51.7% |
| 4 | 0.0081 | 14.0% | 42.5% | 6.38 | 1.36 | 57.5% |
| 5 | 0.0074 | 12.8% | 36.8% | 6.17 | 1.28 | 63.2% |
| 6 | 0.0051 | 8.8% | 27.5% | 5.93 | 1.20 | 72.5% |
| 7 | 0.0033 | 5.8% | 19.5% | 5.75 | 1.17 | 80.5% |
| 8 | 0.0013 | 2.2% | 12.2% | 5.57 | 1.13 | 87.8% |
| 9 | 0.0002 | 0.4% | 6.2% | 5.28 | 1.09 | 93.8% |
| 10 | 0.0000 | 0.0% | 1.2% | 5.12 | 1.06 | 98.8% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0596 |
| PPO regret per step: mean / median | 0.0060 / 0.0000 |
| Share of steps where PPO picks an optimal action | 71.7% |
| Legal actions / optimal actions per step (mean) | 10.43 / 1.32 |
| Share of steps with a unique optimal action | 75.1% |
| Steps with regret per instance (mean) | 2.83 |
| Share of the gap from steps 1–3 | 51.5% |
| Share of the gap from steps 4–6 | 38.3% |
| Share of the gap from steps 7–10 | 10.2% |
| Regret at steps followed by a forced opening: share of regret | 90.0% |
| Regret at steps that are themselves forced openings: share of regret | 67.0% |
| Steps from a regret step to the next forced opening (mean) | 1.26 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0104 | 17.5% | 50.0% | 11.00 | 1.57 | 50.0% |
| 2 | 0.0106 | 17.8% | 50.7% | 10.80 | 1.50 | 49.2% |
| 3 | 0.0096 | 16.2% | 42.5% | 10.75 | 1.46 | 57.5% |
| 4 | 0.0104 | 17.4% | 43.2% | 10.68 | 1.41 | 56.8% |
| 5 | 0.0074 | 12.5% | 37.0% | 10.54 | 1.38 | 63.0% |
| 6 | 0.0050 | 8.4% | 24.2% | 10.38 | 1.29 | 75.8% |
| 7 | 0.0043 | 7.3% | 19.8% | 10.26 | 1.22 | 80.2% |
| 8 | 0.0014 | 2.3% | 9.8% | 10.09 | 1.19 | 90.2% |
| 9 | 0.0004 | 0.6% | 4.8% | 9.93 | 1.11 | 95.2% |
| 10 | 0.0000 | 0.0% | 1.0% | 9.84 | 1.04 | 99.0% |
