# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-10 17:53:43; script `src/scripts/diag_regret.py`; took 9.5 min.
- Model: v4.4.15 structure-aware Mask PPO, 50k instances, 5M (seed 2, 5M, GPU), replayed deterministically on the first 400 instances of the seed-2 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

384 completed instances, 16 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 1.3e-02 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0750 |
| PPO regret per step: mean / median | 0.0050 / 0.0000 |
| Share of steps where PPO picks an optimal action | 63.1% |
| Legal actions / optimal actions per step (mean) | 4.94 / 1.24 |
| Share of steps with a unique optimal action | 79.9% |
| Steps with regret per instance (mean) | 5.53 |
| Share of the gap from steps 1–5 | 49.6% |
| Share of the gap from steps 6–10 | 38.2% |
| Share of the gap from steps 11–15 | 12.2% |
| Regret at steps followed by a forced opening: share of regret | 78.8% |
| Regret at steps that are themselves forced openings: share of regret | 42.6% |
| Steps from a regret step to the next forced opening (mean) | 1.94 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 4.12 / 4 (16 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0073 | 9.8% | 64.6% | 7.22 | 1.53 | 35.4% |
| 2 | 0.0088 | 11.8% | 61.2% | 6.66 | 1.45 | 38.8% |
| 3 | 0.0070 | 9.3% | 59.4% | 6.24 | 1.38 | 40.6% |
| 4 | 0.0079 | 10.6% | 53.4% | 5.87 | 1.35 | 46.6% |
| 5 | 0.0061 | 8.1% | 49.5% | 5.57 | 1.32 | 50.5% |
| 6 | 0.0061 | 8.2% | 46.6% | 5.36 | 1.28 | 53.4% |
| 7 | 0.0066 | 8.8% | 40.6% | 5.11 | 1.23 | 59.4% |
| 8 | 0.0055 | 7.3% | 43.2% | 4.89 | 1.28 | 56.8% |
| 9 | 0.0060 | 8.0% | 38.3% | 4.61 | 1.16 | 61.7% |
| 10 | 0.0044 | 5.9% | 29.4% | 4.36 | 1.21 | 70.6% |
| 11 | 0.0047 | 6.3% | 27.9% | 4.18 | 1.13 | 72.1% |
| 12 | 0.0023 | 3.1% | 18.8% | 3.89 | 1.12 | 81.2% |
| 13 | 0.0017 | 2.3% | 11.7% | 3.56 | 1.09 | 88.3% |
| 14 | 0.0003 | 0.4% | 6.8% | 3.42 | 1.09 | 93.2% |
| 15 | 0.0001 | 0.1% | 1.8% | 3.23 | 1.03 | 98.2% |

## EQ (M = 10)

399 completed instances, 1 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0581 |
| PPO regret per step: mean / median | 0.0058 / 0.0000 |
| Share of steps where PPO picks an optimal action | 71.7% |
| Legal actions / optimal actions per step (mean) | 6.13 / 1.23 |
| Share of steps with a unique optimal action | 80.6% |
| Steps with regret per instance (mean) | 2.83 |
| Share of the gap from steps 1–3 | 48.7% |
| Share of the gap from steps 4–6 | 43.0% |
| Share of the gap from steps 7–10 | 8.3% |
| Regret at steps followed by a forced opening: share of regret | 83.6% |
| Regret at steps that are themselves forced openings: share of regret | 55.6% |
| Steps from a regret step to the next forced opening (mean) | 1.50 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 3.00 / 3 (1 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0093 | 15.9% | 51.1% | 7.34 | 1.39 | 48.9% |
| 2 | 0.0119 | 20.4% | 50.9% | 6.96 | 1.36 | 49.1% |
| 3 | 0.0072 | 12.4% | 40.9% | 6.68 | 1.36 | 59.1% |
| 4 | 0.0095 | 16.3% | 42.1% | 6.53 | 1.30 | 57.9% |
| 5 | 0.0067 | 11.5% | 34.3% | 6.26 | 1.26 | 65.7% |
| 6 | 0.0089 | 15.2% | 30.1% | 5.97 | 1.23 | 69.9% |
| 7 | 0.0031 | 5.3% | 18.8% | 5.75 | 1.20 | 81.2% |
| 8 | 0.0016 | 2.7% | 10.8% | 5.48 | 1.11 | 89.2% |
| 9 | 0.0001 | 0.2% | 3.0% | 5.28 | 1.08 | 97.0% |
| 10 | 0.0000 | 0.0% | 1.3% | 5.08 | 1.04 | 98.7% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0583 |
| PPO regret per step: mean / median | 0.0058 / 0.0000 |
| Share of steps where PPO picks an optimal action | 69.7% |
| Legal actions / optimal actions per step (mean) | 10.40 / 1.31 |
| Share of steps with a unique optimal action | 75.1% |
| Steps with regret per instance (mean) | 3.03 |
| Share of the gap from steps 1–3 | 55.8% |
| Share of the gap from steps 4–6 | 37.2% |
| Share of the gap from steps 7–10 | 6.9% |
| Regret at steps followed by a forced opening: share of regret | 88.0% |
| Regret at steps that are themselves forced openings: share of regret | 64.7% |
| Steps from a regret step to the next forced opening (mean) | 1.30 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0107 | 18.3% | 56.5% | 10.93 | 1.54 | 43.5% |
| 2 | 0.0105 | 18.0% | 55.0% | 10.75 | 1.46 | 45.0% |
| 3 | 0.0114 | 19.5% | 53.5% | 10.68 | 1.46 | 46.5% |
| 4 | 0.0083 | 14.2% | 45.2% | 10.63 | 1.41 | 54.8% |
| 5 | 0.0070 | 12.0% | 36.8% | 10.58 | 1.40 | 63.2% |
| 6 | 0.0064 | 11.0% | 26.5% | 10.39 | 1.30 | 73.5% |
| 7 | 0.0026 | 4.5% | 17.2% | 10.22 | 1.23 | 82.8% |
| 8 | 0.0012 | 2.0% | 9.2% | 10.06 | 1.15 | 90.8% |
| 9 | 0.0002 | 0.4% | 2.8% | 9.97 | 1.11 | 97.2% |
| 10 | 0.0000 | 0.0% | 0.2% | 9.75 | 1.03 | 99.8% |
