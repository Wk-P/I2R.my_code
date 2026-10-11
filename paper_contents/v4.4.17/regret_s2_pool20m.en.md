# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-11 19:28:50; script `src/scripts/diag_regret.py`; took 9.4 min.
- Model: v4.4.17 structure-aware Mask PPO, 50k instances, 20M (seed 2, 20M, GPU), replayed deterministically on the first 400 instances of the seed-2 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

392 completed instances, 8 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 6.1e-04 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0645 |
| PPO regret per step: mean / median | 0.0043 / 0.0000 |
| Share of steps where PPO picks an optimal action | 67.3% |
| Legal actions / optimal actions per step (mean) | 4.91 / 1.25 |
| Share of steps with a unique optimal action | 79.8% |
| Steps with regret per instance (mean) | 4.91 |
| Share of the gap from steps 1–5 | 53.0% |
| Share of the gap from steps 6–10 | 40.2% |
| Share of the gap from steps 11–15 | 6.7% |
| Regret at steps followed by a forced opening: share of regret | 79.3% |
| Regret at steps that are themselves forced openings: share of regret | 44.0% |
| Steps from a regret step to the next forced opening (mean) | 1.94 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 3.50 / 3 (8 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0073 | 11.4% | 59.7% | 7.21 | 1.53 | 40.3% |
| 2 | 0.0072 | 11.1% | 54.6% | 6.66 | 1.49 | 45.4% |
| 3 | 0.0062 | 9.6% | 54.1% | 6.27 | 1.36 | 45.9% |
| 4 | 0.0066 | 10.3% | 49.7% | 5.92 | 1.36 | 50.3% |
| 5 | 0.0069 | 10.7% | 51.5% | 5.59 | 1.30 | 48.5% |
| 6 | 0.0059 | 9.1% | 39.8% | 5.38 | 1.30 | 60.2% |
| 7 | 0.0058 | 8.9% | 42.3% | 5.09 | 1.26 | 57.7% |
| 8 | 0.0060 | 9.2% | 37.5% | 4.85 | 1.24 | 62.5% |
| 9 | 0.0047 | 7.3% | 31.4% | 4.55 | 1.19 | 68.6% |
| 10 | 0.0037 | 5.7% | 22.7% | 4.26 | 1.20 | 77.3% |
| 11 | 0.0019 | 2.9% | 17.6% | 4.02 | 1.13 | 82.4% |
| 12 | 0.0015 | 2.4% | 12.8% | 3.77 | 1.13 | 87.2% |
| 13 | 0.0007 | 1.0% | 9.9% | 3.49 | 1.11 | 90.1% |
| 14 | 0.0002 | 0.4% | 6.6% | 3.39 | 1.08 | 93.4% |
| 15 | 0.0000 | 0.0% | 0.8% | 3.22 | 1.02 | 99.2% |

## EQ (M = 10)

399 completed instances, 1 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0397 |
| PPO regret per step: mean / median | 0.0040 / 0.0000 |
| Share of steps where PPO picks an optimal action | 76.7% |
| Legal actions / optimal actions per step (mean) | 6.16 / 1.22 |
| Share of steps with a unique optimal action | 81.2% |
| Steps with regret per instance (mean) | 2.33 |
| Share of the gap from steps 1–3 | 60.5% |
| Share of the gap from steps 4–6 | 33.5% |
| Share of the gap from steps 7–10 | 6.0% |
| Regret at steps followed by a forced opening: share of regret | 86.7% |
| Regret at steps that are themselves forced openings: share of regret | 60.0% |
| Steps from a regret step to the next forced opening (mean) | 1.42 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 3.00 / 3 (1 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0075 | 18.9% | 44.4% | 7.34 | 1.39 | 55.6% |
| 2 | 0.0086 | 21.6% | 44.6% | 6.99 | 1.34 | 55.4% |
| 3 | 0.0080 | 20.1% | 40.9% | 6.71 | 1.36 | 59.1% |
| 4 | 0.0057 | 14.3% | 32.1% | 6.55 | 1.35 | 67.9% |
| 5 | 0.0042 | 10.5% | 23.8% | 6.30 | 1.23 | 76.2% |
| 6 | 0.0035 | 8.7% | 21.8% | 5.98 | 1.20 | 78.2% |
| 7 | 0.0018 | 4.4% | 12.0% | 5.78 | 1.16 | 88.0% |
| 8 | 0.0004 | 0.9% | 7.3% | 5.49 | 1.12 | 92.7% |
| 9 | 0.0002 | 0.5% | 3.8% | 5.33 | 1.08 | 96.2% |
| 10 | 0.0000 | 0.1% | 2.0% | 5.18 | 1.03 | 98.0% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0411 |
| PPO regret per step: mean / median | 0.0041 / 0.0000 |
| Share of steps where PPO picks an optimal action | 76.7% |
| Legal actions / optimal actions per step (mean) | 10.51 / 1.28 |
| Share of steps with a unique optimal action | 77.3% |
| Steps with regret per instance (mean) | 2.33 |
| Share of the gap from steps 1–3 | 58.9% |
| Share of the gap from steps 4–6 | 34.5% |
| Share of the gap from steps 7–10 | 6.6% |
| Regret at steps followed by a forced opening: share of regret | 87.1% |
| Regret at steps that are themselves forced openings: share of regret | 62.9% |
| Steps from a regret step to the next forced opening (mean) | 1.31 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0084 | 20.5% | 47.0% | 10.93 | 1.54 | 53.0% |
| 2 | 0.0077 | 18.8% | 43.0% | 10.78 | 1.47 | 57.0% |
| 3 | 0.0080 | 19.6% | 41.5% | 10.74 | 1.41 | 58.5% |
| 4 | 0.0065 | 15.8% | 35.2% | 10.70 | 1.41 | 64.8% |
| 5 | 0.0050 | 12.3% | 28.0% | 10.69 | 1.36 | 72.0% |
| 6 | 0.0027 | 6.5% | 17.8% | 10.53 | 1.24 | 82.2% |
| 7 | 0.0016 | 4.0% | 10.2% | 10.37 | 1.19 | 89.8% |
| 8 | 0.0009 | 2.1% | 5.5% | 10.22 | 1.11 | 94.5% |
| 9 | 0.0002 | 0.4% | 3.8% | 10.14 | 1.09 | 96.2% |
| 10 | 0.0000 | 0.0% | 1.0% | 9.96 | 1.03 | 99.0% |
