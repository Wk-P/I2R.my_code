# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-11 05:37:39; script `src/scripts/diag_regret.py`; took 9.9 min.
- Model: v4.4.16 structure-aware Mask PPO, 50k instances, 10M (seed 1, 10M, GPU), replayed deterministically on the first 400 instances of the seed-1 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

391 completed instances, 9 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0694 |
| PPO regret per step: mean / median | 0.0046 / 0.0000 |
| Share of steps where PPO picks an optimal action | 65.4% |
| Legal actions / optimal actions per step (mean) | 4.92 / 1.24 |
| Share of steps with a unique optimal action | 79.6% |
| Steps with regret per instance (mean) | 5.18 |
| Share of the gap from steps 1–5 | 52.3% |
| Share of the gap from steps 6–10 | 41.4% |
| Share of the gap from steps 11–15 | 6.3% |
| Regret at steps followed by a forced opening: share of regret | 85.5% |
| Regret at steps that are themselves forced openings: share of regret | 43.7% |
| Steps from a regret step to the next forced opening (mean) | 1.97 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 2.44 / 1 (9 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0072 | 10.4% | 62.1% | 7.38 | 1.55 | 37.9% |
| 2 | 0.0067 | 9.6% | 62.4% | 6.82 | 1.44 | 37.6% |
| 3 | 0.0071 | 10.2% | 52.7% | 6.39 | 1.36 | 47.3% |
| 4 | 0.0079 | 11.4% | 55.0% | 5.94 | 1.37 | 45.0% |
| 5 | 0.0074 | 10.6% | 49.4% | 5.60 | 1.34 | 50.6% |
| 6 | 0.0071 | 10.3% | 45.5% | 5.31 | 1.26 | 54.5% |
| 7 | 0.0069 | 9.9% | 40.9% | 5.06 | 1.28 | 59.1% |
| 8 | 0.0058 | 8.4% | 36.6% | 4.81 | 1.23 | 63.4% |
| 9 | 0.0050 | 7.2% | 34.0% | 4.53 | 1.15 | 66.0% |
| 10 | 0.0039 | 5.6% | 29.2% | 4.22 | 1.18 | 70.8% |
| 11 | 0.0021 | 3.0% | 19.9% | 3.92 | 1.10 | 80.1% |
| 12 | 0.0008 | 1.2% | 10.5% | 3.69 | 1.12 | 89.5% |
| 13 | 0.0012 | 1.7% | 12.0% | 3.58 | 1.14 | 88.0% |
| 14 | 0.0003 | 0.4% | 6.1% | 3.38 | 1.08 | 93.9% |
| 15 | 0.0000 | 0.0% | 2.0% | 3.18 | 1.04 | 98.0% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0541 |
| PPO regret per step: mean / median | 0.0054 / 0.0000 |
| Share of steps where PPO picks an optimal action | 71.0% |
| Legal actions / optimal actions per step (mean) | 6.12 / 1.25 |
| Share of steps with a unique optimal action | 79.4% |
| Steps with regret per instance (mean) | 2.90 |
| Share of the gap from steps 1–3 | 54.1% |
| Share of the gap from steps 4–6 | 38.2% |
| Share of the gap from steps 7–10 | 7.7% |
| Regret at steps followed by a forced opening: share of regret | 85.3% |
| Regret at steps that are themselves forced openings: share of regret | 56.3% |
| Steps from a regret step to the next forced opening (mean) | 1.48 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0097 | 18.0% | 55.5% | 7.39 | 1.42 | 44.5% |
| 2 | 0.0100 | 18.5% | 55.2% | 6.94 | 1.44 | 44.8% |
| 3 | 0.0095 | 17.6% | 42.5% | 6.64 | 1.34 | 57.5% |
| 4 | 0.0084 | 15.5% | 40.5% | 6.39 | 1.37 | 59.5% |
| 5 | 0.0065 | 12.1% | 32.8% | 6.16 | 1.26 | 67.2% |
| 6 | 0.0057 | 10.6% | 28.5% | 5.96 | 1.21 | 71.5% |
| 7 | 0.0028 | 5.2% | 19.0% | 5.75 | 1.16 | 81.0% |
| 8 | 0.0011 | 2.0% | 9.2% | 5.53 | 1.15 | 90.8% |
| 9 | 0.0003 | 0.5% | 5.8% | 5.25 | 1.09 | 94.2% |
| 10 | 0.0000 | 0.0% | 1.0% | 5.15 | 1.05 | 99.0% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0477 |
| PPO regret per step: mean / median | 0.0048 / 0.0000 |
| Share of steps where PPO picks an optimal action | 73.6% |
| Legal actions / optimal actions per step (mean) | 10.50 / 1.30 |
| Share of steps with a unique optimal action | 76.7% |
| Steps with regret per instance (mean) | 2.64 |
| Share of the gap from steps 1–3 | 51.7% |
| Share of the gap from steps 4–6 | 40.1% |
| Share of the gap from steps 7–10 | 8.2% |
| Regret at steps followed by a forced opening: share of regret | 87.4% |
| Regret at steps that are themselves forced openings: share of regret | 62.7% |
| Steps from a regret step to the next forced opening (mean) | 1.36 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0089 | 18.7% | 51.5% | 11.00 | 1.57 | 48.5% |
| 2 | 0.0082 | 17.1% | 45.0% | 10.83 | 1.48 | 55.0% |
| 3 | 0.0075 | 15.8% | 40.2% | 10.80 | 1.43 | 59.8% |
| 4 | 0.0082 | 17.3% | 42.2% | 10.77 | 1.39 | 57.8% |
| 5 | 0.0061 | 12.8% | 33.2% | 10.64 | 1.34 | 66.8% |
| 6 | 0.0048 | 10.0% | 26.0% | 10.51 | 1.24 | 74.0% |
| 7 | 0.0026 | 5.5% | 16.2% | 10.35 | 1.19 | 83.8% |
| 8 | 0.0010 | 2.1% | 6.5% | 10.21 | 1.18 | 93.5% |
| 9 | 0.0003 | 0.6% | 2.8% | 10.04 | 1.09 | 97.2% |
| 10 | 0.0000 | 0.0% | 0.2% | 9.90 | 1.04 | 99.8% |
