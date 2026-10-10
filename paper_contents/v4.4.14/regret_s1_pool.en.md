# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-10 13:08:17; script `src/scripts/diag_regret.py`; took 10.0 min.
- Model: v4.4.14 structure-aware Mask PPO, 50k training instances (seed 1, 1M, GPU), replayed deterministically on the first 400 instances of the seed-1 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

387 completed instances, 13 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0908 |
| PPO regret per step: mean / median | 0.0061 / 0.0000 |
| Share of steps where PPO picks an optimal action | 59.7% |
| Legal actions / optimal actions per step (mean) | 5.00 / 1.26 |
| Share of steps with a unique optimal action | 78.4% |
| Steps with regret per instance (mean) | 6.04 |
| Share of the gap from steps 1–5 | 48.8% |
| Share of the gap from steps 6–10 | 39.7% |
| Share of the gap from steps 11–15 | 11.4% |
| Regret at steps followed by a forced opening: share of regret | 84.7% |
| Regret at steps that are themselves forced openings: share of regret | 41.6% |
| Steps from a regret step to the next forced opening (mean) | 1.93 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 2.85 / 2 (13 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0105 | 11.5% | 69.5% | 7.41 | 1.55 | 30.5% |
| 2 | 0.0082 | 9.0% | 67.2% | 6.84 | 1.47 | 32.8% |
| 3 | 0.0082 | 9.0% | 58.1% | 6.38 | 1.37 | 41.9% |
| 4 | 0.0097 | 10.7% | 57.9% | 5.91 | 1.43 | 42.1% |
| 5 | 0.0078 | 8.6% | 49.4% | 5.56 | 1.31 | 50.6% |
| 6 | 0.0075 | 8.3% | 49.4% | 5.31 | 1.26 | 50.6% |
| 7 | 0.0095 | 10.4% | 48.3% | 5.23 | 1.24 | 51.7% |
| 8 | 0.0067 | 7.4% | 46.0% | 4.91 | 1.30 | 54.0% |
| 9 | 0.0067 | 7.4% | 41.1% | 4.72 | 1.17 | 58.9% |
| 10 | 0.0055 | 6.1% | 32.6% | 4.51 | 1.18 | 67.4% |
| 11 | 0.0049 | 5.4% | 26.9% | 4.12 | 1.15 | 73.1% |
| 12 | 0.0031 | 3.4% | 27.1% | 3.83 | 1.13 | 72.9% |
| 13 | 0.0017 | 1.9% | 16.0% | 3.65 | 1.16 | 84.0% |
| 14 | 0.0006 | 0.7% | 10.6% | 3.41 | 1.12 | 89.4% |
| 15 | 0.0001 | 0.1% | 3.9% | 3.15 | 1.05 | 96.1% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0795 |
| PPO regret per step: mean / median | 0.0079 / 0.0000 |
| Share of steps where PPO picks an optimal action | 64.8% |
| Legal actions / optimal actions per step (mean) | 6.06 / 1.26 |
| Share of steps with a unique optimal action | 78.1% |
| Steps with regret per instance (mean) | 3.52 |
| Share of the gap from steps 1–3 | 51.3% |
| Share of the gap from steps 4–6 | 36.0% |
| Share of the gap from steps 7–10 | 12.7% |
| Regret at steps followed by a forced opening: share of regret | 89.5% |
| Regret at steps that are themselves forced openings: share of regret | 53.8% |
| Steps from a regret step to the next forced opening (mean) | 1.46 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0145 | 18.2% | 64.0% | 7.39 | 1.42 | 36.0% |
| 2 | 0.0143 | 17.9% | 59.8% | 6.92 | 1.45 | 40.2% |
| 3 | 0.0120 | 15.1% | 50.7% | 6.61 | 1.31 | 49.2% |
| 4 | 0.0124 | 15.6% | 49.2% | 6.34 | 1.37 | 50.7% |
| 5 | 0.0097 | 12.2% | 40.8% | 6.08 | 1.29 | 59.2% |
| 6 | 0.0066 | 8.3% | 35.8% | 5.95 | 1.22 | 64.2% |
| 7 | 0.0053 | 6.7% | 25.2% | 5.68 | 1.22 | 74.8% |
| 8 | 0.0038 | 4.7% | 17.2% | 5.49 | 1.14 | 82.8% |
| 9 | 0.0010 | 1.2% | 8.8% | 5.16 | 1.15 | 91.2% |
| 10 | 0.0000 | 0.0% | 0.8% | 4.95 | 1.09 | 99.2% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0765 |
| PPO regret per step: mean / median | 0.0077 / 0.0000 |
| Share of steps where PPO picks an optimal action | 65.4% |
| Legal actions / optimal actions per step (mean) | 10.35 / 1.34 |
| Share of steps with a unique optimal action | 74.1% |
| Steps with regret per instance (mean) | 3.46 |
| Share of the gap from steps 1–3 | 50.5% |
| Share of the gap from steps 4–6 | 36.3% |
| Share of the gap from steps 7–10 | 13.1% |
| Regret at steps followed by a forced opening: share of regret | 87.6% |
| Regret at steps that are themselves forced openings: share of regret | 63.8% |
| Steps from a regret step to the next forced opening (mean) | 1.25 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0132 | 17.3% | 56.2% | 11.00 | 1.57 | 43.8% |
| 2 | 0.0140 | 18.3% | 57.2% | 10.80 | 1.50 | 42.8% |
| 3 | 0.0115 | 15.0% | 50.7% | 10.73 | 1.47 | 49.2% |
| 4 | 0.0112 | 14.7% | 48.2% | 10.63 | 1.44 | 51.7% |
| 5 | 0.0089 | 11.7% | 43.2% | 10.46 | 1.41 | 56.8% |
| 6 | 0.0076 | 10.0% | 36.2% | 10.32 | 1.28 | 63.7% |
| 7 | 0.0050 | 6.6% | 23.8% | 10.14 | 1.30 | 76.2% |
| 8 | 0.0043 | 5.6% | 21.0% | 10.03 | 1.19 | 79.0% |
| 9 | 0.0007 | 1.0% | 7.8% | 9.80 | 1.14 | 92.2% |
| 10 | 0.0000 | 0.0% | 1.5% | 9.58 | 1.07 | 98.5% |
