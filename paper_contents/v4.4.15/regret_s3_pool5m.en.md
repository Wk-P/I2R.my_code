# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-10 18:03:47; script `src/scripts/diag_regret.py`; took 10.1 min.
- Model: v4.4.15 structure-aware Mask PPO, 50k instances, 5M (seed 3, 5M, GPU), replayed deterministically on the first 400 instances of the seed-3 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

392 completed instances, 8 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 7.4e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0765 |
| PPO regret per step: mean / median | 0.0051 / 0.0000 |
| Share of steps where PPO picks an optimal action | 62.6% |
| Legal actions / optimal actions per step (mean) | 4.94 / 1.23 |
| Share of steps with a unique optimal action | 81.3% |
| Steps with regret per instance (mean) | 5.62 |
| Share of the gap from steps 1–5 | 48.1% |
| Share of the gap from steps 6–10 | 42.2% |
| Share of the gap from steps 11–15 | 9.6% |
| Regret at steps followed by a forced opening: share of regret | 81.4% |
| Regret at steps that are themselves forced openings: share of regret | 41.4% |
| Steps from a regret step to the next forced opening (mean) | 1.93 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 3.75 / 4 (8 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0076 | 9.9% | 64.5% | 7.41 | 1.46 | 35.5% |
| 2 | 0.0080 | 10.5% | 64.8% | 6.85 | 1.47 | 35.2% |
| 3 | 0.0064 | 8.4% | 57.7% | 6.38 | 1.39 | 42.3% |
| 4 | 0.0075 | 9.8% | 53.3% | 5.97 | 1.33 | 46.7% |
| 5 | 0.0073 | 9.5% | 47.2% | 5.56 | 1.29 | 52.8% |
| 6 | 0.0061 | 8.0% | 46.9% | 5.35 | 1.24 | 53.1% |
| 7 | 0.0081 | 10.5% | 44.6% | 5.02 | 1.24 | 55.4% |
| 8 | 0.0071 | 9.3% | 40.3% | 4.77 | 1.23 | 59.7% |
| 9 | 0.0060 | 7.9% | 38.3% | 4.50 | 1.16 | 61.7% |
| 10 | 0.0051 | 6.6% | 34.7% | 4.31 | 1.16 | 65.3% |
| 11 | 0.0031 | 4.1% | 26.8% | 4.08 | 1.13 | 73.2% |
| 12 | 0.0025 | 3.2% | 19.9% | 3.75 | 1.12 | 80.1% |
| 13 | 0.0010 | 1.3% | 12.0% | 3.60 | 1.08 | 88.0% |
| 14 | 0.0007 | 0.9% | 6.4% | 3.37 | 1.09 | 93.6% |
| 15 | 0.0001 | 0.1% | 4.3% | 3.20 | 1.02 | 95.7% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0600 |
| PPO regret per step: mean / median | 0.0060 / 0.0000 |
| Share of steps where PPO picks an optimal action | 70.7% |
| Legal actions / optimal actions per step (mean) | 6.18 / 1.21 |
| Share of steps with a unique optimal action | 81.5% |
| Steps with regret per instance (mean) | 2.93 |
| Share of the gap from steps 1–3 | 52.0% |
| Share of the gap from steps 4–6 | 37.5% |
| Share of the gap from steps 7–10 | 10.5% |
| Regret at steps followed by a forced opening: share of regret | 81.9% |
| Regret at steps that are themselves forced openings: share of regret | 54.0% |
| Steps from a regret step to the next forced opening (mean) | 1.49 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0099 | 16.4% | 58.2% | 7.41 | 1.43 | 41.8% |
| 2 | 0.0102 | 17.0% | 53.0% | 7.04 | 1.35 | 47.0% |
| 3 | 0.0111 | 18.5% | 47.0% | 6.70 | 1.33 | 53.0% |
| 4 | 0.0091 | 15.1% | 41.2% | 6.40 | 1.27 | 58.8% |
| 5 | 0.0076 | 12.7% | 33.0% | 6.25 | 1.20 | 67.0% |
| 6 | 0.0058 | 9.7% | 27.0% | 6.04 | 1.17 | 73.0% |
| 7 | 0.0044 | 7.4% | 17.8% | 5.77 | 1.11 | 82.2% |
| 8 | 0.0016 | 2.6% | 11.5% | 5.60 | 1.11 | 88.5% |
| 9 | 0.0003 | 0.5% | 3.0% | 5.38 | 1.11 | 97.0% |
| 10 | 0.0000 | 0.0% | 1.0% | 5.17 | 1.06 | 99.0% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0631 |
| PPO regret per step: mean / median | 0.0063 / 0.0000 |
| Share of steps where PPO picks an optimal action | 67.8% |
| Legal actions / optimal actions per step (mean) | 10.36 / 1.33 |
| Share of steps with a unique optimal action | 74.8% |
| Steps with regret per instance (mean) | 3.22 |
| Share of the gap from steps 1–3 | 50.8% |
| Share of the gap from steps 4–6 | 36.7% |
| Share of the gap from steps 7–10 | 12.5% |
| Regret at steps followed by a forced opening: share of regret | 85.0% |
| Regret at steps that are themselves forced openings: share of regret | 61.9% |
| Steps from a regret step to the next forced opening (mean) | 1.25 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0124 | 19.7% | 55.8% | 10.91 | 1.51 | 44.2% |
| 2 | 0.0114 | 18.1% | 55.8% | 10.77 | 1.52 | 44.2% |
| 3 | 0.0082 | 13.0% | 53.5% | 10.56 | 1.44 | 46.5% |
| 4 | 0.0094 | 14.8% | 45.0% | 10.62 | 1.46 | 55.0% |
| 5 | 0.0074 | 11.7% | 38.5% | 10.53 | 1.41 | 61.5% |
| 6 | 0.0064 | 10.2% | 32.0% | 10.41 | 1.33 | 68.0% |
| 7 | 0.0046 | 7.3% | 23.0% | 10.28 | 1.28 | 77.0% |
| 8 | 0.0025 | 4.0% | 13.5% | 10.05 | 1.17 | 86.5% |
| 9 | 0.0007 | 1.1% | 4.2% | 9.83 | 1.13 | 95.8% |
| 10 | 0.0001 | 0.1% | 1.0% | 9.64 | 1.05 | 99.0% |
