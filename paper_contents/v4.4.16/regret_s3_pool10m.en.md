# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-11 05:57:24; script `src/scripts/diag_regret.py`; took 10.1 min.
- Model: v4.4.16 structure-aware Mask PPO, 50k instances, 10M (seed 3, 10M, GPU), replayed deterministically on the first 400 instances of the seed-3 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

394 completed instances, 6 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 7.4e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0648 |
| PPO regret per step: mean / median | 0.0043 / 0.0000 |
| Share of steps where PPO picks an optimal action | 66.6% |
| Legal actions / optimal actions per step (mean) | 4.91 / 1.22 |
| Share of steps with a unique optimal action | 81.8% |
| Steps with regret per instance (mean) | 5.02 |
| Share of the gap from steps 1–5 | 53.1% |
| Share of the gap from steps 6–10 | 40.0% |
| Share of the gap from steps 11–15 | 6.9% |
| Regret at steps followed by a forced opening: share of regret | 83.1% |
| Regret at steps that are themselves forced openings: share of regret | 45.0% |
| Steps from a regret step to the next forced opening (mean) | 1.86 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 4.67 / 4 (6 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0071 | 10.9% | 59.1% | 7.40 | 1.46 | 40.9% |
| 2 | 0.0088 | 13.6% | 59.9% | 6.83 | 1.46 | 40.1% |
| 3 | 0.0075 | 11.6% | 57.4% | 6.32 | 1.36 | 42.6% |
| 4 | 0.0053 | 8.2% | 49.0% | 5.89 | 1.33 | 51.0% |
| 5 | 0.0057 | 8.8% | 44.4% | 5.55 | 1.27 | 55.6% |
| 6 | 0.0060 | 9.3% | 39.8% | 5.30 | 1.25 | 60.2% |
| 7 | 0.0053 | 8.1% | 38.3% | 5.01 | 1.22 | 61.7% |
| 8 | 0.0059 | 9.2% | 39.8% | 4.75 | 1.20 | 60.2% |
| 9 | 0.0045 | 6.9% | 31.5% | 4.48 | 1.19 | 68.5% |
| 10 | 0.0042 | 6.5% | 28.2% | 4.29 | 1.17 | 71.8% |
| 11 | 0.0024 | 3.7% | 23.9% | 4.05 | 1.10 | 76.1% |
| 12 | 0.0015 | 2.3% | 16.0% | 3.75 | 1.10 | 84.0% |
| 13 | 0.0003 | 0.5% | 6.6% | 3.52 | 1.09 | 93.4% |
| 14 | 0.0002 | 0.2% | 3.8% | 3.31 | 1.07 | 96.2% |
| 15 | 0.0001 | 0.1% | 3.8% | 3.24 | 1.05 | 96.2% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0520 |
| PPO regret per step: mean / median | 0.0052 / 0.0000 |
| Share of steps where PPO picks an optimal action | 72.7% |
| Legal actions / optimal actions per step (mean) | 6.12 / 1.22 |
| Share of steps with a unique optimal action | 81.1% |
| Steps with regret per instance (mean) | 2.73 |
| Share of the gap from steps 1–3 | 53.6% |
| Share of the gap from steps 4–6 | 39.0% |
| Share of the gap from steps 7–10 | 7.5% |
| Regret at steps followed by a forced opening: share of regret | 85.8% |
| Regret at steps that are themselves forced openings: share of regret | 58.0% |
| Steps from a regret step to the next forced opening (mean) | 1.40 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0101 | 19.4% | 55.8% | 7.41 | 1.43 | 44.2% |
| 2 | 0.0095 | 18.3% | 49.5% | 7.03 | 1.39 | 50.5% |
| 3 | 0.0082 | 15.8% | 44.2% | 6.68 | 1.34 | 55.8% |
| 4 | 0.0082 | 15.7% | 36.2% | 6.38 | 1.28 | 63.7% |
| 5 | 0.0075 | 14.4% | 32.5% | 6.19 | 1.23 | 67.5% |
| 6 | 0.0046 | 8.9% | 22.8% | 5.96 | 1.19 | 77.2% |
| 7 | 0.0023 | 4.5% | 16.5% | 5.67 | 1.14 | 83.5% |
| 8 | 0.0012 | 2.3% | 11.2% | 5.50 | 1.11 | 88.8% |
| 9 | 0.0003 | 0.7% | 4.2% | 5.29 | 1.10 | 95.8% |
| 10 | 0.0000 | 0.0% | 0.2% | 5.09 | 1.04 | 99.8% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0516 |
| PPO regret per step: mean / median | 0.0052 / 0.0000 |
| Share of steps where PPO picks an optimal action | 70.8% |
| Legal actions / optimal actions per step (mean) | 10.41 / 1.31 |
| Share of steps with a unique optimal action | 75.9% |
| Steps with regret per instance (mean) | 2.92 |
| Share of the gap from steps 1–3 | 52.5% |
| Share of the gap from steps 4–6 | 37.6% |
| Share of the gap from steps 7–10 | 9.9% |
| Regret at steps followed by a forced opening: share of regret | 86.2% |
| Regret at steps that are themselves forced openings: share of regret | 66.6% |
| Steps from a regret step to the next forced opening (mean) | 1.28 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0096 | 18.6% | 55.0% | 10.91 | 1.51 | 45.0% |
| 2 | 0.0091 | 17.6% | 53.2% | 10.78 | 1.54 | 46.8% |
| 3 | 0.0084 | 16.3% | 46.5% | 10.62 | 1.43 | 53.5% |
| 4 | 0.0085 | 16.5% | 42.5% | 10.67 | 1.43 | 57.5% |
| 5 | 0.0058 | 11.3% | 34.2% | 10.56 | 1.39 | 65.8% |
| 6 | 0.0051 | 9.9% | 29.0% | 10.46 | 1.29 | 71.0% |
| 7 | 0.0037 | 7.1% | 19.0% | 10.34 | 1.23 | 81.0% |
| 8 | 0.0009 | 1.7% | 8.8% | 10.11 | 1.15 | 91.2% |
| 9 | 0.0006 | 1.1% | 2.8% | 9.91 | 1.12 | 97.2% |
| 10 | 0.0000 | 0.0% | 0.8% | 9.74 | 1.06 | 99.2% |
