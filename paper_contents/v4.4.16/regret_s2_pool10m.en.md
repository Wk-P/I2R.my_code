# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-11 05:47:15; script `src/scripts/diag_regret.py`; took 9.6 min.
- Model: v4.4.16 structure-aware Mask PPO, 50k instances, 10M (seed 2, 10M, GPU), replayed deterministically on the first 400 instances of the seed-2 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

396 completed instances, 4 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 2.3e-04 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0672 |
| PPO regret per step: mean / median | 0.0045 / 0.0000 |
| Share of steps where PPO picks an optimal action | 64.5% |
| Legal actions / optimal actions per step (mean) | 4.95 / 1.24 |
| Share of steps with a unique optimal action | 80.0% |
| Steps with regret per instance (mean) | 5.32 |
| Share of the gap from steps 1–5 | 50.5% |
| Share of the gap from steps 6–10 | 41.9% |
| Share of the gap from steps 11–15 | 7.5% |
| Regret at steps followed by a forced opening: share of regret | 79.3% |
| Regret at steps that are themselves forced openings: share of regret | 44.3% |
| Steps from a regret step to the next forced opening (mean) | 2.13 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 4.75 / 5 (4 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0069 | 10.2% | 64.9% | 7.17 | 1.53 | 35.1% |
| 2 | 0.0066 | 9.9% | 57.1% | 6.64 | 1.46 | 42.9% |
| 3 | 0.0074 | 11.0% | 62.1% | 6.25 | 1.34 | 37.9% |
| 4 | 0.0066 | 9.8% | 52.0% | 5.94 | 1.32 | 48.0% |
| 5 | 0.0065 | 9.7% | 49.0% | 5.66 | 1.32 | 51.0% |
| 6 | 0.0076 | 11.4% | 51.0% | 5.42 | 1.27 | 49.0% |
| 7 | 0.0068 | 10.1% | 42.9% | 5.14 | 1.26 | 57.1% |
| 8 | 0.0057 | 8.4% | 39.1% | 4.94 | 1.29 | 60.9% |
| 9 | 0.0052 | 7.7% | 35.6% | 4.63 | 1.20 | 64.4% |
| 10 | 0.0030 | 4.4% | 25.8% | 4.35 | 1.19 | 74.2% |
| 11 | 0.0028 | 4.2% | 20.7% | 4.12 | 1.17 | 79.3% |
| 12 | 0.0011 | 1.7% | 13.9% | 3.77 | 1.11 | 86.1% |
| 13 | 0.0006 | 0.9% | 8.3% | 3.59 | 1.10 | 91.7% |
| 14 | 0.0005 | 0.8% | 7.1% | 3.42 | 1.07 | 92.9% |
| 15 | 0.0000 | 0.0% | 2.3% | 3.18 | 1.03 | 97.7% |

## EQ (M = 10)

399 completed instances, 1 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0458 |
| PPO regret per step: mean / median | 0.0046 / 0.0000 |
| Share of steps where PPO picks an optimal action | 75.0% |
| Legal actions / optimal actions per step (mean) | 6.17 / 1.22 |
| Share of steps with a unique optimal action | 81.5% |
| Steps with regret per instance (mean) | 2.50 |
| Share of the gap from steps 1–3 | 52.3% |
| Share of the gap from steps 4–6 | 39.9% |
| Share of the gap from steps 7–10 | 7.8% |
| Regret at steps followed by a forced opening: share of regret | 88.2% |
| Regret at steps that are themselves forced openings: share of regret | 56.2% |
| Steps from a regret step to the next forced opening (mean) | 1.49 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 3.00 / 3 (1 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0073 | 15.9% | 45.6% | 7.34 | 1.39 | 54.4% |
| 2 | 0.0089 | 19.5% | 43.1% | 7.01 | 1.35 | 56.9% |
| 3 | 0.0077 | 16.9% | 41.1% | 6.73 | 1.35 | 58.9% |
| 4 | 0.0073 | 16.0% | 37.3% | 6.57 | 1.27 | 62.7% |
| 5 | 0.0060 | 13.1% | 28.8% | 6.32 | 1.26 | 71.2% |
| 6 | 0.0050 | 10.8% | 26.6% | 6.01 | 1.18 | 73.4% |
| 7 | 0.0024 | 5.2% | 13.0% | 5.78 | 1.18 | 87.0% |
| 8 | 0.0007 | 1.5% | 8.0% | 5.49 | 1.10 | 92.0% |
| 9 | 0.0004 | 0.9% | 4.5% | 5.33 | 1.08 | 95.5% |
| 10 | 0.0001 | 0.1% | 2.0% | 5.16 | 1.03 | 98.0% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0476 |
| PPO regret per step: mean / median | 0.0048 / 0.0000 |
| Share of steps where PPO picks an optimal action | 74.6% |
| Legal actions / optimal actions per step (mean) | 10.47 / 1.30 |
| Share of steps with a unique optimal action | 75.9% |
| Steps with regret per instance (mean) | 2.54 |
| Share of the gap from steps 1–3 | 54.3% |
| Share of the gap from steps 4–6 | 37.7% |
| Share of the gap from steps 7–10 | 8.0% |
| Regret at steps followed by a forced opening: share of regret | 89.3% |
| Regret at steps that are themselves forced openings: share of regret | 64.0% |
| Steps from a regret step to the next forced opening (mean) | 1.36 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0092 | 19.4% | 46.8% | 10.93 | 1.54 | 53.2% |
| 2 | 0.0084 | 17.7% | 45.2% | 10.76 | 1.48 | 54.8% |
| 3 | 0.0082 | 17.2% | 43.8% | 10.71 | 1.43 | 56.2% |
| 4 | 0.0069 | 14.5% | 36.8% | 10.68 | 1.42 | 63.2% |
| 5 | 0.0060 | 12.6% | 29.8% | 10.67 | 1.35 | 70.2% |
| 6 | 0.0051 | 10.6% | 22.5% | 10.53 | 1.28 | 77.5% |
| 7 | 0.0018 | 3.8% | 14.2% | 10.35 | 1.23 | 85.8% |
| 8 | 0.0018 | 3.7% | 9.0% | 10.16 | 1.14 | 91.0% |
| 9 | 0.0002 | 0.4% | 3.5% | 10.05 | 1.12 | 96.5% |
| 10 | 0.0000 | 0.1% | 2.5% | 9.87 | 1.04 | 97.5% |
