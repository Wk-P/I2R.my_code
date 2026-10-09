# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-09 23:11:00; script `src/scripts/diag_regret.py`; took 9.9 min.
- Model: v4.4.13 structure-aware Mask PPO, professor's 3-dim state summary (seed 1, 1M, GPU), replayed deterministically on the first 400 instances of the seed-1 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

386 completed instances, 14 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0952 |
| PPO regret per step: mean / median | 0.0063 / 0.0000 |
| Share of steps where PPO picks an optimal action | 59.4% |
| Legal actions / optimal actions per step (mean) | 4.90 / 1.25 |
| Share of steps with a unique optimal action | 79.5% |
| Steps with regret per instance (mean) | 6.09 |
| Share of the gap from steps 1–5 | 48.1% |
| Share of the gap from steps 6–10 | 42.6% |
| Share of the gap from steps 11–15 | 9.3% |
| Regret at steps followed by a forced opening: share of regret | 85.5% |
| Regret at steps that are themselves forced openings: share of regret | 40.2% |
| Steps from a regret step to the next forced opening (mean) | 2.08 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 4.79 / 4 (14 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0101 | 10.6% | 73.6% | 7.40 | 1.55 | 26.4% |
| 2 | 0.0090 | 9.4% | 67.4% | 6.88 | 1.47 | 32.6% |
| 3 | 0.0103 | 10.8% | 63.2% | 6.41 | 1.38 | 36.8% |
| 4 | 0.0077 | 8.1% | 56.7% | 5.94 | 1.41 | 43.3% |
| 5 | 0.0088 | 9.3% | 52.6% | 5.56 | 1.33 | 47.4% |
| 6 | 0.0100 | 10.5% | 51.8% | 5.27 | 1.24 | 48.2% |
| 7 | 0.0098 | 10.3% | 51.3% | 5.09 | 1.24 | 48.7% |
| 8 | 0.0066 | 6.9% | 39.9% | 4.81 | 1.23 | 60.1% |
| 9 | 0.0076 | 8.0% | 44.0% | 4.57 | 1.16 | 56.0% |
| 10 | 0.0066 | 6.9% | 33.7% | 4.25 | 1.16 | 66.3% |
| 11 | 0.0039 | 4.1% | 24.4% | 3.92 | 1.12 | 75.6% |
| 12 | 0.0030 | 3.1% | 18.9% | 3.66 | 1.14 | 81.1% |
| 13 | 0.0015 | 1.6% | 18.1% | 3.50 | 1.16 | 81.9% |
| 14 | 0.0004 | 0.4% | 8.3% | 3.23 | 1.11 | 91.7% |
| 15 | 0.0001 | 0.1% | 5.4% | 3.01 | 1.05 | 94.6% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0733 |
| PPO regret per step: mean / median | 0.0073 / 0.0000 |
| Share of steps where PPO picks an optimal action | 67.9% |
| Legal actions / optimal actions per step (mean) | 6.02 / 1.26 |
| Share of steps with a unique optimal action | 78.2% |
| Steps with regret per instance (mean) | 3.21 |
| Share of the gap from steps 1–3 | 49.9% |
| Share of the gap from steps 4–6 | 38.6% |
| Share of the gap from steps 7–10 | 11.4% |
| Regret at steps followed by a forced opening: share of regret | 89.0% |
| Regret at steps that are themselves forced openings: share of regret | 55.2% |
| Steps from a regret step to the next forced opening (mean) | 1.43 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0126 | 17.2% | 57.0% | 7.39 | 1.42 | 43.0% |
| 2 | 0.0123 | 16.7% | 53.8% | 6.93 | 1.46 | 46.2% |
| 3 | 0.0118 | 16.0% | 47.8% | 6.62 | 1.35 | 52.2% |
| 4 | 0.0128 | 17.5% | 44.8% | 6.34 | 1.36 | 55.2% |
| 5 | 0.0083 | 11.3% | 38.0% | 6.04 | 1.31 | 62.0% |
| 6 | 0.0072 | 9.9% | 32.8% | 5.83 | 1.19 | 67.2% |
| 7 | 0.0042 | 5.7% | 22.5% | 5.62 | 1.19 | 77.5% |
| 8 | 0.0035 | 4.7% | 17.2% | 5.39 | 1.17 | 82.8% |
| 9 | 0.0007 | 1.0% | 7.2% | 5.11 | 1.11 | 92.8% |
| 10 | 0.0000 | 0.0% | 0.2% | 4.94 | 1.07 | 99.8% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0899 |
| PPO regret per step: mean / median | 0.0090 / 0.0000 |
| Share of steps where PPO picks an optimal action | 64.4% |
| Legal actions / optimal actions per step (mean) | 10.22 / 1.34 |
| Share of steps with a unique optimal action | 73.3% |
| Steps with regret per instance (mean) | 3.56 |
| Share of the gap from steps 1–3 | 53.6% |
| Share of the gap from steps 4–6 | 36.4% |
| Share of the gap from steps 7–10 | 9.9% |
| Regret at steps followed by a forced opening: share of regret | 91.6% |
| Regret at steps that are themselves forced openings: share of regret | 65.7% |
| Steps from a regret step to the next forced opening (mean) | 1.24 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0166 | 18.4% | 61.5% | 11.00 | 1.57 | 38.5% |
| 2 | 0.0179 | 19.9% | 63.2% | 10.80 | 1.49 | 36.8% |
| 3 | 0.0138 | 15.3% | 53.8% | 10.71 | 1.45 | 46.2% |
| 4 | 0.0137 | 15.2% | 50.5% | 10.59 | 1.44 | 49.5% |
| 5 | 0.0105 | 11.7% | 43.2% | 10.36 | 1.47 | 56.8% |
| 6 | 0.0086 | 9.5% | 37.8% | 10.15 | 1.33 | 62.3% |
| 7 | 0.0053 | 5.9% | 23.5% | 9.91 | 1.28 | 76.5% |
| 8 | 0.0023 | 2.5% | 15.0% | 9.76 | 1.22 | 85.0% |
| 9 | 0.0014 | 1.5% | 7.2% | 9.57 | 1.12 | 92.8% |
| 10 | 0.0000 | 0.0% | 0.5% | 9.36 | 1.06 | 99.5% |
