# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-09 21:14:05; script `src/scripts/diag_regret.py`; took 10.3 min.
- Model: v4.4.12 structure-aware Mask PPO, global plan + sequential correction (seed 1, 1M, GPU), replayed deterministically on the first 400 instances of the seed-1 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

388 completed instances, 12 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0967 |
| PPO regret per step: mean / median | 0.0064 / 0.0000 |
| Share of steps where PPO picks an optimal action | 55.8% |
| Legal actions / optimal actions per step (mean) | 5.18 / 1.27 |
| Share of steps with a unique optimal action | 78.0% |
| Steps with regret per instance (mean) | 6.63 |
| Share of the gap from steps 1–5 | 48.8% |
| Share of the gap from steps 6–10 | 40.5% |
| Share of the gap from steps 11–15 | 10.6% |
| Regret at steps followed by a forced opening: share of regret | 80.3% |
| Regret at steps that are themselves forced openings: share of regret | 33.5% |
| Steps from a regret step to the next forced opening (mean) | 2.22 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 5.00 / 4 (12 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0100 | 10.4% | 75.0% | 7.41 | 1.55 | 25.0% |
| 2 | 0.0094 | 9.8% | 69.3% | 6.96 | 1.47 | 30.7% |
| 3 | 0.0099 | 10.2% | 67.0% | 6.59 | 1.36 | 33.0% |
| 4 | 0.0089 | 9.2% | 63.7% | 6.27 | 1.40 | 36.3% |
| 5 | 0.0090 | 9.3% | 57.0% | 5.98 | 1.32 | 43.0% |
| 6 | 0.0087 | 9.0% | 60.1% | 5.75 | 1.30 | 39.9% |
| 7 | 0.0086 | 8.9% | 52.6% | 5.54 | 1.28 | 47.4% |
| 8 | 0.0093 | 9.7% | 52.1% | 5.28 | 1.27 | 47.9% |
| 9 | 0.0062 | 6.5% | 43.6% | 4.95 | 1.20 | 56.4% |
| 10 | 0.0063 | 6.5% | 40.2% | 4.63 | 1.19 | 59.8% |
| 11 | 0.0051 | 5.3% | 32.5% | 4.22 | 1.13 | 67.5% |
| 12 | 0.0036 | 3.7% | 23.2% | 3.84 | 1.15 | 76.8% |
| 13 | 0.0012 | 1.2% | 14.7% | 3.69 | 1.19 | 85.3% |
| 14 | 0.0003 | 0.4% | 7.5% | 3.39 | 1.13 | 92.5% |
| 15 | 0.0001 | 0.1% | 4.4% | 3.15 | 1.03 | 95.6% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0801 |
| PPO regret per step: mean / median | 0.0080 / 0.0000 |
| Share of steps where PPO picks an optimal action | 66.4% |
| Legal actions / optimal actions per step (mean) | 5.98 / 1.27 |
| Share of steps with a unique optimal action | 77.7% |
| Steps with regret per instance (mean) | 3.36 |
| Share of the gap from steps 1–3 | 50.5% |
| Share of the gap from steps 4–6 | 38.0% |
| Share of the gap from steps 7–10 | 11.5% |
| Regret at steps followed by a forced opening: share of regret | 95.6% |
| Regret at steps that are themselves forced openings: share of regret | 53.6% |
| Steps from a regret step to the next forced opening (mean) | 1.47 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0150 | 18.7% | 60.5% | 7.39 | 1.42 | 39.5% |
| 2 | 0.0128 | 16.0% | 55.8% | 6.91 | 1.46 | 44.2% |
| 3 | 0.0127 | 15.8% | 47.5% | 6.57 | 1.33 | 52.5% |
| 4 | 0.0134 | 16.7% | 44.5% | 6.28 | 1.36 | 55.5% |
| 5 | 0.0094 | 11.7% | 39.2% | 6.02 | 1.32 | 60.8% |
| 6 | 0.0077 | 9.6% | 34.2% | 5.76 | 1.20 | 65.8% |
| 7 | 0.0045 | 5.7% | 26.0% | 5.60 | 1.22 | 74.0% |
| 8 | 0.0038 | 4.8% | 19.0% | 5.33 | 1.16 | 81.0% |
| 9 | 0.0008 | 1.0% | 7.8% | 5.05 | 1.14 | 92.2% |
| 10 | 0.0000 | 0.0% | 1.2% | 4.86 | 1.06 | 98.8% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0722 |
| PPO regret per step: mean / median | 0.0072 / 0.0000 |
| Share of steps where PPO picks an optimal action | 65.3% |
| Legal actions / optimal actions per step (mean) | 10.43 / 1.33 |
| Share of steps with a unique optimal action | 75.1% |
| Steps with regret per instance (mean) | 3.46 |
| Share of the gap from steps 1–3 | 48.1% |
| Share of the gap from steps 4–6 | 38.5% |
| Share of the gap from steps 7–10 | 13.3% |
| Regret at steps followed by a forced opening: share of regret | 84.0% |
| Regret at steps that are themselves forced openings: share of regret | 61.4% |
| Steps from a regret step to the next forced opening (mean) | 1.34 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0130 | 17.9% | 56.2% | 11.00 | 1.57 | 43.8% |
| 2 | 0.0115 | 15.9% | 53.8% | 10.79 | 1.52 | 46.2% |
| 3 | 0.0103 | 14.3% | 51.5% | 10.73 | 1.49 | 48.5% |
| 4 | 0.0107 | 14.8% | 48.5% | 10.66 | 1.42 | 51.5% |
| 5 | 0.0095 | 13.2% | 44.2% | 10.53 | 1.39 | 55.8% |
| 6 | 0.0077 | 10.6% | 35.8% | 10.37 | 1.28 | 64.2% |
| 7 | 0.0046 | 6.4% | 27.8% | 10.30 | 1.27 | 72.2% |
| 8 | 0.0027 | 3.7% | 17.0% | 10.21 | 1.18 | 83.0% |
| 9 | 0.0015 | 2.0% | 9.0% | 9.95 | 1.11 | 91.0% |
| 10 | 0.0009 | 1.2% | 2.8% | 9.77 | 1.05 | 97.2% |
