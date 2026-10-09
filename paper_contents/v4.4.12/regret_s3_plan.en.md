# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-09 21:34:03; script `src/scripts/diag_regret.py`; took 10.1 min.
- Model: v4.4.12 structure-aware Mask PPO, global plan + sequential correction (seed 3, 1M, GPU), replayed deterministically on the first 400 instances of the seed-3 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

392 completed instances, 8 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 7.4e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0912 |
| PPO regret per step: mean / median | 0.0061 / 0.0000 |
| Share of steps where PPO picks an optimal action | 58.7% |
| Legal actions / optimal actions per step (mean) | 4.92 / 1.25 |
| Share of steps with a unique optimal action | 79.9% |
| Steps with regret per instance (mean) | 6.20 |
| Share of the gap from steps 1–5 | 50.5% |
| Share of the gap from steps 6–10 | 39.9% |
| Share of the gap from steps 11–15 | 9.6% |
| Regret at steps followed by a forced opening: share of regret | 83.2% |
| Regret at steps that are themselves forced openings: share of regret | 38.5% |
| Steps from a regret step to the next forced opening (mean) | 2.11 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 4.50 / 3 (8 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0099 | 10.9% | 71.9% | 7.41 | 1.46 | 28.1% |
| 2 | 0.0093 | 10.1% | 68.9% | 6.81 | 1.47 | 31.1% |
| 3 | 0.0083 | 9.1% | 59.7% | 6.33 | 1.43 | 40.3% |
| 4 | 0.0078 | 8.6% | 56.6% | 5.92 | 1.35 | 43.4% |
| 5 | 0.0107 | 11.7% | 62.8% | 5.58 | 1.28 | 37.2% |
| 6 | 0.0074 | 8.1% | 49.7% | 5.37 | 1.30 | 50.3% |
| 7 | 0.0088 | 9.6% | 45.2% | 5.07 | 1.31 | 54.8% |
| 8 | 0.0079 | 8.7% | 43.6% | 4.81 | 1.23 | 56.4% |
| 9 | 0.0070 | 7.6% | 43.1% | 4.58 | 1.16 | 56.9% |
| 10 | 0.0054 | 5.9% | 38.0% | 4.31 | 1.19 | 62.0% |
| 11 | 0.0042 | 4.6% | 31.4% | 4.09 | 1.15 | 68.6% |
| 12 | 0.0023 | 2.5% | 21.7% | 3.71 | 1.14 | 78.3% |
| 13 | 0.0016 | 1.8% | 15.6% | 3.43 | 1.10 | 84.4% |
| 14 | 0.0005 | 0.6% | 8.2% | 3.22 | 1.12 | 91.8% |
| 15 | 0.0000 | 0.0% | 3.3% | 3.14 | 1.03 | 96.7% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0822 |
| PPO regret per step: mean / median | 0.0082 / 0.0000 |
| Share of steps where PPO picks an optimal action | 65.7% |
| Legal actions / optimal actions per step (mean) | 6.05 / 1.23 |
| Share of steps with a unique optimal action | 80.0% |
| Steps with regret per instance (mean) | 3.43 |
| Share of the gap from steps 1–3 | 51.3% |
| Share of the gap from steps 4–6 | 37.1% |
| Share of the gap from steps 7–10 | 11.6% |
| Regret at steps followed by a forced opening: share of regret | 91.0% |
| Regret at steps that are themselves forced openings: share of regret | 53.1% |
| Steps from a regret step to the next forced opening (mean) | 1.52 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0143 | 17.3% | 61.0% | 7.41 | 1.43 | 39.0% |
| 2 | 0.0145 | 17.7% | 57.0% | 7.00 | 1.34 | 43.0% |
| 3 | 0.0134 | 16.3% | 51.7% | 6.62 | 1.30 | 48.2% |
| 4 | 0.0110 | 13.4% | 45.8% | 6.32 | 1.29 | 54.2% |
| 5 | 0.0096 | 11.7% | 40.8% | 6.10 | 1.23 | 59.2% |
| 6 | 0.0098 | 12.0% | 35.5% | 5.93 | 1.21 | 64.5% |
| 7 | 0.0049 | 6.0% | 26.0% | 5.64 | 1.15 | 74.0% |
| 8 | 0.0040 | 4.8% | 17.2% | 5.44 | 1.14 | 82.8% |
| 9 | 0.0006 | 0.8% | 7.5% | 5.15 | 1.15 | 92.5% |
| 10 | 0.0000 | 0.0% | 0.2% | 4.91 | 1.06 | 99.8% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0737 |
| PPO regret per step: mean / median | 0.0074 / 0.0000 |
| Share of steps where PPO picks an optimal action | 67.0% |
| Legal actions / optimal actions per step (mean) | 10.19 / 1.36 |
| Share of steps with a unique optimal action | 72.4% |
| Steps with regret per instance (mean) | 3.30 |
| Share of the gap from steps 1–3 | 46.0% |
| Share of the gap from steps 4–6 | 34.4% |
| Share of the gap from steps 7–10 | 19.7% |
| Regret at steps followed by a forced opening: share of regret | 88.0% |
| Regret at steps that are themselves forced openings: share of regret | 63.8% |
| Steps from a regret step to the next forced opening (mean) | 1.27 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0112 | 15.2% | 50.2% | 10.91 | 1.51 | 49.8% |
| 2 | 0.0115 | 15.6% | 50.7% | 10.76 | 1.56 | 49.2% |
| 3 | 0.0112 | 15.1% | 52.5% | 10.53 | 1.46 | 47.5% |
| 4 | 0.0109 | 14.8% | 42.5% | 10.49 | 1.46 | 57.5% |
| 5 | 0.0074 | 10.1% | 38.2% | 10.32 | 1.47 | 61.8% |
| 6 | 0.0070 | 9.5% | 36.0% | 10.18 | 1.37 | 64.0% |
| 7 | 0.0060 | 8.1% | 27.0% | 10.02 | 1.28 | 73.0% |
| 8 | 0.0041 | 5.6% | 17.2% | 9.79 | 1.21 | 82.8% |
| 9 | 0.0011 | 1.5% | 10.5% | 9.57 | 1.19 | 89.5% |
| 10 | 0.0033 | 4.4% | 5.0% | 9.33 | 1.09 | 95.0% |
