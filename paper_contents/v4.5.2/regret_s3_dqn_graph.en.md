# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 21:18:11; script `src/scripts/diag_regret.py`; took 9.3 min.
- Model: v4.5.2 Transformer Mask-DQN (50k) (seed 3, 20M), replayed deterministically on the first 400 instances of the seed-3 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

387 completed instances, 13 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 7.4e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0927 |
| PPO regret per step: mean / median | 0.0062 / 0.0000 |
| Share of steps where PPO picks an optimal action | 62.8% |
| Legal actions / optimal actions per step (mean) | 4.84 / 1.25 |
| Share of steps with a unique optimal action | 79.7% |
| Steps with regret per instance (mean) | 5.57 |
| Share of the gap from steps 1–5 | 48.3% |
| Share of the gap from steps 6–10 | 43.8% |
| Share of the gap from steps 11–15 | 7.9% |
| Regret at steps followed by a forced opening: share of regret | 87.1% |
| Regret at steps that are themselves forced openings: share of regret | 40.9% |
| Steps from a regret step to the next forced opening (mean) | 2.04 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 4.92 / 5 (13 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0084 | 9.0% | 66.7% | 7.43 | 1.47 | 33.3% |
| 2 | 0.0089 | 9.6% | 64.3% | 6.88 | 1.47 | 35.7% |
| 3 | 0.0094 | 10.1% | 59.2% | 6.41 | 1.45 | 40.8% |
| 4 | 0.0099 | 10.7% | 54.8% | 5.97 | 1.37 | 45.2% |
| 5 | 0.0082 | 8.8% | 51.4% | 5.60 | 1.28 | 48.6% |
| 6 | 0.0085 | 9.1% | 45.7% | 5.32 | 1.33 | 54.3% |
| 7 | 0.0097 | 10.5% | 46.0% | 5.01 | 1.27 | 54.0% |
| 8 | 0.0081 | 8.8% | 35.9% | 4.72 | 1.26 | 64.1% |
| 9 | 0.0073 | 7.9% | 38.2% | 4.50 | 1.15 | 61.8% |
| 10 | 0.0070 | 7.5% | 31.8% | 4.13 | 1.20 | 68.2% |
| 11 | 0.0039 | 4.2% | 24.3% | 3.85 | 1.15 | 75.7% |
| 12 | 0.0019 | 2.1% | 16.8% | 3.48 | 1.13 | 83.2% |
| 13 | 0.0012 | 1.3% | 11.4% | 3.29 | 1.11 | 88.6% |
| 14 | 0.0003 | 0.3% | 5.9% | 3.07 | 1.12 | 94.1% |
| 15 | 0.0001 | 0.1% | 4.9% | 2.90 | 1.05 | 95.1% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0352 |
| PPO regret per step: mean / median | 0.0035 / 0.0000 |
| Share of steps where PPO picks an optimal action | 77.0% |
| Legal actions / optimal actions per step (mean) | 6.20 / 1.21 |
| Share of steps with a unique optimal action | 82.2% |
| Steps with regret per instance (mean) | 2.29 |
| Share of the gap from steps 1–3 | 54.4% |
| Share of the gap from steps 4–6 | 37.6% |
| Share of the gap from steps 7–10 | 7.9% |
| Regret at steps followed by a forced opening: share of regret | 82.5% |
| Regret at steps that are themselves forced openings: share of regret | 56.4% |
| Steps from a regret step to the next forced opening (mean) | 1.51 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0068 | 19.5% | 51.2% | 7.41 | 1.43 | 48.8% |
| 2 | 0.0061 | 17.5% | 45.8% | 7.05 | 1.36 | 54.2% |
| 3 | 0.0062 | 17.5% | 36.2% | 6.70 | 1.35 | 63.7% |
| 4 | 0.0050 | 14.2% | 31.0% | 6.42 | 1.26 | 69.0% |
| 5 | 0.0050 | 14.3% | 24.2% | 6.25 | 1.20 | 75.8% |
| 6 | 0.0032 | 9.2% | 17.2% | 6.04 | 1.16 | 82.8% |
| 7 | 0.0019 | 5.3% | 12.8% | 5.80 | 1.10 | 87.2% |
| 8 | 0.0005 | 1.6% | 6.2% | 5.68 | 1.11 | 93.8% |
| 9 | 0.0004 | 1.0% | 4.2% | 5.45 | 1.10 | 95.8% |
| 10 | 0.0000 | 0.0% | 0.5% | 5.24 | 1.03 | 99.5% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0286 |
| PPO regret per step: mean / median | 0.0029 / 0.0000 |
| Share of steps where PPO picks an optimal action | 82.9% |
| Legal actions / optimal actions per step (mean) | 10.48 / 1.29 |
| Share of steps with a unique optimal action | 77.6% |
| Steps with regret per instance (mean) | 1.71 |
| Share of the gap from steps 1–3 | 50.7% |
| Share of the gap from steps 4–6 | 41.4% |
| Share of the gap from steps 7–10 | 7.9% |
| Regret at steps followed by a forced opening: share of regret | 92.7% |
| Regret at steps that are themselves forced openings: share of regret | 69.5% |
| Steps from a regret step to the next forced opening (mean) | 1.26 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0048 | 16.6% | 35.2% | 10.91 | 1.51 | 64.8% |
| 2 | 0.0048 | 16.8% | 33.2% | 10.79 | 1.54 | 66.8% |
| 3 | 0.0049 | 17.2% | 30.2% | 10.65 | 1.35 | 69.8% |
| 4 | 0.0047 | 16.3% | 28.7% | 10.73 | 1.41 | 71.2% |
| 5 | 0.0048 | 16.6% | 21.5% | 10.64 | 1.38 | 78.5% |
| 6 | 0.0024 | 8.4% | 11.0% | 10.51 | 1.30 | 89.0% |
| 7 | 0.0009 | 3.0% | 5.0% | 10.37 | 1.22 | 95.0% |
| 8 | 0.0010 | 3.5% | 4.2% | 10.20 | 1.10 | 95.8% |
| 9 | 0.0004 | 1.4% | 1.2% | 10.12 | 1.08 | 98.8% |
| 10 | 0.0000 | 0.0% | 0.2% | 9.93 | 1.04 | 99.8% |
