# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 22:11:54; script `src/scripts/diag_regret.py`; took 9.3 min.
- Model: v4.5.2 Transformer Mask-DDQN (50k) (seed 3, 20M), replayed deterministically on the first 400 instances of the seed-3 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

392 completed instances, 8 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 7.4e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0839 |
| PPO regret per step: mean / median | 0.0056 / 0.0000 |
| Share of steps where PPO picks an optimal action | 59.2% |
| Legal actions / optimal actions per step (mean) | 5.02 / 1.25 |
| Share of steps with a unique optimal action | 79.8% |
| Steps with regret per instance (mean) | 6.11 |
| Share of the gap from steps 1–5 | 53.6% |
| Share of the gap from steps 6–10 | 38.1% |
| Share of the gap from steps 11–15 | 8.2% |
| Regret at steps followed by a forced opening: share of regret | 86.4% |
| Regret at steps that are themselves forced openings: share of regret | 44.4% |
| Steps from a regret step to the next forced opening (mean) | 2.06 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 4.62 / 4 (8 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0095 | 11.4% | 71.9% | 7.42 | 1.46 | 28.1% |
| 2 | 0.0097 | 11.6% | 67.9% | 6.88 | 1.45 | 32.1% |
| 3 | 0.0088 | 10.5% | 62.8% | 6.39 | 1.45 | 37.2% |
| 4 | 0.0093 | 11.1% | 61.0% | 6.02 | 1.35 | 39.0% |
| 5 | 0.0076 | 9.1% | 54.3% | 5.65 | 1.30 | 45.7% |
| 6 | 0.0073 | 8.7% | 50.8% | 5.53 | 1.27 | 49.2% |
| 7 | 0.0077 | 9.2% | 48.0% | 5.21 | 1.29 | 52.0% |
| 8 | 0.0054 | 6.4% | 43.9% | 4.94 | 1.26 | 56.1% |
| 9 | 0.0066 | 7.8% | 39.3% | 4.69 | 1.19 | 60.7% |
| 10 | 0.0051 | 6.0% | 32.1% | 4.39 | 1.20 | 67.9% |
| 11 | 0.0029 | 3.5% | 27.3% | 4.13 | 1.17 | 72.7% |
| 12 | 0.0023 | 2.7% | 20.9% | 3.79 | 1.12 | 79.1% |
| 13 | 0.0012 | 1.4% | 15.6% | 3.55 | 1.10 | 84.4% |
| 14 | 0.0005 | 0.5% | 10.5% | 3.36 | 1.11 | 89.5% |
| 15 | 0.0001 | 0.1% | 5.4% | 3.33 | 1.04 | 94.6% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0392 |
| PPO regret per step: mean / median | 0.0039 / 0.0000 |
| Share of steps where PPO picks an optimal action | 77.6% |
| Legal actions / optimal actions per step (mean) | 6.11 / 1.21 |
| Share of steps with a unique optimal action | 81.7% |
| Steps with regret per instance (mean) | 2.24 |
| Share of the gap from steps 1–3 | 52.3% |
| Share of the gap from steps 4–6 | 37.9% |
| Share of the gap from steps 7–10 | 9.9% |
| Regret at steps followed by a forced opening: share of regret | 87.3% |
| Regret at steps that are themselves forced openings: share of regret | 62.8% |
| Steps from a regret step to the next forced opening (mean) | 1.39 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0067 | 17.2% | 44.0% | 7.41 | 1.43 | 56.0% |
| 2 | 0.0064 | 16.3% | 42.0% | 7.01 | 1.35 | 58.0% |
| 3 | 0.0074 | 18.8% | 38.0% | 6.62 | 1.33 | 62.0% |
| 4 | 0.0058 | 14.8% | 29.5% | 6.32 | 1.26 | 70.5% |
| 5 | 0.0059 | 15.0% | 29.2% | 6.14 | 1.22 | 70.8% |
| 6 | 0.0032 | 8.1% | 18.2% | 5.94 | 1.17 | 81.8% |
| 7 | 0.0028 | 7.0% | 12.8% | 5.67 | 1.13 | 87.2% |
| 8 | 0.0009 | 2.2% | 5.8% | 5.50 | 1.11 | 94.2% |
| 9 | 0.0002 | 0.6% | 3.5% | 5.30 | 1.09 | 96.5% |
| 10 | 0.0000 | 0.0% | 1.0% | 5.16 | 1.04 | 99.0% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0385 |
| PPO regret per step: mean / median | 0.0039 / 0.0000 |
| Share of steps where PPO picks an optimal action | 82.0% |
| Legal actions / optimal actions per step (mean) | 10.42 / 1.30 |
| Share of steps with a unique optimal action | 76.6% |
| Steps with regret per instance (mean) | 1.79 |
| Share of the gap from steps 1–3 | 47.0% |
| Share of the gap from steps 4–6 | 42.4% |
| Share of the gap from steps 7–10 | 10.6% |
| Regret at steps followed by a forced opening: share of regret | 88.6% |
| Regret at steps that are themselves forced openings: share of regret | 66.7% |
| Steps from a regret step to the next forced opening (mean) | 1.26 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0057 | 14.8% | 34.2% | 10.91 | 1.51 | 65.8% |
| 2 | 0.0059 | 15.3% | 36.8% | 10.78 | 1.56 | 63.2% |
| 3 | 0.0065 | 16.9% | 31.0% | 10.62 | 1.40 | 69.0% |
| 4 | 0.0042 | 11.0% | 25.8% | 10.66 | 1.42 | 74.2% |
| 5 | 0.0063 | 16.4% | 19.8% | 10.58 | 1.37 | 80.2% |
| 6 | 0.0058 | 15.0% | 16.2% | 10.44 | 1.27 | 83.8% |
| 7 | 0.0022 | 5.6% | 9.5% | 10.30 | 1.23 | 90.5% |
| 8 | 0.0015 | 4.0% | 4.8% | 10.13 | 1.14 | 95.2% |
| 9 | 0.0004 | 1.0% | 1.0% | 9.99 | 1.09 | 99.0% |
| 10 | 0.0000 | 0.0% | 0.5% | 9.81 | 1.05 | 99.5% |
