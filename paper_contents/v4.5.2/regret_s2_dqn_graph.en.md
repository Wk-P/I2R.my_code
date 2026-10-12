# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 21:08:55; script `src/scripts/diag_regret.py`; took 8.6 min.
- Model: v4.5.2 Transformer Mask-DQN (50k) (seed 2, 20M), replayed deterministically on the first 400 instances of the seed-2 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

392 completed instances, 8 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0874 |
| PPO regret per step: mean / median | 0.0058 / 0.0000 |
| Share of steps where PPO picks an optimal action | 62.0% |
| Legal actions / optimal actions per step (mean) | 4.88 / 1.25 |
| Share of steps with a unique optimal action | 79.8% |
| Steps with regret per instance (mean) | 5.70 |
| Share of the gap from steps 1–5 | 46.5% |
| Share of the gap from steps 6–10 | 43.1% |
| Share of the gap from steps 11–15 | 10.4% |
| Regret at steps followed by a forced opening: share of regret | 86.5% |
| Regret at steps that are themselves forced openings: share of regret | 38.6% |
| Steps from a regret step to the next forced opening (mean) | 2.11 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 6.00 / 6 (8 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0083 | 9.5% | 70.2% | 7.21 | 1.53 | 29.8% |
| 2 | 0.0078 | 8.9% | 62.5% | 6.67 | 1.46 | 37.5% |
| 3 | 0.0082 | 9.4% | 57.7% | 6.30 | 1.36 | 42.3% |
| 4 | 0.0080 | 9.2% | 53.6% | 5.98 | 1.36 | 46.4% |
| 5 | 0.0083 | 9.5% | 51.3% | 5.63 | 1.32 | 48.7% |
| 6 | 0.0090 | 10.3% | 51.5% | 5.37 | 1.26 | 48.5% |
| 7 | 0.0087 | 10.0% | 46.2% | 5.11 | 1.28 | 53.8% |
| 8 | 0.0076 | 8.7% | 42.6% | 4.88 | 1.26 | 57.4% |
| 9 | 0.0070 | 8.0% | 33.9% | 4.54 | 1.18 | 66.1% |
| 10 | 0.0054 | 6.2% | 30.1% | 4.27 | 1.20 | 69.9% |
| 11 | 0.0044 | 5.0% | 24.7% | 4.00 | 1.17 | 75.3% |
| 12 | 0.0026 | 3.0% | 15.1% | 3.67 | 1.12 | 84.9% |
| 13 | 0.0014 | 1.6% | 16.1% | 3.35 | 1.11 | 83.9% |
| 14 | 0.0004 | 0.5% | 7.7% | 3.21 | 1.10 | 92.3% |
| 15 | 0.0003 | 0.4% | 7.4% | 3.06 | 1.03 | 92.6% |

## EQ (M = 10)

399 completed instances, 1 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0362 |
| PPO regret per step: mean / median | 0.0036 / 0.0000 |
| Share of steps where PPO picks an optimal action | 79.2% |
| Legal actions / optimal actions per step (mean) | 6.15 / 1.22 |
| Share of steps with a unique optimal action | 81.7% |
| Steps with regret per instance (mean) | 2.08 |
| Share of the gap from steps 1–3 | 54.3% |
| Share of the gap from steps 4–6 | 38.7% |
| Share of the gap from steps 7–10 | 7.0% |
| Regret at steps followed by a forced opening: share of regret | 82.0% |
| Regret at steps that are themselves forced openings: share of regret | 58.0% |
| Steps from a regret step to the next forced opening (mean) | 1.44 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 3.00 / 3 (1 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0063 | 17.5% | 39.3% | 7.34 | 1.39 | 60.7% |
| 2 | 0.0075 | 20.7% | 37.3% | 6.97 | 1.35 | 62.7% |
| 3 | 0.0058 | 16.1% | 33.8% | 6.68 | 1.36 | 66.2% |
| 4 | 0.0056 | 15.4% | 32.6% | 6.53 | 1.29 | 67.4% |
| 5 | 0.0049 | 13.5% | 26.6% | 6.28 | 1.24 | 73.4% |
| 6 | 0.0036 | 9.8% | 18.0% | 5.97 | 1.20 | 82.0% |
| 7 | 0.0020 | 5.4% | 11.8% | 5.77 | 1.17 | 88.2% |
| 8 | 0.0005 | 1.5% | 6.0% | 5.52 | 1.10 | 94.0% |
| 9 | 0.0000 | 0.0% | 0.8% | 5.35 | 1.08 | 99.2% |
| 10 | 0.0000 | 0.0% | 1.3% | 5.14 | 1.03 | 98.7% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0305 |
| PPO regret per step: mean / median | 0.0031 / 0.0000 |
| Share of steps where PPO picks an optimal action | 81.5% |
| Legal actions / optimal actions per step (mean) | 10.46 / 1.28 |
| Share of steps with a unique optimal action | 77.2% |
| Steps with regret per instance (mean) | 1.85 |
| Share of the gap from steps 1–3 | 57.0% |
| Share of the gap from steps 4–6 | 37.5% |
| Share of the gap from steps 7–10 | 5.5% |
| Regret at steps followed by a forced opening: share of regret | 85.1% |
| Regret at steps that are themselves forced openings: share of regret | 68.3% |
| Steps from a regret step to the next forced opening (mean) | 1.32 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0059 | 19.3% | 37.5% | 10.93 | 1.54 | 62.5% |
| 2 | 0.0053 | 17.4% | 33.5% | 10.76 | 1.47 | 66.5% |
| 3 | 0.0062 | 20.3% | 35.5% | 10.67 | 1.41 | 64.5% |
| 4 | 0.0046 | 15.0% | 29.2% | 10.63 | 1.38 | 70.8% |
| 5 | 0.0041 | 13.5% | 22.2% | 10.58 | 1.36 | 77.8% |
| 6 | 0.0027 | 9.0% | 14.0% | 10.47 | 1.26 | 86.0% |
| 7 | 0.0013 | 4.4% | 6.2% | 10.30 | 1.19 | 93.8% |
| 8 | 0.0003 | 1.0% | 4.0% | 10.20 | 1.12 | 96.0% |
| 9 | 0.0000 | 0.1% | 1.5% | 10.08 | 1.08 | 98.5% |
| 10 | 0.0000 | 0.1% | 1.2% | 9.97 | 1.03 | 98.8% |
