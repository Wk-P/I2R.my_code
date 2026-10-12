# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 22:02:36; script `src/scripts/diag_regret.py`; took 8.8 min.
- Model: v4.5.2 Transformer Mask-DDQN (50k) (seed 2, 20M), replayed deterministically on the first 400 instances of the seed-2 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

388 completed instances, 12 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0868 |
| PPO regret per step: mean / median | 0.0058 / 0.0000 |
| Share of steps where PPO picks an optimal action | 58.5% |
| Legal actions / optimal actions per step (mean) | 5.02 / 1.25 |
| Share of steps with a unique optimal action | 79.0% |
| Steps with regret per instance (mean) | 6.22 |
| Share of the gap from steps 1–5 | 53.2% |
| Share of the gap from steps 6–10 | 37.5% |
| Share of the gap from steps 11–15 | 9.3% |
| Regret at steps followed by a forced opening: share of regret | 78.2% |
| Regret at steps that are themselves forced openings: share of regret | 37.6% |
| Steps from a regret step to the next forced opening (mean) | 2.16 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 5.17 / 6 (12 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0100 | 11.5% | 75.8% | 7.22 | 1.54 | 24.2% |
| 2 | 0.0103 | 11.8% | 66.2% | 6.76 | 1.45 | 33.8% |
| 3 | 0.0091 | 10.4% | 65.5% | 6.41 | 1.35 | 34.5% |
| 4 | 0.0100 | 11.5% | 62.6% | 6.07 | 1.36 | 37.4% |
| 5 | 0.0068 | 7.9% | 54.9% | 5.77 | 1.34 | 45.1% |
| 6 | 0.0081 | 9.3% | 49.7% | 5.54 | 1.28 | 50.3% |
| 7 | 0.0075 | 8.6% | 48.2% | 5.23 | 1.24 | 51.8% |
| 8 | 0.0069 | 8.0% | 45.4% | 4.99 | 1.27 | 54.6% |
| 9 | 0.0053 | 6.1% | 37.6% | 4.67 | 1.19 | 62.4% |
| 10 | 0.0047 | 5.4% | 32.0% | 4.40 | 1.21 | 68.0% |
| 11 | 0.0041 | 4.7% | 34.5% | 4.15 | 1.16 | 65.5% |
| 12 | 0.0021 | 2.4% | 20.6% | 3.85 | 1.16 | 79.4% |
| 13 | 0.0014 | 1.7% | 17.3% | 3.57 | 1.12 | 82.7% |
| 14 | 0.0003 | 0.4% | 8.0% | 3.49 | 1.09 | 92.0% |
| 15 | 0.0001 | 0.1% | 3.9% | 3.26 | 1.03 | 96.1% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0378 |
| PPO regret per step: mean / median | 0.0038 / 0.0000 |
| Share of steps where PPO picks an optimal action | 78.1% |
| Legal actions / optimal actions per step (mean) | 6.13 / 1.22 |
| Share of steps with a unique optimal action | 81.7% |
| Steps with regret per instance (mean) | 2.19 |
| Share of the gap from steps 1–3 | 53.3% |
| Share of the gap from steps 4–6 | 40.9% |
| Share of the gap from steps 7–10 | 5.8% |
| Regret at steps followed by a forced opening: share of regret | 87.8% |
| Regret at steps that are themselves forced openings: share of regret | 64.5% |
| Steps from a regret step to the next forced opening (mean) | 1.45 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0064 | 16.9% | 38.8% | 7.33 | 1.39 | 61.3% |
| 2 | 0.0072 | 19.1% | 40.0% | 6.96 | 1.34 | 60.0% |
| 3 | 0.0065 | 17.3% | 35.5% | 6.67 | 1.31 | 64.5% |
| 4 | 0.0069 | 18.2% | 32.2% | 6.51 | 1.28 | 67.8% |
| 5 | 0.0051 | 13.6% | 27.3% | 6.22 | 1.26 | 72.8% |
| 6 | 0.0034 | 9.1% | 20.8% | 5.93 | 1.20 | 79.2% |
| 7 | 0.0016 | 4.3% | 14.2% | 5.71 | 1.16 | 85.8% |
| 8 | 0.0004 | 1.0% | 7.0% | 5.48 | 1.08 | 93.0% |
| 9 | 0.0002 | 0.4% | 2.2% | 5.33 | 1.09 | 97.8% |
| 10 | 0.0000 | 0.0% | 0.8% | 5.12 | 1.04 | 99.2% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0398 |
| PPO regret per step: mean / median | 0.0040 / 0.0000 |
| Share of steps where PPO picks an optimal action | 80.1% |
| Legal actions / optimal actions per step (mean) | 10.35 / 1.30 |
| Share of steps with a unique optimal action | 76.4% |
| Steps with regret per instance (mean) | 1.99 |
| Share of the gap from steps 1–3 | 71.5% |
| Share of the gap from steps 4–6 | 27.1% |
| Share of the gap from steps 7–10 | 1.4% |
| Regret at steps followed by a forced opening: share of regret | 96.2% |
| Regret at steps that are themselves forced openings: share of regret | 68.5% |
| Steps from a regret step to the next forced opening (mean) | 1.16 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0100 | 25.2% | 43.5% | 10.93 | 1.54 | 56.5% |
| 2 | 0.0097 | 24.2% | 40.8% | 10.74 | 1.47 | 59.2% |
| 3 | 0.0088 | 22.1% | 39.8% | 10.67 | 1.43 | 60.2% |
| 4 | 0.0051 | 12.7% | 27.5% | 10.58 | 1.44 | 72.5% |
| 5 | 0.0028 | 7.0% | 21.0% | 10.47 | 1.40 | 79.0% |
| 6 | 0.0029 | 7.4% | 15.0% | 10.32 | 1.29 | 85.0% |
| 7 | 0.0004 | 0.9% | 5.5% | 10.13 | 1.24 | 94.5% |
| 8 | 0.0001 | 0.2% | 2.5% | 10.01 | 1.12 | 97.5% |
| 9 | 0.0001 | 0.2% | 2.5% | 9.88 | 1.08 | 97.5% |
| 10 | 0.0000 | 0.1% | 1.0% | 9.75 | 1.01 | 99.0% |
