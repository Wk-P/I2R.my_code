# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 20:51:18; script `src/scripts/diag_regret.py`; took 8.9 min.
- Model: v4.5.2 MLP Mask-DQN (50k) (seed 3, 20M), replayed deterministically on the first 400 instances of the seed-3 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

333 completed instances, 67 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 7.4e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1951 |
| PPO regret per step: mean / median | 0.0130 / 0.0030 |
| Share of steps where PPO picks an optimal action | 38.8% |
| Legal actions / optimal actions per step (mean) | 4.81 / 1.23 |
| Share of steps with a unique optimal action | 81.0% |
| Steps with regret per instance (mean) | 9.19 |
| Share of the gap from steps 1–5 | 45.3% |
| Share of the gap from steps 6–10 | 37.0% |
| Share of the gap from steps 11–15 | 17.7% |
| Regret at steps followed by a forced opening: share of regret | 92.8% |
| Regret at steps that are themselves forced openings: share of regret | 48.9% |
| Steps from a regret step to the next forced opening (mean) | 1.67 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 7.22 / 7 (67 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0165 | 8.4% | 79.6% | 7.47 | 1.47 | 20.4% |
| 2 | 0.0167 | 8.6% | 74.5% | 6.93 | 1.49 | 25.5% |
| 3 | 0.0178 | 9.1% | 74.8% | 6.44 | 1.37 | 25.2% |
| 4 | 0.0183 | 9.4% | 72.4% | 6.00 | 1.33 | 27.6% |
| 5 | 0.0192 | 9.8% | 74.2% | 5.65 | 1.31 | 25.8% |
| 6 | 0.0149 | 7.6% | 68.8% | 5.41 | 1.24 | 31.2% |
| 7 | 0.0176 | 9.0% | 72.4% | 4.99 | 1.28 | 27.6% |
| 8 | 0.0131 | 6.7% | 63.1% | 4.75 | 1.26 | 36.9% |
| 9 | 0.0142 | 7.3% | 64.3% | 4.43 | 1.16 | 35.7% |
| 10 | 0.0124 | 6.4% | 53.5% | 4.08 | 1.20 | 46.5% |
| 11 | 0.0107 | 5.5% | 55.6% | 3.75 | 1.14 | 44.4% |
| 12 | 0.0085 | 4.3% | 45.3% | 3.37 | 1.08 | 54.7% |
| 13 | 0.0071 | 3.7% | 44.1% | 3.14 | 1.09 | 55.9% |
| 14 | 0.0053 | 2.7% | 38.7% | 2.93 | 1.09 | 61.3% |
| 15 | 0.0029 | 1.5% | 37.5% | 2.82 | 1.01 | 62.5% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1632 |
| PPO regret per step: mean / median | 0.0163 / 0.0027 |
| Share of steps where PPO picks an optimal action | 42.3% |
| Legal actions / optimal actions per step (mean) | 6.03 / 1.21 |
| Share of steps with a unique optimal action | 81.9% |
| Steps with regret per instance (mean) | 5.77 |
| Share of the gap from steps 1–3 | 44.5% |
| Share of the gap from steps 4–6 | 37.4% |
| Share of the gap from steps 7–10 | 18.1% |
| Regret at steps followed by a forced opening: share of regret | 94.1% |
| Regret at steps that are themselves forced openings: share of regret | 57.9% |
| Steps from a regret step to the next forced opening (mean) | 1.36 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0229 | 14.0% | 78.8% | 7.41 | 1.43 | 21.2% |
| 2 | 0.0257 | 15.8% | 78.0% | 7.06 | 1.32 | 22.0% |
| 3 | 0.0240 | 14.7% | 72.2% | 6.73 | 1.27 | 27.8% |
| 4 | 0.0220 | 13.5% | 63.0% | 6.47 | 1.27 | 37.0% |
| 5 | 0.0212 | 13.0% | 64.2% | 6.22 | 1.19 | 35.8% |
| 6 | 0.0179 | 11.0% | 60.5% | 5.93 | 1.14 | 39.5% |
| 7 | 0.0130 | 7.9% | 52.2% | 5.53 | 1.17 | 47.8% |
| 8 | 0.0086 | 5.3% | 42.5% | 5.29 | 1.15 | 57.5% |
| 9 | 0.0047 | 2.9% | 32.2% | 5.02 | 1.12 | 67.8% |
| 10 | 0.0033 | 2.0% | 33.0% | 4.66 | 1.04 | 67.0% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.2051 |
| PPO regret per step: mean / median | 0.0205 / 0.0110 |
| Share of steps where PPO picks an optimal action | 33.3% |
| Legal actions / optimal actions per step (mean) | 10.48 / 1.31 |
| Share of steps with a unique optimal action | 75.8% |
| Steps with regret per instance (mean) | 6.67 |
| Share of the gap from steps 1–3 | 41.0% |
| Share of the gap from steps 4–6 | 34.2% |
| Share of the gap from steps 7–10 | 24.8% |
| Regret at steps followed by a forced opening: share of regret | 90.0% |
| Regret at steps that are themselves forced openings: share of regret | 65.3% |
| Steps from a regret step to the next forced opening (mean) | 1.33 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0264 | 12.9% | 86.0% | 10.91 | 1.51 | 14.0% |
| 2 | 0.0267 | 13.0% | 82.5% | 10.86 | 1.51 | 17.5% |
| 3 | 0.0309 | 15.1% | 82.2% | 10.80 | 1.33 | 17.8% |
| 4 | 0.0248 | 12.1% | 66.8% | 10.98 | 1.40 | 33.2% |
| 5 | 0.0249 | 12.1% | 70.8% | 10.89 | 1.38 | 29.2% |
| 6 | 0.0205 | 10.0% | 59.8% | 10.67 | 1.27 | 40.2% |
| 7 | 0.0171 | 8.3% | 61.3% | 10.40 | 1.27 | 38.8% |
| 8 | 0.0156 | 7.6% | 58.8% | 10.03 | 1.22 | 41.2% |
| 9 | 0.0103 | 5.0% | 51.7% | 9.71 | 1.12 | 48.2% |
| 10 | 0.0078 | 3.8% | 47.2% | 9.54 | 1.06 | 52.8% |
