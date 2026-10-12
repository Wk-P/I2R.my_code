# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 20:42:21; script `src/scripts/diag_regret.py`; took 8.5 min.
- Model: v4.5.2 MLP Mask-DQN (50k) (seed 2, 20M), replayed deterministically on the first 400 instances of the seed-2 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

330 completed instances, 70 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 9.9e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1968 |
| PPO regret per step: mean / median | 0.0131 / 0.0036 |
| Share of steps where PPO picks an optimal action | 37.4% |
| Legal actions / optimal actions per step (mean) | 4.86 / 1.23 |
| Share of steps with a unique optimal action | 80.4% |
| Steps with regret per instance (mean) | 9.39 |
| Share of the gap from steps 1–5 | 42.6% |
| Share of the gap from steps 6–10 | 38.3% |
| Share of the gap from steps 11–15 | 19.0% |
| Regret at steps followed by a forced opening: share of regret | 93.4% |
| Regret at steps that are themselves forced openings: share of regret | 49.9% |
| Steps from a regret step to the next forced opening (mean) | 1.59 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 6.64 / 7 (70 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0152 | 7.7% | 80.0% | 7.29 | 1.54 | 20.0% |
| 2 | 0.0171 | 8.7% | 76.7% | 6.77 | 1.46 | 23.3% |
| 3 | 0.0181 | 9.2% | 72.7% | 6.41 | 1.36 | 27.3% |
| 4 | 0.0180 | 9.1% | 73.9% | 6.08 | 1.37 | 26.1% |
| 5 | 0.0156 | 7.9% | 71.2% | 5.79 | 1.26 | 28.8% |
| 6 | 0.0183 | 9.3% | 73.0% | 5.50 | 1.24 | 27.0% |
| 7 | 0.0149 | 7.6% | 68.8% | 5.19 | 1.24 | 31.2% |
| 8 | 0.0169 | 8.6% | 67.6% | 4.91 | 1.26 | 32.4% |
| 9 | 0.0139 | 7.1% | 64.5% | 4.52 | 1.20 | 35.5% |
| 10 | 0.0115 | 5.8% | 61.8% | 4.13 | 1.15 | 38.2% |
| 11 | 0.0127 | 6.4% | 59.1% | 3.85 | 1.13 | 40.9% |
| 12 | 0.0092 | 4.7% | 48.5% | 3.51 | 1.12 | 51.5% |
| 13 | 0.0074 | 3.8% | 47.9% | 3.10 | 1.09 | 52.1% |
| 14 | 0.0053 | 2.7% | 41.2% | 3.06 | 1.08 | 58.8% |
| 15 | 0.0029 | 1.5% | 31.8% | 2.75 | 1.02 | 68.2% |

## EQ (M = 10)

398 completed instances, 2 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1658 |
| PPO regret per step: mean / median | 0.0166 / 0.0024 |
| Share of steps where PPO picks an optimal action | 43.4% |
| Legal actions / optimal actions per step (mean) | 5.98 / 1.22 |
| Share of steps with a unique optimal action | 80.9% |
| Steps with regret per instance (mean) | 5.66 |
| Share of the gap from steps 1–3 | 46.9% |
| Share of the gap from steps 4–6 | 35.5% |
| Share of the gap from steps 7–10 | 17.6% |
| Regret at steps followed by a forced opening: share of regret | 94.4% |
| Regret at steps that are themselves forced openings: share of regret | 54.8% |
| Steps from a regret step to the next forced opening (mean) | 1.34 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 2.50 / 2 (2 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0268 | 16.2% | 79.4% | 7.35 | 1.39 | 20.6% |
| 2 | 0.0279 | 16.8% | 77.1% | 7.04 | 1.31 | 22.9% |
| 3 | 0.0231 | 13.9% | 69.6% | 6.79 | 1.34 | 30.4% |
| 4 | 0.0199 | 12.0% | 69.6% | 6.59 | 1.26 | 30.4% |
| 5 | 0.0229 | 13.8% | 62.3% | 6.28 | 1.23 | 37.7% |
| 6 | 0.0160 | 9.6% | 52.5% | 5.89 | 1.21 | 47.5% |
| 7 | 0.0137 | 8.3% | 47.5% | 5.48 | 1.21 | 52.5% |
| 8 | 0.0070 | 4.2% | 37.9% | 5.11 | 1.14 | 62.1% |
| 9 | 0.0049 | 2.9% | 35.4% | 4.76 | 1.10 | 64.6% |
| 10 | 0.0036 | 2.2% | 34.4% | 4.51 | 1.03 | 65.6% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1944 |
| PPO regret per step: mean / median | 0.0194 / 0.0080 |
| Share of steps where PPO picks an optimal action | 36.5% |
| Legal actions / optimal actions per step (mean) | 10.53 / 1.34 |
| Share of steps with a unique optimal action | 72.5% |
| Steps with regret per instance (mean) | 6.35 |
| Share of the gap from steps 1–3 | 45.2% |
| Share of the gap from steps 4–6 | 33.2% |
| Share of the gap from steps 7–10 | 21.7% |
| Regret at steps followed by a forced opening: share of regret | 92.6% |
| Regret at steps that are themselves forced openings: share of regret | 59.3% |
| Steps from a regret step to the next forced opening (mean) | 1.37 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0284 | 14.6% | 84.0% | 10.93 | 1.54 | 16.0% |
| 2 | 0.0289 | 14.9% | 81.2% | 10.84 | 1.50 | 18.8% |
| 3 | 0.0305 | 15.7% | 79.8% | 10.90 | 1.39 | 20.2% |
| 4 | 0.0241 | 12.4% | 71.8% | 10.93 | 1.38 | 28.2% |
| 5 | 0.0224 | 11.5% | 68.5% | 10.94 | 1.37 | 31.5% |
| 6 | 0.0180 | 9.2% | 60.8% | 10.76 | 1.34 | 39.2% |
| 7 | 0.0162 | 8.3% | 55.8% | 10.48 | 1.33 | 44.2% |
| 8 | 0.0129 | 6.7% | 48.5% | 10.11 | 1.22 | 51.5% |
| 9 | 0.0082 | 4.2% | 43.5% | 9.84 | 1.18 | 56.5% |
| 10 | 0.0048 | 2.5% | 41.2% | 9.57 | 1.10 | 58.8% |
