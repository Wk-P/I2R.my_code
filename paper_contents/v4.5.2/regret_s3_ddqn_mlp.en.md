# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 21:44:40; script `src/scripts/diag_regret.py`; took 9.0 min.
- Model: v4.5.2 MLP Mask-DDQN (50k) (seed 3, 20M), replayed deterministically on the first 400 instances of the seed-3 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

321 completed instances, 79 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 7.4e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1895 |
| PPO regret per step: mean / median | 0.0126 / 0.0028 |
| Share of steps where PPO picks an optimal action | 39.1% |
| Legal actions / optimal actions per step (mean) | 4.82 / 1.22 |
| Share of steps with a unique optimal action | 81.9% |
| Steps with regret per instance (mean) | 9.14 |
| Share of the gap from steps 1–5 | 46.4% |
| Share of the gap from steps 6–10 | 36.3% |
| Share of the gap from steps 11–15 | 17.3% |
| Regret at steps followed by a forced opening: share of regret | 94.4% |
| Regret at steps that are themselves forced openings: share of regret | 51.4% |
| Steps from a regret step to the next forced opening (mean) | 1.58 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 7.23 / 7 (79 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0169 | 8.9% | 75.4% | 7.52 | 1.48 | 24.6% |
| 2 | 0.0180 | 9.5% | 79.4% | 6.94 | 1.38 | 20.6% |
| 3 | 0.0160 | 8.5% | 72.3% | 6.47 | 1.38 | 27.7% |
| 4 | 0.0191 | 10.1% | 73.5% | 6.07 | 1.26 | 26.5% |
| 5 | 0.0180 | 9.5% | 75.7% | 5.65 | 1.26 | 24.3% |
| 6 | 0.0164 | 8.6% | 72.0% | 5.35 | 1.28 | 28.0% |
| 7 | 0.0158 | 8.3% | 73.2% | 5.03 | 1.21 | 26.8% |
| 8 | 0.0119 | 6.3% | 57.3% | 4.79 | 1.23 | 42.7% |
| 9 | 0.0133 | 7.0% | 62.9% | 4.41 | 1.18 | 37.1% |
| 10 | 0.0113 | 6.0% | 58.3% | 3.98 | 1.17 | 41.7% |
| 11 | 0.0090 | 4.8% | 49.2% | 3.73 | 1.16 | 50.8% |
| 12 | 0.0088 | 4.6% | 45.8% | 3.35 | 1.11 | 54.2% |
| 13 | 0.0070 | 3.7% | 42.4% | 3.24 | 1.09 | 57.6% |
| 14 | 0.0051 | 2.7% | 37.7% | 2.92 | 1.09 | 62.3% |
| 15 | 0.0030 | 1.6% | 38.6% | 2.83 | 1.03 | 61.4% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1647 |
| PPO regret per step: mean / median | 0.0165 / 0.0033 |
| Share of steps where PPO picks an optimal action | 41.3% |
| Legal actions / optimal actions per step (mean) | 6.04 / 1.20 |
| Share of steps with a unique optimal action | 82.0% |
| Steps with regret per instance (mean) | 5.87 |
| Share of the gap from steps 1–3 | 44.5% |
| Share of the gap from steps 4–6 | 37.1% |
| Share of the gap from steps 7–10 | 18.4% |
| Regret at steps followed by a forced opening: share of regret | 93.9% |
| Regret at steps that are themselves forced openings: share of regret | 58.0% |
| Steps from a regret step to the next forced opening (mean) | 1.34 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0240 | 14.5% | 78.2% | 7.41 | 1.43 | 21.8% |
| 2 | 0.0253 | 15.4% | 77.5% | 7.04 | 1.31 | 22.5% |
| 3 | 0.0240 | 14.6% | 71.2% | 6.74 | 1.27 | 28.7% |
| 4 | 0.0230 | 14.0% | 71.8% | 6.43 | 1.23 | 28.2% |
| 5 | 0.0219 | 13.3% | 70.5% | 6.23 | 1.18 | 29.5% |
| 6 | 0.0162 | 9.9% | 55.8% | 5.95 | 1.16 | 44.2% |
| 7 | 0.0131 | 7.9% | 49.5% | 5.55 | 1.17 | 50.5% |
| 8 | 0.0099 | 6.0% | 44.0% | 5.29 | 1.14 | 56.0% |
| 9 | 0.0047 | 2.9% | 36.2% | 5.04 | 1.12 | 63.7% |
| 10 | 0.0027 | 1.6% | 31.8% | 4.70 | 1.04 | 68.2% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.2075 |
| PPO regret per step: mean / median | 0.0208 / 0.0113 |
| Share of steps where PPO picks an optimal action | 33.1% |
| Legal actions / optimal actions per step (mean) | 10.49 / 1.32 |
| Share of steps with a unique optimal action | 74.6% |
| Steps with regret per instance (mean) | 6.69 |
| Share of the gap from steps 1–3 | 40.3% |
| Share of the gap from steps 4–6 | 34.5% |
| Share of the gap from steps 7–10 | 25.3% |
| Regret at steps followed by a forced opening: share of regret | 88.9% |
| Regret at steps that are themselves forced openings: share of regret | 64.1% |
| Steps from a regret step to the next forced opening (mean) | 1.35 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0261 | 12.6% | 86.2% | 10.91 | 1.51 | 13.8% |
| 2 | 0.0279 | 13.4% | 82.0% | 10.84 | 1.48 | 18.0% |
| 3 | 0.0296 | 14.3% | 78.2% | 10.79 | 1.32 | 21.8% |
| 4 | 0.0261 | 12.6% | 73.2% | 10.93 | 1.41 | 26.8% |
| 5 | 0.0218 | 10.5% | 69.0% | 10.90 | 1.38 | 31.0% |
| 6 | 0.0236 | 11.4% | 68.2% | 10.73 | 1.30 | 31.8% |
| 7 | 0.0194 | 9.3% | 64.0% | 10.41 | 1.28 | 36.0% |
| 8 | 0.0154 | 7.4% | 57.2% | 10.06 | 1.23 | 42.8% |
| 9 | 0.0108 | 5.2% | 50.0% | 9.79 | 1.18 | 50.0% |
| 10 | 0.0069 | 3.3% | 40.5% | 9.55 | 1.08 | 59.5% |
