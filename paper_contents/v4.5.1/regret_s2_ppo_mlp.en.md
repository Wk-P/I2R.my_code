# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 00:22:52; script `src/scripts/diag_regret.py`; took 9.4 min.
- Model: v4.5.1 MLP Mask PPO (50k instances) (seed 2, 20M), replayed deterministically on the first 400 instances of the seed-2 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

355 completed instances, 45 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 8.4e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1804 |
| PPO regret per step: mean / median | 0.0120 / 0.0033 |
| Share of steps where PPO picks an optimal action | 40.5% |
| Legal actions / optimal actions per step (mean) | 5.09 / 1.25 |
| Share of steps with a unique optimal action | 79.9% |
| Steps with regret per instance (mean) | 8.92 |
| Share of the gap from steps 1–5 | 44.6% |
| Share of the gap from steps 6–10 | 40.6% |
| Share of the gap from steps 11–15 | 14.8% |
| Regret at steps followed by a forced opening: share of regret | 96.0% |
| Regret at steps that are themselves forced openings: share of regret | 40.3% |
| Steps from a regret step to the next forced opening (mean) | 1.95 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 6.18 / 7 (45 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0134 | 7.4% | 80.8% | 7.27 | 1.53 | 19.2% |
| 2 | 0.0153 | 8.5% | 82.0% | 6.84 | 1.48 | 18.0% |
| 3 | 0.0163 | 9.0% | 77.5% | 6.53 | 1.38 | 22.5% |
| 4 | 0.0183 | 10.1% | 81.7% | 6.22 | 1.32 | 18.3% |
| 5 | 0.0171 | 9.5% | 74.9% | 6.03 | 1.33 | 25.1% |
| 6 | 0.0153 | 8.5% | 67.3% | 5.77 | 1.25 | 32.7% |
| 7 | 0.0154 | 8.5% | 65.9% | 5.51 | 1.23 | 34.1% |
| 8 | 0.0153 | 8.5% | 64.8% | 5.21 | 1.27 | 35.2% |
| 9 | 0.0151 | 8.4% | 62.3% | 4.86 | 1.23 | 37.7% |
| 10 | 0.0121 | 6.7% | 56.1% | 4.48 | 1.20 | 43.9% |
| 11 | 0.0093 | 5.2% | 49.3% | 4.17 | 1.16 | 50.7% |
| 12 | 0.0083 | 4.6% | 41.7% | 3.73 | 1.12 | 58.3% |
| 13 | 0.0045 | 2.5% | 31.0% | 3.34 | 1.10 | 69.0% |
| 14 | 0.0026 | 1.5% | 26.5% | 3.30 | 1.08 | 73.5% |
| 15 | 0.0019 | 1.1% | 30.4% | 3.06 | 1.03 | 69.6% |

## EQ (M = 10)

399 completed instances, 1 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1428 |
| PPO regret per step: mean / median | 0.0143 / 0.0001 |
| Share of steps where PPO picks an optimal action | 50.0% |
| Legal actions / optimal actions per step (mean) | 5.76 / 1.23 |
| Share of steps with a unique optimal action | 80.5% |
| Steps with regret per instance (mean) | 5.00 |
| Share of the gap from steps 1–3 | 48.9% |
| Share of the gap from steps 4–6 | 37.3% |
| Share of the gap from steps 7–10 | 13.8% |
| Regret at steps followed by a forced opening: share of regret | 95.1% |
| Regret at steps that are themselves forced openings: share of regret | 61.0% |
| Steps from a regret step to the next forced opening (mean) | 1.27 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 1.00 / 1 (1 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0228 | 15.9% | 71.2% | 7.34 | 1.39 | 28.8% |
| 2 | 0.0237 | 16.6% | 72.4% | 6.93 | 1.35 | 27.6% |
| 3 | 0.0235 | 16.4% | 66.9% | 6.58 | 1.33 | 33.1% |
| 4 | 0.0185 | 13.0% | 60.4% | 6.32 | 1.24 | 39.6% |
| 5 | 0.0199 | 13.9% | 57.6% | 5.96 | 1.24 | 42.4% |
| 6 | 0.0149 | 10.4% | 47.1% | 5.55 | 1.19 | 52.9% |
| 7 | 0.0096 | 6.7% | 38.6% | 5.14 | 1.19 | 61.4% |
| 8 | 0.0049 | 3.4% | 32.1% | 4.83 | 1.16 | 67.9% |
| 9 | 0.0033 | 2.3% | 27.6% | 4.54 | 1.12 | 72.4% |
| 10 | 0.0019 | 1.3% | 26.1% | 4.41 | 1.06 | 73.9% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1505 |
| PPO regret per step: mean / median | 0.0150 / 0.0008 |
| Share of steps where PPO picks an optimal action | 48.1% |
| Legal actions / optimal actions per step (mean) | 9.93 / 1.30 |
| Share of steps with a unique optimal action | 75.5% |
| Steps with regret per instance (mean) | 5.19 |
| Share of the gap from steps 1–3 | 49.3% |
| Share of the gap from steps 4–6 | 33.6% |
| Share of the gap from steps 7–10 | 17.1% |
| Regret at steps followed by a forced opening: share of regret | 94.5% |
| Regret at steps that are themselves forced openings: share of regret | 68.1% |
| Steps from a regret step to the next forced opening (mean) | 1.19 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0257 | 17.1% | 72.8% | 10.93 | 1.54 | 27.3% |
| 2 | 0.0255 | 17.0% | 70.8% | 10.73 | 1.45 | 29.2% |
| 3 | 0.0230 | 15.3% | 68.8% | 10.63 | 1.40 | 31.2% |
| 4 | 0.0185 | 12.3% | 61.3% | 10.47 | 1.38 | 38.8% |
| 5 | 0.0186 | 12.3% | 54.8% | 10.25 | 1.38 | 45.2% |
| 6 | 0.0134 | 8.9% | 49.8% | 9.87 | 1.28 | 50.2% |
| 7 | 0.0109 | 7.2% | 43.2% | 9.54 | 1.22 | 56.8% |
| 8 | 0.0080 | 5.3% | 36.8% | 9.18 | 1.19 | 63.2% |
| 9 | 0.0046 | 3.1% | 34.5% | 8.97 | 1.14 | 65.5% |
| 10 | 0.0022 | 1.5% | 26.5% | 8.71 | 1.04 | 73.5% |
