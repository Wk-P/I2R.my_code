# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 00:32:47; script `src/scripts/diag_regret.py`; took 9.9 min.
- Model: v4.5.1 MLP Mask PPO (50k instances) (seed 3, 20M), replayed deterministically on the first 400 instances of the seed-3 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

361 completed instances, 39 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 6.6e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1707 |
| PPO regret per step: mean / median | 0.0114 / 0.0022 |
| Share of steps where PPO picks an optimal action | 42.0% |
| Legal actions / optimal actions per step (mean) | 5.05 / 1.23 |
| Share of steps with a unique optimal action | 81.1% |
| Steps with regret per instance (mean) | 8.69 |
| Share of the gap from steps 1–5 | 45.5% |
| Share of the gap from steps 6–10 | 39.3% |
| Share of the gap from steps 11–15 | 15.2% |
| Regret at steps followed by a forced opening: share of regret | 93.2% |
| Regret at steps that are themselves forced openings: share of regret | 40.5% |
| Steps from a regret step to the next forced opening (mean) | 1.89 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 6.59 / 6 (39 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0120 | 7.0% | 79.5% | 7.45 | 1.46 | 20.5% |
| 2 | 0.0149 | 8.7% | 80.1% | 6.95 | 1.44 | 19.9% |
| 3 | 0.0174 | 10.2% | 78.4% | 6.55 | 1.34 | 21.6% |
| 4 | 0.0170 | 9.9% | 73.1% | 6.21 | 1.37 | 26.9% |
| 5 | 0.0164 | 9.6% | 70.4% | 5.85 | 1.32 | 29.6% |
| 6 | 0.0161 | 9.4% | 71.5% | 5.65 | 1.23 | 28.5% |
| 7 | 0.0161 | 9.4% | 67.6% | 5.36 | 1.21 | 32.4% |
| 8 | 0.0130 | 7.6% | 62.3% | 5.08 | 1.23 | 37.7% |
| 9 | 0.0121 | 7.1% | 58.7% | 4.67 | 1.19 | 41.3% |
| 10 | 0.0098 | 5.8% | 49.3% | 4.35 | 1.18 | 50.7% |
| 11 | 0.0097 | 5.7% | 49.0% | 4.02 | 1.15 | 51.0% |
| 12 | 0.0067 | 3.9% | 39.1% | 3.65 | 1.11 | 60.9% |
| 13 | 0.0043 | 2.5% | 34.1% | 3.44 | 1.07 | 65.9% |
| 14 | 0.0031 | 1.8% | 27.7% | 3.25 | 1.10 | 72.3% |
| 15 | 0.0022 | 1.3% | 28.5% | 3.19 | 1.03 | 71.5% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1417 |
| PPO regret per step: mean / median | 0.0142 / 0.0002 |
| Share of steps where PPO picks an optimal action | 49.9% |
| Legal actions / optimal actions per step (mean) | 5.86 / 1.21 |
| Share of steps with a unique optimal action | 81.9% |
| Steps with regret per instance (mean) | 5.01 |
| Share of the gap from steps 1–3 | 46.8% |
| Share of the gap from steps 4–6 | 37.1% |
| Share of the gap from steps 7–10 | 16.1% |
| Regret at steps followed by a forced opening: share of regret | 95.6% |
| Regret at steps that are themselves forced openings: share of regret | 55.1% |
| Steps from a regret step to the next forced opening (mean) | 1.34 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0235 | 16.6% | 74.8% | 7.41 | 1.43 | 25.2% |
| 2 | 0.0209 | 14.8% | 69.0% | 7.00 | 1.33 | 31.0% |
| 3 | 0.0219 | 15.5% | 65.5% | 6.62 | 1.30 | 34.5% |
| 4 | 0.0195 | 13.7% | 60.5% | 6.29 | 1.23 | 39.5% |
| 5 | 0.0167 | 11.8% | 60.8% | 5.98 | 1.17 | 39.2% |
| 6 | 0.0163 | 11.5% | 48.2% | 5.67 | 1.16 | 51.7% |
| 7 | 0.0097 | 6.8% | 36.8% | 5.30 | 1.13 | 63.2% |
| 8 | 0.0070 | 5.0% | 32.2% | 5.04 | 1.14 | 67.8% |
| 9 | 0.0041 | 2.9% | 28.7% | 4.78 | 1.12 | 71.2% |
| 10 | 0.0020 | 1.4% | 24.5% | 4.46 | 1.06 | 75.5% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1507 |
| PPO regret per step: mean / median | 0.0151 / 0.0014 |
| Share of steps where PPO picks an optimal action | 45.6% |
| Legal actions / optimal actions per step (mean) | 9.89 / 1.32 |
| Share of steps with a unique optimal action | 74.6% |
| Steps with regret per instance (mean) | 5.44 |
| Share of the gap from steps 1–3 | 45.2% |
| Share of the gap from steps 4–6 | 38.2% |
| Share of the gap from steps 7–10 | 16.6% |
| Regret at steps followed by a forced opening: share of regret | 94.4% |
| Regret at steps that are themselves forced openings: share of regret | 69.2% |
| Steps from a regret step to the next forced opening (mean) | 1.16 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0252 | 16.7% | 76.8% | 10.91 | 1.51 | 23.2% |
| 2 | 0.0218 | 14.4% | 71.2% | 10.76 | 1.52 | 28.7% |
| 3 | 0.0211 | 14.0% | 72.0% | 10.53 | 1.38 | 28.0% |
| 4 | 0.0218 | 14.4% | 64.5% | 10.49 | 1.40 | 35.5% |
| 5 | 0.0186 | 12.3% | 59.5% | 10.21 | 1.38 | 40.5% |
| 6 | 0.0172 | 11.4% | 54.8% | 9.88 | 1.29 | 45.2% |
| 7 | 0.0102 | 6.8% | 42.8% | 9.56 | 1.27 | 57.2% |
| 8 | 0.0074 | 4.9% | 41.8% | 9.08 | 1.23 | 58.2% |
| 9 | 0.0049 | 3.2% | 30.2% | 8.82 | 1.15 | 69.8% |
| 10 | 0.0026 | 1.7% | 30.5% | 8.69 | 1.07 | 69.5% |
