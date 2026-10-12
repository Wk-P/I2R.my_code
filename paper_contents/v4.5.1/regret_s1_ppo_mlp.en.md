# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 00:13:29; script `src/scripts/diag_regret.py`; took 9.8 min.
- Model: v4.5.1 MLP Mask PPO (50k instances) (seed 1, 20M), replayed deterministically on the first 400 instances of the seed-1 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

365 completed instances, 35 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1704 |
| PPO regret per step: mean / median | 0.0114 / 0.0024 |
| Share of steps where PPO picks an optimal action | 41.5% |
| Legal actions / optimal actions per step (mean) | 5.06 / 1.25 |
| Share of steps with a unique optimal action | 80.0% |
| Steps with regret per instance (mean) | 8.77 |
| Share of the gap from steps 1–5 | 45.2% |
| Share of the gap from steps 6–10 | 41.5% |
| Share of the gap from steps 11–15 | 13.3% |
| Regret at steps followed by a forced opening: share of regret | 94.7% |
| Regret at steps that are themselves forced openings: share of regret | 41.9% |
| Steps from a regret step to the next forced opening (mean) | 1.88 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 6.23 / 6 (35 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0133 | 7.8% | 81.4% | 7.47 | 1.55 | 18.6% |
| 2 | 0.0150 | 8.8% | 81.9% | 7.01 | 1.45 | 18.1% |
| 3 | 0.0160 | 9.4% | 77.3% | 6.64 | 1.36 | 22.7% |
| 4 | 0.0151 | 8.9% | 75.6% | 6.31 | 1.35 | 24.4% |
| 5 | 0.0176 | 10.3% | 72.9% | 5.98 | 1.30 | 27.1% |
| 6 | 0.0160 | 9.4% | 71.5% | 5.72 | 1.28 | 28.5% |
| 7 | 0.0146 | 8.6% | 66.0% | 5.43 | 1.28 | 34.0% |
| 8 | 0.0145 | 8.5% | 60.3% | 5.10 | 1.23 | 39.7% |
| 9 | 0.0131 | 7.7% | 60.8% | 4.72 | 1.18 | 39.2% |
| 10 | 0.0125 | 7.3% | 54.2% | 4.37 | 1.19 | 45.8% |
| 11 | 0.0071 | 4.1% | 44.4% | 3.94 | 1.12 | 55.6% |
| 12 | 0.0063 | 3.7% | 39.2% | 3.55 | 1.14 | 60.8% |
| 13 | 0.0043 | 2.5% | 35.1% | 3.48 | 1.14 | 64.9% |
| 14 | 0.0029 | 1.7% | 30.1% | 3.24 | 1.08 | 69.9% |
| 15 | 0.0021 | 1.2% | 26.6% | 3.01 | 1.05 | 73.4% |

## EQ (M = 10)

399 completed instances, 1 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1406 |
| PPO regret per step: mean / median | 0.0141 / 0.0000 |
| Share of steps where PPO picks an optimal action | 51.0% |
| Legal actions / optimal actions per step (mean) | 5.77 / 1.24 |
| Share of steps with a unique optimal action | 80.0% |
| Steps with regret per instance (mean) | 4.90 |
| Share of the gap from steps 1–3 | 50.4% |
| Share of the gap from steps 4–6 | 36.2% |
| Share of the gap from steps 7–10 | 13.4% |
| Regret at steps followed by a forced opening: share of regret | 95.1% |
| Regret at steps that are themselves forced openings: share of regret | 58.4% |
| Steps from a regret step to the next forced opening (mean) | 1.28 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 3.00 / 3 (1 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0237 | 16.9% | 71.4% | 7.40 | 1.42 | 28.6% |
| 2 | 0.0213 | 15.2% | 67.7% | 6.92 | 1.42 | 32.3% |
| 3 | 0.0258 | 18.4% | 68.2% | 6.58 | 1.30 | 31.8% |
| 4 | 0.0204 | 14.5% | 63.7% | 6.22 | 1.32 | 36.3% |
| 5 | 0.0148 | 10.5% | 51.4% | 5.85 | 1.27 | 48.6% |
| 6 | 0.0156 | 11.1% | 47.4% | 5.52 | 1.17 | 52.6% |
| 7 | 0.0074 | 5.3% | 36.3% | 5.23 | 1.18 | 63.7% |
| 8 | 0.0064 | 4.5% | 32.6% | 4.88 | 1.14 | 67.4% |
| 9 | 0.0028 | 2.0% | 24.3% | 4.58 | 1.11 | 75.7% |
| 10 | 0.0024 | 1.7% | 27.6% | 4.51 | 1.05 | 72.4% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1560 |
| PPO regret per step: mean / median | 0.0156 / 0.0014 |
| Share of steps where PPO picks an optimal action | 45.5% |
| Legal actions / optimal actions per step (mean) | 9.97 / 1.30 |
| Share of steps with a unique optimal action | 76.3% |
| Steps with regret per instance (mean) | 5.45 |
| Share of the gap from steps 1–3 | 46.3% |
| Share of the gap from steps 4–6 | 38.1% |
| Share of the gap from steps 7–10 | 15.7% |
| Regret at steps followed by a forced opening: share of regret | 94.3% |
| Regret at steps that are themselves forced openings: share of regret | 69.6% |
| Steps from a regret step to the next forced opening (mean) | 1.17 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0241 | 15.5% | 72.0% | 11.00 | 1.57 | 28.0% |
| 2 | 0.0236 | 15.1% | 72.0% | 10.80 | 1.47 | 28.0% |
| 3 | 0.0245 | 15.7% | 67.0% | 10.71 | 1.40 | 33.0% |
| 4 | 0.0240 | 15.4% | 67.5% | 10.56 | 1.37 | 32.5% |
| 5 | 0.0185 | 11.9% | 58.0% | 10.27 | 1.31 | 42.0% |
| 6 | 0.0169 | 10.8% | 52.5% | 9.94 | 1.27 | 47.5% |
| 7 | 0.0119 | 7.6% | 47.5% | 9.59 | 1.25 | 52.5% |
| 8 | 0.0060 | 3.9% | 43.8% | 9.16 | 1.21 | 56.2% |
| 9 | 0.0041 | 2.6% | 31.8% | 8.94 | 1.14 | 68.2% |
| 10 | 0.0024 | 1.6% | 33.2% | 8.74 | 1.07 | 66.8% |
