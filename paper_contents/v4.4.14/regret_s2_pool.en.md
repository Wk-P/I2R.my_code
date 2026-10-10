# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-10 13:18:01; script `src/scripts/diag_regret.py`; took 9.7 min.
- Model: v4.4.14 structure-aware Mask PPO, 50k training instances (seed 2, 1M, GPU), replayed deterministically on the first 400 instances of the seed-2 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

385 completed instances, 15 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0909 |
| PPO regret per step: mean / median | 0.0061 / 0.0000 |
| Share of steps where PPO picks an optimal action | 59.1% |
| Legal actions / optimal actions per step (mean) | 5.04 / 1.26 |
| Share of steps with a unique optimal action | 78.6% |
| Steps with regret per instance (mean) | 6.14 |
| Share of the gap from steps 1–5 | 53.6% |
| Share of the gap from steps 6–10 | 37.3% |
| Share of the gap from steps 11–15 | 9.1% |
| Regret at steps followed by a forced opening: share of regret | 82.4% |
| Regret at steps that are themselves forced openings: share of regret | 37.9% |
| Steps from a regret step to the next forced opening (mean) | 2.33 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 5.40 / 5 (15 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0102 | 11.2% | 75.6% | 7.24 | 1.53 | 24.4% |
| 2 | 0.0111 | 12.2% | 69.1% | 6.75 | 1.47 | 30.9% |
| 3 | 0.0093 | 10.2% | 69.4% | 6.42 | 1.36 | 30.6% |
| 4 | 0.0092 | 10.2% | 60.5% | 6.12 | 1.40 | 39.5% |
| 5 | 0.0090 | 9.9% | 60.5% | 5.83 | 1.32 | 39.5% |
| 6 | 0.0094 | 10.3% | 51.7% | 5.57 | 1.30 | 48.3% |
| 7 | 0.0062 | 6.8% | 47.5% | 5.29 | 1.26 | 52.5% |
| 8 | 0.0071 | 7.8% | 43.1% | 5.06 | 1.30 | 56.9% |
| 9 | 0.0059 | 6.5% | 37.4% | 4.77 | 1.18 | 62.6% |
| 10 | 0.0053 | 5.9% | 33.8% | 4.49 | 1.21 | 66.2% |
| 11 | 0.0041 | 4.5% | 25.2% | 4.14 | 1.17 | 74.8% |
| 12 | 0.0025 | 2.7% | 19.0% | 3.81 | 1.14 | 81.0% |
| 13 | 0.0014 | 1.5% | 13.5% | 3.55 | 1.12 | 86.5% |
| 14 | 0.0003 | 0.3% | 7.0% | 3.41 | 1.13 | 93.0% |
| 15 | 0.0000 | 0.0% | 0.8% | 3.22 | 1.03 | 99.2% |

## EQ (M = 10)

399 completed instances, 1 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0710 |
| PPO regret per step: mean / median | 0.0071 / 0.0000 |
| Share of steps where PPO picks an optimal action | 68.5% |
| Legal actions / optimal actions per step (mean) | 6.04 / 1.23 |
| Share of steps with a unique optimal action | 80.7% |
| Steps with regret per instance (mean) | 3.15 |
| Share of the gap from steps 1–3 | 50.0% |
| Share of the gap from steps 4–6 | 35.7% |
| Share of the gap from steps 7–10 | 14.3% |
| Regret at steps followed by a forced opening: share of regret | 89.0% |
| Regret at steps that are themselves forced openings: share of regret | 54.8% |
| Steps from a regret step to the next forced opening (mean) | 1.43 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 3.00 / 3 (1 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0121 | 17.0% | 55.1% | 7.34 | 1.39 | 44.9% |
| 2 | 0.0130 | 18.3% | 53.9% | 6.93 | 1.37 | 46.1% |
| 3 | 0.0105 | 14.7% | 47.1% | 6.61 | 1.33 | 52.9% |
| 4 | 0.0105 | 14.8% | 45.4% | 6.43 | 1.30 | 54.6% |
| 5 | 0.0078 | 10.9% | 38.6% | 6.14 | 1.24 | 61.4% |
| 6 | 0.0072 | 10.1% | 30.3% | 5.87 | 1.19 | 69.7% |
| 7 | 0.0054 | 7.6% | 23.1% | 5.70 | 1.19 | 76.9% |
| 8 | 0.0031 | 4.3% | 14.3% | 5.36 | 1.15 | 85.7% |
| 9 | 0.0017 | 2.3% | 6.8% | 5.10 | 1.11 | 93.2% |
| 10 | 0.0000 | 0.0% | 0.5% | 4.88 | 1.04 | 99.5% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0757 |
| PPO regret per step: mean / median | 0.0076 / 0.0000 |
| Share of steps where PPO picks an optimal action | 65.9% |
| Legal actions / optimal actions per step (mean) | 10.25 / 1.34 |
| Share of steps with a unique optimal action | 72.8% |
| Steps with regret per instance (mean) | 3.41 |
| Share of the gap from steps 1–3 | 54.0% |
| Share of the gap from steps 4–6 | 35.8% |
| Share of the gap from steps 7–10 | 10.1% |
| Regret at steps followed by a forced opening: share of regret | 92.3% |
| Regret at steps that are themselves forced openings: share of regret | 67.5% |
| Steps from a regret step to the next forced opening (mean) | 1.25 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0134 | 17.7% | 57.0% | 10.93 | 1.54 | 43.0% |
| 2 | 0.0140 | 18.4% | 61.0% | 10.73 | 1.48 | 39.0% |
| 3 | 0.0136 | 17.9% | 55.5% | 10.63 | 1.49 | 44.5% |
| 4 | 0.0105 | 13.9% | 47.0% | 10.52 | 1.45 | 53.0% |
| 5 | 0.0088 | 11.6% | 38.8% | 10.41 | 1.43 | 61.3% |
| 6 | 0.0078 | 10.3% | 35.5% | 10.21 | 1.33 | 64.5% |
| 7 | 0.0047 | 6.2% | 26.8% | 10.07 | 1.29 | 73.2% |
| 8 | 0.0020 | 2.6% | 12.8% | 9.84 | 1.21 | 87.2% |
| 9 | 0.0009 | 1.3% | 5.8% | 9.70 | 1.15 | 94.2% |
| 10 | 0.0000 | 0.0% | 0.8% | 9.51 | 1.06 | 99.2% |
