# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 21:53:50; script `src/scripts/diag_regret.py`; took 9.2 min.
- Model: v4.5.2 Transformer Mask-DDQN (50k) (seed 1, 20M), replayed deterministically on the first 400 instances of the seed-1 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

390 completed instances, 10 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0915 |
| PPO regret per step: mean / median | 0.0061 / 0.0000 |
| Share of steps where PPO picks an optimal action | 57.3% |
| Legal actions / optimal actions per step (mean) | 5.06 / 1.26 |
| Share of steps with a unique optimal action | 78.5% |
| Steps with regret per instance (mean) | 6.40 |
| Share of the gap from steps 1–5 | 46.6% |
| Share of the gap from steps 6–10 | 44.1% |
| Share of the gap from steps 11–15 | 9.4% |
| Regret at steps followed by a forced opening: share of regret | 77.1% |
| Regret at steps that are themselves forced openings: share of regret | 41.3% |
| Steps from a regret step to the next forced opening (mean) | 2.08 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 5.30 / 4 (10 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0087 | 9.5% | 73.6% | 7.41 | 1.55 | 26.4% |
| 2 | 0.0084 | 9.2% | 67.9% | 6.86 | 1.48 | 32.1% |
| 3 | 0.0097 | 10.6% | 62.1% | 6.43 | 1.42 | 37.9% |
| 4 | 0.0089 | 9.7% | 58.7% | 6.02 | 1.38 | 41.3% |
| 5 | 0.0069 | 7.5% | 54.4% | 5.70 | 1.33 | 45.6% |
| 6 | 0.0097 | 10.6% | 60.3% | 5.46 | 1.28 | 39.7% |
| 7 | 0.0093 | 10.2% | 51.5% | 5.32 | 1.27 | 48.5% |
| 8 | 0.0074 | 8.1% | 48.5% | 5.12 | 1.22 | 51.5% |
| 9 | 0.0080 | 8.8% | 43.8% | 4.84 | 1.19 | 56.2% |
| 10 | 0.0058 | 6.4% | 35.4% | 4.54 | 1.19 | 64.6% |
| 11 | 0.0038 | 4.2% | 30.0% | 4.19 | 1.16 | 70.0% |
| 12 | 0.0030 | 3.2% | 22.1% | 3.83 | 1.13 | 77.9% |
| 13 | 0.0014 | 1.5% | 18.5% | 3.65 | 1.15 | 81.5% |
| 14 | 0.0004 | 0.4% | 9.7% | 3.42 | 1.11 | 90.3% |
| 15 | 0.0001 | 0.1% | 3.8% | 3.16 | 1.04 | 96.2% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0420 |
| PPO regret per step: mean / median | 0.0042 / 0.0000 |
| Share of steps where PPO picks an optimal action | 74.6% |
| Legal actions / optimal actions per step (mean) | 6.13 / 1.25 |
| Share of steps with a unique optimal action | 80.0% |
| Steps with regret per instance (mean) | 2.54 |
| Share of the gap from steps 1–3 | 55.8% |
| Share of the gap from steps 4–6 | 37.5% |
| Share of the gap from steps 7–10 | 6.7% |
| Regret at steps followed by a forced opening: share of regret | 85.6% |
| Regret at steps that are themselves forced openings: share of regret | 61.6% |
| Steps from a regret step to the next forced opening (mean) | 1.34 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0073 | 17.4% | 50.7% | 7.39 | 1.42 | 49.2% |
| 2 | 0.0076 | 18.1% | 44.2% | 6.93 | 1.41 | 55.8% |
| 3 | 0.0086 | 20.4% | 41.0% | 6.64 | 1.31 | 59.0% |
| 4 | 0.0076 | 18.0% | 34.8% | 6.38 | 1.35 | 65.2% |
| 5 | 0.0050 | 12.0% | 29.2% | 6.14 | 1.31 | 70.8% |
| 6 | 0.0032 | 7.5% | 23.0% | 5.94 | 1.23 | 77.0% |
| 7 | 0.0019 | 4.4% | 15.8% | 5.76 | 1.17 | 84.2% |
| 8 | 0.0008 | 1.8% | 8.2% | 5.60 | 1.11 | 91.8% |
| 9 | 0.0002 | 0.4% | 5.0% | 5.33 | 1.09 | 95.0% |
| 10 | 0.0000 | 0.1% | 2.5% | 5.17 | 1.04 | 97.5% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0265 |
| PPO regret per step: mean / median | 0.0026 / 0.0000 |
| Share of steps where PPO picks an optimal action | 82.3% |
| Legal actions / optimal actions per step (mean) | 10.54 / 1.30 |
| Share of steps with a unique optimal action | 76.9% |
| Steps with regret per instance (mean) | 1.77 |
| Share of the gap from steps 1–3 | 57.4% |
| Share of the gap from steps 4–6 | 37.2% |
| Share of the gap from steps 7–10 | 5.3% |
| Regret at steps followed by a forced opening: share of regret | 91.9% |
| Regret at steps that are themselves forced openings: share of regret | 67.8% |
| Steps from a regret step to the next forced opening (mean) | 1.28 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0045 | 16.9% | 34.2% | 11.00 | 1.57 | 65.8% |
| 2 | 0.0049 | 18.5% | 33.8% | 10.82 | 1.55 | 66.2% |
| 3 | 0.0058 | 22.1% | 31.8% | 10.81 | 1.42 | 68.2% |
| 4 | 0.0049 | 18.6% | 28.5% | 10.79 | 1.42 | 71.5% |
| 5 | 0.0029 | 11.1% | 21.5% | 10.65 | 1.34 | 78.5% |
| 6 | 0.0020 | 7.6% | 13.2% | 10.53 | 1.25 | 86.8% |
| 7 | 0.0010 | 3.7% | 6.5% | 10.40 | 1.23 | 93.5% |
| 8 | 0.0003 | 1.2% | 4.2% | 10.27 | 1.12 | 95.8% |
| 9 | 0.0001 | 0.3% | 2.2% | 10.13 | 1.08 | 97.8% |
| 10 | 0.0000 | 0.1% | 1.0% | 9.95 | 1.03 | 99.0% |
