# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-09 23:30:14; script `src/scripts/diag_regret.py`; took 9.8 min.
- Model: v4.4.13 structure-aware Mask PPO, professor's 3-dim state summary (seed 3, 1M, GPU), replayed deterministically on the first 400 instances of the seed-3 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

387 completed instances, 13 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 7.4e-03 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0947 |
| PPO regret per step: mean / median | 0.0063 / 0.0000 |
| Share of steps where PPO picks an optimal action | 61.2% |
| Legal actions / optimal actions per step (mean) | 4.78 / 1.25 |
| Share of steps with a unique optimal action | 79.9% |
| Steps with regret per instance (mean) | 5.82 |
| Share of the gap from steps 1–5 | 52.4% |
| Share of the gap from steps 6–10 | 38.4% |
| Share of the gap from steps 11–15 | 9.2% |
| Regret at steps followed by a forced opening: share of regret | 91.5% |
| Regret at steps that are themselves forced openings: share of regret | 48.9% |
| Steps from a regret step to the next forced opening (mean) | 1.74 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 4.92 / 4 (13 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0110 | 11.7% | 73.6% | 7.43 | 1.46 | 26.4% |
| 2 | 0.0100 | 10.5% | 64.9% | 6.81 | 1.44 | 35.1% |
| 3 | 0.0103 | 10.9% | 59.4% | 6.28 | 1.43 | 40.6% |
| 4 | 0.0084 | 8.9% | 58.7% | 5.79 | 1.35 | 41.3% |
| 5 | 0.0099 | 10.4% | 54.3% | 5.40 | 1.30 | 45.7% |
| 6 | 0.0084 | 8.8% | 49.1% | 5.12 | 1.32 | 50.9% |
| 7 | 0.0081 | 8.5% | 43.4% | 4.82 | 1.29 | 56.6% |
| 8 | 0.0061 | 6.4% | 36.2% | 4.58 | 1.23 | 63.8% |
| 9 | 0.0087 | 9.2% | 38.0% | 4.36 | 1.16 | 62.0% |
| 10 | 0.0051 | 5.3% | 30.7% | 4.15 | 1.20 | 69.3% |
| 11 | 0.0040 | 4.2% | 29.5% | 3.94 | 1.15 | 70.5% |
| 12 | 0.0027 | 2.8% | 22.5% | 3.60 | 1.14 | 77.5% |
| 13 | 0.0018 | 1.9% | 14.5% | 3.39 | 1.14 | 85.5% |
| 14 | 0.0002 | 0.2% | 4.1% | 3.14 | 1.11 | 95.9% |
| 15 | 0.0000 | 0.0% | 2.8% | 2.98 | 1.03 | 97.2% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0841 |
| PPO regret per step: mean / median | 0.0084 / 0.0000 |
| Share of steps where PPO picks an optimal action | 66.8% |
| Legal actions / optimal actions per step (mean) | 6.01 / 1.23 |
| Share of steps with a unique optimal action | 79.5% |
| Steps with regret per instance (mean) | 3.33 |
| Share of the gap from steps 1–3 | 53.0% |
| Share of the gap from steps 4–6 | 35.2% |
| Share of the gap from steps 7–10 | 11.7% |
| Regret at steps followed by a forced opening: share of regret | 91.6% |
| Regret at steps that are themselves forced openings: share of regret | 56.6% |
| Steps from a regret step to the next forced opening (mean) | 1.37 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0144 | 17.1% | 63.0% | 7.41 | 1.43 | 37.0% |
| 2 | 0.0161 | 19.1% | 61.3% | 7.00 | 1.34 | 38.8% |
| 3 | 0.0142 | 16.8% | 50.0% | 6.58 | 1.33 | 50.0% |
| 4 | 0.0107 | 12.8% | 43.8% | 6.25 | 1.28 | 56.2% |
| 5 | 0.0097 | 11.6% | 36.8% | 6.03 | 1.24 | 63.2% |
| 6 | 0.0092 | 10.9% | 32.8% | 5.86 | 1.19 | 67.2% |
| 7 | 0.0053 | 6.3% | 22.8% | 5.55 | 1.16 | 77.2% |
| 8 | 0.0042 | 5.0% | 16.2% | 5.39 | 1.14 | 83.8% |
| 9 | 0.0003 | 0.4% | 5.2% | 5.20 | 1.15 | 94.8% |
| 10 | 0.0000 | 0.0% | 0.8% | 4.84 | 1.09 | 99.2% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0860 |
| PPO regret per step: mean / median | 0.0086 / 0.0000 |
| Share of steps where PPO picks an optimal action | 65.0% |
| Legal actions / optimal actions per step (mean) | 10.15 / 1.34 |
| Share of steps with a unique optimal action | 73.8% |
| Steps with regret per instance (mean) | 3.50 |
| Share of the gap from steps 1–3 | 50.5% |
| Share of the gap from steps 4–6 | 35.6% |
| Share of the gap from steps 7–10 | 13.9% |
| Regret at steps followed by a forced opening: share of regret | 90.2% |
| Regret at steps that are themselves forced openings: share of regret | 63.0% |
| Steps from a regret step to the next forced opening (mean) | 1.28 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0157 | 18.3% | 58.0% | 10.91 | 1.51 | 42.0% |
| 2 | 0.0146 | 17.0% | 57.2% | 10.76 | 1.54 | 42.8% |
| 3 | 0.0131 | 15.2% | 55.8% | 10.53 | 1.40 | 44.2% |
| 4 | 0.0137 | 16.0% | 51.0% | 10.52 | 1.44 | 49.0% |
| 5 | 0.0088 | 10.3% | 38.5% | 10.28 | 1.42 | 61.5% |
| 6 | 0.0080 | 9.3% | 33.0% | 10.12 | 1.35 | 67.0% |
| 7 | 0.0066 | 7.7% | 27.5% | 9.98 | 1.27 | 72.5% |
| 8 | 0.0042 | 4.9% | 20.2% | 9.73 | 1.20 | 79.8% |
| 9 | 0.0012 | 1.4% | 6.8% | 9.48 | 1.18 | 93.2% |
| 10 | 0.0000 | 0.0% | 2.0% | 9.23 | 1.09 | 98.0% |
