# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 21:00:17; script `src/scripts/diag_regret.py`; took 9.0 min.
- Model: v4.5.2 Transformer Mask-DQN (50k) (seed 1, 20M), replayed deterministically on the first 400 instances of the seed-1 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

390 completed instances, 10 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0825 |
| PPO regret per step: mean / median | 0.0055 / 0.0000 |
| Share of steps where PPO picks an optimal action | 61.1% |
| Legal actions / optimal actions per step (mean) | 5.04 / 1.25 |
| Share of steps with a unique optimal action | 79.1% |
| Steps with regret per instance (mean) | 5.84 |
| Share of the gap from steps 1–5 | 52.0% |
| Share of the gap from steps 6–10 | 39.6% |
| Share of the gap from steps 11–15 | 8.4% |
| Regret at steps followed by a forced opening: share of regret | 74.0% |
| Regret at steps that are themselves forced openings: share of regret | 40.8% |
| Steps from a regret step to the next forced opening (mean) | 1.90 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 2.30 / 2 (10 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0086 | 10.5% | 67.4% | 7.40 | 1.55 | 32.6% |
| 2 | 0.0081 | 9.8% | 65.6% | 6.85 | 1.47 | 34.4% |
| 3 | 0.0092 | 11.1% | 60.5% | 6.42 | 1.37 | 39.5% |
| 4 | 0.0083 | 10.1% | 57.2% | 6.02 | 1.39 | 42.8% |
| 5 | 0.0087 | 10.5% | 50.5% | 5.66 | 1.33 | 49.5% |
| 6 | 0.0074 | 9.0% | 49.0% | 5.40 | 1.26 | 51.0% |
| 7 | 0.0077 | 9.4% | 43.8% | 5.22 | 1.27 | 56.2% |
| 8 | 0.0067 | 8.1% | 43.1% | 5.01 | 1.24 | 56.9% |
| 9 | 0.0054 | 6.5% | 37.4% | 4.76 | 1.21 | 62.6% |
| 10 | 0.0055 | 6.6% | 34.1% | 4.51 | 1.16 | 65.9% |
| 11 | 0.0035 | 4.3% | 25.9% | 4.17 | 1.15 | 74.1% |
| 12 | 0.0019 | 2.3% | 20.8% | 3.87 | 1.14 | 79.2% |
| 13 | 0.0012 | 1.4% | 17.7% | 3.71 | 1.12 | 82.3% |
| 14 | 0.0004 | 0.4% | 9.7% | 3.45 | 1.12 | 90.3% |
| 15 | 0.0000 | 0.0% | 1.0% | 3.23 | 1.02 | 99.0% |

## EQ (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0341 |
| PPO regret per step: mean / median | 0.0034 / 0.0000 |
| Share of steps where PPO picks an optimal action | 78.4% |
| Legal actions / optimal actions per step (mean) | 6.13 / 1.23 |
| Share of steps with a unique optimal action | 81.1% |
| Steps with regret per instance (mean) | 2.16 |
| Share of the gap from steps 1–3 | 57.2% |
| Share of the gap from steps 4–6 | 36.5% |
| Share of the gap from steps 7–10 | 6.3% |
| Regret at steps followed by a forced opening: share of regret | 83.2% |
| Regret at steps that are themselves forced openings: share of regret | 60.7% |
| Steps from a regret step to the next forced opening (mean) | 1.42 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0060 | 17.7% | 47.8% | 7.39 | 1.42 | 52.2% |
| 2 | 0.0071 | 20.8% | 43.8% | 6.95 | 1.39 | 56.2% |
| 3 | 0.0064 | 18.7% | 37.2% | 6.65 | 1.30 | 62.7% |
| 4 | 0.0059 | 17.2% | 30.0% | 6.37 | 1.33 | 70.0% |
| 5 | 0.0038 | 11.2% | 21.5% | 6.14 | 1.27 | 78.5% |
| 6 | 0.0028 | 8.1% | 16.8% | 5.92 | 1.19 | 83.2% |
| 7 | 0.0017 | 4.8% | 8.5% | 5.72 | 1.16 | 91.5% |
| 8 | 0.0004 | 1.1% | 5.2% | 5.59 | 1.11 | 94.8% |
| 9 | 0.0001 | 0.2% | 3.8% | 5.36 | 1.07 | 96.2% |
| 10 | 0.0000 | 0.1% | 1.5% | 5.25 | 1.04 | 98.5% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.0242 |
| PPO regret per step: mean / median | 0.0024 / 0.0000 |
| Share of steps where PPO picks an optimal action | 82.7% |
| Legal actions / optimal actions per step (mean) | 10.59 / 1.30 |
| Share of steps with a unique optimal action | 77.1% |
| Steps with regret per instance (mean) | 1.73 |
| Share of the gap from steps 1–3 | 64.8% |
| Share of the gap from steps 4–6 | 31.2% |
| Share of the gap from steps 7–10 | 4.0% |
| Regret at steps followed by a forced opening: share of regret | 92.7% |
| Regret at steps that are themselves forced openings: share of regret | 68.2% |
| Steps from a regret step to the next forced opening (mean) | 1.35 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0050 | 20.5% | 38.5% | 11.00 | 1.57 | 61.5% |
| 2 | 0.0057 | 23.5% | 37.0% | 10.84 | 1.52 | 63.0% |
| 3 | 0.0050 | 20.7% | 29.8% | 10.83 | 1.44 | 70.2% |
| 4 | 0.0039 | 16.1% | 25.5% | 10.85 | 1.42 | 74.5% |
| 5 | 0.0025 | 10.2% | 18.8% | 10.71 | 1.35 | 81.2% |
| 6 | 0.0012 | 5.0% | 10.8% | 10.59 | 1.22 | 89.2% |
| 7 | 0.0004 | 1.7% | 5.0% | 10.49 | 1.21 | 95.0% |
| 8 | 0.0004 | 1.5% | 4.5% | 10.32 | 1.15 | 95.5% |
| 9 | 0.0002 | 0.9% | 2.8% | 10.23 | 1.07 | 97.2% |
| 10 | 0.0000 | 0.0% | 0.2% | 10.05 | 1.02 | 99.8% |
