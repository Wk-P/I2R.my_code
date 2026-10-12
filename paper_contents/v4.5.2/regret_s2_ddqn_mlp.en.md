# Per-step regret decomposition (measured with the ILP, no training)

- Generated 2026-10-12 21:35:42; script `src/scripts/diag_regret.py`; took 8.6 min.
- Model: v4.5.2 MLP Mask-DDQN (50k) (seed 2, 20M), replayed deterministically on the first 400 instances of the seed-2 test split.
- V*(s_t) = the final AR reachable when the placements made so far are fixed and the rest is completed optimally by the ILP (Dinkelbach); Regret(s_t, a) = V*(s_t) − V*(after a). Summed along the policy's own trajectory it equals the instance's absolute gap (ILP AR − AR).
- Optimal action = a legal action with Regret ≤ 0.0001 (tolerance against ILP numerical error). Empty ECUs of equal capacity are interchangeable: solved once, but each counts as an action.
- EXIT instances are reported separately: the first step after which the ILP has no feasible completion.

## LT (M = 15)

326 completed instances, 74 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 1.2e-02 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.2130 |
| PPO regret per step: mean / median | 0.0142 / 0.0042 |
| Share of steps where PPO picks an optimal action | 36.7% |
| Legal actions / optimal actions per step (mean) | 4.89 / 1.23 |
| Share of steps with a unique optimal action | 80.8% |
| Steps with regret per instance (mean) | 9.50 |
| Share of the gap from steps 1–5 | 41.1% |
| Share of the gap from steps 6–10 | 36.5% |
| Share of the gap from steps 11–15 | 22.5% |
| Regret at steps followed by a forced opening: share of regret | 86.1% |
| Regret at steps that are themselves forced openings: share of regret | 44.8% |
| Steps from a regret step to the next forced opening (mean) | 1.64 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 6.99 / 7 (74 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0148 | 7.0% | 81.6% | 7.30 | 1.53 | 18.4% |
| 2 | 0.0171 | 8.0% | 76.7% | 6.77 | 1.42 | 23.3% |
| 3 | 0.0199 | 9.4% | 76.1% | 6.44 | 1.35 | 23.9% |
| 4 | 0.0169 | 8.0% | 73.6% | 6.09 | 1.29 | 26.4% |
| 5 | 0.0187 | 8.8% | 74.8% | 5.81 | 1.28 | 25.2% |
| 6 | 0.0157 | 7.4% | 70.6% | 5.42 | 1.24 | 29.4% |
| 7 | 0.0179 | 8.4% | 68.4% | 5.17 | 1.21 | 31.6% |
| 8 | 0.0159 | 7.5% | 66.3% | 4.90 | 1.24 | 33.7% |
| 9 | 0.0145 | 6.8% | 60.7% | 4.50 | 1.20 | 39.3% |
| 10 | 0.0136 | 6.4% | 60.4% | 4.23 | 1.18 | 39.6% |
| 11 | 0.0124 | 5.8% | 56.1% | 3.92 | 1.14 | 43.9% |
| 12 | 0.0095 | 4.5% | 51.2% | 3.60 | 1.13 | 48.8% |
| 13 | 0.0111 | 5.2% | 49.7% | 3.19 | 1.10 | 50.3% |
| 14 | 0.0094 | 4.4% | 48.2% | 3.09 | 1.13 | 51.8% |
| 15 | 0.0055 | 2.6% | 35.3% | 2.87 | 1.05 | 64.7% |

## EQ (M = 10)

398 completed instances, 2 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.0e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.1735 |
| PPO regret per step: mean / median | 0.0174 / 0.0039 |
| Share of steps where PPO picks an optimal action | 40.2% |
| Legal actions / optimal actions per step (mean) | 5.98 / 1.22 |
| Share of steps with a unique optimal action | 81.3% |
| Steps with regret per instance (mean) | 5.98 |
| Share of the gap from steps 1–3 | 42.2% |
| Share of the gap from steps 4–6 | 35.3% |
| Share of the gap from steps 7–10 | 22.4% |
| Regret at steps followed by a forced opening: share of regret | 92.6% |
| Regret at steps that are themselves forced openings: share of regret | 58.5% |
| Steps from a regret step to the next forced opening (mean) | 1.35 |
| EXIT instances: first step with no feasible completion (1-based, mean / median) | 3.00 / 3 (2 instances) |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0263 | 15.1% | 81.4% | 7.35 | 1.39 | 18.6% |
| 2 | 0.0246 | 14.2% | 74.1% | 6.99 | 1.33 | 25.9% |
| 3 | 0.0224 | 12.9% | 72.4% | 6.74 | 1.30 | 27.6% |
| 4 | 0.0252 | 14.5% | 72.9% | 6.54 | 1.23 | 27.1% |
| 5 | 0.0197 | 11.3% | 62.1% | 6.26 | 1.25 | 37.9% |
| 6 | 0.0165 | 9.5% | 59.3% | 5.90 | 1.19 | 40.7% |
| 7 | 0.0164 | 9.5% | 55.5% | 5.53 | 1.20 | 44.5% |
| 8 | 0.0125 | 7.2% | 50.0% | 5.17 | 1.16 | 50.0% |
| 9 | 0.0067 | 3.9% | 41.2% | 4.83 | 1.11 | 58.8% |
| 10 | 0.0033 | 1.9% | 29.4% | 4.51 | 1.04 | 70.6% |

## GT (M = 10)

400 completed instances, 0 with EXIT / no feasible completion on the way. Max deviation between the sum of per-step regrets and the absolute gap: 5.1e-07 (telescoping check).

| Metric | Value |
|---|---|
| Mean absolute gap (= mean Σ regret) | 0.2171 |
| PPO regret per step: mean / median | 0.0217 / 0.0122 |
| Share of steps where PPO picks an optimal action | 32.1% |
| Legal actions / optimal actions per step (mean) | 10.57 / 1.32 |
| Share of steps with a unique optimal action | 74.4% |
| Steps with regret per instance (mean) | 6.79 |
| Share of the gap from steps 1–3 | 41.9% |
| Share of the gap from steps 4–6 | 32.9% |
| Share of the gap from steps 7–10 | 25.2% |
| Regret at steps followed by a forced opening: share of regret | 89.5% |
| Regret at steps that are themselves forced openings: share of regret | 62.3% |
| Steps from a regret step to the next forced opening (mean) | 1.40 |

By step:

| Step | Mean regret | Share of gap | Instances with regret | Legal actions | Optimal actions | PPO picks optimal |
|---|---|---|---|---|---|---|
| 1 | 0.0288 | 13.3% | 84.2% | 10.93 | 1.54 | 15.8% |
| 2 | 0.0318 | 14.6% | 83.0% | 10.83 | 1.47 | 17.0% |
| 3 | 0.0305 | 14.0% | 79.2% | 10.88 | 1.37 | 20.8% |
| 4 | 0.0267 | 12.3% | 75.2% | 10.95 | 1.34 | 24.8% |
| 5 | 0.0239 | 11.0% | 67.0% | 10.99 | 1.34 | 33.0% |
| 6 | 0.0208 | 9.6% | 66.5% | 10.81 | 1.30 | 33.5% |
| 7 | 0.0195 | 9.0% | 66.0% | 10.53 | 1.29 | 34.0% |
| 8 | 0.0166 | 7.7% | 58.5% | 10.20 | 1.22 | 41.5% |
| 9 | 0.0121 | 5.6% | 52.8% | 9.92 | 1.19 | 47.2% |
| 10 | 0.0064 | 2.9% | 46.0% | 9.69 | 1.11 | 54.0% |
