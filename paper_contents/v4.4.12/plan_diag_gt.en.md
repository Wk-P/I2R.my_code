# What the global plan P does (GT, v4.4.12 models)

- Generated 2026-10-09 22:38:41; script `src/scripts/diag_plan.py`; took 11.0 min.
- Along the plan-conditioned policy's own deterministic trajectories (first 400 instances of each seed's test split); at every step the ECU logit is P[i_t, j] + Δℓ_j(s_t). Recorded: std of the P row and of Δℓ over the legal ECUs; action taken = argmax(P + Δℓ), counterfactual = argmax(Δℓ) (what the per-step score alone would pick).
- Both actions' regrets are computed with the ILP at the same state (placements so far fixed, the rest completed optimally, as in `diag_regret.py`), i.e. a counterfactual comparison at the same state.
- "P changes the action" = the two actions differ; "P better / worse" = regret differs by more than 1e−4; "net gain" = Σ(counterfactual regret − actual regret) / steps in the phase, positive = regret P removes per step.

## Seed 1 (exp_id 9d101677, 400 completed instances, 0 EXIT)

| Steps | n | Actual regret | Median std(P) / std(Δℓ) | P changes action | When changed: P's regret | When changed: counterfactual regret | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|---|---|---|
| 1–3 | 1200 | 0.0116 | 0.04 | 2.8% | 0.0079 | 0.0134 | 48.5% | 27.3% | +1.53 |
| 4–6 | 1200 | 0.0093 | 0.05 | 2.5% | 0.0105 | 0.0104 | 56.7% | 26.7% | -0.03 |
| 7–10 | 1600 | 0.0024 | 0.05 | 1.8% | 0.0098 | 0.0106 | 62.1% | 17.2% | +0.14 |
| all | 4000 | 0.0072 | 0.05 | 2.3% | 0.0093 | 0.0115 | 55.4% | 23.9% | +0.51 |

By step:

| Step | Actual regret | std(P)/std(Δℓ) | P changes action | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|
| 1 | 0.0130 | 0.04 | 2.2% | 55.6% | 22.2% | +2.83 |
| 2 | 0.0115 | 0.04 | 2.2% | 55.6% | 22.2% | +0.04 |
| 3 | 0.0103 | 0.05 | 3.8% | 40.0% | 33.3% | +1.71 |
| 4 | 0.0107 | 0.05 | 2.8% | 45.5% | 36.4% | -0.10 |
| 5 | 0.0095 | 0.05 | 2.5% | 70.0% | 20.0% | +1.81 |
| 6 | 0.0077 | 0.04 | 2.2% | 55.6% | 22.2% | -1.79 |
| 7 | 0.0046 | 0.04 | 2.8% | 54.5% | 18.2% | +1.69 |
| 8 | 0.0027 | 0.04 | 2.2% | 88.9% | 11.1% | +2.28 |
| 9 | 0.0015 | 0.05 | 2.2% | 44.4% | 22.2% | -3.41 |
| 10 | 0.0009 | 0.07 | 0.0% | nan% | nan% | +0.00 |

## Seed 2 (exp_id c3b72808, 400 completed instances, 0 EXIT)

| Steps | n | Actual regret | Median std(P) / std(Δℓ) | P changes action | When changed: P's regret | When changed: counterfactual regret | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|---|---|---|
| 1–3 | 1200 | 0.0117 | 0.01 | 0.8% | 0.0133 | 0.0228 | 88.9% | 11.1% | +0.71 |
| 4–6 | 1200 | 0.0095 | 0.01 | 0.7% | 0.0181 | 0.0250 | 62.5% | 25.0% | +0.45 |
| 7–10 | 1600 | 0.0019 | 0.01 | 0.4% | 0.0263 | 0.0044 | 66.7% | 33.3% | -0.82 |
| all | 4000 | 0.0071 | 0.01 | 0.6% | 0.0184 | 0.0188 | 73.9% | 21.7% | +0.02 |

By step:

| Step | Actual regret | std(P)/std(Δℓ) | P changes action | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|
| 1 | 0.0122 | 0.02 | 0.5% | 100.0% | 0.0% | +0.31 |
| 2 | 0.0118 | 0.01 | 0.5% | 100.0% | 0.0% | +0.58 |
| 3 | 0.0111 | 0.01 | 1.2% | 80.0% | 20.0% | +1.25 |
| 4 | 0.0115 | 0.01 | 0.8% | 33.3% | 33.3% | +0.08 |
| 5 | 0.0104 | 0.01 | 0.5% | 100.0% | 0.0% | +0.86 |
| 6 | 0.0066 | 0.01 | 0.8% | 66.7% | 33.3% | +0.42 |
| 7 | 0.0045 | 0.01 | 0.0% | nan% | nan% | +0.00 |
| 8 | 0.0024 | 0.01 | 1.0% | 50.0% | 50.0% | -3.34 |
| 9 | 0.0008 | 0.01 | 0.5% | 100.0% | 0.0% | +0.07 |
| 10 | 0.0000 | 0.01 | 0.0% | nan% | nan% | +0.00 |

## Seed 3 (exp_id 07735885, 400 completed instances, 0 EXIT)

| Steps | n | Actual regret | Median std(P) / std(Δℓ) | P changes action | When changed: P's regret | When changed: counterfactual regret | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|---|---|---|
| 1–3 | 1200 | 0.0113 | 0.05 | 2.0% | 0.0158 | 0.0152 | 50.0% | 29.2% | -0.12 |
| 4–6 | 1200 | 0.0084 | 0.05 | 1.7% | 0.0113 | 0.0157 | 45.0% | 20.0% | +0.73 |
| 7–10 | 1600 | 0.0036 | 0.06 | 1.8% | 0.0354 | 0.0069 | 50.0% | 42.9% | -4.99 |
| all | 4000 | 0.0074 | 0.06 | 1.8% | 0.0222 | 0.0121 | 48.6% | 31.9% | -1.81 |

By step:

| Step | Actual regret | std(P)/std(Δℓ) | P changes action | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|
| 1 | 0.0112 | 0.06 | 1.8% | 71.4% | 14.3% | +1.19 |
| 2 | 0.0115 | 0.05 | 2.2% | 44.4% | 33.3% | -0.38 |
| 3 | 0.0112 | 0.05 | 2.0% | 37.5% | 37.5% | -1.16 |
| 4 | 0.0109 | 0.05 | 2.5% | 40.0% | 20.0% | +1.33 |
| 5 | 0.0074 | 0.05 | 1.5% | 33.3% | 33.3% | -0.89 |
| 6 | 0.0070 | 0.05 | 1.0% | 75.0% | 0.0% | +1.75 |
| 7 | 0.0060 | 0.05 | 1.0% | 50.0% | 50.0% | -2.86 |
| 8 | 0.0041 | 0.05 | 2.2% | 66.7% | 11.1% | +1.01 |
| 9 | 0.0011 | 0.06 | 1.2% | 100.0% | 0.0% | +1.35 |
| 10 | 0.0033 | 0.07 | 2.5% | 10.0% | 90.0% | -19.45 |
