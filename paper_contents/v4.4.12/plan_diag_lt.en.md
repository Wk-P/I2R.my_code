# What the global plan P does (LT, v4.4.12 models)

- Generated 2026-10-09 22:21:40; script `src/scripts/diag_plan.py`; took 21.3 min.
- Along the plan-conditioned policy's own deterministic trajectories (first 400 instances of each seed's test split); at every step the ECU logit is P[i_t, j] + Δℓ_j(s_t). Recorded: std of the P row and of Δℓ over the legal ECUs; action taken = argmax(P + Δℓ), counterfactual = argmax(Δℓ) (what the per-step score alone would pick).
- Both actions' regrets are computed with the ILP at the same state (placements so far fixed, the rest completed optimally, as in `diag_regret.py`), i.e. a counterfactual comparison at the same state.
- "P changes the action" = the two actions differ; "P better / worse" = regret differs by more than 1e−4; "net gain" = Σ(counterfactual regret − actual regret) / steps in the phase, positive = regret P removes per step.

## Seed 1 (exp_id 1ddcc834, 388 completed instances, 12 EXIT)

| Steps | n | Actual regret | Median std(P) / std(Δℓ) | P changes action | When changed: P's regret | When changed: counterfactual regret | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|---|---|---|
| 1–5 | 1977 | 0.0093 | 0.02 | 1.0% | 0.0110 | 0.0115 | 50.0% | 45.0% | +0.05 |
| 6–10 | 1951 | 0.0078 | 0.01 | 0.5% | 0.0184 | 0.0172 | 77.8% | 22.2% | -0.06 |
| 11–15 | 1940 | 0.0021 | 0.01 | 0.2% | 0.0006 | 0.0108 | 100.0% | 0.0% | +0.16 |
| all | 5868 | 0.0064 | 0.01 | 0.5% | 0.0121 | 0.0130 | 62.5% | 34.4% | +0.05 |

By step:

| Step | Actual regret | std(P)/std(Δℓ) | P changes action | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|
| 1 | 0.0098 | 0.02 | 2.0% | 37.5% | 50.0% | -0.37 |
| 2 | 0.0094 | 0.02 | 1.0% | 25.0% | 75.0% | -0.92 |
| 3 | 0.0098 | 0.02 | 0.8% | 66.7% | 33.3% | -0.13 |
| 4 | 0.0088 | 0.02 | 0.5% | 50.0% | 50.0% | +0.14 |
| 5 | 0.0089 | 0.01 | 0.8% | 100.0% | 0.0% | +1.55 |
| 6 | 0.0086 | 0.01 | 0.8% | 100.0% | 0.0% | +0.40 |
| 7 | 0.0085 | 0.01 | 0.5% | 100.0% | 0.0% | +0.62 |
| 8 | 0.0093 | 0.01 | 0.3% | 0.0% | 100.0% | -1.49 |
| 9 | 0.0062 | 0.01 | 0.3% | 100.0% | 0.0% | +0.41 |
| 10 | 0.0063 | 0.01 | 0.5% | 50.0% | 50.0% | -0.23 |
| 11 | 0.0051 | 0.01 | 0.5% | 100.0% | 0.0% | +0.75 |
| 12 | 0.0036 | 0.01 | 0.3% | 100.0% | 0.0% | +0.04 |
| 13 | 0.0012 | 0.01 | 0.0% | nan% | nan% | +0.00 |
| 14 | 0.0003 | 0.01 | 0.0% | nan% | nan% | +0.00 |
| 15 | 0.0001 | 0.00 | 0.0% | nan% | nan% | +0.00 |

## Seed 2 (exp_id ba469276, 384 completed instances, 16 EXIT)

| Steps | n | Actual regret | Median std(P) / std(Δℓ) | P changes action | When changed: P's regret | When changed: counterfactual regret | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|---|---|---|
| 1–5 | 1975 | 0.0096 | 0.02 | 2.2% | 0.0072 | 0.0082 | 60.5% | 32.6% | +0.21 |
| 6–10 | 1940 | 0.0079 | 0.01 | 0.8% | 0.0098 | 0.0212 | 46.7% | 46.7% | +0.88 |
| 11–15 | 1920 | 0.0076 | 0.01 | 0.8% | 0.0077 | 0.0393 | 60.0% | 33.3% | +2.47 |
| all | 5835 | 0.0084 | 0.02 | 1.3% | 0.0079 | 0.0173 | 57.5% | 35.6% | +1.18 |

By step:

| Step | Actual regret | std(P)/std(Δℓ) | P changes action | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|
| 1 | 0.0098 | 0.03 | 3.0% | 58.3% | 41.7% | +0.44 |
| 2 | 0.0098 | 0.03 | 3.8% | 60.0% | 26.7% | -0.14 |
| 3 | 0.0100 | 0.02 | 1.8% | 71.4% | 28.6% | +0.24 |
| 4 | 0.0092 | 0.02 | 1.3% | 80.0% | 20.0% | +0.69 |
| 5 | 0.0092 | 0.02 | 1.0% | 25.0% | 50.0% | -0.18 |
| 6 | 0.0096 | 0.02 | 1.5% | 16.7% | 83.3% | -1.76 |
| 7 | 0.0074 | 0.01 | 0.5% | 0.0% | 50.0% | -0.04 |
| 8 | 0.0076 | 0.01 | 1.0% | 75.0% | 25.0% | +1.65 |
| 9 | 0.0071 | 0.01 | 0.3% | 100.0% | 0.0% | +1.51 |
| 10 | 0.0075 | 0.01 | 0.5% | 100.0% | 0.0% | +3.08 |
| 11 | 0.0056 | 0.01 | 0.3% | 100.0% | 0.0% | +0.13 |
| 12 | 0.0036 | 0.01 | 0.5% | 0.0% | 100.0% | -0.18 |
| 13 | 0.0023 | 0.01 | 0.5% | 50.0% | 50.0% | -0.27 |
| 14 | 0.0028 | 0.02 | 1.0% | 50.0% | 25.0% | -0.04 |
| 15 | 0.0236 | 0.03 | 1.6% | 83.3% | 16.7% | +12.73 |

## Seed 3 (exp_id d5b2ea85, 392 completed instances, 8 EXIT)

| Steps | n | Actual regret | Median std(P) / std(Δℓ) | P changes action | When changed: P's regret | When changed: counterfactual regret | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|---|---|---|
| 1–5 | 1978 | 0.0092 | 0.00 | 0.0% | nan | nan | nan% | nan% | +0.00 |
| 6–10 | 1969 | 0.0073 | 0.00 | 0.0% | nan | nan | nan% | nan% | +0.00 |
| 11–15 | 1961 | 0.0018 | 0.00 | 0.0% | nan | nan | nan% | nan% | +0.00 |
| all | 5908 | 0.0061 | 0.00 | 0.0% | nan | nan | nan% | nan% | +0.00 |

By step:

| Step | Actual regret | std(P)/std(Δℓ) | P changes action | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|
| 1 | 0.0099 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 2 | 0.0093 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 3 | 0.0084 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 4 | 0.0079 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 5 | 0.0108 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 6 | 0.0074 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 7 | 0.0088 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 8 | 0.0079 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 9 | 0.0070 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 10 | 0.0056 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 11 | 0.0042 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 12 | 0.0023 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 13 | 0.0016 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 14 | 0.0005 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 15 | 0.0000 | 0.00 | 0.0% | nan% | nan% | +0.00 |
