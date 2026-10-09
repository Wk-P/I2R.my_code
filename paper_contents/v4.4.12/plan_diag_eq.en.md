# What the global plan P does (EQ, v4.4.12 models)

- Generated 2026-10-09 22:27:42; script `src/scripts/diag_plan.py`; took 6.0 min.
- Along the plan-conditioned policy's own deterministic trajectories (first 400 instances of each seed's test split); at every step the ECU logit is P[i_t, j] + Δℓ_j(s_t). Recorded: std of the P row and of Δℓ over the legal ECUs; action taken = argmax(P + Δℓ), counterfactual = argmax(Δℓ) (what the per-step score alone would pick).
- Both actions' regrets are computed with the ILP at the same state (placements so far fixed, the rest completed optimally, as in `diag_regret.py`), i.e. a counterfactual comparison at the same state.
- "P changes the action" = the two actions differ; "P better / worse" = regret differs by more than 1e−4; "net gain" = Σ(counterfactual regret − actual regret) / steps in the phase, positive = regret P removes per step.

## Seed 1 (exp_id 79607f84, 400 completed instances, 0 EXIT)

| Steps | n | Actual regret | Median std(P) / std(Δℓ) | P changes action | When changed: P's regret | When changed: counterfactual regret | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|---|---|---|
| 1–3 | 1200 | 0.0135 | 0.03 | 0.8% | 0.0272 | 0.0188 | 33.3% | 66.7% | -0.63 |
| 4–6 | 1200 | 0.0102 | 0.02 | 0.9% | 0.0125 | 0.0262 | 81.8% | 18.2% | +1.25 |
| 7–10 | 1600 | 0.0023 | 0.02 | 0.4% | 0.0101 | 0.0016 | 14.3% | 57.1% | -0.37 |
| all | 4000 | 0.0080 | 0.02 | 0.7% | 0.0168 | 0.0174 | 48.1% | 44.4% | +0.04 |

By step:

| Step | Actual regret | std(P)/std(Δℓ) | P changes action | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|
| 1 | 0.0150 | 0.03 | 0.5% | 0.0% | 100.0% | -2.16 |
| 2 | 0.0128 | 0.03 | 0.8% | 33.3% | 66.7% | -0.20 |
| 3 | 0.0127 | 0.03 | 1.0% | 50.0% | 50.0% | +0.47 |
| 4 | 0.0134 | 0.02 | 1.2% | 80.0% | 20.0% | -0.11 |
| 5 | 0.0094 | 0.02 | 0.8% | 100.0% | 0.0% | +3.23 |
| 6 | 0.0077 | 0.02 | 0.8% | 66.7% | 33.3% | +0.65 |
| 7 | 0.0045 | 0.02 | 0.8% | 33.3% | 33.3% | -1.40 |
| 8 | 0.0038 | 0.02 | 0.5% | 0.0% | 100.0% | -0.05 |
| 9 | 0.0008 | 0.02 | 0.2% | 0.0% | 0.0% | +0.00 |
| 10 | 0.0000 | 0.02 | 0.2% | 0.0% | 100.0% | -0.04 |

## Seed 2 (exp_id 4e353913, 399 completed instances, 1 EXIT)

| Steps | n | Actual regret | Median std(P) / std(Δℓ) | P changes action | When changed: P's regret | When changed: counterfactual regret | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|---|---|---|
| 1–3 | 1197 | 0.0148 | 0.02 | 1.4% | 0.0296 | 0.0149 | 35.3% | 58.8% | -2.09 |
| 4–6 | 1197 | 0.0098 | 0.02 | 1.3% | 0.0087 | 0.0178 | 75.0% | 18.8% | +1.21 |
| 7–10 | 1596 | 0.0021 | 0.02 | 0.7% | 0.0229 | 0.0067 | 63.6% | 36.4% | -1.12 |
| all | 3990 | 0.0082 | 0.02 | 1.1% | 0.0203 | 0.0139 | 56.8% | 38.6% | -0.71 |

By step:

| Step | Actual regret | std(P)/std(Δℓ) | P changes action | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|
| 1 | 0.0157 | 0.02 | 1.5% | 16.7% | 83.3% | -2.90 |
| 2 | 0.0154 | 0.02 | 1.5% | 50.0% | 33.3% | -1.95 |
| 3 | 0.0133 | 0.02 | 1.3% | 40.0% | 60.0% | -1.41 |
| 4 | 0.0124 | 0.03 | 1.0% | 100.0% | 0.0% | +2.46 |
| 5 | 0.0087 | 0.02 | 1.3% | 80.0% | 0.0% | +3.35 |
| 6 | 0.0083 | 0.02 | 1.8% | 57.1% | 42.9% | -2.17 |
| 7 | 0.0054 | 0.02 | 1.3% | 60.0% | 40.0% | -1.87 |
| 8 | 0.0025 | 0.02 | 1.3% | 60.0% | 40.0% | -2.70 |
| 9 | 0.0006 | 0.02 | 0.3% | 100.0% | 0.0% | +0.09 |
| 10 | 0.0000 | 0.01 | 0.0% | nan% | nan% | +0.00 |

## Seed 3 (exp_id 3748f95a, 400 completed instances, 0 EXIT)

| Steps | n | Actual regret | Median std(P) / std(Δℓ) | P changes action | When changed: P's regret | When changed: counterfactual regret | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|---|---|---|
| 1–3 | 1200 | 0.0141 | 0.01 | 0.2% | 0.0269 | 0.0098 | 33.3% | 66.7% | -0.43 |
| 4–6 | 1200 | 0.0102 | 0.00 | 0.3% | 0.0382 | 0.0024 | 25.0% | 50.0% | -1.19 |
| 7–10 | 1600 | 0.0024 | 0.00 | 0.0% | nan | nan | nan% | nan% | +0.00 |
| all | 4000 | 0.0082 | 0.00 | 0.2% | 0.0333 | 0.0056 | 28.6% | 57.1% | -0.49 |

By step:

| Step | Actual regret | std(P)/std(Δℓ) | P changes action | P better | P worse | Net gain / step (×1e−4) |
|---|---|---|---|---|---|---|
| 1 | 0.0143 | 0.01 | 0.2% | 0.0% | 100.0% | -0.20 |
| 2 | 0.0145 | 0.01 | 0.0% | nan% | nan% | +0.00 |
| 3 | 0.0134 | 0.00 | 0.5% | 50.0% | 50.0% | -1.08 |
| 4 | 0.0110 | 0.00 | 0.2% | 100.0% | 0.0% | +0.23 |
| 5 | 0.0096 | 0.00 | 0.5% | 0.0% | 100.0% | -3.81 |
| 6 | 0.0098 | 0.00 | 0.2% | 0.0% | 0.0% | +0.00 |
| 7 | 0.0049 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 8 | 0.0040 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 9 | 0.0006 | 0.00 | 0.0% | nan% | nan% | +0.00 |
| 10 | 0.0000 | 0.00 | 0.0% | nan% | nan% | +0.00 |
