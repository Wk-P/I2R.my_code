# v4.4.13 The professor's 3-dim state summary as the global input

- Generated 2026-10-09 23:30:14
- Single change (`--glob3`): the input of the global token is exactly the fixed 3-dimensional, step-varying summary [M_rem / M, AR_t, σ_util,t] -- remaining services (normalised by M), current average utilisation of the active ECUs, and the std of their utilisation -- projected ℝ³ → ℝ¹²⁸ as the global token; the two other quantities of the baseline global token (unplaced demand / total capacity, free capacity / total capacity) are dropped. Each scenario is trained separately with a constant M, so M_rem and M_rem / M are equivalent after the linear projection.
- Smoke tests: on 1018 states the 3-dim input matches the values computed directly from the environment (error 1.2e−7); ECU permutation equivariance; mask unchanged. Parameters 450.2k → 450.1k.
- Everything else unchanged: ECU / service tokens, relation biases, all heads, ar_pen reward, MaskablePPO, lr 3e-4, entropy coefficient 0.005, GAE λ = 0.95, γ = 1, EXIT, descending demand, GPU, 1M steps.
- Difference from v4.4.6: v4.4.6 added σ_util on top of the existing 4-dim global input (5 dims); this version keeps only the three quantities the professor specified.
- Baseline = the v4.4.5 three-seed GPU baseline, reused; manifest of the new models: `logs/v4.4.13/manifest.json`.
- Every model is evaluated on the test split of its own training seed; the relative gap is computed per seed and then averaged; tables show the mean ± sample std over three seeds.
- Steps 1–5 regret and optimal-action rates come from `src/scripts/diag_regret.py` (first 400 instances of each seed's test split). Noise reference: three-seed SD 0.4–0.7pp for the same configuration, a rerun of the same seed can differ by 1.4–2.0pp; the EQ baseline of 10.2% is probably high (several variants give 9.4–9.5%).

## Three-seed summary

| Scenario | Model | Relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | Step-1 optimal-action rate | EXIT rate | Active ECUs − ILP | Forced openings / instance |
|---|---|---|---|---|---|---|---|---|
| LT | baseline (4-dim global token) | 11.5% ± 0.4% | 0.0533 ± 0.0029 | 32.8% ± 0.9% | 25.1% ± 2.0% | 3.5% ± 1.6% | +0.78 ± +0.06 | 5.60 ± 0.14 |
| LT | professor's 3-dim state summary | 11.2% ± 0.8% | 0.0494 ± 0.0035 | 37.8% ± 0.4% | 27.5% ± 1.9% | 3.7% ± 0.5% | +0.90 ± +0.07 | 6.26 ± 0.68 |
| EQ | baseline (4-dim global token) | 10.2% ± 0.4% | 0.0683 ± 0.0024 | 48.3% ± 1.2% | 38.5% ± 1.8% | 0.1% ± 0.1% | +0.83 ± +0.05 | 4.85 ± 0.11 |
| EQ | professor's 3-dim state summary | 9.5% ± 0.8% | 0.0628 ± 0.0044 | 48.6% ± 3.4% | 38.6% ± 3.8% | 0.1% ± 0.1% | +0.74 ± +0.05 | 4.76 ± 0.08 |
| GT | baseline (4-dim global token) | 8.7% ± 0.7% | 0.0590 ± 0.0052 | 50.3% ± 2.3% | 44.1% ± 1.2% | 0.0% ± 0.0% | +1.02 ± +0.04 | 5.41 ± 0.15 |
| GT | professor's 3-dim state summary | 9.9% ± 0.5% | 0.0707 ± 0.0041 | 46.2% ± 1.5% | 40.3% ± 1.8% | 0.0% ± 0.0% | +1.15 ± +0.04 | 5.52 ± 0.09 |

Average over the three scenarios (per seed first, then mean ± SD over seeds):

| Model | Relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | EXIT rate |
|---|---|---|---|---|
| baseline (4-dim global token) | 10.1% ± 0.3% | 0.0602 ± 0.0025 | 43.8% ± 0.9% | 1.2% ± 0.5% |
| professor's 3-dim state summary | 10.2% ± 0.6% | 0.0610 ± 0.0028 | 44.2% ± 1.3% | 1.2% ± 0.2% |

## Per seed

| Scenario | Seed | Model | Relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | Step-1 optimal-action rate | EXIT rate | Active ECUs − ILP | Forced openings / instance |
|---|---|---|---|---|---|---|---|---|---|
| LT | 1 | baseline (4-dim global token) | 11.5% | 0.0546 | 33.5% | 27.4% | 5.2% | +0.78 | 5.66 |
| LT | 1 | professor's 3-dim state summary | 10.8% | 0.0458 | 37.3% | 26.4% | 3.5% | +0.83 | 5.50 |
| LT | 2 | baseline (4-dim global token) | 11.1% | 0.0499 | 33.2% | 23.7% | 3.0% | +0.72 | 5.44 |
| LT | 2 | professor's 3-dim state summary | 12.2% | 0.0528 | 38.1% | 29.8% | 4.2% | +0.97 | 6.79 |
| LT | 3 | baseline (4-dim global token) | 11.9% | 0.0552 | 31.8% | 24.3% | 2.2% | +0.84 | 5.70 |
| LT | 3 | professor's 3-dim state summary | 10.7% | 0.0496 | 37.8% | 26.4% | 3.2% | +0.89 | 6.49 |
| EQ | 1 | baseline (4-dim global token) | 10.4% | 0.0707 | 47.8% | 36.5% | 0.0% | +0.86 | 4.85 |
| EQ | 1 | professor's 3-dim state summary | 8.6% | 0.0577 | 51.7% | 43.0% | 0.0% | +0.70 | 4.70 |
| EQ | 2 | baseline (4-dim global token) | 9.7% | 0.0659 | 47.5% | 39.3% | 0.2% | +0.77 | 4.75 |
| EQ | 2 | professor's 3-dim state summary | 10.0% | 0.0657 | 44.9% | 35.8% | 0.2% | +0.73 | 4.72 |
| EQ | 3 | baseline (4-dim global token) | 10.4% | 0.0682 | 49.7% | 39.8% | 0.0% | +0.86 | 4.97 |
| EQ | 3 | professor's 3-dim state summary | 9.8% | 0.0650 | 49.1% | 37.0% | 0.0% | +0.80 | 4.85 |
| GT | 1 | baseline (4-dim global token) | 9.3% | 0.0637 | 51.0% | 43.5% | 0.0% | +0.99 | 5.28 |
| GT | 1 | professor's 3-dim state summary | 9.9% | 0.0725 | 45.5% | 38.5% | 0.0% | +1.18 | 5.43 |
| GT | 2 | baseline (4-dim global token) | 8.9% | 0.0599 | 47.7% | 43.2% | 0.0% | +1.01 | 5.39 |
| GT | 2 | professor's 3-dim state summary | 10.5% | 0.0736 | 45.1% | 40.5% | 0.0% | +1.11 | 5.60 |
| GT | 3 | baseline (4-dim global token) | 8.0% | 0.0534 | 52.2% | 45.5% | 0.0% | +1.07 | 5.57 |
| GT | 3 | professor's 3-dim state summary | 9.5% | 0.0660 | 47.9% | 42.0% | 0.0% | +1.16 | 5.52 |
