# v4.4.12 Global-plan-conditioned policy (global plan + sequential correction)

- Generated 2026-10-09 21:34:04
- Single change: the policy goes from reactive to plan-conditioned (`--plan`, `src/paper_rl/graph_net.PlanEncoder`). A separate plan encoder (same structure as the main encoder, its own parameters) reads the instance's initial state s_0 (recovered exactly from any s_t by clearing the assignment and setting t = 0) and forms a service × ECU plan matrix P_ij = ⟨W_s h_i(s_0), W_e h_j(s_0)⟩ / √d, constant over the episode; the ECU logit at each step is P[i_t, j] + the usual per-step score. W_e is zero-initialised, so P ≡ 0 and the policy equals the baseline at the start of training. Parameters 450k → 880k (all of the increase is the plan encoder).
- Smoke tests: identical logits to the plain network at zero initialisation; P exactly constant within an episode; ECU permutation equivariance with a random W_e (error 1.5e−6); mask and EXIT unchanged; no illegal action when sampling the policy.
- Everything else unchanged: EXIT head, critic, ar_pen reward, MaskablePPO, lr 3e-4, entropy coefficient 0.005, GAE λ = 0.95, γ = 1, descending demand, GPU, 1M steps; no ILP, no BC, one deterministic rollout at test time.
- Baseline = the v4.4.5 three-seed GPU baseline, reused; manifest of the new models: `logs/v4.4.12/manifest.json`.
- Every model is evaluated on the test split of its own training seed; the relative gap is computed per seed and then averaged; tables show the mean ± sample std over three seeds.
- Steps 1–5 regret and optimal-action rates come from `src/scripts/diag_regret.py` (first 400 instances of each seed's test split). Noise reference: three-seed SD 0.4–0.7pp for the same configuration, a rerun of the same seed can differ by 1.4–2.0pp; the EQ baseline of 10.2% is probably high (several variants give 9.4–9.5%).
- Order of judgement (fixed in advance): first whether steps 1–5 regret falls and the optimal-action rate rises, then whether extra ECUs fall, and only then the gap.

## Three-seed summary

| Scenario | Model | Relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | Step-1 optimal-action rate | EXIT rate | Active ECUs − ILP | Forced openings / instance |
|---|---|---|---|---|---|---|---|---|
| LT | baseline (reactive policy) | 11.5% ± 0.4% | 0.0533 ± 0.0029 | 32.8% ± 0.9% | 25.1% ± 2.0% | 3.5% ± 1.6% | +0.78 ± +0.06 | 5.60 ± 0.14 |
| LT | global plan + sequential correction | 11.9% ± 2.1% | 0.0473 ± 0.0013 | 33.7% ± 2.3% | 25.6% ± 2.2% | 3.0% ± 1.0% | +0.78 ± +0.16 | 5.15 ± 0.18 |
| EQ | baseline (reactive policy) | 10.2% ± 0.4% | 0.0683 ± 0.0024 | 48.3% ± 1.2% | 38.5% ± 1.8% | 0.1% ± 0.1% | +0.83 ± +0.05 | 4.85 ± 0.11 |
| EQ | global plan + sequential correction | 9.5% ± 0.1% | 0.0638 ± 0.0014 | 49.6% ± 0.9% | 39.4% ± 0.3% | 0.1% ± 0.1% | +0.76 ± +0.04 | 4.80 ± 0.16 |
| GT | baseline (reactive policy) | 8.7% ± 0.7% | 0.0590 ± 0.0052 | 50.3% ± 2.3% | 44.1% ± 1.2% | 0.0% ± 0.0% | +1.02 ± +0.04 | 5.41 ± 0.15 |
| GT | global plan + sequential correction | 8.0% ± 0.1% | 0.0548 ± 0.0025 | 50.8% ± 2.1% | 46.7% ± 3.0% | 0.0% ± 0.0% | +0.92 ± +0.16 | 5.21 ± 0.29 |

Average over the three scenarios (per seed first, then mean ± SD over seeds):

| Model | Relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | EXIT rate |
|---|---|---|---|---|
| baseline (reactive policy) | 10.1% ± 0.3% | 0.0602 ± 0.0025 | 43.8% ± 0.9% | 1.2% ± 0.5% |
| global plan + sequential correction | 9.8% ± 0.7% | 0.0553 ± 0.0017 | 44.7% ± 1.1% | 1.0% ± 0.4% |

## Per seed

| Scenario | Seed | Model | Relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | Step-1 optimal-action rate | EXIT rate | Active ECUs − ILP | Forced openings / instance |
|---|---|---|---|---|---|---|---|---|---|
| LT | 1 | baseline (reactive policy) | 11.5% | 0.0546 | 33.5% | 27.4% | 5.2% | +0.78 | 5.66 |
| LT | 1 | global plan + sequential correction | 11.0% | 0.0472 | 33.6% | 25.0% | 3.0% | +0.64 | 5.02 |
| LT | 2 | baseline (reactive policy) | 11.1% | 0.0499 | 33.2% | 23.7% | 3.0% | +0.72 | 5.44 |
| LT | 2 | global plan + sequential correction | 14.3% | 0.0487 | 31.5% | 23.7% | 4.0% | +0.95 | 5.08 |
| LT | 3 | baseline (reactive policy) | 11.9% | 0.0552 | 31.8% | 24.3% | 2.2% | +0.84 | 5.70 |
| LT | 3 | global plan + sequential correction | 10.3% | 0.0460 | 36.0% | 28.1% | 2.0% | +0.74 | 5.36 |
| EQ | 1 | baseline (reactive policy) | 10.4% | 0.0707 | 47.8% | 36.5% | 0.0% | +0.86 | 4.85 |
| EQ | 1 | global plan + sequential correction | 9.4% | 0.0632 | 50.5% | 39.5% | 0.0% | +0.79 | 4.96 |
| EQ | 2 | baseline (reactive policy) | 9.7% | 0.0659 | 47.5% | 39.3% | 0.2% | +0.77 | 4.75 |
| EQ | 2 | global plan + sequential correction | 9.6% | 0.0654 | 49.7% | 39.6% | 0.2% | +0.76 | 4.78 |
| EQ | 3 | baseline (reactive policy) | 10.4% | 0.0682 | 49.7% | 39.8% | 0.0% | +0.86 | 4.97 |
| EQ | 3 | global plan + sequential correction | 9.6% | 0.0628 | 48.8% | 39.0% | 0.0% | +0.72 | 4.65 |
| GT | 1 | baseline (reactive policy) | 9.3% | 0.0637 | 51.0% | 43.5% | 0.0% | +0.99 | 5.28 |
| GT | 1 | global plan + sequential correction | 7.9% | 0.0549 | 49.1% | 43.8% | 0.0% | +0.76 | 4.90 |
| GT | 2 | baseline (reactive policy) | 8.9% | 0.0599 | 47.7% | 43.2% | 0.0% | +1.01 | 5.39 |
| GT | 2 | global plan + sequential correction | 7.9% | 0.0571 | 50.2% | 46.5% | 0.0% | +0.92 | 5.26 |
| GT | 3 | baseline (reactive policy) | 8.0% | 0.0534 | 52.2% | 45.5% | 0.0% | +1.07 | 5.57 |
| GT | 3 | global plan + sequential correction | 8.1% | 0.0522 | 53.2% | 49.8% | 0.0% | +1.07 | 5.47 |
