# v4.4.14 PPO with 50k training instances

- Generated 2026-10-10 13:28:14
- Single change: the training instances. 1600 (the seed's 80% split of the 2000 v4.3.1.4 instances) → 50,000 ILP-verified instances (`results/sup_diag/<scenario>.json`: same generator, p = 0.6, every instance has an ILP optimum, disjoint from all 2000 v4.3.1.4 instances), one drawn uniformly at every reset. The test split is unchanged (400 per seed).
- Everything else unchanged: structure-aware network, ar_pen reward, mask, descending demand, MaskablePPO, lr 3e-4, entropy coefficient 0.005, GAE λ = 0.95, γ = 1, EXIT, GPU, 1M steps; no ILP in training.
- Training-set relative gap: the baseline on its own 1600 training instances (`diag_datasize.py`); the new models on a fixed 1600-instance subset of the 50k training pool; both deterministic rollouts.
- Reference (data-size diagnostic): the same network imitating the ILP reaches a test gap of 10.1 / 5.7 / 6.1% (LT), 7.8 / 4.4 / 3.5% (EQ) and 7.2 / 4.3 / 3.7% (GT) with 1600 / 10k / 50k instances.
- Baseline = the v4.4.5 three-seed GPU baseline, reused; manifest of the new models: `logs/v4.4.14/manifest.json`.
- Every model is evaluated on the test split of its own training seed; the relative gap is computed per seed and then averaged; tables show the mean ± sample std over three seeds.
- Steps 1–5 regret and optimal-action rates come from `src/scripts/diag_regret.py` (first 400 instances of each seed's test split). Noise reference: three-seed SD 0.4–0.7pp for the same configuration, a rerun of the same seed can differ by 1.4–2.0pp.

## Three-seed summary

| Scenario | Model | Test relative gap | Train relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | Step-1 optimal-action rate | EXIT rate | Active ECUs − ILP | Forced openings / instance |
|---|---|---|---|---|---|---|---|---|---|
| LT | baseline (1600 training instances) | 11.5% ± 0.4% | 11.1% ± 0.2% | 0.0533 ± 0.0029 | 32.8% ± 0.9% | 25.1% ± 2.0% | 3.5% ± 1.6% | +0.78 ± +0.06 | 5.60 ± 0.14 |
| LT | 50k training instances | 10.4% ± 0.2% | 10.3% ± 0.3% | 0.0470 ± 0.0024 | 35.9% ± 3.4% | 26.6% ± 3.4% | 3.2% ± 0.6% | +0.66 ± +0.02 | 5.40 ± 0.38 |
| EQ | baseline (1600 training instances) | 10.2% ± 0.4% | 7.8% ± 0.2% | 0.0683 ± 0.0024 | 48.3% ± 1.2% | 38.5% ± 1.8% | 0.1% ± 0.1% | +0.83 ± +0.05 | 4.85 ± 0.11 |
| EQ | 50k training instances | 8.4% ± 0.8% | 8.5% ± 0.5% | 0.0561 ± 0.0059 | 49.9% ± 2.5% | 40.9% ± 4.5% | 0.1% ± 0.1% | +0.63 ± +0.07 | 4.59 ± 0.21 |
| GT | baseline (1600 training instances) | 8.7% ± 0.7% | 7.0% ± 0.1% | 0.0590 ± 0.0052 | 50.3% ± 2.3% | 44.1% ± 1.2% | 0.0% ± 0.0% | +1.02 ± +0.04 | 5.41 ± 0.15 |
| GT | 50k training instances | 9.1% ± 1.2% | 8.9% ± 1.4% | 0.0646 ± 0.0089 | 48.2% ± 0.6% | 42.7% ± 1.3% | 0.0% ± 0.0% | +1.11 ± +0.22 | 5.51 ± 0.28 |

Average over the three scenarios (per seed first, then mean ± SD over seeds):

| Model | Test relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | EXIT rate |
|---|---|---|---|---|
| baseline (1600 training instances) | 10.1% ± 0.3% | 0.0602 ± 0.0025 | 43.8% ± 0.9% | 1.2% ± 0.5% |
| 50k training instances | 9.3% ± 0.3% | 0.0559 ± 0.0020 | 44.7% ± 0.4% | 1.1% ± 0.2% |

## Per seed

| Scenario | Seed | Model | Test relative gap | Train relative gap | Steps 1–5 regret | Steps 1–5 optimal-action rate | Step-1 optimal-action rate | EXIT rate | Active ECUs − ILP | Forced openings / instance |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | 1 | baseline (1600 training instances) | 11.5% | 11.2% | 0.0546 | 33.5% | 27.4% | 5.2% | +0.78 | 5.66 |
| LT | 1 | 50k training instances | 10.3% | 10.2% | 0.0443 | 39.6% | 30.5% | 3.2% | +0.65 | 5.79 |
| LT | 2 | baseline (1600 training instances) | 11.1% | 10.9% | 0.0499 | 33.2% | 23.7% | 3.0% | +0.72 | 5.44 |
| LT | 2 | 50k training instances | 10.3% | 10.6% | 0.0488 | 33.0% | 24.4% | 3.7% | +0.65 | 5.02 |
| LT | 3 | baseline (1600 training instances) | 11.9% | 11.1% | 0.0552 | 31.8% | 24.3% | 2.2% | +0.84 | 5.70 |
| LT | 3 | 50k training instances | 10.7% | 10.0% | 0.0480 | 35.2% | 24.9% | 2.5% | +0.68 | 5.39 |
| EQ | 1 | baseline (1600 training instances) | 10.4% | 7.9% | 0.0707 | 47.8% | 36.5% | 0.0% | +0.86 | 4.85 |
| EQ | 1 | 50k training instances | 9.3% | 9.0% | 0.0628 | 47.1% | 36.0% | 0.0% | +0.68 | 4.72 |
| EQ | 2 | baseline (1600 training instances) | 9.7% | 7.6% | 0.0659 | 47.5% | 39.3% | 0.2% | +0.77 | 4.75 |
| EQ | 2 | 50k training instances | 8.3% | 8.1% | 0.0537 | 52.0% | 44.9% | 0.2% | +0.67 | 4.70 |
| EQ | 3 | baseline (1600 training instances) | 10.4% | 8.0% | 0.0682 | 49.7% | 39.8% | 0.0% | +0.86 | 4.97 |
| EQ | 3 | 50k training instances | 7.7% | 8.3% | 0.0517 | 50.7% | 41.8% | 0.0% | +0.55 | 4.34 |
| GT | 1 | baseline (1600 training instances) | 9.3% | 7.1% | 0.0637 | 51.0% | 43.5% | 0.0% | +0.99 | 5.28 |
| GT | 1 | 50k training instances | 8.4% | 7.9% | 0.0588 | 48.9% | 43.8% | 0.0% | +0.95 | 5.28 |
| GT | 2 | baseline (1600 training instances) | 8.9% | 7.2% | 0.0599 | 47.7% | 43.2% | 0.0% | +1.01 | 5.39 |
| GT | 2 | 50k training instances | 8.3% | 8.3% | 0.0602 | 48.2% | 43.0% | 0.0% | +1.02 | 5.45 |
| GT | 3 | baseline (1600 training instances) | 8.0% | 6.9% | 0.0534 | 52.2% | 45.5% | 0.0% | +1.07 | 5.57 |
| GT | 3 | 50k training instances | 10.4% | 10.5% | 0.0748 | 47.8% | 41.2% | 0.0% | +1.36 | 5.82 |
