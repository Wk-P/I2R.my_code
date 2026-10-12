# v4.5.1 MLP vs Transformer: Mask PPO (50k instances, 20M)

- Generated 2026-10-12 00:32:47; script `src/scripts/run_v4.5.py --version 4.5.1`.
- Every variant: 3 seeds × 3 scenarios, 20M steps each, checkpoints of the same run at 1 / 3 / 5 / 7.5 / 10 / 15M; ar_pen reward, γ = 1, full episode, EXIT, descending demand; PPO / DQN hyper-parameters from `paper_rl/config.py`, identical for MLP and Transformer; Transformer trained on GPU, MLP on CPU. 50k instances = the ILP-verified pool (`results/sup_diag/<scenario>.json`).
- Test gap: each seed's own 400 test instances; training-set gap: 1600 training instances (a fixed subset of the 50k pool, or the 1600-instance split itself); deterministic rollouts; mean ± sample std.
- Manifest: `logs/v4.5.1/manifest.json`.

## Learning curve: test relative gap

### LT

| Steps | MLP Mask PPO (50k instances) | Transformer Mask PPO (50k instances) |
|---|---|---|
| 1M | 23.2% ± 0.1% | 10.4% ± 0.1% |
| 3M | 21.8% ± 0.6% | 9.1% ± 0.2% |
| 5M | 20.8% ± 0.6% | 8.7% ± 0.2% |
| 7.5M | 20.2% ± 0.7% | 8.2% ± 0.2% |
| 10M | 20.6% ± 0.3% | 7.6% ± 0.3% |
| 15M | 20.0% ± 0.4% | 7.4% ± 0.2% |
| 20M | 19.6% ± 0.6% | 7.0% ± 0.3% |

### EQ

| Steps | MLP Mask PPO (50k instances) | Transformer Mask PPO (50k instances) |
|---|---|---|
| 1M | 16.9% ± 0.4% | 8.6% ± 0.5% |
| 3M | 16.8% ± 0.3% | 7.6% ± 0.5% |
| 5M | 16.8% ± 0.2% | 6.8% ± 0.1% |
| 7.5M | 16.8% ± 0.1% | 6.0% ± 0.7% |
| 10M | 16.8% ± 0.5% | 5.9% ± 0.5% |
| 15M | 16.7% ± 0.2% | 5.3% ± 0.5% |
| 20M | 16.5% ± 0.1% | 4.9% ± 0.3% |

### GT

| Steps | MLP Mask PPO (50k instances) | Transformer Mask PPO (50k instances) |
|---|---|---|
| 1M | 16.3% ± 0.4% | 8.9% ± 0.9% |
| 3M | 16.3% ± 0.2% | 7.4% ± 0.8% |
| 5M | 16.5% ± 0.4% | 6.6% ± 0.3% |
| 7.5M | 16.6% ± 0.3% | 5.8% ± 0.2% |
| 10M | 16.5% ± 0.3% | 5.4% ± 0.3% |
| 15M | 16.8% ± 0.5% | 5.0% ± 0.3% |
| 20M | 16.8% ± 0.3% | 4.6% ± 0.2% |

## Learning curve: mean over the three scenarios (per seed first)

| Steps | MLP Mask PPO (50k instances) | Transformer Mask PPO (50k instances) |
|---|---|---|
| 1M | 18.8% ± 0.3% | 9.3% ± 0.2% |
| 3M | 18.3% ± 0.3% | 8.0% ± 0.4% |
| 5M | 18.0% ± 0.4% | 7.4% ± 0.2% |
| 7.5M | 17.9% ± 0.1% | 6.7% ± 0.2% |
| 10M | 18.0% ± 0.3% | 6.3% ± 0.2% |
| 15M | 17.8% ± 0.1% | 5.9% ± 0.2% |
| 20M | 17.6% ± 0.2% | 5.5% ± 0.0% |

## 20M: test and training-set gap, early decisions

| Scenario | Model | Test relative gap | Train relative gap | Test EXIT rate | Steps 1–5 regret | Steps 1–5 optimal-action rate |
|---|---|---|---|---|---|---|
| LT | MLP Mask PPO (50k instances) | 19.6% ± 0.6% | 19.6% ± 0.2% | 9.9% ± 1.3% | 0.0784 ± 0.0018 | 22.2% ± 1.5% |
| LT | Transformer Mask PPO (50k instances) | 7.0% ± 0.3% | 6.7% ± 0.1% | 1.8% ± 0.3% | — | — |
| EQ | MLP Mask PPO (50k instances) | 16.5% ± 0.1% | 16.4% ± 0.1% | 0.2% ± 0.1% | 0.1056 ± 0.0029 | 34.6% ± 0.9% |
| EQ | Transformer Mask PPO (50k instances) | 4.9% ± 0.3% | 5.0% ± 0.3% | 0.1% ± 0.1% | — | — |
| GT | MLP Mask PPO (50k instances) | 16.8% ± 0.3% | 16.6% ± 0.1% | 0.0% ± 0.0% | 0.1115 ± 0.0031 | 32.7% ± 1.6% |
| GT | Transformer Mask PPO (50k instances) | 4.6% ± 0.2% | 4.2% ± 0.4% | 0.0% ± 0.0% | — | — |

## Test gap per seed (20M)

| Scenario | Model | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| LT | MLP Mask PPO (50k instances) | 19.3% | 20.3% | 19.2% |
| LT | Transformer Mask PPO (50k instances) | 6.8% | 7.3% | 6.8% |
| EQ | MLP Mask PPO (50k instances) | 16.5% | 16.6% | 16.5% |
| EQ | Transformer Mask PPO (50k instances) | 5.1% | 4.6% | 5.0% |
| GT | MLP Mask PPO (50k instances) | 17.2% | 16.6% | 16.6% |
| GT | Transformer Mask PPO (50k instances) | 4.5% | 4.5% | 4.9% |
