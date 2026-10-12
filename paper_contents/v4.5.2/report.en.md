# v4.5.2 MLP vs Transformer: Mask-DQN / Mask-DDQN (50k instances, 20M)

- Generated 2026-10-12 22:11:54; script `src/scripts/run_v4.5.py --version 4.5.2`.
- Every variant: 3 seeds × 3 scenarios, 20M steps each, checkpoints of the same run at 1 / 3 / 5 / 7.5 / 10 / 15M; ar_pen reward, γ = 1, full episode, EXIT, descending demand; PPO / DQN hyper-parameters from `paper_rl/config.py`, identical for MLP and Transformer; Transformer trained on GPU, MLP on CPU. 50k instances = the ILP-verified pool (`results/sup_diag/<scenario>.json`).
- Test gap: each seed's own 400 test instances; training-set gap: 1600 training instances (a fixed subset of the 50k pool, or the 1600-instance split itself); deterministic rollouts; mean ± sample std.
- Manifest: `logs/v4.5.2/manifest.json`.

## Learning curve: test relative gap

### LT

| Steps | MLP Mask-DQN (50k) | Transformer Mask-DQN (50k) | MLP Mask-DDQN (50k) | Transformer Mask-DDQN (50k) |
|---|---|---|---|---|
| 1M | 23.3% ± 0.4% | 19.9% ± 1.2% | 23.3% ± 0.7% | 20.0% ± 0.7% |
| 3M | 23.5% ± 0.5% | 17.7% ± 0.5% | 23.0% ± 0.5% | 17.2% ± 1.6% |
| 5M | 22.7% ± 0.3% | 15.3% ± 1.7% | 23.1% ± 0.5% | 14.0% ± 0.9% |
| 7.5M | 22.8% ± 0.4% | 12.3% ± 0.2% | 23.0% ± 0.5% | 12.0% ± 0.6% |
| 10M | 22.7% ± 0.1% | 11.5% ± 0.8% | 22.4% ± 0.5% | 12.2% ± 1.3% |
| 15M | 21.9% ± 0.6% | 10.4% ± 0.8% | 22.8% ± 0.7% | 11.7% ± 0.7% |
| 20M | 22.0% ± 0.1% | 9.9% ± 0.5% | 22.5% ± 1.3% | 9.9% ± 0.5% |

### EQ

| Steps | MLP Mask-DQN (50k) | Transformer Mask-DQN (50k) | MLP Mask-DDQN (50k) | Transformer Mask-DDQN (50k) |
|---|---|---|---|---|
| 1M | 17.8% ± 0.4% | 13.2% ± 1.0% | 18.2% ± 1.1% | 13.4% ± 1.2% |
| 3M | 17.9% ± 1.5% | 7.9% ± 0.5% | 17.2% ± 0.5% | 11.2% ± 0.7% |
| 5M | 17.9% ± 0.2% | 6.7% ± 0.3% | 18.0% ± 0.4% | 7.7% ± 1.0% |
| 7.5M | 18.5% ± 0.4% | 6.0% ± 0.6% | 18.0% ± 0.2% | 6.7% ± 1.1% |
| 10M | 19.5% ± 0.4% | 5.4% ± 0.2% | 18.0% ± 0.3% | 5.6% ± 0.3% |
| 15M | 18.6% ± 0.1% | 6.0% ± 2.1% | 18.6% ± 0.5% | 6.0% ± 1.7% |
| 20M | 19.0% ± 0.4% | 4.1% ± 0.1% | 19.6% ± 0.5% | 4.6% ± 0.3% |

### GT

| Steps | MLP Mask-DQN (50k) | Transformer Mask-DQN (50k) | MLP Mask-DDQN (50k) | Transformer Mask-DDQN (50k) |
|---|---|---|---|---|
| 1M | 18.8% ± 0.3% | 13.5% ± 0.3% | 17.9% ± 1.0% | 12.1% ± 0.1% |
| 3M | 18.1% ± 1.0% | 7.7% ± 0.3% | 18.5% ± 0.1% | 7.3% ± 0.5% |
| 5M | 19.2% ± 0.5% | 6.2% ± 0.7% | 19.2% ± 0.2% | 5.9% ± 0.0% |
| 7.5M | 20.3% ± 0.4% | 5.5% ± 1.1% | 20.4% ± 0.8% | 5.0% ± 0.9% |
| 10M | 21.8% ± 0.9% | 4.0% ± 0.7% | 21.5% ± 0.7% | 8.3% ± 3.7% |
| 15M | 23.0% ± 0.3% | 5.3% ± 2.7% | 22.3% ± 0.6% | 4.3% ± 1.1% |
| 20M | 22.4% ± 0.9% | 3.1% ± 0.4% | 23.1% ± 0.7% | 3.8% ± 0.8% |

## Learning curve: mean over the three scenarios (per seed first)

| Steps | MLP Mask-DQN (50k) | Transformer Mask-DQN (50k) | MLP Mask-DDQN (50k) | Transformer Mask-DDQN (50k) |
|---|---|---|---|---|
| 1M | 20.0% ± 0.2% | 15.5% ± 0.5% | 19.8% ± 0.6% | 15.1% ± 0.4% |
| 3M | 19.8% ± 0.6% | 11.1% ± 0.4% | 19.6% ± 0.3% | 11.9% ± 0.8% |
| 5M | 19.9% ± 0.1% | 9.4% ± 0.6% | 20.1% ± 0.1% | 9.2% ± 0.4% |
| 7.5M | 20.5% ± 0.1% | 7.9% ± 0.5% | 20.4% ± 0.3% | 7.9% ± 0.8% |
| 10M | 21.3% ± 0.2% | 7.0% ± 0.1% | 20.6% ± 0.3% | 8.7% ± 1.4% |
| 15M | 21.1% ± 0.3% | 7.2% ± 0.9% | 21.3% ± 0.2% | 7.3% ± 1.2% |
| 20M | 21.1% ± 0.2% | 5.7% ± 0.3% | 21.7% ± 0.8% | 6.1% ± 0.1% |

## 20M: test and training-set gap, early decisions

| Scenario | Model | Test relative gap | Train relative gap | Test EXIT rate | Steps 1–5 regret | Steps 1–5 optimal-action rate |
|---|---|---|---|---|---|---|
| LT | MLP Mask-DQN (50k) | 22.0% ± 0.1% | 21.7% ± 0.1% | 16.8% ± 0.6% | 0.0866 ± 0.0024 | 24.6% ± 0.7% |
| LT | Transformer Mask-DQN (50k) | 9.9% ± 0.5% | 9.7% ± 0.6% | 2.6% ± 0.6% | 0.0427 ± 0.0021 | 40.5% ± 0.6% |
| LT | MLP Mask-DDQN (50k) | 22.5% ± 1.3% | 22.5% ± 0.9% | 16.8% ± 4.0% | 0.0858 ± 0.0034 | 24.0% ± 0.7% |
| LT | Transformer Mask-DDQN (50k) | 9.9% ± 0.5% | 9.9% ± 0.3% | 2.5% ± 0.5% | 0.0446 ± 0.0018 | 36.0% ± 0.9% |
| EQ | MLP Mask-DQN (50k) | 19.0% ± 0.4% | 18.8% ± 0.1% | 0.2% ± 0.3% | 0.1162 ± 0.0042 | 29.2% ± 1.2% |
| EQ | Transformer Mask-DQN (50k) | 4.1% ± 0.1% | 4.3% ± 0.2% | 0.1% ± 0.1% | 0.0295 ± 0.0006 | 64.1% ± 1.9% |
| EQ | MLP Mask-DDQN (50k) | 19.6% ± 0.5% | 19.3% ± 0.2% | 0.2% ± 0.3% | 0.1171 ± 0.0018 | 27.3% ± 1.1% |
| EQ | Transformer Mask-DDQN (50k) | 4.6% ± 0.3% | 4.8% ± 0.3% | 0.0% ± 0.0% | 0.0335 ± 0.0023 | 62.9% ± 2.7% |
| GT | MLP Mask-DQN (50k) | 22.4% ± 0.9% | 22.7% ± 0.9% | 0.0% ± 0.0% | 0.1366 ± 0.0046 | 22.2% ± 0.8% |
| GT | Transformer Mask-DQN (50k) | 3.1% ± 0.4% | 3.0% ± 0.4% | 0.0% ± 0.0% | 0.0240 ± 0.0020 | 69.6% ± 1.0% |
| GT | MLP Mask-DDQN (50k) | 23.1% ± 0.7% | 23.2% ± 1.1% | 0.0% ± 0.0% | 0.1375 ± 0.0054 | 22.0% ± 0.4% |
| GT | Transformer Mask-DDQN (50k) | 3.8% ± 0.8% | 3.8% ± 0.8% | 0.0% ± 0.0% | 0.0294 ± 0.0067 | 68.7% ± 2.8% |

## Test gap per seed (20M)

| Scenario | Model | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| LT | MLP Mask-DQN (50k) | 21.9% | 22.1% | 21.9% |
| LT | Transformer Mask-DQN (50k) | 9.4% | 9.9% | 10.4% |
| LT | MLP Mask-DDQN (50k) | 22.4% | 23.9% | 21.3% |
| LT | Transformer Mask-DDQN (50k) | 10.4% | 9.8% | 9.5% |
| EQ | MLP Mask-DQN (50k) | 18.6% | 19.3% | 19.0% |
| EQ | Transformer Mask-DQN (50k) | 4.0% | 4.2% | 4.1% |
| EQ | MLP Mask-DDQN (50k) | 19.3% | 20.2% | 19.2% |
| EQ | Transformer Mask-DDQN (50k) | 4.9% | 4.4% | 4.6% |
| GT | MLP Mask-DQN (50k) | 23.2% | 21.4% | 22.6% |
| GT | Transformer Mask-DQN (50k) | 2.7% | 3.4% | 3.1% |
| GT | MLP Mask-DDQN (50k) | 22.5% | 23.9% | 22.8% |
| GT | Transformer Mask-DDQN (50k) | 2.9% | 4.4% | 4.2% |
