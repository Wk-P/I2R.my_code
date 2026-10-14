# v5.0.3.2 No λ_max: linear λ + 0.5(J_C − d) vs normalized λ + 0.05(J_C − d)/max(J_C, d), λ uncapped, observed as λ/(1+λ), no warm-up (seed 1, 20M, 50k instances)

- Generated 2026-10-14 02:39:11; script `src/scripts/run_v5.py --version 5.0.3.2`.
- Every variant: 1 seed(s) (1) × 3 scenarios, 20M steps each, checkpoints of the same run at 1M / 3M / 5M / 7.5M / 10M / 15M; ar_pen reward, γ = 1, full episode, EXIT for Maskable only, descending demand; PPO / DQN hyper-parameters from `paper_rl/config.py`; structure-aware network (Transformer) trained on GPU. 50k instances = the ILP-verified pool (`results/sup_diag/<scenario>.json`).
- Test gap: each seed's own 400 test instances; training-set gap: 1600 training instances (a fixed subset of the 50k pool, or the 1600-instance split itself); deterministic rollouts; mean ± sample std.
- Manifest: `logs/v5.0.3.2/manifest.json`.

## λ dynamics (during training, from training_curve.csv)

J_C = violations per episode (capacity + privacy); target d = 0.001. 'First at cap' = first episode with λ ≥ λ_max; 'fall' = from the peak to the end of training.

| Scenario | Model | λ_max | Final λ | Max λ | First at cap (episode) | Share at cap | Decreases | Fall | J_C first 1000 | J_C last 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · lagrangian · linear | none | 9529.692 | 9529.692 | — | 0.0% | 0 | 0.000 | 1.981 | 1.9630 |
| LT | PPO · lagrangian · normalized | none | 65.898 | 65.898 | — | 0.0% | 0 | 0.000 | 1.981 | 0.0730 |
| LT | DQN · lagrangian · linear | none | 1904.044 | 1904.044 | — | 0.0% | 0 | 0.000 | 1.981 | 0.2560 |
| LT | DQN · lagrangian · normalized | none | 66.512 | 66.512 | — | 0.0% | 0 | 0.000 | 1.981 | 0.1260 |
| LT | DDQN · lagrangian · linear | none | 1980.371 | 1980.371 | — | 0.0% | 0 | 0.000 | 1.981 | 0.2130 |
| LT | DDQN · lagrangian · normalized | none | 66.457 | 66.457 | — | 0.0% | 0 | 0.000 | 1.981 | 0.1140 |
| EQ | PPO · lagrangian · linear | none | 29.614 | 29.614 | — | 0.0% | 568 | 0.000 | 1.820 | 0.0030 |
| EQ | PPO · lagrangian · normalized | none | 8.509 | 11.659 | — | 0.0% | 576 | 3.150 | 1.820 | 0.0010 |
| EQ | DQN · lagrangian · linear | none | 1210.651 | 1210.664 | — | 0.0% | 513 | 0.014 | 1.815 | 0.0020 |
| EQ | DQN · lagrangian · normalized | none | 39.753 | 49.940 | — | 0.0% | 405 | 10.188 | 1.815 | 0.0020 |
| EQ | DDQN · lagrangian · linear | none | 1210.310 | 1210.311 | — | 0.0% | 390 | 0.001 | 1.815 | 0.0010 |
| EQ | DDQN · lagrangian · normalized | none | 34.349 | 49.968 | — | 0.0% | 464 | 15.618 | 1.815 | 0.0000 |
| GT | PPO · lagrangian · linear | none | 23.557 | 23.796 | — | 0.0% | 1338 | 0.238 | 1.645 | 0.0000 |
| GT | PPO · lagrangian · normalized | none | 0.000 | 7.379 | — | 0.0% | 546 | 7.379 | 1.645 | 0.0000 |
| GT | DQN · lagrangian · linear | none | 893.386 | 893.763 | — | 0.0% | 984 | 0.377 | 1.666 | 0.0000 |
| GT | DQN · lagrangian · normalized | none | 0.851 | 49.875 | — | 0.0% | 988 | 49.024 | 1.666 | 0.0000 |
| GT | DDQN · lagrangian · linear | none | 885.962 | 886.247 | — | 0.0% | 988 | 0.285 | 1.666 | 0.0000 |
| GT | DDQN · lagrangian · normalized | none | 0.785 | 49.875 | — | 0.0% | 990 | 49.089 | 1.666 | 0.0000 |

## Learning curve: test relative gap

### LT

| Steps | PPO · lagrangian · linear | PPO · lagrangian · normalized | DQN · lagrangian · linear | DQN · lagrangian · normalized | DDQN · lagrangian · linear | DDQN · lagrangian · normalized |
|---|---|---|---|---|---|---|
| 1M | 26.8% | 17.5% | 30.1% | 30.9% | 24.1% | 30.6% |
| 3M | — | 16.7% | 27.0% | 22.2% | 32.5% | 28.8% |
| 5M | 36.1% | 21.4% | 21.7% | 23.1% | 21.4% | 28.1% |
| 7.5M | — | 20.1% | 27.6% | 20.6% | 32.8% | 30.9% |
| 10M | — | 24.7% | 31.1% | 22.2% | 24.7% | 22.2% |
| 15M | — | 21.4% | 20.8% | 28.0% | 24.1% | 29.3% |
| 20M | — | 26.1% | 21.4% | 31.9% | 25.3% | 31.2% |

### EQ

| Steps | PPO · lagrangian · linear | PPO · lagrangian · normalized | DQN · lagrangian · linear | DQN · lagrangian · normalized | DDQN · lagrangian · linear | DDQN · lagrangian · normalized |
|---|---|---|---|---|---|---|
| 1M | 15.5% | 34.6% | 18.2% | 48.9% | 33.0% | 46.2% |
| 3M | 9.8% | 12.5% | 34.7% | 51.0% | 36.8% | 49.9% |
| 5M | 9.3% | 9.9% | 32.6% | 50.6% | 32.8% | 49.3% |
| 7.5M | 9.3% | 9.3% | 19.4% | 39.3% | 36.2% | 50.0% |
| 10M | 9.0% | 9.3% | 18.5% | 29.2% | 20.1% | 29.2% |
| 15M | 7.7% | 8.2% | 14.4% | 17.7% | 17.2% | 16.7% |
| 20M | 6.7% | 7.5% | 17.2% | 16.4% | 17.3% | 15.8% |

### GT

| Steps | PPO · lagrangian · linear | PPO · lagrangian · normalized | DQN · lagrangian · linear | DQN · lagrangian · normalized | DDQN · lagrangian · linear | DDQN · lagrangian · normalized |
|---|---|---|---|---|---|---|
| 1M | 14.8% | 15.2% | 27.3% | 50.8% | 46.5% | 51.8% |
| 3M | 11.4% | 11.4% | 57.0% | 31.5% | 50.9% | 48.0% |
| 5M | 10.1% | 10.3% | 39.5% | 28.7% | 48.5% | 22.4% |
| 7.5M | 10.1% | 8.9% | 37.6% | 26.6% | 20.6% | 28.4% |
| 10M | 9.5% | 8.3% | 16.7% | 26.4% | 38.7% | 16.7% |
| 15M | 8.4% | 7.0% | 14.1% | 8.0% | 17.5% | 7.7% |
| 20M | 7.3% | 5.9% | 12.0% | 5.7% | 12.8% | 9.1% |

## Learning curve: mean over the three scenarios (per seed first)

| Steps | PPO · lagrangian · linear | PPO · lagrangian · normalized | DQN · lagrangian · linear | DQN · lagrangian · normalized | DDQN · lagrangian · linear | DDQN · lagrangian · normalized |
|---|---|---|---|---|---|---|
| 1M | 19.0% | 22.4% | 25.2% | 43.5% | 34.5% | 42.8% |
| 3M | — | 13.6% | 39.6% | 34.9% | 40.1% | 42.2% |
| 5M | 18.5% | 13.8% | 31.3% | 34.1% | 34.2% | 33.3% |
| 7.5M | — | 12.8% | 28.2% | 28.9% | 29.9% | 36.4% |
| 10M | — | 14.1% | 22.1% | 25.9% | 27.8% | 22.7% |
| 15M | — | 12.2% | 16.4% | 17.9% | 19.6% | 17.9% |
| 20M | — | 13.2% | 16.9% | 18.0% | 18.5% | 18.7% |

## Learning curve: test failure rate (EXIT or any violation)

The gap is computed on the remaining instances (no EXIT, no violation); with a high failure rate it covers only the easier instances.

### LT

| Steps | PPO · lagrangian · linear | PPO · lagrangian · normalized | DQN · lagrangian · linear | DQN · lagrangian · normalized | DDQN · lagrangian · linear | DDQN · lagrangian · normalized |
|---|---|---|---|---|---|---|
| 1M | 9.3% | 4.7% | 27.2% | 56.5% | 30.5% | 48.3% |
| 3M | 100.0% | 6.0% | 42.5% | 9.5% | 36.8% | 16.2% |
| 5M | 99.0% | 3.7% | 23.8% | 50.0% | 36.8% | 19.2% |
| 7.5M | 100.0% | 4.2% | 36.8% | 31.5% | 37.3% | 11.0% |
| 10M | 100.0% | 3.7% | 52.0% | 13.2% | 57.2% | 8.5% |
| 15M | 100.0% | 5.0% | 38.5% | 18.5% | 28.2% | 6.8% |
| 20M | 100.0% | 4.7% | 17.0% | 10.8% | 29.0% | 7.3% |

### EQ

| Steps | PPO · lagrangian · linear | PPO · lagrangian · normalized | DQN · lagrangian · linear | DQN · lagrangian · normalized | DDQN · lagrangian · linear | DDQN · lagrangian · normalized |
|---|---|---|---|---|---|---|
| 1M | 0.5% | 0.0% | 0.5% | 0.2% | 0.5% | 0.2% |
| 3M | 0.0% | 0.0% | 0.0% | 0.2% | 0.2% | 0.2% |
| 5M | 0.5% | 0.2% | 0.0% | 0.2% | 0.0% | 0.2% |
| 7.5M | 0.0% | 0.0% | 0.5% | 0.0% | 0.2% | 0.2% |
| 10M | 2.5% | 0.0% | 0.5% | 0.5% | 0.2% | 0.0% |
| 15M | 0.5% | 0.0% | 0.0% | 0.2% | 0.2% | 0.0% |
| 20M | 0.2% | 0.0% | 0.2% | 0.2% | 0.2% | 0.0% |

### GT

| Steps | PPO · lagrangian · linear | PPO · lagrangian · normalized | DQN · lagrangian · linear | DQN · lagrangian · normalized | DDQN · lagrangian · linear | DDQN · lagrangian · normalized |
|---|---|---|---|---|---|---|
| 1M | 0.0% | 15.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| 3M | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 5M | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 7.5M | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 10M | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 15M | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 20M | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |


## 20M: test and training-set gap, violations, early decisions

| Scenario | Model | Test relative gap | Train relative gap | Test failure rate (EXIT / violation) | Train failure rate | Capacity violation rate | Privacy violation rate | Steps 1–5 regret | Steps 1–5 optimal-action rate |
|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · lagrangian · linear | — | — | 100.0% | 100.0% | 96.8% | 100.0% | — | — |
| LT | PPO · lagrangian · normalized | 26.1% | 25.2% | 4.7% | 3.9% | 2.8% | 2.5% | — | — |
| LT | DQN · lagrangian · linear | 21.4% | 22.1% | 17.0% | 15.9% | 3.5% | 16.2% | — | — |
| LT | DQN · lagrangian · normalized | 31.9% | 31.3% | 10.8% | 8.8% | 3.5% | 9.0% | — | — |
| LT | DDQN · lagrangian · linear | 25.3% | 25.0% | 29.0% | 28.8% | 11.5% | 28.2% | — | — |
| LT | DDQN · lagrangian · normalized | 31.2% | 30.9% | 7.3% | 6.0% | 3.5% | 5.5% | — | — |
| EQ | PPO · lagrangian · linear | 6.7% | 6.3% | 0.2% | 0.8% | 0.2% | 0.2% | — | — |
| EQ | PPO · lagrangian · normalized | 7.5% | 7.7% | 0.0% | 0.2% | 0.0% | 0.0% | — | — |
| EQ | DQN · lagrangian · linear | 17.2% | 17.0% | 0.2% | 0.1% | 0.0% | 0.2% | — | — |
| EQ | DQN · lagrangian · normalized | 16.4% | 16.4% | 0.2% | 0.1% | 0.2% | 0.0% | — | — |
| EQ | DDQN · lagrangian · linear | 17.3% | 17.2% | 0.2% | 0.1% | 0.2% | 0.0% | — | — |
| EQ | DDQN · lagrangian · normalized | 15.8% | 16.2% | 0.0% | 0.2% | 0.0% | 0.0% | — | — |
| GT | PPO · lagrangian · linear | 7.3% | 7.1% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | PPO · lagrangian · normalized | 5.9% | 5.4% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DQN · lagrangian · linear | 12.0% | 11.7% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DQN · lagrangian · normalized | 5.7% | 5.5% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DDQN · lagrangian · linear | 12.8% | 12.5% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DDQN · lagrangian · normalized | 9.1% | 9.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |

Regret is defined over the legal ECUs, i.e. for Maskable only; the other mechanisms may execute violating actions (—).


## Test gap per seed (20M)

| Scenario | Model | Seed 1 |
|---|---|---|
| LT | PPO · lagrangian · linear | — |
| LT | PPO · lagrangian · normalized | 26.1% |
| LT | DQN · lagrangian · linear | 21.4% |
| LT | DQN · lagrangian · normalized | 31.9% |
| LT | DDQN · lagrangian · linear | 25.3% |
| LT | DDQN · lagrangian · normalized | 31.2% |
| EQ | PPO · lagrangian · linear | 6.7% |
| EQ | PPO · lagrangian · normalized | 7.5% |
| EQ | DQN · lagrangian · linear | 17.2% |
| EQ | DQN · lagrangian · normalized | 16.4% |
| EQ | DDQN · lagrangian · linear | 17.3% |
| EQ | DDQN · lagrangian · normalized | 15.8% |
| GT | PPO · lagrangian · linear | 7.3% |
| GT | PPO · lagrangian · normalized | 5.9% |
| GT | DQN · lagrangian · linear | 12.0% |
| GT | DQN · lagrangian · normalized | 5.7% |
| GT | DDQN · lagrangian · linear | 12.8% |
| GT | DDQN · lagrangian · normalized | 9.1% |
