# v5.0.3.1 λ update sub-experiment: linear λ + 0.5(J_C − d) vs normalized λ + 0.05(J_C − d)/max(J_C, d), no warm-up, λ_max = 5 (seed 1, 1M, 50k instances)

- Generated 2026-10-13 19:12:14; script `src/scripts/run_v5.py --version 5.0.3.1`.
- Every variant: 1 seed(s) (1) × 3 scenarios, 1M steps each, checkpoints of the same run at 0.5M; ar_pen reward, γ = 1, full episode, EXIT for Maskable only, descending demand; PPO / DQN hyper-parameters from `paper_rl/config.py`; structure-aware network (Transformer) trained on GPU. 50k instances = the ILP-verified pool (`results/sup_diag/<scenario>.json`).
- Test gap: each seed's own 400 test instances; training-set gap: 1600 training instances (a fixed subset of the 50k pool, or the 1600-instance split itself); deterministic rollouts; mean ± sample std.
- Manifest: `logs/v5.0.3.1/manifest.json`.

## λ dynamics (during training, from training_curve.csv)

J_C = violations per episode (capacity + privacy); target d = 0.001. 'First at cap' = first episode with λ ≥ λ_max; 'fall' = from the peak to the end of training.

| Scenario | Model | λ_max | Final λ | Max λ | First at cap (episode) | Share at cap | Decreases | Fall | J_C first 1000 | J_C last 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · lagrangian · linear | 5 | 5.000 | 5.000 | 2000 | 97.0% | 0 | 0.000 | 1.981 | 0.1830 |
| LT | PPO · lagrangian · normalized | 5 | 3.391 | 3.391 | — | 0.0% | 0 | 0.000 | 1.981 | 0.1400 |
| LT | DQN · lagrangian · linear | 5 | 5.000 | 5.000 | 2000 | 97.0% | 0 | 0.000 | 1.980 | 0.1530 |
| LT | DQN · lagrangian · normalized | 5 | 3.389 | 3.389 | — | 0.0% | 0 | 0.000 | 1.980 | 0.1410 |
| LT | DDQN · lagrangian · linear | 5 | 5.000 | 5.000 | 2000 | 97.0% | 0 | 0.000 | 1.980 | 0.2580 |
| LT | DDQN · lagrangian · normalized | 5 | 3.388 | 3.388 | — | 0.0% | 0 | 0.000 | 1.980 | 0.1060 |
| EQ | PPO · lagrangian · linear | 5 | 5.000 | 5.000 | 3000 | 97.0% | 0 | 0.000 | 1.820 | 0.0980 |
| EQ | PPO · lagrangian · normalized | 5 | 5.000 | 5.000 | 99000 | 1.3% | 0 | 0.000 | 1.820 | 0.0880 |
| EQ | DQN · lagrangian · linear | 5 | 4.991 | 5.000 | 3000 | 80.0% | 28 | 0.009 | 1.816 | 0.0000 |
| EQ | DQN · lagrangian · normalized | 5 | 1.789 | 2.597 | — | 0.0% | 22 | 0.808 | 1.816 | 0.0000 |
| EQ | DDQN · lagrangian · linear | 5 | 4.997 | 5.000 | 3000 | 97.0% | 19 | 0.003 | 1.816 | 0.0000 |
| EQ | DDQN · lagrangian · normalized | 5 | 2.202 | 2.645 | — | 0.0% | 18 | 0.442 | 1.816 | 0.0000 |
| GT | PPO · lagrangian · linear | 5 | 5.000 | 5.000 | 3000 | 97.0% | 1 | 0.000 | 1.645 | 0.7660 |
| GT | PPO · lagrangian · normalized | 5 | 5.000 | 5.000 | 99000 | 1.3% | 0 | 0.000 | 1.645 | 0.2360 |
| GT | DQN · lagrangian · linear | 5 | 4.976 | 5.000 | 3000 | 58.0% | 49 | 0.024 | 1.681 | 0.0000 |
| GT | DQN · lagrangian · normalized | 5 | 0.146 | 2.596 | — | 0.0% | 49 | 2.450 | 1.681 | 0.0000 |
| GT | DDQN · lagrangian · linear | 5 | 4.976 | 5.000 | 3000 | 58.0% | 48 | 0.024 | 1.681 | 0.0030 |
| GT | DDQN · lagrangian · normalized | 5 | 0.346 | 2.596 | — | 0.0% | 47 | 2.250 | 1.681 | 0.0000 |

## Learning curve: test relative gap

### LT

| Steps | PPO · lagrangian · linear | PPO · lagrangian · normalized | DQN · lagrangian · linear | DQN · lagrangian · normalized | DDQN · lagrangian · linear | DDQN · lagrangian · normalized |
|---|---|---|---|---|---|---|
| 0.5M | 25.8% | 23.2% | 24.5% | 30.1% | 21.4% | 31.0% |
| 1M | 25.1% | 18.9% | 28.1% | 31.8% | 23.3% | 25.5% |

### EQ

| Steps | PPO · lagrangian · linear | PPO · lagrangian · normalized | DQN · lagrangian · linear | DQN · lagrangian · normalized | DDQN · lagrangian · linear | DDQN · lagrangian · normalized |
|---|---|---|---|---|---|---|
| 0.5M | 17.0% | 30.7% | 32.7% | 36.4% | 37.4% | 32.2% |
| 1M | 17.9% | 38.4% | 16.5% | 16.6% | 17.2% | 19.3% |

### GT

| Steps | PPO · lagrangian · linear | PPO · lagrangian · normalized | DQN · lagrangian · linear | DQN · lagrangian · normalized | DDQN · lagrangian · linear | DDQN · lagrangian · normalized |
|---|---|---|---|---|---|---|
| 0.5M | 29.5% | 28.5% | 30.8% | 33.4% | 44.4% | 34.1% |
| 1M | 41.8% | 37.7% | 15.9% | 16.3% | 16.2% | 16.1% |

## Learning curve: mean over the three scenarios (per seed first)

| Steps | PPO · lagrangian · linear | PPO · lagrangian · normalized | DQN · lagrangian · linear | DQN · lagrangian · normalized | DDQN · lagrangian · linear | DDQN · lagrangian · normalized |
|---|---|---|---|---|---|---|
| 0.5M | 24.1% | 27.5% | 29.3% | 33.3% | 34.4% | 32.5% |
| 1M | 28.3% | 31.7% | 20.2% | 21.6% | 18.9% | 20.3% |

## Learning curve: test failure rate (EXIT or any violation)

The gap is computed on the remaining instances (no EXIT, no violation); with a high failure rate it covers only the easier instances.

### LT

| Steps | PPO · lagrangian · linear | PPO · lagrangian · normalized | DQN · lagrangian · linear | DQN · lagrangian · normalized | DDQN · lagrangian · linear | DDQN · lagrangian · normalized |
|---|---|---|---|---|---|---|
| 0.5M | 7.5% | 6.0% | 38.5% | 9.3% | 32.8% | 8.5% |
| 1M | 9.5% | 5.8% | 11.0% | 10.0% | 25.5% | 7.5% |

### EQ

| Steps | PPO · lagrangian · linear | PPO · lagrangian · normalized | DQN · lagrangian · linear | DQN · lagrangian · normalized | DDQN · lagrangian · linear | DDQN · lagrangian · normalized |
|---|---|---|---|---|---|---|
| 0.5M | 45.2% | 16.8% | 0.0% | 0.5% | 0.5% | 0.5% |
| 1M | 0.7% | 74.0% | 0.0% | 0.2% | 0.2% | 0.5% |

### GT

| Steps | PPO · lagrangian · linear | PPO · lagrangian · normalized | DQN · lagrangian · linear | DQN · lagrangian · normalized | DDQN · lagrangian · linear | DDQN · lagrangian · normalized |
|---|---|---|---|---|---|---|
| 0.5M | 76.0% | 89.8% | 0.0% | 0.0% | 0.0% | 0.0% |
| 1M | 7.3% | 2.2% | 0.0% | 0.0% | 0.0% | 0.0% |


## 1M: test and training-set gap, violations, early decisions

| Scenario | Model | Test relative gap | Train relative gap | Test failure rate (EXIT / violation) | Train failure rate | Capacity violation rate | Privacy violation rate | Steps 1–5 regret | Steps 1–5 optimal-action rate |
|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · lagrangian · linear | 25.1% | 25.2% | 9.5% | 8.8% | 3.2% | 8.5% | — | — |
| LT | PPO · lagrangian · normalized | 18.9% | 18.4% | 5.8% | 5.1% | 3.5% | 5.2% | — | — |
| LT | DQN · lagrangian · linear | 28.1% | 28.2% | 11.0% | 9.7% | 5.0% | 10.8% | — | — |
| LT | DQN · lagrangian · normalized | 31.8% | 31.2% | 10.0% | 9.6% | 5.8% | 5.5% | — | — |
| LT | DDQN · lagrangian · linear | 23.3% | 23.9% | 25.5% | 23.4% | 5.8% | 24.2% | — | — |
| LT | DDQN · lagrangian · normalized | 25.5% | 25.9% | 7.5% | 7.2% | 1.8% | 6.5% | — | — |
| EQ | PPO · lagrangian · linear | 17.9% | 18.2% | 0.7% | 0.2% | 0.8% | 0.2% | — | — |
| EQ | PPO · lagrangian · normalized | 38.4% | 39.1% | 74.0% | 77.9% | 52.5% | 49.5% | — | — |
| EQ | DQN · lagrangian · linear | 16.5% | 16.8% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DQN · lagrangian · normalized | 16.6% | 16.4% | 0.2% | 0.1% | 0.0% | 0.2% | — | — |
| EQ | DDQN · lagrangian · linear | 17.2% | 17.1% | 0.2% | 0.1% | 0.2% | 0.0% | — | — |
| EQ | DDQN · lagrangian · normalized | 19.3% | 18.5% | 0.5% | 0.1% | 0.5% | 0.0% | — | — |
| GT | PPO · lagrangian · linear | 41.8% | 43.0% | 7.3% | 7.0% | 7.2% | 2.2% | — | — |
| GT | PPO · lagrangian · normalized | 37.7% | 37.4% | 2.2% | 4.1% | 2.0% | 0.2% | — | — |
| GT | DQN · lagrangian · linear | 15.9% | 16.1% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DQN · lagrangian · normalized | 16.3% | 16.4% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DDQN · lagrangian · linear | 16.2% | 16.2% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DDQN · lagrangian · normalized | 16.1% | 16.2% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |

Regret is defined over the legal ECUs, i.e. for Maskable only; the other mechanisms may execute violating actions (—).


## Test gap per seed (1M)

| Scenario | Model | Seed 1 |
|---|---|---|
| LT | PPO · lagrangian · linear | 25.1% |
| LT | PPO · lagrangian · normalized | 18.9% |
| LT | DQN · lagrangian · linear | 28.1% |
| LT | DQN · lagrangian · normalized | 31.8% |
| LT | DDQN · lagrangian · linear | 23.3% |
| LT | DDQN · lagrangian · normalized | 25.5% |
| EQ | PPO · lagrangian · linear | 17.9% |
| EQ | PPO · lagrangian · normalized | 38.4% |
| EQ | DQN · lagrangian · linear | 16.5% |
| EQ | DQN · lagrangian · normalized | 16.6% |
| EQ | DDQN · lagrangian · linear | 17.2% |
| EQ | DDQN · lagrangian · normalized | 19.3% |
| GT | PPO · lagrangian · linear | 41.8% |
| GT | PPO · lagrangian · normalized | 37.7% |
| GT | DQN · lagrangian · linear | 15.9% |
| GT | DQN · lagrangian · normalized | 16.3% |
| GT | DDQN · lagrangian · linear | 16.2% |
| GT | DDQN · lagrangian · normalized | 16.1% |
