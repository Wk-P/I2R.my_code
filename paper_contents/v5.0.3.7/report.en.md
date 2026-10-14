# v5.0.3.7 hyper-params lagrangian: the usual Safe RL setting (OmniSafe PPOLag) -- λ init 0.001, Adam (lr 0.035) on λ, λ ≥ 0 uncapped, J_C of the behaviour policy, d = 0; PPO updates once per rollout, DQN / DDQN every 1000 episodes (ε 1.0→0 over a fixed 2M steps) (seed 1, 5M, 50k instances)

- Generated 2026-10-14 21:00:08; script `src/scripts/run_v5.py --version 5.0.3.7`.
- Every variant: 1 seed(s) (1) × 3 scenario(s) (LT / EQ / GT), 5M steps each, checkpoints of the same run at 1M / 2M / 3M / 4M; ar_pen reward, γ = 1, full episode, EXIT for Maskable only, descending demand; PPO / DQN hyper-parameters from `paper_rl/config.py`; structure-aware network (Transformer) trained on GPU. 50k instances = the ILP-verified pool (`results/sup_diag/<scenario>.json`).
- Test gap: each seed's own 400 test instances; training-set gap: 1600 training instances (a fixed subset of the 50k pool, or the 1600-instance split itself); deterministic rollouts; mean ± sample std.
- Manifest: `logs/v5.0.3.7/manifest.json`.

## λ dynamics (during training, from training_curve.csv)

J_C = violations per episode (capacity + privacy); target d = 0.001. 'First at cap' = first episode with λ ≥ λ_max; 'fall' = from the peak to the end of training.

| Scenario | Model | λ_max | Final λ | Max λ | First at cap (episode) | Share at cap | Decreases | Fall | J_C first 1000 | J_C last 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · lagrangian · hyper-params | none | 1.261 | 1.261 | — | 0.0% | 0 | 0.000 | 1.981 | 0.0540 |
| LT | DQN · lagrangian · hyper-params | none | 3.512 | 3.512 | — | 0.0% | 0 | 0.000 | 1.983 | 0.1110 |
| LT | DDQN · lagrangian · hyper-params | none | 3.461 | 3.461 | — | 0.0% | 0 | 0.000 | 1.983 | 0.0960 |
| EQ | PPO · lagrangian · hyper-params | none | 1.330 | 1.330 | — | 0.0% | 0 | 0.000 | 1.820 | 0.0040 |
| EQ | DQN · lagrangian · hyper-params | none | 4.646 | 4.646 | — | 0.0% | 0 | 0.000 | 1.812 | 0.0020 |
| EQ | DDQN · lagrangian · hyper-params | none | 4.606 | 4.606 | — | 0.0% | 0 | 0.000 | 1.811 | 0.0020 |
| GT | PPO · lagrangian · hyper-params | none | 1.025 | 1.025 | — | 0.0% | 0 | 0.000 | 1.645 | 0.1950 |
| GT | DQN · lagrangian · hyper-params | none | 5.012 | 5.012 | — | 0.0% | 0 | 0.000 | 1.672 | 0.0000 |
| GT | DDQN · lagrangian · hyper-params | none | 4.897 | 4.897 | — | 0.0% | 0 | 0.000 | 1.672 | 0.0000 |

### λ and J_C (from lambda_updates.csv, last update before each step)

Each cell: λ / behaviour-policy J_C / J_C used for the update (greedy: ε = 0 episodes).

| Scenario | Model | 1M | 2M | 3M | 4M | 5M |
|---|---|---|---|---|---|---|
| LT | PPO · lagrangian · hyper-params | 0.85 / 0.210 / 0.210 | 0.99 / 0.105 / 0.105 | 1.09 / 0.084 / 0.084 | 1.18 / 0.058 / 0.058 | 1.26 / 0.068 / 0.068 |
| EQ | PPO · lagrangian · hyper-params | 1.05 / 0.012 / 0.012 | 1.28 / 0.015 / 0.015 | 1.30 / 0.001 / 0.001 | 1.32 / 0.005 / 0.005 | 1.33 / 0.001 / 0.001 |
| GT | PPO · lagrangian · hyper-params | 0.68 / 0.346 / 0.346 | 0.91 / 0.002 / 0.002 | 0.93 / 0.029 / 0.029 | 0.95 / 0.001 / 0.001 | 1.02 / 0.003 / 0.003 |
| LT | DQN · lagrangian · hyper-params | 2.11 / 5.224 / 5.224 | 3.22 / 0.223 / 0.223 | 3.35 / 0.124 / 0.124 | 3.44 / 0.100 / 0.100 | 3.51 / 0.138 / 0.138 |
| EQ | DQN · lagrangian · hyper-params | 3.07 / 2.179 / 2.179 | 4.58 / 0.041 / 0.041 | 4.61 / 0.000 / 0.000 | 4.64 / 0.001 / 0.001 | 4.65 / 0.002 / 0.002 |
| GT | DQN · lagrangian · hyper-params | 3.15 / 1.751 / 1.751 | 4.91 / 0.029 / 0.029 | 4.95 / 0.000 / 0.000 | 5.01 / 0.000 / 0.000 | 5.01 / 0.000 / 0.000 |
| LT | DDQN · lagrangian · hyper-params | 2.10 / 4.939 / 4.939 | 3.21 / 0.247 / 0.247 | 3.31 / 0.090 / 0.090 | 3.38 / 0.094 / 0.094 | 3.46 / 0.109 / 0.109 |
| EQ | DDQN · lagrangian · hyper-params | 3.06 / 2.099 / 2.099 | 4.56 / 0.041 / 0.041 | 4.60 / 0.000 / 0.000 | 4.60 / 0.001 / 0.001 | 4.61 / 0.000 / 0.000 |
| GT | DDQN · lagrangian · hyper-params | 3.11 / 1.740 / 1.740 | 4.83 / 0.029 / 0.029 | 4.89 / 0.000 / 0.000 | 4.89 / 0.000 / 0.000 | 4.90 / 0.000 / 0.000 |

## Learning curve: test relative gap

### LT

| Steps | PPO · lagrangian · hyper-params | DQN · lagrangian · hyper-params | DDQN · lagrangian · hyper-params |
|---|---|---|---|
| 1M | 18.1% | 27.1% | 27.7% |
| 2M | 12.4% | 31.8% | 29.7% |
| 3M | 10.1% | 24.8% | 27.2% |
| 4M | 9.8% | 26.8% | 26.5% |
| 5M | 9.3% | 31.2% | 20.3% |

### EQ

| Steps | PPO · lagrangian · hyper-params | DQN · lagrangian · hyper-params | DDQN · lagrangian · hyper-params |
|---|---|---|---|
| 1M | 17.0% | 46.3% | 44.1% |
| 2M | 12.1% | 36.5% | 19.9% |
| 3M | 10.5% | 17.5% | 16.9% |
| 4M | 9.7% | 16.7% | 17.6% |
| 5M | 9.7% | 17.0% | 14.1% |

### GT

| Steps | PPO · lagrangian · hyper-params | DQN · lagrangian · hyper-params | DDQN · lagrangian · hyper-params |
|---|---|---|---|
| 1M | 12.4% | 30.7% | 45.4% |
| 2M | 10.3% | 16.0% | 22.4% |
| 3M | 10.7% | 15.8% | 16.2% |
| 4M | 10.6% | 16.1% | 14.8% |
| 5M | 10.9% | 12.2% | 14.6% |

## Learning curve: mean over the three scenarios (per seed first)

| Steps | PPO · lagrangian · hyper-params | DQN · lagrangian · hyper-params | DDQN · lagrangian · hyper-params |
|---|---|---|---|
| 1M | 15.8% | 34.7% | 39.1% |
| 2M | 11.6% | 28.1% | 24.0% |
| 3M | 10.4% | 19.3% | 20.1% |
| 4M | 10.0% | 19.8% | 19.6% |
| 5M | 10.0% | 20.2% | 16.3% |

## Learning curve: test failure rate (EXIT or any violation)

The gap is computed on the remaining instances (no EXIT, no violation); with a high failure rate it covers only the easier instances.

### LT

| Steps | PPO · lagrangian · hyper-params | DQN · lagrangian · hyper-params | DDQN · lagrangian · hyper-params |
|---|---|---|---|
| 1M | 5.5% | 7.8% | 6.8% |
| 2M | 3.7% | 9.0% | 7.3% |
| 3M | 4.5% | 32.8% | 8.8% |
| 4M | 3.7% | 7.0% | 6.2% |
| 5M | 2.7% | 7.3% | 6.0% |

### EQ

| Steps | PPO · lagrangian · hyper-params | DQN · lagrangian · hyper-params | DDQN · lagrangian · hyper-params |
|---|---|---|---|
| 1M | 16.8% | 0.2% | 0.0% |
| 2M | 0.2% | 0.2% | 0.5% |
| 3M | 0.2% | 0.0% | 0.2% |
| 4M | 0.0% | 0.0% | 0.0% |
| 5M | 0.2% | 0.0% | 0.0% |

### GT

| Steps | PPO · lagrangian · hyper-params | DQN · lagrangian · hyper-params | DDQN · lagrangian · hyper-params |
|---|---|---|---|
| 1M | 5.0% | 0.0% | 0.0% |
| 2M | 0.2% | 0.0% | 0.0% |
| 3M | 1.0% | 0.0% | 0.0% |
| 4M | 0.2% | 0.0% | 0.0% |
| 5M | 0.7% | 0.0% | 0.0% |


## 5M: test and training-set gap, violations, early decisions

| Scenario | Model | Test relative gap | Train relative gap | Test failure rate (EXIT / violation) | Train failure rate | Capacity violation rate | Privacy violation rate | Steps 1–5 regret | Steps 1–5 optimal-action rate |
|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · lagrangian · hyper-params | 9.3% | 9.2% | 2.7% | 2.7% | 2.5% | 1.2% | — | — |
| LT | DQN · lagrangian · hyper-params | 31.2% | 31.0% | 7.3% | 6.6% | 3.5% | 6.0% | — | — |
| LT | DDQN · lagrangian · hyper-params | 20.3% | 20.1% | 6.0% | 5.8% | 2.2% | 4.5% | — | — |
| EQ | PPO · lagrangian · hyper-params | 9.7% | 9.0% | 0.2% | 0.2% | 0.2% | 0.0% | — | — |
| EQ | DQN · lagrangian · hyper-params | 17.0% | 17.3% | 0.0% | 0.2% | 0.0% | 0.0% | — | — |
| EQ | DDQN · lagrangian · hyper-params | 14.1% | 14.2% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| GT | PPO · lagrangian · hyper-params | 10.9% | 10.7% | 0.7% | 0.8% | 0.2% | 0.5% | — | — |
| GT | DQN · lagrangian · hyper-params | 12.2% | 11.3% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DDQN · lagrangian · hyper-params | 14.6% | 14.6% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |

Regret is defined over the legal ECUs, i.e. for Maskable only; the other mechanisms may execute violating actions (—).


## Test gap per seed (5M)

| Scenario | Model | Seed 1 |
|---|---|---|
| LT | PPO · lagrangian · hyper-params | 9.3% |
| LT | DQN · lagrangian · hyper-params | 31.2% |
| LT | DDQN · lagrangian · hyper-params | 20.3% |
| EQ | PPO · lagrangian · hyper-params | 9.7% |
| EQ | DQN · lagrangian · hyper-params | 17.0% |
| EQ | DDQN · lagrangian · hyper-params | 14.1% |
| GT | PPO · lagrangian · hyper-params | 10.9% |
| GT | DQN · lagrangian · hyper-params | 12.2% |
| GT | DDQN · lagrangian · hyper-params | 14.6% |
