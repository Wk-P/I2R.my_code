# v5.1.0 lagrangian 20M version: the v5.0.3.7 setting (λ init 0.001, Adam lr 0.035, λ ≥ 0 uncapped, J_C of the behaviour policy, d = 0; PPO updates once per rollout, DQN / DDQN every 1000 episodes, ε 1.0→0 over a fixed 2M steps), extended to 20M (seed 1, 20M, 50k instances)

- Generated 2026-10-15 04:56:53; script `src/scripts/run_v5.py --version 5.1.0`.
- Every variant: 1 seed(s) (1) × 3 scenario(s) (LT / EQ / GT), 20M steps each, checkpoints of the same run at 5M / 10M / 15M; ar_pen reward, γ = 1, full episode, EXIT for Maskable only, descending demand; PPO / DQN hyper-parameters from `paper_rl/config.py`; structure-aware network (Transformer) trained on GPU. 50k instances = the ILP-verified pool (`results/sup_diag/<scenario>.json`).
- Test gap: each seed's own 400 test instances; training-set gap: 1600 training instances (a fixed subset of the 50k pool, or the 1600-instance split itself); deterministic rollouts; mean ± sample std.
- Manifest: `logs/v5.1.0/manifest.json`.

## λ dynamics (during training, from training_curve.csv)

J_C = violations per episode (capacity + privacy); target d = 0. 'First at cap' = first episode with λ ≥ λ_max; 'fall' = from the peak to the end of training.

| Scenario | Model | λ_max | Final λ | Max λ | First at cap (episode) | Share at cap | Decreases | Fall | J_C first 1000 | J_C last 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · lagrangian 20M version | none | 2.971 | 2.971 | — | 0.0% | 0 | 0.000 | 1.981 | 0.0440 |
| LT | DQN · lagrangian 20M version | none | 5.597 | 5.597 | — | 0.0% | 0 | 0.000 | 1.983 | 0.0960 |
| LT | DDQN · lagrangian 20M version | none | 5.307 | 5.307 | — | 0.0% | 0 | 0.000 | 1.983 | 0.0890 |
| EQ | PPO · lagrangian 20M version | none | 1.554 | 1.554 | — | 0.0% | 0 | 0.000 | 1.820 | 0.0090 |
| EQ | DQN · lagrangian 20M version | none | 4.735 | 4.735 | — | 0.0% | 0 | 0.000 | 1.812 | 0.0000 |
| EQ | DDQN · lagrangian 20M version | none | 5.079 | 5.079 | — | 0.0% | 0 | 0.000 | 1.811 | 0.0000 |
| GT | PPO · lagrangian 20M version | none | 1.623 | 1.623 | — | 0.0% | 0 | 0.000 | 1.645 | 0.0000 |
| GT | DQN · lagrangian 20M version | none | 5.097 | 5.097 | — | 0.0% | 0 | 0.000 | 1.672 | 0.0000 |
| GT | DDQN · lagrangian 20M version | none | 4.923 | 4.923 | — | 0.0% | 0 | 0.000 | 1.672 | 0.0000 |

### λ and J_C (from lambda_updates.csv, last update before each step)

Each cell: λ / behaviour-policy J_C / J_C used for the update (greedy: ε = 0 episodes).

| Scenario | Model | 5M | 10M | 15M | 20M |
|---|---|---|---|---|---|
| LT | PPO · lagrangian 20M version | 1.26 / 0.068 / 0.068 | 1.75 / 0.034 / 0.034 | 2.26 / 0.043 / 0.043 | 2.97 / 0.053 / 0.053 |
| EQ | PPO · lagrangian 20M version | 1.33 / 0.001 / 0.001 | 1.44 / 0.000 / 0.000 | 1.50 / 0.001 / 0.001 | 1.55 / 0.025 / 0.025 |
| GT | PPO · lagrangian 20M version | 1.02 / 0.003 / 0.003 | 1.28 / 0.000 / 0.000 | 1.34 / 0.000 / 0.000 | 1.62 / 0.001 / 0.001 |
| LT | DQN · lagrangian 20M version | 3.51 / 0.138 / 0.138 | 4.06 / 0.085 / 0.085 | 4.75 / 0.092 / 0.092 | 5.60 / 0.089 / 0.089 |
| EQ | DQN · lagrangian 20M version | 4.65 / 0.002 / 0.002 | 4.71 / 0.001 / 0.001 | 4.72 / 0.001 / 0.001 | 4.74 / 0.001 / 0.001 |
| GT | DQN · lagrangian 20M version | 5.01 / 0.000 / 0.000 | 5.04 / 0.000 / 0.000 | 5.05 / 0.000 / 0.000 | 5.10 / 0.000 / 0.000 |
| LT | DDQN · lagrangian 20M version | 3.46 / 0.109 / 0.109 | 3.89 / 0.098 / 0.098 | 4.53 / 0.096 / 0.096 | 5.31 / 0.081 / 0.081 |
| EQ | DDQN · lagrangian 20M version | 4.61 / 0.000 / 0.000 | 4.65 / 0.001 / 0.001 | 4.93 / 0.002 / 0.002 | 5.08 / 0.001 / 0.001 |
| GT | DDQN · lagrangian 20M version | 4.90 / 0.000 / 0.000 | 4.91 / 0.000 / 0.000 | 4.92 / 0.000 / 0.000 | 4.92 / 0.000 / 0.000 |

## Learning curve: test relative gap

### LT

| Steps | PPO · lagrangian 20M version | DQN · lagrangian 20M version | DDQN · lagrangian 20M version |
|---|---|---|---|
| 5M | 9.3% | 31.1% | 20.4% |
| 10M | 7.8% | 24.9% | 30.5% |
| 15M | 7.0% | 26.9% | 26.8% |
| 20M | 6.7% | 25.9% | 26.7% |

### EQ

| Steps | PPO · lagrangian 20M version | DQN · lagrangian 20M version | DDQN · lagrangian 20M version |
|---|---|---|---|
| 5M | 9.7% | 17.2% | 14.2% |
| 10M | 7.6% | 11.7% | 16.7% |
| 15M | 6.9% | 8.0% | 16.9% |
| 20M | 6.0% | 8.1% | 11.9% |

### GT

| Steps | PPO · lagrangian 20M version | DQN · lagrangian 20M version | DDQN · lagrangian 20M version |
|---|---|---|---|
| 5M | 11.1% | 12.2% | 14.7% |
| 10M | 9.6% | 7.9% | 10.7% |
| 15M | 8.9% | 6.1% | 7.3% |
| 20M | 7.2% | 3.4% | 6.5% |

## Learning curve: mean over the three scenarios (per seed first)

| Steps | PPO · lagrangian 20M version | DQN · lagrangian 20M version | DDQN · lagrangian 20M version |
|---|---|---|---|
| 5M | 10.0% | 20.2% | 16.4% |
| 10M | 8.3% | 14.8% | 19.3% |
| 15M | 7.6% | 13.7% | 17.0% |
| 20M | 6.6% | 12.5% | 15.0% |

## Learning curve: test failure rate (EXIT or any violation)

The gap is computed on the remaining instances (no EXIT, no violation); with a high failure rate it covers only the easier instances.

### LT

| Steps | PPO · lagrangian 20M version | DQN · lagrangian 20M version | DDQN · lagrangian 20M version |
|---|---|---|---|
| 5M | 2.2% | 7.5% | 6.0% |
| 10M | 2.7% | 7.8% | 6.5% |
| 15M | 2.7% | 6.0% | 7.3% |
| 20M | 2.7% | 6.8% | 6.0% |

### EQ

| Steps | PPO · lagrangian 20M version | DQN · lagrangian 20M version | DDQN · lagrangian 20M version |
|---|---|---|---|
| 5M | 0.5% | 0.0% | 0.0% |
| 10M | 0.0% | 0.0% | 0.0% |
| 15M | 0.0% | 0.0% | 0.0% |
| 20M | 0.0% | 0.0% | 0.0% |

### GT

| Steps | PPO · lagrangian 20M version | DQN · lagrangian 20M version | DDQN · lagrangian 20M version |
|---|---|---|---|
| 5M | 4.0% | 0.0% | 0.0% |
| 10M | 0.0% | 0.0% | 0.0% |
| 15M | 0.2% | 0.0% | 0.0% |
| 20M | 0.0% | 0.0% | 0.0% |


## 20M: test and training-set gap, violations, early decisions

| Scenario | Model | Test relative gap | Train relative gap | Test failure rate (EXIT / violation) | Train failure rate | Capacity violation rate | Privacy violation rate | Steps 1–5 regret | Steps 1–5 optimal-action rate |
|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · lagrangian 20M version | 6.7% | 6.4% | 2.7% | 2.2% | 2.8% | 0.2% | — | — |
| LT | DQN · lagrangian 20M version | 25.9% | 25.3% | 6.8% | 5.8% | 2.0% | 5.2% | — | — |
| LT | DDQN · lagrangian 20M version | 26.7% | 26.5% | 6.0% | 6.4% | 3.8% | 3.8% | — | — |
| EQ | PPO · lagrangian 20M version | 6.0% | 6.1% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DQN · lagrangian 20M version | 8.1% | 8.5% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DDQN · lagrangian 20M version | 11.9% | 12.0% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| GT | PPO · lagrangian 20M version | 7.2% | 7.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DQN · lagrangian 20M version | 3.4% | 3.4% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DDQN · lagrangian 20M version | 6.5% | 6.2% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |

Regret is defined over the legal ECUs, i.e. for Maskable only; the other mechanisms may execute violating actions (—).


## Test gap per seed (20M)

| Scenario | Model | Seed 1 |
|---|---|---|
| LT | PPO · lagrangian 20M version | 6.7% |
| LT | DQN · lagrangian 20M version | 25.9% |
| LT | DDQN · lagrangian 20M version | 26.7% |
| EQ | PPO · lagrangian 20M version | 6.0% |
| EQ | DQN · lagrangian 20M version | 8.1% |
| EQ | DDQN · lagrangian 20M version | 11.9% |
| GT | PPO · lagrangian 20M version | 7.2% |
| GT | DQN · lagrangian 20M version | 3.4% |
| GT | DDQN · lagrangian 20M version | 6.5% |
