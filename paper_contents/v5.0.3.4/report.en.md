# v5.0.3.4 DQN / DDQN lagrangian λ research: λ_max = 5, normalized update η = 0.025, ε from 1.0 to 0 over a fixed 2M steps, J_C of the greedy (ε = 0) policy (100 episodes per update) (seed 1, 5M, 50k instances)

- Generated 2026-10-14 10:40:16; script `src/scripts/run_v5.py --version 5.0.3.4`.
- Every variant: 1 seed(s) (1) × 3 scenario(s) (LT / EQ / GT), 5M steps each, checkpoints of the same run at 1M / 2M / 3M / 4M; ar_pen reward, γ = 1, full episode, EXIT for Maskable only, descending demand; PPO / DQN hyper-parameters from `paper_rl/config.py`; structure-aware network (Transformer) trained on GPU. 50k instances = the ILP-verified pool (`results/sup_diag/<scenario>.json`).
- Test gap: each seed's own 400 test instances; training-set gap: 1600 training instances (a fixed subset of the 50k pool, or the 1600-instance split itself); deterministic rollouts; mean ± sample std.
- Manifest: `logs/v5.0.3.4/manifest.json`.

## λ dynamics (during training, from training_curve.csv)

J_C = violations per episode (capacity + privacy); target d = 0.001. 'First at cap' = first episode with λ ≥ λ_max; 'fall' = from the peak to the end of training.

| Scenario | Model | λ_max | Final λ | Max λ | First at cap (episode) | Share at cap | Decreases | Fall | J_C first 1000 | J_C last 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | DQN · lagrangian · λ research | 5 | 5.000 | 5.000 | 198000 | 40.6% | 0 | 0.000 | 1.983 | 0.1780 |
| LT | DDQN · lagrangian · λ research | 5 | 5.000 | 5.000 | 198000 | 40.6% | 0 | 0.000 | 1.983 | 0.0950 |
| EQ | DQN · lagrangian · λ research | 5 | 0.101 | 0.208 | — | 0.0% | 168 | 0.107 | 1.810 | 0.0860 |
| EQ | DDQN · lagrangian · λ research | 5 | 0.119 | 0.192 | — | 0.0% | 188 | 0.073 | 1.812 | 0.0100 |
| GT | DQN · lagrangian · λ research | 5 | 0.149 | 0.215 | — | 0.0% | 141 | 0.067 | 1.669 | 0.0360 |
| GT | DDQN · lagrangian · λ research | 5 | 0.049 | 0.170 | — | 0.0% | 56 | 0.121 | 1.669 | 0.0620 |

### λ and J_C (from lambda_updates.csv, last update before each step)

Each cell: λ / behaviour-policy J_C / J_C used for the update (greedy: ε = 0 episodes).

| Scenario | Model | 1M | 2M | 3M | 4M | 5M |
|---|---|---|---|---|---|---|
| LT | DQN · lagrangian · λ research | 1.74 / 5.075 / 0.240 | 3.39 / 0.189 / 0.120 | 5.00 / 0.257 / 0.250 | 5.00 / 0.148 / 0.180 | 5.00 / 0.408 / 0.100 |
| EQ | DQN · lagrangian · λ research | 0.02 / 2.541 / 0.020 | 0.00 / 0.046 / 0.000 | 0.00 / 0.001 / 0.000 | 0.07 / 0.108 / 0.110 | 0.10 / 0.015 / 0.070 |
| GT | DQN · lagrangian · λ research | 0.00 / 1.946 / 0.000 | 0.05 / 0.134 / 0.000 | 0.07 / 0.023 / 0.000 | 0.00 / 0.005 / 0.000 | 0.15 / 0.239 / 0.100 |
| LT | DDQN · lagrangian · λ research | 1.74 / 4.969 / 0.140 | 3.39 / 0.213 / 0.120 | 5.00 / 0.123 / 0.190 | 5.00 / 0.103 / 0.170 | 5.00 / 0.122 / 0.070 |
| EQ | DDQN · lagrangian · λ research | 0.02 / 2.467 / 0.020 | 0.00 / 0.048 / 0.000 | 0.06 / 0.003 / 0.000 | 0.12 / 0.038 / 0.020 | 0.12 / 0.003 / 0.000 |
| GT | DDQN · lagrangian · λ research | 0.00 / 1.926 / 0.000 | 0.00 / 0.056 / 0.000 | 0.00 / 0.000 / 0.000 | 0.07 / 0.094 / 0.210 | 0.05 / 0.090 / 0.300 |

## Learning curve: test relative gap

### LT

| Steps | DQN · lagrangian · λ research | DDQN · lagrangian · λ research |
|---|---|---|
| 1M | 33.1% | 30.7% |
| 2M | 28.9% | 29.4% |
| 3M | 20.8% | 28.6% |
| 4M | 24.4% | 31.3% |
| 5M | 24.4% | 30.8% |

### EQ

| Steps | DQN · lagrangian · λ research | DDQN · lagrangian · λ research |
|---|---|---|
| 1M | 29.2% | 42.1% |
| 2M | 31.8% | 32.0% |
| 3M | 14.7% | 16.7% |
| 4M | 12.6% | 14.4% |
| 5M | 13.6% | 12.0% |

### GT

| Steps | DQN · lagrangian · λ research | DDQN · lagrangian · λ research |
|---|---|---|
| 1M | 27.1% | 28.3% |
| 2M | 26.2% | 20.5% |
| 3M | 20.2% | 14.3% |
| 4M | 15.2% | 15.7% |
| 5M | 16.1% | 16.0% |

## Learning curve: mean over the three scenarios (per seed first)

| Steps | DQN · lagrangian · λ research | DDQN · lagrangian · λ research |
|---|---|---|
| 1M | 29.8% | 33.7% |
| 2M | 29.0% | 27.3% |
| 3M | 18.6% | 19.9% |
| 4M | 17.4% | 20.5% |
| 5M | 18.0% | 19.6% |

## Learning curve: test failure rate (EXIT or any violation)

The gap is computed on the remaining instances (no EXIT, no violation); with a high failure rate it covers only the easier instances.

### LT

| Steps | DQN · lagrangian · λ research | DDQN · lagrangian · λ research |
|---|---|---|
| 1M | 7.0% | 8.8% |
| 2M | 9.5% | 7.5% |
| 3M | 7.3% | 7.0% |
| 4M | 7.3% | 9.8% |
| 5M | 7.8% | 6.2% |

### EQ

| Steps | DQN · lagrangian · λ research | DDQN · lagrangian · λ research |
|---|---|---|
| 1M | 0.5% | 0.2% |
| 2M | 0.5% | 0.5% |
| 3M | 0.2% | 0.5% |
| 4M | 4.5% | 0.2% |
| 5M | 24.0% | 0.5% |

### GT

| Steps | DQN · lagrangian · λ research | DDQN · lagrangian · λ research |
|---|---|---|
| 1M | 0.0% | 0.0% |
| 2M | 0.0% | 0.0% |
| 3M | 0.5% | 0.0% |
| 4M | 0.5% | 4.5% |
| 5M | 3.0% | 3.0% |


## 5M: test and training-set gap, violations, early decisions

| Scenario | Model | Test relative gap | Train relative gap | Test failure rate (EXIT / violation) | Train failure rate | Capacity violation rate | Privacy violation rate | Steps 1–5 regret | Steps 1–5 optimal-action rate |
|---|---|---|---|---|---|---|---|---|---|
| LT | DQN · lagrangian · λ research | 24.4% | 24.0% | 7.8% | 6.5% | 3.8% | 7.2% | — | — |
| LT | DDQN · lagrangian · λ research | 30.8% | 30.3% | 6.2% | 6.3% | 2.2% | 5.2% | — | — |
| EQ | DQN · lagrangian · λ research | 13.6% | 14.2% | 24.0% | 21.0% | 0.0% | 24.0% | — | — |
| EQ | DDQN · lagrangian · λ research | 12.0% | 12.4% | 0.5% | 0.6% | 0.2% | 0.5% | — | — |
| GT | DQN · lagrangian · λ research | 16.1% | 15.7% | 3.0% | 5.2% | 2.2% | 3.0% | — | — |
| GT | DDQN · lagrangian · λ research | 16.0% | 16.2% | 3.0% | 2.8% | 0.2% | 3.0% | — | — |

Regret is defined over the legal ECUs, i.e. for Maskable only; the other mechanisms may execute violating actions (—).


## Test gap per seed (5M)

| Scenario | Model | Seed 1 |
|---|---|---|
| LT | DQN · lagrangian · λ research | 24.4% |
| LT | DDQN · lagrangian · λ research | 30.8% |
| EQ | DQN · lagrangian · λ research | 13.6% |
| EQ | DDQN · lagrangian · λ research | 12.0% |
| GT | DQN · lagrangian · λ research | 16.1% |
| GT | DDQN · lagrangian · λ research | 16.0% |
