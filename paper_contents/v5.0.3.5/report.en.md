# v5.0.3.5 DQN series 20M: λ_max = 5, normalized update η = 0.025, ε from 1.0 to 0 over a fixed 2M steps, J_C of the greedy (ε = 0) policy (100 episodes per update) (seed 1, 20M, 50k instances)

- Generated 2026-10-14 13:56:59; script `src/scripts/run_v5.py --version 5.0.3.5`.
- Every variant: 1 seed(s) (1) × 3 scenario(s) (LT / EQ / GT), 20M steps each, checkpoints of the same run at 5M / 10M / 15M; ar_pen reward, γ = 1, full episode, EXIT for Maskable only, descending demand; PPO / DQN hyper-parameters from `paper_rl/config.py`; structure-aware network (Transformer) trained on GPU. 50k instances = the ILP-verified pool (`results/sup_diag/<scenario>.json`).
- Test gap: each seed's own 400 test instances; training-set gap: 1600 training instances (a fixed subset of the 50k pool, or the 1600-instance split itself); deterministic rollouts; mean ± sample std.
- Manifest: `logs/v5.0.3.5/manifest.json`.

## λ dynamics (during training, from training_curve.csv)

J_C = violations per episode (capacity + privacy); target d = 0.001. 'First at cap' = first episode with λ ≥ λ_max; 'fall' = from the peak to the end of training.

| Scenario | Model | λ_max | Final λ | Max λ | First at cap (episode) | Share at cap | Decreases | Fall | J_C first 1000 | J_C last 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | DQN · lagrangian · DQN series 20M | 5 | 5.000 | 5.000 | 198000 | 85.1% | 0 | 0.000 | 1.983 | 0.1030 |
| LT | DDQN · lagrangian · DQN series 20M | 5 | 5.000 | 5.000 | 198000 | 85.1% | 0 | 0.000 | 1.983 | 0.0870 |
| EQ | DQN · lagrangian · DQN series 20M | 5 | 0.098 | 0.241 | — | 0.0% | 893 | 0.143 | 1.810 | 0.0610 |
| EQ | DDQN · lagrangian · DQN series 20M | 5 | 0.000 | 0.323 | — | 0.0% | 885 | 0.323 | 1.812 | 0.0000 |
| GT | DQN · lagrangian · DQN series 20M | 5 | 0.136 | 0.245 | — | 0.0% | 836 | 0.109 | 1.669 | 0.0020 |
| GT | DDQN · lagrangian · DQN series 20M | 5 | 0.212 | 0.264 | — | 0.0% | 583 | 0.052 | 1.669 | 0.0100 |

### λ and J_C (from lambda_updates.csv, last update before each step)

Each cell: λ / behaviour-policy J_C / J_C used for the update (greedy: ε = 0 episodes).

| Scenario | Model | 5M | 10M | 15M | 20M |
|---|---|---|---|---|---|
| LT | DQN · lagrangian · DQN series 20M | 5.00 / 0.408 / 0.100 | 5.00 / 0.096 / 0.170 | 5.00 / 0.094 / 0.140 | 5.00 / 0.100 / 0.090 |
| EQ | DQN · lagrangian · DQN series 20M | 0.10 / 0.015 / 0.070 | 0.06 / 0.015 / 0.000 | 0.12 / 0.003 / 0.000 | 0.10 / 0.001 / 0.000 |
| GT | DQN · lagrangian · DQN series 20M | 0.15 / 0.239 / 0.100 | 0.07 / 0.003 / 0.000 | 0.18 / 0.000 / 0.000 | 0.14 / 0.001 / 0.000 |
| LT | DDQN · lagrangian · DQN series 20M | 5.00 / 0.122 / 0.070 | 5.00 / 0.099 / 0.150 | 5.00 / 0.098 / 0.220 | 5.00 / 0.076 / 0.080 |
| EQ | DDQN · lagrangian · DQN series 20M | 0.12 / 0.003 / 0.000 | 0.12 / 0.009 / 0.000 | 0.12 / 0.012 / 0.000 | 0.00 / 0.002 / 0.000 |
| GT | DDQN · lagrangian · DQN series 20M | 0.05 / 0.090 / 0.300 | 0.00 / 0.001 / 0.000 | 0.07 / 0.003 / 0.000 | 0.21 / 0.012 / 0.010 |

## Learning curve: test relative gap

### LT

| Steps | DQN · lagrangian · DQN series 20M | DDQN · lagrangian · DQN series 20M |
|---|---|---|
| 5M | 24.3% | 30.6% |
| 10M | 22.8% | 24.4% |
| 15M | 22.8% | 19.9% |
| 20M | 31.4% | 18.7% |

### EQ

| Steps | DQN · lagrangian · DQN series 20M | DDQN · lagrangian · DQN series 20M |
|---|---|---|
| 5M | 13.4% | 12.2% |
| 10M | 9.6% | 7.2% |
| 15M | 8.2% | 6.7% |
| 20M | 6.2% | 9.7% |

### GT

| Steps | DQN · lagrangian · DQN series 20M | DDQN · lagrangian · DQN series 20M |
|---|---|---|
| 5M | 16.1% | 16.1% |
| 10M | 17.0% | 11.2% |
| 15M | 9.2% | 10.2% |
| 20M | 7.1% | 8.1% |

## Learning curve: mean over the three scenarios (per seed first)

| Steps | DQN · lagrangian · DQN series 20M | DDQN · lagrangian · DQN series 20M |
|---|---|---|
| 5M | 17.9% | 19.7% |
| 10M | 16.5% | 14.3% |
| 15M | 13.4% | 12.2% |
| 20M | 14.9% | 12.1% |

## Learning curve: test failure rate (EXIT or any violation)

The gap is computed on the remaining instances (no EXIT, no violation); with a high failure rate it covers only the easier instances.

### LT

| Steps | DQN · lagrangian · DQN series 20M | DDQN · lagrangian · DQN series 20M |
|---|---|---|
| 5M | 7.5% | 6.2% |
| 10M | 6.8% | 6.2% |
| 15M | 6.0% | 7.0% |
| 20M | 6.8% | 6.5% |

### EQ

| Steps | DQN · lagrangian · DQN series 20M | DDQN · lagrangian · DQN series 20M |
|---|---|---|
| 5M | 23.8% | 0.5% |
| 10M | 0.0% | 1.7% |
| 15M | 0.7% | 2.2% |
| 20M | 4.2% | 0.0% |

### GT

| Steps | DQN · lagrangian · DQN series 20M | DDQN · lagrangian · DQN series 20M |
|---|---|---|
| 5M | 3.2% | 2.7% |
| 10M | 0.7% | 0.0% |
| 15M | 0.2% | 0.0% |
| 20M | 0.0% | 1.0% |


## 20M: test and training-set gap, violations, early decisions

| Scenario | Model | Test relative gap | Train relative gap | Test failure rate (EXIT / violation) | Train failure rate | Capacity violation rate | Privacy violation rate | Steps 1–5 regret | Steps 1–5 optimal-action rate |
|---|---|---|---|---|---|---|---|---|---|
| LT | DQN · lagrangian · DQN series 20M | 31.4% | 31.1% | 6.8% | 7.0% | 5.0% | 2.8% | — | — |
| LT | DDQN · lagrangian · DQN series 20M | 18.7% | 18.1% | 6.5% | 6.1% | 5.0% | 4.8% | — | — |
| EQ | DQN · lagrangian · DQN series 20M | 6.2% | 6.2% | 4.2% | 3.3% | 0.0% | 4.2% | — | — |
| EQ | DDQN · lagrangian · DQN series 20M | 9.7% | 9.3% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| GT | DQN · lagrangian · DQN series 20M | 7.1% | 6.6% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DDQN · lagrangian · DQN series 20M | 8.1% | 8.0% | 1.0% | 0.9% | 0.0% | 1.0% | — | — |

Regret is defined over the legal ECUs, i.e. for Maskable only; the other mechanisms may execute violating actions (—).


## Test gap per seed (20M)

| Scenario | Model | Seed 1 |
|---|---|---|
| LT | DQN · lagrangian · DQN series 20M | 31.4% |
| LT | DDQN · lagrangian · DQN series 20M | 18.7% |
| EQ | DQN · lagrangian · DQN series 20M | 6.2% |
| EQ | DDQN · lagrangian · DQN series 20M | 9.7% |
| GT | DQN · lagrangian · DQN series 20M | 7.1% |
| GT | DDQN · lagrangian · DQN series 20M | 8.1% |
