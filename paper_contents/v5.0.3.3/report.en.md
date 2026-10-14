# v5.0.3.3 Exploration decoupled from the λ update: ε reaches 0 at a fixed 2M steps, J_C of the greedy (ε = 0) policy (100 episodes per update), normalized update η = 0.05, λ uncapped (seed 1, 5M, 50k instances)

- Generated 2026-10-14 04:09:27; script `src/scripts/run_v5.py --version 5.0.3.3`.
- Every variant: 1 seed(s) (1) × 3 scenario(s) (LT / EQ / GT), 5M steps each, checkpoints of the same run at 1M / 2M / 3M / 4M; ar_pen reward, γ = 1, full episode, EXIT for Maskable only, descending demand; PPO / DQN hyper-parameters from `paper_rl/config.py`; structure-aware network (Transformer) trained on GPU. 50k instances = the ILP-verified pool (`results/sup_diag/<scenario>.json`).
- Test gap: each seed's own 400 test instances; training-set gap: 1600 training instances (a fixed subset of the 50k pool, or the 1600-instance split itself); deterministic rollouts; mean ± sample std.
- Manifest: `logs/v5.0.3.3/manifest.json`.

## λ dynamics (during training, from training_curve.csv)

J_C = violations per episode (capacity + privacy); target d = 0.001. 'First at cap' = first episode with λ ≥ λ_max; 'fall' = from the peak to the end of training.

| Scenario | Model | λ_max | Final λ | Max λ | First at cap (episode) | Share at cap | Decreases | Fall | J_C first 1000 | J_C last 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | DQN · lagrangian · greedy J_C | none | 16.679 | 16.679 | — | 0.0% | 0 | 0.000 | 1.983 | 0.1540 |
| LT | DDQN · lagrangian · greedy J_C | none | 16.636 | 16.636 | — | 0.0% | 0 | 0.000 | 1.983 | 0.1990 |
| EQ | DQN · lagrangian · greedy J_C | none | 0.156 | 0.241 | — | 0.0% | 191 | 0.085 | 1.810 | 0.0110 |
| EQ | DDQN · lagrangian · greedy J_C | none | 0.050 | 0.239 | — | 0.0% | 189 | 0.189 | 1.812 | 0.1930 |
| GT | DQN · lagrangian · greedy J_C | none | 0.000 | 0.284 | — | 0.0% | 93 | 0.284 | 1.669 | 0.0140 |
| GT | DDQN · lagrangian · greedy J_C | none | 0.000 | 0.375 | — | 0.0% | 137 | 0.375 | 1.669 | 0.0040 |

### λ and J_C (from lambda_updates.csv, last update before each step)

Each cell: λ / behaviour-policy J_C / J_C used for the update (greedy: ε = 0 episodes).

| Scenario | Model | 1M | 2M | 3M | 4M | 5M |
|---|---|---|---|---|---|---|
| LT | DQN · lagrangian · greedy J_C | 3.39 / 5.827 / 0.720 | 6.74 / 0.731 / 0.740 | 10.03 / 0.348 / 0.410 | 13.36 / 0.193 / 0.580 | 16.68 / 0.177 / 0.160 |
| EQ | DQN · lagrangian · greedy J_C | 0.05 / 2.562 / 0.020 | 0.14 / 0.185 / 0.100 | 0.09 / 0.015 / 0.000 | 0.11 / 0.055 / 0.050 | 0.16 / 0.108 / 0.160 |
| GT | DQN · lagrangian · greedy J_C | 0.00 / 1.922 / 0.000 | 0.00 / 0.035 / 0.000 | 0.00 / 0.000 / 0.000 | 0.00 / 0.000 / 0.000 | 0.00 / 0.003 / 0.000 |
| LT | DDQN · lagrangian · greedy J_C | 3.38 / 4.953 / 0.160 | 6.70 / 0.237 / 0.110 | 9.98 / 0.169 / 0.230 | 13.31 / 0.189 / 0.240 | 16.64 / 0.313 / 0.100 |
| EQ | DDQN · lagrangian · greedy J_C | 0.13 / 2.388 / 0.020 | 0.00 / 0.049 / 0.000 | 0.10 / 0.007 / 0.000 | 0.09 / 0.004 / 0.000 | 0.05 / 0.053 / 0.210 |
| GT | DDQN · lagrangian · greedy J_C | 0.00 / 1.967 / 0.000 | 0.08 / 0.304 / 0.150 | 0.20 / 0.002 / 0.010 | 0.00 / 0.012 / 0.000 | 0.00 / 0.004 / 0.000 |

## Learning curve: test relative gap

### LT

| Steps | DQN · lagrangian · greedy J_C | DDQN · lagrangian · greedy J_C |
|---|---|---|
| 1M | 19.7% | 29.8% |
| 2M | 24.4% | 27.3% |
| 3M | 26.6% | 26.0% |
| 4M | 31.2% | 30.1% |
| 5M | 25.7% | 24.8% |

### EQ

| Steps | DQN · lagrangian · greedy J_C | DDQN · lagrangian · greedy J_C |
|---|---|---|
| 1M | 41.2% | 38.0% |
| 2M | 32.0% | 28.1% |
| 3M | 14.6% | 18.9% |
| 4M | 12.7% | 16.9% |
| 5M | 11.8% | 16.9% |

### GT

| Steps | DQN · lagrangian · greedy J_C | DDQN · lagrangian · greedy J_C |
|---|---|---|
| 1M | 27.0% | 48.4% |
| 2M | 27.3% | 16.5% |
| 3M | 13.7% | 16.7% |
| 4M | 14.0% | 15.2% |
| 5M | 10.6% | 13.3% |

## Learning curve: mean over the three scenarios (per seed first)

| Steps | DQN · lagrangian · greedy J_C | DDQN · lagrangian · greedy J_C |
|---|---|---|
| 1M | 29.3% | 38.7% |
| 2M | 27.9% | 24.0% |
| 3M | 18.3% | 20.5% |
| 4M | 19.3% | 20.7% |
| 5M | 16.1% | 18.4% |

## Learning curve: test failure rate (EXIT or any violation)

The gap is computed on the remaining instances (no EXIT, no violation); with a high failure rate it covers only the easier instances.

### LT

| Steps | DQN · lagrangian · greedy J_C | DDQN · lagrangian · greedy J_C |
|---|---|---|
| 1M | 44.8% | 8.8% |
| 2M | 44.2% | 30.0% |
| 3M | 17.2% | 15.5% |
| 4M | 29.0% | 17.0% |
| 5M | 12.7% | 9.5% |

### EQ

| Steps | DQN · lagrangian · greedy J_C | DDQN · lagrangian · greedy J_C |
|---|---|---|
| 1M | 0.2% | 0.2% |
| 2M | 0.7% | 2.0% |
| 3M | 5.5% | 0.5% |
| 4M | 0.7% | 0.2% |
| 5M | 0.7% | 7.8% |

### GT

| Steps | DQN · lagrangian · greedy J_C | DDQN · lagrangian · greedy J_C |
|---|---|---|
| 1M | 0.0% | 0.0% |
| 2M | 1.5% | 2.0% |
| 3M | 0.0% | 0.0% |
| 4M | 0.0% | 21.3% |
| 5M | 2.2% | 0.7% |


## 5M: test and training-set gap, violations, early decisions

| Scenario | Model | Test relative gap | Train relative gap | Test failure rate (EXIT / violation) | Train failure rate | Capacity violation rate | Privacy violation rate | Steps 1–5 regret | Steps 1–5 optimal-action rate |
|---|---|---|---|---|---|---|---|---|---|
| LT | DQN · lagrangian · greedy J_C | 25.7% | 26.2% | 12.7% | 14.2% | 3.0% | 12.5% | — | — |
| LT | DDQN · lagrangian · greedy J_C | 24.8% | 24.5% | 9.5% | 9.3% | 4.2% | 9.0% | — | — |
| EQ | DQN · lagrangian · greedy J_C | 11.8% | 12.0% | 0.7% | 0.4% | 0.0% | 0.8% | — | — |
| EQ | DDQN · lagrangian · greedy J_C | 16.9% | 17.0% | 7.8% | 4.6% | 0.2% | 7.5% | — | — |
| GT | DQN · lagrangian · greedy J_C | 10.6% | 10.4% | 2.2% | 2.4% | 0.0% | 2.2% | — | — |
| GT | DDQN · lagrangian · greedy J_C | 13.3% | 13.2% | 0.7% | 0.7% | 0.0% | 0.8% | — | — |

Regret is defined over the legal ECUs, i.e. for Maskable only; the other mechanisms may execute violating actions (—).


## Test gap per seed (5M)

| Scenario | Model | Seed 1 |
|---|---|---|
| LT | DQN · lagrangian · greedy J_C | 25.7% |
| LT | DDQN · lagrangian · greedy J_C | 24.8% |
| EQ | DQN · lagrangian · greedy J_C | 11.8% |
| EQ | DDQN · lagrangian · greedy J_C | 16.9% |
| GT | DQN · lagrangian · greedy J_C | 10.6% |
| GT | DDQN · lagrangian · greedy J_C | 13.3% |
