# v5.4.0 verication un/lag: unconstraint and Lagrangian (objective − λc_t, λ setting of v5.1.0) under the objective reward (terminal M·AR, violations ignored); DQN / DDQN ε 1.0→0 over a fixed 2M steps (seed 1, 20M, 50k instances)

- Generated 2026-10-16 19:32:16; script `src/scripts/run_v5.py --version 5.4.0`.
- Every variant: 1 seed(s) (1) × 3 scenario(s) (LT / EQ / GT), 20M steps each, checkpoints of the same run at 5M / 10M / 15M; ar_pen reward, γ = 1, full episode, EXIT for Maskable only, descending demand; PPO / DQN hyper-parameters from `paper_rl/config.py`; structure-aware network (Transformer) trained on GPU. 50k instances = the ILP-verified pool (`results/sup_diag/<scenario>.json`).
- Test gap: each seed's own 400 test instances; training-set gap: 1600 training instances (a fixed subset of the 50k pool, or the 1600-instance split itself); deterministic rollouts; mean ± sample std.
- Manifest: `logs/v5.4.0/manifest.json`.

## λ dynamics (during training, from training_curve.csv)

J_C = violations per episode (capacity + privacy); target d = 0. 'First at cap' = first episode with λ ≥ λ_max; 'fall' = from the peak to the end of training.

| Scenario | Model | λ_max | Final λ | Max λ | First at cap (episode) | Share at cap | Decreases | Fall | J_C first 1000 | J_C last 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · lagrangian · objective | none | 14.929 | 14.929 | — | 0.0% | 0 | 0.000 | 1.981 | 0.0490 |
| LT | DQN · lagrangian · objective | none | 9.591 | 9.591 | — | 0.0% | 0 | 0.000 | 1.984 | 0.0250 |
| LT | DDQN · lagrangian · objective | none | 3.734 | 3.734 | — | 0.0% | 0 | 0.000 | 1.985 | 0.0150 |
| EQ | PPO · lagrangian · objective | none | 7.525 | 7.525 | — | 0.0% | 0 | 0.000 | 1.820 | 0.0010 |
| EQ | DQN · lagrangian · objective | none | 4.747 | 4.747 | — | 0.0% | 0 | 0.000 | 1.810 | 0.0000 |
| EQ | DDQN · lagrangian · objective | none | 4.686 | 4.686 | — | 0.0% | 0 | 0.000 | 1.812 | 0.0000 |
| GT | PPO · lagrangian · objective | none | 17.387 | 17.387 | — | 0.0% | 0 | 0.000 | 1.645 | 0.0000 |
| GT | DQN · lagrangian · objective | none | 4.725 | 4.725 | — | 0.0% | 0 | 0.000 | 1.672 | 0.0000 |
| GT | DDQN · lagrangian · objective | none | 4.768 | 4.768 | — | 0.0% | 0 | 0.000 | 1.672 | 0.0000 |

### λ and J_C (from lambda_updates.csv, last update before each step)

Each cell: λ / behaviour-policy J_C / J_C used for the update (greedy: ε = 0 episodes).

| Scenario | Model | 5M | 10M | 15M | 20M |
|---|---|---|---|---|---|
| LT | PPO · lagrangian · objective | 8.97 / 27.707 / 27.707 | 14.85 / 0.090 / 0.090 | 14.89 / 0.059 / 0.059 | 14.93 / 0.093 / 0.093 |
| EQ | PPO · lagrangian · objective | 7.52 / 0.000 / 0.000 | 7.52 / 0.000 / 0.000 | 7.52 / 0.000 / 0.000 | 7.53 / 0.000 / 0.000 |
| GT | PPO · lagrangian · objective | 9.09 / 17.922 / 17.922 | 17.33 / 0.130 / 0.130 | 17.39 / 0.000 / 0.000 | 17.39 / 0.000 / 0.000 |
| LT | DQN · lagrangian · objective | 9.20 / 1.867 / 1.867 | 9.42 / 0.037 / 0.037 | 9.50 / 0.056 / 0.056 | 9.59 / 0.021 / 0.021 |
| EQ | DQN · lagrangian · objective | 4.65 / 0.000 / 0.000 | 4.67 / 0.001 / 0.001 | 4.72 / 0.000 / 0.000 | 4.75 / 0.000 / 0.000 |
| GT | DQN · lagrangian · objective | 4.71 / 0.000 / 0.000 | 4.71 / 0.000 / 0.000 | 4.72 / 0.000 / 0.000 | 4.72 / 0.000 / 0.000 |
| LT | DDQN · lagrangian · objective | 3.43 / 0.031 / 0.031 | 3.51 / 0.013 / 0.013 | 3.61 / 0.014 / 0.014 | 3.73 / 0.010 / 0.010 |
| EQ | DDQN · lagrangian · objective | 4.66 / 0.000 / 0.000 | 4.67 / 0.000 / 0.000 | 4.67 / 0.000 / 0.000 | 4.69 / 0.001 / 0.001 |
| GT | DDQN · lagrangian · objective | 4.71 / 0.006 / 0.006 | 4.73 / 0.000 / 0.000 | 4.74 / 0.000 / 0.000 | 4.77 / 0.000 / 0.000 |

## Learning curve: test relative gap

### LT

| Steps | PPO · unconstraint · objective | DQN · unconstraint · objective | DDQN · unconstraint · objective | PPO · lagrangian · objective | DQN · lagrangian · objective | DDQN · lagrangian · objective |
|---|---|---|---|---|---|---|
| 5M | — | — | — | — | 15.6% | 8.0% |
| 10M | — | — | — | 14.6% | 11.2% | 7.3% |
| 15M | — | — | — | 12.4% | 9.8% | 6.1% |
| 20M | — | — | — | 11.8% | 9.1% | 5.3% |

### EQ

| Steps | PPO · unconstraint · objective | DQN · unconstraint · objective | DDQN · unconstraint · objective | PPO · lagrangian · objective | DQN · lagrangian · objective | DDQN · lagrangian · objective |
|---|---|---|---|---|---|---|
| 5M | — | — | — | 11.2% | 5.5% | 5.3% |
| 10M | — | — | — | 5.1% | 5.4% | 4.4% |
| 15M | — | — | — | 4.3% | 4.2% | 3.3% |
| 20M | — | — | — | 3.9% | 3.5% | 4.0% |

### GT

| Steps | PPO · unconstraint · objective | DQN · unconstraint · objective | DDQN · unconstraint · objective | PPO · lagrangian · objective | DQN · lagrangian · objective | DDQN · lagrangian · objective |
|---|---|---|---|---|---|---|
| 5M | — | — | — | — | 12.4% | 7.7% |
| 10M | — | — | — | 17.6% | 4.9% | 4.1% |
| 15M | — | — | — | 4.3% | 3.6% | 4.7% |
| 20M | — | — | — | 3.5% | 2.5% | 2.9% |

## Learning curve: mean over the three scenarios (per seed first)

| Steps | PPO · unconstraint · objective | DQN · unconstraint · objective | DDQN · unconstraint · objective | PPO · lagrangian · objective | DQN · lagrangian · objective | DDQN · lagrangian · objective |
|---|---|---|---|---|---|---|
| 5M | — | — | — | — | 11.2% | 7.0% |
| 10M | — | — | — | 12.5% | 7.2% | 5.3% |
| 15M | — | — | — | 7.0% | 5.9% | 4.7% |
| 20M | — | — | — | 6.4% | 5.0% | 4.1% |

## Learning curve: test failure rate (EXIT or any violation)

The gap is computed on the remaining instances (no EXIT, no violation); with a high failure rate it covers only the easier instances.

### LT

| Steps | PPO · unconstraint · objective | DQN · unconstraint · objective | DDQN · unconstraint · objective | PPO · lagrangian · objective | DQN · lagrangian · objective | DDQN · lagrangian · objective |
|---|---|---|---|---|---|---|
| 5M | 100.0% | 100.0% | 100.0% | 100.0% | 9.3% | 2.5% |
| 10M | 100.0% | 100.0% | 100.0% | 5.5% | 2.5% | 1.7% |
| 15M | 100.0% | 100.0% | 100.0% | 4.5% | 3.7% | 1.7% |
| 20M | 100.0% | 100.0% | 100.0% | 5.2% | 2.7% | 1.5% |

### EQ

| Steps | PPO · unconstraint · objective | DQN · unconstraint · objective | DDQN · unconstraint · objective | PPO · lagrangian · objective | DQN · lagrangian · objective | DDQN · lagrangian · objective |
|---|---|---|---|---|---|---|
| 5M | 100.0% | 100.0% | 100.0% | 0.2% | 0.0% | 0.0% |
| 10M | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| 15M | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| 20M | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |

### GT

| Steps | PPO · unconstraint · objective | DQN · unconstraint · objective | DDQN · unconstraint · objective | PPO · lagrangian · objective | DQN · lagrangian · objective | DDQN · lagrangian · objective |
|---|---|---|---|---|---|---|
| 5M | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| 10M | 100.0% | 100.0% | 100.0% | 4.5% | 0.0% | 0.0% |
| 15M | 100.0% | 100.0% | 100.0% | 0.2% | 0.0% | 0.0% |
| 20M | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |


## 20M: test and training-set gap, violations, early decisions

| Scenario | Model | Test relative gap | Train relative gap | Test failure rate (EXIT / violation) | Train failure rate | Capacity violation rate | Privacy violation rate | Steps 1–5 regret | Steps 1–5 optimal-action rate |
|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| LT | DQN · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| LT | DDQN · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| LT | PPO · lagrangian · objective | 11.8% | 11.1% | 5.2% | 4.2% | 3.2% | 4.5% | — | — |
| LT | DQN · lagrangian · objective | 9.1% | 9.0% | 2.7% | 1.9% | 1.5% | 1.5% | — | — |
| LT | DDQN · lagrangian · objective | 5.3% | 5.2% | 1.5% | 0.7% | 1.2% | 0.5% | — | — |
| EQ | PPO · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| EQ | DQN · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| EQ | DDQN · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| EQ | PPO · lagrangian · objective | 3.9% | 3.5% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DQN · lagrangian · objective | 3.5% | 3.3% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DDQN · lagrangian · objective | 4.0% | 4.2% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| GT | PPO · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| GT | DQN · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| GT | DDQN · unconstraint · objective | — | — | 100.0% | 100.0% | 100.0% | 100.0% | — | — |
| GT | PPO · lagrangian · objective | 3.5% | 3.2% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DQN · lagrangian · objective | 2.5% | 2.6% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DDQN · lagrangian · objective | 2.9% | 3.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |

Regret is defined over the legal ECUs, i.e. for Maskable only; the other mechanisms may execute violating actions (—).


## Test gap per seed (20M)

| Scenario | Model | Seed 1 |
|---|---|---|
| LT | PPO · unconstraint · objective | — |
| LT | DQN · unconstraint · objective | — |
| LT | DDQN · unconstraint · objective | — |
| LT | PPO · lagrangian · objective | 11.8% |
| LT | DQN · lagrangian · objective | 9.1% |
| LT | DDQN · lagrangian · objective | 5.3% |
| EQ | PPO · unconstraint · objective | — |
| EQ | DQN · unconstraint · objective | — |
| EQ | DDQN · unconstraint · objective | — |
| EQ | PPO · lagrangian · objective | 3.9% |
| EQ | DQN · lagrangian · objective | 3.5% |
| EQ | DDQN · lagrangian · objective | 4.0% |
| GT | PPO · unconstraint · objective | — |
| GT | DQN · unconstraint · objective | — |
| GT | DDQN · unconstraint · objective | — |
| GT | PPO · lagrangian · objective | 3.5% |
| GT | DQN · lagrangian · objective | 2.5% |
| GT | DDQN · lagrangian · objective | 2.9% |
