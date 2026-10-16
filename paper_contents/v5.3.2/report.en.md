# v5.3.2 unconstraint: PPO / DQN / DDQN, DQN / DDQN ε 1.0→0 over a fixed 2M steps (seed 1, 20M, 50k instances)

- Generated 2026-10-16 01:32:52; script `src/scripts/run_v5.py --version 5.3.2`.
- Every variant: 1 seed(s) (1) × 3 scenario(s) (LT / EQ / GT), 20M steps each, checkpoints of the same run at 5M / 10M / 15M; ar_pen reward, γ = 1, full episode, EXIT for Maskable only, descending demand; PPO / DQN hyper-parameters from `paper_rl/config.py`; structure-aware network (Transformer) trained on GPU. 50k instances = the ILP-verified pool (`results/sup_diag/<scenario>.json`).
- Test gap: each seed's own 400 test instances; training-set gap: 1600 training instances (a fixed subset of the 50k pool, or the 1600-instance split itself); deterministic rollouts; mean ± sample std.
- Manifest: `logs/v5.3.2/manifest.json`.

## Learning curve: test relative gap

### LT

| Steps | PPO · unconstraint | DQN · unconstraint | DDQN · unconstraint |
|---|---|---|---|
| 5M | 9.5% | 30.4% | 33.7% |
| 10M | 8.8% | 28.1% | 30.1% |
| 15M | 7.7% | 25.4% | 23.0% |
| 20M | 7.4% | 25.2% | 28.0% |

### EQ

| Steps | PPO · unconstraint | DQN · unconstraint | DDQN · unconstraint |
|---|---|---|---|
| 5M | 10.3% | 24.0% | 25.1% |
| 10M | 7.7% | 27.7% | 19.4% |
| 15M | 6.8% | 36.8% | 23.3% |
| 20M | 5.6% | 31.9% | 29.3% |

### GT

| Steps | PPO · unconstraint | DQN · unconstraint | DDQN · unconstraint |
|---|---|---|---|
| 5M | 9.1% | 16.3% | 24.0% |
| 10M | 7.8% | 16.0% | 22.0% |
| 15M | 6.7% | 16.9% | 15.6% |
| 20M | 6.3% | 18.7% | 15.8% |

## Learning curve: mean over the three scenarios (per seed first)

| Steps | PPO · unconstraint | DQN · unconstraint | DDQN · unconstraint |
|---|---|---|---|
| 5M | 9.6% | 23.6% | 27.6% |
| 10M | 8.1% | 23.9% | 23.8% |
| 15M | 7.1% | 26.4% | 20.6% |
| 20M | 6.4% | 25.3% | 24.4% |

## Learning curve: test failure rate (EXIT or any violation)

The gap is computed on the remaining instances (no EXIT, no violation); with a high failure rate it covers only the easier instances.

### LT

| Steps | PPO · unconstraint | DQN · unconstraint | DDQN · unconstraint |
|---|---|---|---|
| 5M | 3.5% | 93.8% | 72.8% |
| 10M | 2.7% | 96.5% | 93.8% |
| 15M | 2.5% | 91.5% | 78.8% |
| 20M | 2.5% | 93.0% | 86.2% |

### EQ

| Steps | PPO · unconstraint | DQN · unconstraint | DDQN · unconstraint |
|---|---|---|---|
| 5M | 0.5% | 42.5% | 12.5% |
| 10M | 0.5% | 19.0% | 49.8% |
| 15M | 0.0% | 0.5% | 7.3% |
| 20M | 0.0% | 2.0% | 1.2% |

### GT

| Steps | PPO · unconstraint | DQN · unconstraint | DDQN · unconstraint |
|---|---|---|---|
| 5M | 0.0% | 0.0% | 0.0% |
| 10M | 0.2% | 0.0% | 0.2% |
| 15M | 0.2% | 17.5% | 0.0% |
| 20M | 0.0% | 13.5% | 35.2% |


## 20M: test and training-set gap, violations, early decisions

| Scenario | Model | Test relative gap | Train relative gap | Test failure rate (EXIT / violation) | Train failure rate | Capacity violation rate | Privacy violation rate | Steps 1–5 regret | Steps 1–5 optimal-action rate |
|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · unconstraint | 7.4% | 6.9% | 2.5% | 1.4% | 1.5% | 2.5% | — | — |
| LT | DQN · unconstraint | 25.2% | 24.9% | 93.0% | 91.7% | 3.0% | 92.8% | — | — |
| LT | DDQN · unconstraint | 28.0% | 27.2% | 86.2% | 84.4% | 7.8% | 85.8% | — | — |
| EQ | PPO · unconstraint | 5.6% | 5.8% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DQN · unconstraint | 31.9% | 32.0% | 2.0% | 1.1% | 1.8% | 1.0% | — | — |
| EQ | DDQN · unconstraint | 29.3% | 28.7% | 1.2% | 0.5% | 0.0% | 1.2% | — | — |
| GT | PPO · unconstraint | 6.3% | 5.6% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DQN · unconstraint | 18.7% | 18.6% | 13.5% | 10.5% | 0.0% | 13.5% | — | — |
| GT | DDQN · unconstraint | 15.8% | 15.3% | 35.2% | 35.5% | 0.5% | 35.2% | — | — |

Regret is defined over the legal ECUs, i.e. for Maskable only; the other mechanisms may execute violating actions (—).


## Test gap per seed (20M)

| Scenario | Model | Seed 1 |
|---|---|---|
| LT | PPO · unconstraint | 7.4% |
| LT | DQN · unconstraint | 25.2% |
| LT | DDQN · unconstraint | 28.0% |
| EQ | PPO · unconstraint | 5.6% |
| EQ | DQN · unconstraint | 31.9% |
| EQ | DDQN · unconstraint | 29.3% |
| GT | PPO · unconstraint | 6.3% |
| GT | DQN · unconstraint | 18.7% |
| GT | DDQN · unconstraint | 15.8% |
