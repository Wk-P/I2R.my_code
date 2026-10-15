# v5.3.1 BFT Repair: PPO / DQN / DDQN, DQN / DDQN ε 1.0→0 over a fixed 2M steps (seed 1, 20M, 50k instances)

- Generated 2026-10-15 20:22:59; script `src/scripts/run_v5.py --version 5.3.1`.
- Every variant: 1 seed(s) (1) × 3 scenario(s) (LT / EQ / GT), 20M steps each, checkpoints of the same run at 5M / 10M / 15M; ar_pen reward, γ = 1, full episode, EXIT for Maskable only, descending demand; PPO / DQN hyper-parameters from `paper_rl/config.py`; structure-aware network (Transformer) trained on GPU. 50k instances = the ILP-verified pool (`results/sup_diag/<scenario>.json`).
- Test gap: each seed's own 400 test instances; training-set gap: 1600 training instances (a fixed subset of the 50k pool, or the 1600-instance split itself); deterministic rollouts; mean ± sample std.
- Manifest: `logs/v5.3.1/manifest.json`.

## Learning curve: test relative gap

### LT

| Steps | PPO · BFT Repair | DQN · BFT Repair | DDQN · BFT Repair |
|---|---|---|---|
| 5M | 10.0% | 16.7% | 16.9% |
| 10M | 9.1% | 14.8% | 13.7% |
| 15M | 8.2% | 14.2% | 14.7% |
| 20M | 7.8% | 11.6% | 13.7% |

### EQ

| Steps | PPO · BFT Repair | DQN · BFT Repair | DDQN · BFT Repair |
|---|---|---|---|
| 5M | 8.2% | 5.0% | 5.1% |
| 10M | 7.7% | 4.2% | 4.8% |
| 15M | 7.5% | 5.3% | 9.3% |
| 20M | 7.2% | 3.7% | 3.8% |

### GT

| Steps | PPO · BFT Repair | DQN · BFT Repair | DDQN · BFT Repair |
|---|---|---|---|
| 5M | 8.8% | 8.7% | 8.5% |
| 10M | 7.6% | 4.0% | 3.6% |
| 15M | 6.8% | 5.6% | 10.7% |
| 20M | 6.6% | 3.9% | 2.7% |

## Learning curve: mean over the three scenarios (per seed first)

| Steps | PPO · BFT Repair | DQN · BFT Repair | DDQN · BFT Repair |
|---|---|---|---|
| 5M | 9.0% | 10.1% | 10.2% |
| 10M | 8.1% | 7.6% | 7.4% |
| 15M | 7.5% | 8.4% | 11.6% |
| 20M | 7.2% | 6.4% | 6.7% |

## Learning curve: test failure rate (EXIT or any violation)

The gap is computed on the remaining instances (no EXIT, no violation); with a high failure rate it covers only the easier instances.

### LT

| Steps | PPO · BFT Repair | DQN · BFT Repair | DDQN · BFT Repair |
|---|---|---|---|
| 5M | 2.7% | 9.8% | 8.5% |
| 10M | 3.0% | 5.5% | 3.2% |
| 15M | 1.7% | 9.5% | 3.2% |
| 20M | 2.5% | 3.7% | 5.5% |

### EQ

| Steps | PPO · BFT Repair | DQN · BFT Repair | DDQN · BFT Repair |
|---|---|---|---|
| 5M | 0.0% | 0.0% | 0.0% |
| 10M | 0.0% | 0.0% | 0.0% |
| 15M | 0.0% | 0.0% | 0.0% |
| 20M | 0.0% | 0.0% | 0.0% |

### GT

| Steps | PPO · BFT Repair | DQN · BFT Repair | DDQN · BFT Repair |
|---|---|---|---|
| 5M | 0.0% | 0.0% | 0.0% |
| 10M | 0.0% | 0.0% | 0.0% |
| 15M | 0.0% | 0.0% | 0.0% |
| 20M | 0.0% | 0.0% | 0.0% |


## 20M: test and training-set gap, violations, early decisions

| Scenario | Model | Test relative gap | Train relative gap | Test failure rate (EXIT / violation) | Train failure rate | Capacity violation rate | Privacy violation rate | Steps 1–5 regret | Steps 1–5 optimal-action rate |
|---|---|---|---|---|---|---|---|---|---|
| LT | PPO · BFT Repair | 7.8% | 7.9% | 2.5% | 1.6% | 1.8% | 2.0% | — | — |
| LT | DQN · BFT Repair | 11.6% | 11.3% | 3.7% | 2.8% | 1.2% | 3.5% | — | — |
| LT | DDQN · BFT Repair | 13.7% | 13.6% | 5.5% | 4.3% | 1.2% | 5.2% | — | — |
| EQ | PPO · BFT Repair | 7.2% | 6.5% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DQN · BFT Repair | 3.7% | 3.4% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| EQ | DDQN · BFT Repair | 3.8% | 3.9% | 0.0% | 0.1% | 0.0% | 0.0% | — | — |
| GT | PPO · BFT Repair | 6.6% | 6.1% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DQN · BFT Repair | 3.9% | 3.5% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| GT | DDQN · BFT Repair | 2.7% | 2.8% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |

Regret is defined over the legal ECUs, i.e. for Maskable only; the other mechanisms may execute violating actions (—).


## Test gap per seed (20M)

| Scenario | Model | Seed 1 |
|---|---|---|
| LT | PPO · BFT Repair | 7.8% |
| LT | DQN · BFT Repair | 11.6% |
| LT | DDQN · BFT Repair | 13.7% |
| EQ | PPO · BFT Repair | 7.2% |
| EQ | DQN · BFT Repair | 3.7% |
| EQ | DDQN · BFT Repair | 3.8% |
| GT | PPO · BFT Repair | 6.6% |
| GT | DQN · BFT Repair | 3.9% |
| GT | DDQN · BFT Repair | 2.7% |
