# Per-instance time: ILP vs the 12 variants

- Generated 2026-10-29 15:58:56; script `src/scripts/bench_time_v6.py`; took 1.4 min; CPU only, no training.
- Instances: the first 200 of the fixed v6.0.4 test set per scenario, the same for every method; ms per instance, one CPU thread.
- RL: the seed-1 model of every summary cell, deterministic full episode (reset + mask + forward + step), 5 warm-up episodes. ILP: `solve_ilp_max_ar` (Dinkelbach, CBC single thread), model building and every iteration included. Speed-up = ILP mean / variant mean.

| Method | LT | GT | EQ | Mean | Speed-up (mean) |
|---|---|---|---|---|---|
| ILP (Dinkelbach, 3.06 iterations) | 3192.5 | 807.7 | 543.3 | 1514.5 | 1× |
| Mask PPO | 74.6 | 49.9 | 46.8 | 57.1 | 24× |
| Mask DQN | 67.2 | 41.8 | 41.9 | 50.3 | 27× |
| Mask Double DQN | 64.9 | 43.5 | 41.9 | 50.1 | 27× |
| Repair PPO | 71.4 | 48.4 | 45.6 | 55.1 | 24× |
| Repair DQN | 64.7 | 43.7 | 50.6 | 53.0 | 26× |
| Repair Double DQN | 65.3 | 41.7 | 41.9 | 49.6 | 27× |
| Lagrangian PPO | 69.7 | 47.5 | 43.3 | 53.5 | 25× |
| Lagrangian DQN | 66.5 | 42.9 | 41.8 | 50.4 | 27× |
| Lagrangian Double DQN | 63.0 | 43.1 | 41.8 | 49.3 | 27× |
| Unconstraint PPO | 69.4 | 44.7 | 43.9 | 52.6 | 25× |
| Unconstraint DQN | 66.3 | 42.1 | 41.1 | 49.8 | 27× |
| Unconstraint Double DQN | 63.9 | 43.4 | 39.7 | 49.0 | 27× |

- Speed-up is the mean of the per-scenario speed-ups.
