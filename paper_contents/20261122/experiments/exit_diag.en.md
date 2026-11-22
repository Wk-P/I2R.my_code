# v6.0.2 Mask EXIT attribution diagnostic

- Generated 2026-10-22 14:36:07; script `src/scripts/diag_exit_v6.py`; took 0.6 min; no training.
- Models: the v6.0.0 Maskable final models (PPO / DQN / DDQN, seeds 1–3), replayed deterministically on each seed's 400 test instances; the replayed EXIT counts match results.json run by run.
- For every EXIT instance, with the first t placements of the trajectory fixed:
  - **t_inf**: the smallest t for which the ILP finds no feasible completion, i.e. the step that made the instance unsolvable;
  - **t_cheap**: the smallest t at which some unplaced service has no legal ECU left (a one-step necessary condition on the current state);
  - **t_exit**: services placed before the EXIT. Always t_inf ≤ t_cheap ≤ t_exit.
- **Latent steps** = t_exit − t_inf: steps placed after the instance was already unsolvable. **Visible steps** = t_exit − t_cheap.

## 1. EXIT instances

| Scen | Algo | EXIT instances / total | Rate |
|---|---|---|---|
| EQ | PPO | 1 / 1200 | 0.083% |
| EQ | DQN | 1 / 1200 | 0.083% |
| LT | PPO | 22 / 1200 | 1.833% |
| LT | DQN | 45 / 1200 | 3.750% |
| LT | DDQN | 43 / 1200 | 3.583% |

## 2. When does the instance become unsolvable

| Group | Instances | mean t_inf | mean t_cheap | mean t_exit | mean latent | latent = 0 | latent ≥ 3 | t_cheap = t_inf |
|---|---|---|---|---|---|---|---|---|
| EQ PPO | 1 | 3.000 | 5.000 | 5.000 | 2.000 | 0 / 1 | 0 / 1 | 0 / 1 |
| EQ DQN | 1 | 2.000 | 6.000 | 6.000 | 4.000 | 0 / 1 | 1 / 1 | 0 / 1 |
| EQ all | 2 | 2.500 | 5.500 | 5.500 | 3.000 | 0 / 2 | 1 / 2 | 0 / 2 |
| LT PPO | 22 | 4.455 | 11.773 | 12.091 | 7.636 | 1 / 22 | 21 / 22 | 1 / 22 |
| LT DQN | 45 | 4.267 | 11.467 | 11.978 | 7.711 | 0 / 45 | 44 / 45 | 0 / 45 |
| LT DDQN | 43 | 4.535 | 11.535 | 11.860 | 7.326 | 0 / 43 | 38 / 43 | 0 / 43 |
| LT all | 110 | 4.409 | 11.555 | 11.955 | 7.545 | 1 / 110 | 103 / 110 | 1 / 110 |

### Latent-step distribution (t_exit − t_inf)

| Group | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EQ PPO | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| EQ DQN | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| EQ all | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| LT PPO | 1 | 0 | 0 | 1 | 3 | 2 | 1 | 1 | 2 | 4 | 3 | 1 | 2 | 1 |
| LT DQN | 0 | 0 | 1 | 2 | 3 | 6 | 4 | 5 | 4 | 7 | 6 | 3 | 2 | 2 |
| LT DDQN | 0 | 2 | 3 | 1 | 4 | 2 | 3 | 7 | 6 | 3 | 4 | 3 | 2 | 3 |
| LT all | 1 | 2 | 4 | 4 | 10 | 10 | 8 | 13 | 12 | 14 | 13 | 7 | 6 | 6 |

## 3. Why every ECU is masked at the EXIT

For the service that cannot be placed at the EXIT, each ECU's blocking reason; instances are classified over all ECUs.

| Group | Instances | ECUs capacity only | ECUs privacy only | ECUs both | Instances: all capacity | Instances: all privacy | Instances: mixed |
|---|---|---|---|---|---|---|---|
| EQ PPO | 1 | 80.000% | 20.000% | 0.000% | 0 / 1 | 0 / 1 | 1 / 1 |
| EQ DQN | 1 | 60.000% | 10.000% | 30.000% | 0 / 1 | 0 / 1 | 1 / 1 |
| EQ all | 2 | 70.000% | 15.000% | 15.000% | 0 / 2 | 0 / 2 | 2 / 2 |
| LT PPO | 22 | 21.364% | 24.545% | 54.091% | 0 / 22 | 0 / 22 | 22 / 22 |
| LT DQN | 45 | 18.889% | 39.111% | 42.000% | 1 / 45 | 0 / 45 | 44 / 45 |
| LT DDQN | 43 | 20.000% | 36.744% | 43.256% | 1 / 43 | 0 / 43 | 42 / 43 |
| LT all | 110 | 19.818% | 35.273% | 44.909% | 2 / 110 | 0 / 110 | 108 / 110 |
