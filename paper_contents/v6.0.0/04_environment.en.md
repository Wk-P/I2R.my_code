# 4. Environment: variables, observation (state), action space

Code: `PlacementEnv` (gymnasium.Env) in `src/paper_rl/env.py`, shared by all 12 variants.

## 4.1 Instance (one drawn at every reset)

| variable | meaning | distribution (data generation) |
|---|---|---|
| c_n | capacity of ECU n (containers), N of them | uniform{50, 55, …, 195} |
| e_m | demand of service m (containers), M of them | uniform{10, 15, …, 95} |
| conflict sets | K = 10; two services of the same set may not share an ECU | each service joins each set independently with probability q, sets with < 2 members dropped; two services conflict with probability p = 1 − (1 − q²)^K = 0.6 |
| ar_star | the instance's ILP AR (Dinkelbach), for logging and evaluation only, not in the observation |  |

Every instance is kept only if the ILP solves it to optimality.

## 4.2 Internal variables (per episode)

| variable | meaning |
|---|---|
| t | current step = the service to place (position in descending-demand order), 0 … M |
| remaining_n | free capacity c_n − load of ECU n (negative after a capacity violation) |
| hosted_n | services executed on ECU n (violating ones included) |
| partners_m | services sharing at least one conflict set with m |
| legal_ru / legal_ecus | sum of e_m / c_n over legal placements, ECUs hosting a legal placement → AR |
| exec_ru / exec_ecus | every executed placement (violations included) → AR_obj (lagrangian reward) |
| valid_placed | number of legally placed services |
| cap_violations / conflict_violations | capacity / privacy violation counts |
| repairs | number of repair replacements |
| dead_end | whether a step with no feasible ECU occurred |
| exited | whether the episode ended by EXIT (maskable only) |
| λ | dual variable (lagrangian only), fixed within an episode |

## 4.3 Observation and state

- The observation the environment emits is the full state (fully observable, Markov): `obs_mode = raw` (the command line says `--obs base`, but `--net graph` forces raw; results.json records raw).
- Observation vector = [c (N) | e in placement order (M) | conflict-graph adjacency (M × M) | ECU index + 1 of every service, 0 = unplaced (M) | t (1)], lagrangian appends λ / (1 + λ) (1).
- Observation space: `gymnasium.spaces.Box(low=-1, high=1e4, float32)`, not normalised (capacities and demands are raw container counts); normalisation happens inside the network (02_network §2.2).
- About an "observer": there is no separate observer class. The observation is produced by `PlacementEnv._raw_obs()`; feature extraction (feasibility, loads, relation matrices, ...) is done by the network's `build()`. PPO and DQN / DDQN receive the same observation and run the same feature extraction.

## 4.4 Action space

- Discrete action space `gymnasium.spaces.Discrete` (not continuous): action a ∈ {0, …, N − 1} = place the current service on ECU a; maskable has one extra EXIT action a = N.
- Each step decides one service's ECU (`action_mode = ecu`); the service order is fixed (descending demand).

## 4.5 Dimensions per scenario and strategy (from instantiating the environment)

| scenario | strategy | N | M | observation size | action space |
|---|---|---|---|---|---|
| LT | maskable | 10 | 15 | 266 | Discrete(11) |
| LT | repair | 10 | 15 | 266 | Discrete(10) |
| LT | lagrangian | 10 | 15 | 267 | Discrete(10) |
| LT | unconstraint | 10 | 15 | 266 | Discrete(10) |
| GT | maskable | 15 | 10 | 136 | Discrete(16) |
| GT | repair | 15 | 10 | 136 | Discrete(15) |
| GT | lagrangian | 15 | 10 | 137 | Discrete(15) |
| GT | unconstraint | 15 | 10 | 136 | Discrete(15) |
| EQ | maskable | 10 | 10 | 131 | Discrete(11) |
| EQ | repair | 10 | 10 | 131 | Discrete(10) |
| EQ | lagrangian | 10 | 10 | 132 | Discrete(10) |
| EQ | unconstraint | 10 | 10 | 131 | Discrete(10) |

## 4.6 One step (`step`)

1. maskable chooses EXIT: mark exited, failure reward, the episode ends.
2. repair: an infeasible action is replaced by the feasible ECU with the largest e_m / c_n when one exists; otherwise the action is kept (dead_end).
3. Violations: capacity (remaining_n < e_m), privacy (a conflict partner on the ECU); the placement is executed and remaining, hosted and the AR terms are updated.
4. t ← t + 1; at t = M the episode ends with the terminal reward (01_algorithms §1.3). lagrangian subtracts λ·c_t at every step.

## 4.7 Differences between the 12 variants

| aspect | maskable | repair | lagrangian | unconstraint |
|---|---|---|---|---|
| observation | raw state | raw state | raw state + λ / (1 + λ) | raw state |
| action space | Discrete(N + 1) | Discrete(N) | Discrete(N) | Discrete(N) |
| infeasible action | masked; EXIT only when none is feasible | replaced by the env; executed (violation) when none is feasible | executed, violation charged −λ·c_t | executed, violation only recorded |
| reward | ar_pen | ar_pen | ar_obj − λ·c_t | ar_pen |
| can violate | no | dead ends only | yes | yes |

| aspect | PPO | DQN | DDQN (Double DQN) |
|---|---|---|---|
| type | on-policy actor-critic | off-policy, replay + target network | off-policy, replay + target network |
| output | action logits (categorical) + V(s) | Q(s, a) | Q(s, a) |
| exploration in training | sampling from the categorical distribution (entropy coefficient 0.005) | ε-greedy, ε linear from 1 to 0 over the first 2M steps | as DQN |
| evaluation | deterministic: argmax logit | argmax Q | argmax Q |
| mask source for maskable | environment `action_masks()` (+ the same mask inside the network) | computed by the network from the observation | computed by the network from the observation |
| parallel envs | 40 | 12 | 12 |

## 4.8 info fields (returned every step, for logging and evaluation)

`ar`, `ar_obj`, `ar_star`, `valid_placed`, `services_placed`, `capacity_violations`, `conflict_violations`, `repairs`, `cap_violated`, `conflict_violated`, `dead_end`, `exited`, `ecus_used`, `inst_idx`, `viol_rate_ep`, `cost_ep`
