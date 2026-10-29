# 1. The 12 variants: strategy × RL_algorithm

## 1.1 Common setting

- Task: place M services on N ECUs. An instance gives the capacities c_n, the demands e_m and the privacy conflict sets (K = 10).
- Constraints: capacity -- the demands placed on ECU n sum to at most c_n; privacy -- two services of the same conflict set may not share an ECU.
- Objective AR (average resource utilisation) = Σ_{legal placements (m,n)} e_m / c_n ÷ the number of ECUs hosting at least one legal placement. A violating placement adds nothing to AR.
- Sequential decision: services are sorted by descending demand (`order = desc`); step t chooses an ECU for the t-th service; an episode has at most M steps.
- `full_episode = True` (every run): no early stop; at a dead end (no feasible ECU for the current service) maskable can only choose EXIT, the other strategies execute the placement and record the violation. The only early end is maskable's EXIT action.
- Scenarios: LT (N=10, M=15, scarce), EQ (N=10, M=10), GT (N=15, M=10, ample).
- Discount γ = 1.0 (every run), so the return is the sum of an episode's rewards.

## 1.2 The four strategies (constraint handling)

**maskable (paper name Mask)**

- Action mask = the ECUs that satisfy both capacity (remaining capacity ≥ e_m) and privacy (no service conflicting with m on the ECU) for the current service m.
- An extra EXIT action (index N): valid only when no ECU is feasible, and then the only valid action. Choosing EXIT ends the episode at once with the failure reward. Mask therefore never violates by construction.
- Reward: `ar_pen` (see 1.3).

**repair**

- If the chosen action is infeasible, the environment replaces it by the feasible ECU with the largest e_m / c_n (i.e. the feasible ECU of smallest capacity, lowest index on ties) and counts it in `repairs`.
- If no ECU is feasible, the agent's own action is executed and the violation recorded (full_episode). Repair therefore violates only at a dead end.
- The replacement happens inside the environment; the replay buffer / rollout stores the action the agent chose. Reward: `ar_pen`.

**lagrangian (standard Lagrangian)**

- Actions are executed as chosen and violations recorded. Per-step cost c_t = 1[capacity violated] + 1[privacy violated] ∈ {0, 1, 2}.
- Per-step reward r_t = r_t^obj − λ·c_t with r_t^obj the `ar_obj` reward (see 1.3); λ is the dual variable, updated by Adam (settings in 03_training), fixed within an episode and fed to the network as the last observation entry (λ / (1 + λ)).

**unconstraint (paper name Unconstraint)**

- No constraint handling: actions are executed as chosen and violations only recorded. Reward: the full `ar_pen` (v6.0.0 definition); a violation makes the episode a failure with a negative reward.
- The state contains the feasibility features (as for the other strategies, feas_in = True), but nothing enforces feasibility.

## 1.3 Rewards

Let valid = the number of legally placed services. Every non-terminal step pays 0; the terminal step pays:

- `ar_pen` (maskable / repair / unconstraint): AR ∈ (0, 1] if all M services are placed legally; otherwise (EXIT included) −(1 − valid / M) ∈ [−1, 0).
- `ar_obj` (lagrangian): AR_obj = the mean of min(load_n / c_n, 1) over the ECUs with any executed placement (violating ones included); no failure penalty -- violations enter only through −λ·c_t every step.

## 1.4 The three RL_algorithms

- **PPO**: on-policy actor-critic (clipped probability-ratio objective + GAE). maskable uses sb3-contrib's `MaskablePPO`, the others stable-baselines3's `PPO`; the policy is always our `GraphMaskablePolicy` (02_network), actor and critic sharing one encoder.
- **DQN**: off-policy value learning with experience replay and a target network, TD target r + γ·max_a' Q_target(s', a'). maskable uses our `MaskableDQN`, the others stable-baselines3's `DQN`.
- **DDQN = Double DQN** (van Hasselt et al., 2016): the online network picks the next-state action a* = argmax_a' Q_online(s', a'), the target network evaluates it, TD target r + γ·Q_target(s', a*), reducing DQN's Q-value overestimation. maskable uses our `MaskableDDQN`, the others our `DoubleDQN` (`src/shared/dqn_variants.py`, subclasses of SB3 `DQN` that only override `train()`).
- The Q-network has the same structure as the PPO policy (02_network): every ECU token gives Q(s, ECU n); for maskable the global token also gives Q(s, EXIT).

## 1.5 The 12 variants

| strategy | RL_algorithm | class | reward | actions | constraint handling |
|---|---|---|---|---|---|
| maskable | PPO | `sb3_contrib.MaskablePPO` | `ar_pen` | N + 1 (EXIT) | from the environment's `action_masks()` every step, stored in the rollout and used in the update; the network also sets infeasible logits to −1e9 by the same feasibility |
| maskable | DQN | `MaskableDQN` | `ar_pen` | N + 1 (EXIT) | computed by the network from the raw observation (`graph_net.build`, the same feasibility as the environment); ε-greedy samples among valid actions only, the greedy action and the TD-target max range over valid actions only |
| maskable | DDQN (Double DQN) | `MaskableDDQN` | `ar_pen` | N + 1 (EXIT) | as DQN; the online network's argmax in the TD target ranges over the next state's valid actions only |
| repair | PPO | `stable_baselines3.PPO` | `ar_pen` | N | no mask; the environment replaces infeasible actions |
| repair | DQN | `stable_baselines3.DQN` | `ar_pen` | N | no mask; the environment replaces infeasible actions |
| repair | DDQN (Double DQN) | `DoubleDQN` | `ar_pen` | N | no mask; the environment replaces infeasible actions |
| lagrangian | PPO | `stable_baselines3.PPO` | `ar_obj` − λ·c_t | N | no mask; −λ·c_t penalty |
| lagrangian | DQN | `stable_baselines3.DQN` | `ar_obj` − λ·c_t | N | no mask; −λ·c_t penalty |
| lagrangian | DDQN (Double DQN) | `DoubleDQN` | `ar_obj` − λ·c_t | N | no mask; −λ·c_t penalty |
| unconstraint | PPO | `stable_baselines3.PPO` | `ar_pen` | N | none |
| unconstraint | DQN | `stable_baselines3.DQN` | `ar_pen` | N | none |
| unconstraint | DDQN (Double DQN) | `DoubleDQN` | `ar_pen` | N | none |

## 1.6 Training and evaluation

- Every run trains for 20,000,000 environment steps on the 50,000 ILP-verified instances of `results/sup_diag/<scenario>.json` (p = 0.6, K = 10), one drawn at random at every reset.
- Summary evaluation: the final model, deterministic, on the fixed v6.0.4 test set (`data/testset_v6.0.4/<scenario>.json`, 5000 ILP-optimal instances per scenario, disjoint from the training pool; `src/scripts/eval_testset_v6_0_4.py`), with the run's own settings (Lagrangian: the λ at the end of training).
- gap = 1 − mean(AR) / mean(ILP AR); ILP AR from Dinkelbach's method (`shared/ilp_utils.solve_ilp_max_ar`).

### Lagrangian: λ at the end of training

| RL_algorithm | scenario | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| PPO | LT | 2.0258 | 2.3696 | 2.2095 |
| PPO | GT | 1.6404 | 0.9867 | 1.0642 |
| PPO | EQ | 1.4388 | 1.6365 | 1.7882 |
| DQN | LT | 4.6051 | 4.5378 | 4.4091 |
| DQN | GT | 4.8633 | 4.8822 | 4.9638 |
| DQN | EQ | 5.0588 | 5.1759 | 4.8931 |
| DDQN (Double DQN) | LT | 4.5790 | 5.4718 | 5.0258 |
| DDQN (Double DQN) | GT | 5.1293 | 4.9786 | 5.0154 |
| DDQN (Double DQN) | EQ | 5.0747 | 5.4400 | 5.1392 |
