# Per-Algorithm Pseudocode

This document gives **algorithm-level decision/constraint-handling** pseudocode for each of the 6 algorithms (as distinct from the "overall experiment pipeline" and "generic single-episode logic" pseudocode already in Section 13 of `论文概要3.md`, which covers the outer flow shared by all algorithms — this document covers what differs between algorithms in how they handle constraints).

The pseudocode maps directly to the real implementation in `scenarios/lt/{algo}/env.py` and `run_all.py` (using the lt scenario as the reference; whether the reward *formula* itself is unified across the three scenarios is covered in Section 12.6 of `论文概要3.md` — what's described here is the constraint-handling *mechanism*, which is consistent across the three scenarios; only the reward formula's numeric coefficients differ). Everything below is current as of 2026-09-10 (after the formal 5-seed campaign completed), **the experiments are no longer being modified** — this document is a read-only writeup.

---

## PPO (P3, unconstrained baseline)

Uses `stable_baselines3.PPO` (the non-maskable version); the action space is an unrestricted discrete choice over N ECUs. The environment does implement `action_masks()` (capacity dimension), but `run_all.py` never wraps the env with `ActionMasker` and never passes it to `MaskablePPO` — so the masking method is a leftover interface that has zero effect during training.

```
for t = 1..M:
    a_t ← policy.sample(all N ECUs)          # no constraint at all, including capacity
    cap_violated      ← remaining_vms[a_t] < svc[t].requirement
    conflict_violated ← has_conflict(a_t, t)
    # violations are only counted, never affect whether this step executes or its reward
    ru ← svc[t].requirement / initial_vms[a_t]
    total_ru += ru                            # ru is added to the AR numerator even on violation
    place(a_t, svc[t])                        # forced execution; remaining_vms may go negative / conflicts may coexist
if t == M:
    reward ← graded terminal AR reward (Eq. 12.3); degrades to a valid_placed-ratio penalty if violations > 0
return episode_success (= violations == 0), final AR
```

**Key point**: whether constraints are satisfied is only reflected in the info dict and the final success judgment — never in the per-step reward, never in the action space. This is a deliberately designed "does-nothing" floor baseline meant to set off how the other five algorithms each handle constraints; its low success_rate is by design (5-seed campaign result: only 35.0% success_rate on the lt scenario).

---

## PPO+Mask (P4, hard masking)

Uses `sb3_contrib.MaskablePPO` + an `ActionMasker`-wrapped env. The env exposes `action_masks()`, which zeroes out the probability of infeasible actions *before* sampling (rather than penalizing after the fact).

```
def action_masks(state):
    svc ← services[t]
    base_mask[j] ← (remaining_vms[j] >= svc.requirement) AND NOT has_conflict(j, t)   # hard constraints: capacity + conflict
    if base_mask is all False:
        return base_mask                       # no way out; let the env judge the violation post hoc
    # one-step lookahead: on top of base_mask, further exclude any ECU
    # that would provably strand some future service
    refined_mask[j] ← base_mask[j] AND NOT lethal_after(j)
    return refined_mask if any(refined_mask) else base_mask   # if lookahead kills every option, fall back to base_mask rather than an empty mask

for t = 1..M:
    mask ← action_masks()
    a_t ← policy.sample(only ECUs with mask=True)   # sampling space is hard-restricted; infeasible actions get probability 0
    place(a_t, svc[t])                              # by construction: as long as mask isn't all-False, this step cannot violate
if t == M:
    reward ← graded terminal reward (Eq. 12.3); lt scenario additionally applies the AR-weight curriculum w and potential-based shaping (Eq. 12.4)
```

**Key point**: constraint handling happens *before* action sampling — the only one of the four constraint mechanisms that achieves a theoretical zero-violation guarantee at both train and deployment time (campaign result: CapViol% = ConflictViol% = 0.00% across lt/eq/gt). The cost is needing to enumerate the feasible action set, and the one-step lookahead is itself a heuristic approximation (it does not guarantee a globally optimal feasible sequence).

---

## PPO+Lagrangian (P5, Lagrangian dual soft constraint)

The capacity dimension is still masked (hard constraint — only capacity-sufficient ECUs are allowed); the conflict dimension is instead handled with a **Lagrangian dual-ascent** soft penalty that adapts over training, rather than a hard prohibition — the `env.py` docstring explicitly states "Hard capacity (masked), Lagrangian conflict (adaptive penalty)".

```
init: λ ← LAMBDA_INIT, base_penalty ← 0.2, LAMBDA_LR, LAMBDA_TARGET, LAMBDA_MAX(=10.0)

# inside the environment (every step):
for t = 1..M:
    a_t ← policy.sample(ECU set restricted by the capacity mask)   # conflict is NOT masked; a conflicting ECU can still be picked
    c_t ← 1.0 if has_conflict(a_t, t) else 0.0    # conflict-violation indicator, not a continuous penalty
    match_gain ← svc[t].requirement / initial_vms[a_t]
    lagrange_penalty ← -(λ + base_penalty) * c_t  # soft penalty, grows with λ, never blocks the choice
    forced_overflow_penalty ← -2.0 if cap_violated else 0.0  # capacity is masked away in theory; only triggers in extreme fallback
    total_ru += match_gain                        # a conflict does NOT zero out ru; it's suppressed via λ, not by deducting AR
    step_reward ← match_gain/n_active + lagrange_penalty + forced_overflow_penalty
if t == M:
    reward ← graded terminal reward (Eq. 12.3); the violations == 0 check includes the conflict count

# training callback (LagrangeCallback, runs one dual-ascent update every LAMBDA_UPDATE_WINDOW episodes):
avg_viol ← moving average of the conflict-violation rate over that window of episodes
λ ← clip(λ + LAMBDA_LR * (avg_viol - LAMBDA_TARGET), 0, LAMBDA_MAX)   # dual ascent: more violations -> larger λ -> heavier penalty
env.set_lambda(λ)                                # broadcast the new λ to every parallel env instance
```

**Key point**: this is the only one of the four constraint mechanisms that maintains a genuine dual variable and updates it via a gradient-style rule keyed on the violation rate (not a fixed penalty coefficient); λ only acts on the conflict constraint — the capacity constraint is still a hard mask. Strictly speaking this is a "hard constraint (capacity) + soft constraint (conflict)" hybrid, not a pure Lagrangian method — the paper needs to state this precisely rather than describing it broadly as "Lagrangian handles all constraints."

---

## PPO+Repair (P6, ppo_opt, heuristic repair fallback)

The policy samples over the fully unconstrained action space; if the chosen action would violate a constraint, the environment immediately **reassigns** it to a feasible ECU using a best-fit heuristic and continues — the repair happens inside `env.step()`, and the policy itself is never aware a repair occurred (it only observes the post-repair outcome).

```
def best_fit_repair(svc_idx):
    svc ← services[svc_idx]
    valid ← {j : remaining_vms[j] >= svc.requirement AND svc_idx ∈ ecu_allowed[j]}   # both capacity and conflict satisfied
    if valid is empty: return None                 # cannot be repaired
    return argmax_{j∈valid} (svc.requirement / initial_vms[j])   # best-fit: pick the feasible ECU with the highest utilization

for t = 1..M:
    a_t ← policy.sample(all N ECUs)                # unmasked; the policy may pick any ECU
    cap_violated      ← remaining_vms[a_t] < svc[t].requirement
    conflict_violated ← has_conflict(a_t, t)
    if cap_violated OR conflict_violated:
        a_repaired ← best_fit_repair(t)
        if a_repaired is None:
            return reward = -M, episode terminates immediately   # unrepairable -> immediate failure
        a_t ← a_repaired                           # the policy's original action is silently overridden
        repairs += 1
    place(a_t, svc[t])                              # guarantee: whenever a repair is possible, this step will not actually violate
if t == M:
    reward ← graded terminal reward (Eq. 12.3), success = zero repairs triggered the whole episode   # identical formula to ppo_mask/ppo/dqn/ddqn, no extra term
```

⚠**Correction (2026-09-11)**: an earlier version of this section included `repair_penalty ← -0.1` (a per-step penalty when a repair fires) and `terminal_bonus ← AR*(1-repair_rate)` (an extra terminal bonus scaled by how rarely repairs fired). Those were transcribed from the **design docstring** at the top of `env.py`, but checking the actual `step()` code directly shows both `repair_penalty` and `terminal_bonus` (really `step_reward`) are computed as local variables and then **never added to the `reward` that gets returned** — written but never wired in, so the docstring's intended design and the reward formula that's actually in effect don't match. The reward `ppo_opt` actually uses is structurally identical to ppo_mask/ppo_lagrangian/ppo/dqn/ddqn: `M*(2·AR-1)` on success, `-M*(1-valid_placed/M)` on failure — no term tied to whether a repair fired this step. In other words, the policy gets zero step-level signal that a repair even happened; the only way a repair can influence anything is indirectly, through however it changes the AR that eventually gets delivered. This lines up with — and actually strengthens — the "Key point" behavioral finding below (the policy leans on repair without restraint on eq/gt): **the -0.1 penalty that was supposed to discourage this behavior was never actually active in the reward the policy was trained on**, which is an even more direct explanation for why the policy shows no hesitation about relying on repair once resources are abundant.

**Key point**: the actual "zero violation" guarantee comes from the environment's built-in repair fallback, not from the policy having learned to avoid violations — in the 5-seed campaign, CapViol%/ConflictViol% is 24.30%/54.25% on lt but close to 100% on eq/gt. This was verified by replaying the saved model weights episode-by-episode with the matching `TRAIN_SEED` (the replayed numbers match `summary.csv` exactly): on lt, 94.9% of *successful* episodes never trigger a repair at all, meaning resource scarcity forces the policy to genuinely learn constraint-avoiding placement; on eq/gt, resources are abundant enough that repair essentially never fails, success is nearly guaranteed regardless of the policy's choices, and the `-0.1` repair penalty is negligible next to the terminal reward's scale — so the policy has zero incentive to learn real constraint avoidance (on eq, a successful episode triggers a repair on roughly 7.7 of its 10 steps, on average). This is not a training failure or a statistics bug — it is the policy strategically exploiting the heuristic repair mechanism (a form of reward hacking): the more abundant the resources, the lazier the policy gets and the more it leans on repair. This is the code- and data-level basis for the paper's recurring point that "ppo_opt's high success rate is mainly credit to the environment's repair mechanism, not evidence the policy autonomously learned to satisfy constraints."

---

## DQN

Standard off-policy Q-learning. The action space matches the PPO family (a discrete choice among N ECUs); no masking, no repair, no dual penalty — constraint handling is structurally identical to plain PPO (P3): violations are only recorded, the action space is never restricted, and no extra penalty is applied. The reward formula is the same graded terminal reward as Eq. 12.3.

```
init: replay_buffer, q_net, q_net_target ← copy(q_net)
for env_step = 1..TOTAL_STEPS:
    a_t ← epsilon_greedy(q_net, obs_t)             # epsilon decays over DQN_EXPLORATION_FRACTION of training
    obs_{t+1}, r_t, done ← env.step(a_t)           # same env as P3: violations unpenalized, only recorded
    replay_buffer.add(obs_t, a_t, r_t, obs_{t+1}, done)
    if step % TRAIN_FREQ == 0:
        batch ← replay_buffer.sample(BATCH_SIZE)
        target_q ← r + (1-done) * gamma * max_a' q_net_target(obs_{t+1}, a')   # vanilla DQN: the target network both selects and evaluates
        loss ← smooth_l1(q_net(obs_t, a_t), target_q)
        gradient-descent update on q_net
    if step % TARGET_UPDATE == 0:
        q_net_target ← copy(q_net)
```

---

## DDQN

Subclasses `stable_baselines3.DQN`, overriding only the target-Q computation inside `train()` (Double DQN, van Hasselt et al. 2016). Everything else (exploration schedule, replay buffer, environment, reward) is identical to DQN — this is the sole controlled difference.

```
class DDQN(DQN):
    def train(gradient_steps, batch_size):
        for _ in range(gradient_steps):
            batch ← replay_buffer.sample(batch_size)
            # the only difference from DQN: action selection and value evaluation are decoupled onto two networks
            best_actions ← argmax_a' q_net(obs_{t+1}, a')          # the ONLINE network selects the action (not the target network)
            next_q       ← q_net_target(obs_{t+1})[best_actions]   # the TARGET network only evaluates that action's value
            target_q ← r + (1-done) * gamma * next_q
            loss ← smooth_l1(q_net(obs_t, a_t), target_q)
            gradient-descent update on q_net
        # target-network sync frequency, exploration schedule, and environment constraint handling are all identical to DQN
```

**Key point**: DQN uses `q_net_target` to both "pick the action" and "evaluate its value," which is prone to Q-value overestimation. DDQN hands "pick the action" to the online network `q_net` and "evaluate the value" to `q_net_target`, decoupling the two roles to suppress the overestimation bias — this is the only code-level difference between DQN and DDQN; the action space, state space, and constraint-handling behavior (no masking / no repair / violations unpenalized in both) and the environment itself are fully shared between them.

---

## Constraint-Handling Mechanism Overview (All 6 Algorithms)

| Algorithm | Action space restricted? | When constraints are handled | Can a violation occur? | Zero-violation guarantee (train/deploy) |
|---|---|---|---|---|
| PPO (P3) | No — free sampling over the full space | Never — recorded only | Yes, and never corrected | None |
| PPO+Mask (P4) | Yes — hard mask before sampling (capacity + conflict + one-step lookahead) | Before sampling | In theory, no (whenever the mask isn't all-False) | Yes (structural guarantee) |
| PPO+Lagrangian (P5) | Partial — capacity hard-masked, conflict unmasked | Capacity: before sampling; conflict: soft penalty after the fact + dual-ascent update of λ | Conflict violations can occur (soft constraint) | No strict guarantee — relies on λ suppressing the violation rate once training has converged |
| PPO+Repair (P6 / ppo_opt) | No — free sampling over the full space | After the fact: reassigned in-env via best-fit | The policy's raw action often violates, but is silently repaired | Conditional guarantee (fails when repair is impossible) |
| DQN | No — free sampling over the full space | Never — recorded only | Yes, and never corrected | None |
| DDQN | No — free sampling over the full space | Never — recorded only (shares the env with DQN) | Yes, and never corrected | None |

The mechanism descriptions above complement the formula-/pipeline-level pseudocode in Sections 12 and 13 of `论文概要3.md`: those sections describe "how the reward is computed and how the whole experiment pipeline runs"; this document describes "what each algorithm actually does at the action/constraint level."
