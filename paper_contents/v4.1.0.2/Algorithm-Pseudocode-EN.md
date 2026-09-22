# Per-Algorithm Pseudocode (English)

This document mirrors `paper_contents/各算法伪代码.md` in English. It gives the **core decision/constraint-handling mechanism** pseudocode for each of the 6 algorithms — as distinct from the "overall experiment pipeline" and "shared per-episode logic" pseudocode already in `论文概要3.md` §13, which covers what's common across all algorithms; this document covers what differs between them.

The pseudocode maps directly to `scenarios/lt/{algo}/env.py` and `run_all.py` (lt scenario as the reference; see each section for whether the mechanism is identical across lt/eq/gt). **This version reflects the final state on the `final_paper_experiments` branch, tag `v4.1.0.2`** — i.e. after the "dead-code penalty" audit, fix, ablation, and partial revert are all complete. The full investigation trail is in `paper_contents/v4.1.0_changelog.md`; this document only states the final conclusions.

---

## PPO (P3, unconstrained baseline) — never touched

Uses plain `stable_baselines3.PPO` (not the Maskable variant). The action space is an unrestricted discrete choice over N ECUs. The environment does implement `action_masks()` (capacity dimension), but `run_all.py` never wraps the environment with `ActionMasker` or passes it to `MaskablePPO` — so the masking method is a leftover interface that is never actually active during training.

```
for t = 1..M:
    a_t <- policy.sample(all N ECUs)          # no restriction at all, including capacity
    cap_violated      <- remaining_vms[a_t] < svc[t].requirement
    conflict_violated <- has_conflict(a_t, t)
    # violations are only counted, never block execution or change this step's reward
    ru <- svc[t].requirement / initial_vms[a_t]
    total_ru += ru                            # ru is added to the AR numerator even on violation
    place(a_t, svc[t])                        # forced execution; remaining_vms can go negative / conflicts can co-occur
if t == M:
    reward <- graded terminal AR reward (§12.3-style), degrades to a valid_placed-fraction penalty if violations>0
return episode success (=violations==0), final AR
```

**Key point**: whether constraints are satisfied only shows up in the info dict and the final success flag — it never appears in the per-step reward and never restricts the action space. This is the deliberately "does nothing" floor baseline used to set off the other 5 algorithms. It is the **only one of the 6 algorithms with zero code changes across the entire v4.1.0 series** (the other 5 all had their non-terminal reward touched at some point; in the end only `ppo_lagrangian` was reverted).

---

## PPO+Mask (P4, hard masking) — the only algorithm that genuinely uses action masking

Trained with `sb3_contrib.MaskablePPO` + `ActionMasker` wrapping the environment (verified genuinely active, not a dormant interface). The env exposes `action_masks()`, and infeasible actions are given zero probability before sampling.

```
def action_masks(state):
    svc <- services[t]
    base_mask[j] <- (remaining_vms[j] >= svc.requirement) AND NOT has_conflict(j, t)   # hard: capacity + conflict
    if base_mask is all False:
        return base_mask                       # no way out; let the environment record a violation after the fact
    refined_mask[j] <- base_mask[j] AND NOT lethal_after(j)   # one-step lookahead, excludes choices that strand a later service
    return refined_mask if refined_mask has any True else base_mask

for t = 1..M:
    mask <- action_masks()
    a_t <- policy.sample(only ECUs with mask=True)      # sampling space is hard-restricted; infeasible actions get probability 0
    violated <- (forced-overflow fallback triggers only when mask is all False; picks the ECU with the most remaining capacity)
    reward_t <- violation_penalty(-2.0, nonzero only on the forced-overflow fallback) + shaping (off by default)
    place(a_t, svc[t])
if t == M:
    reward <- §12.3-style graded terminal reward (lt scenario additionally has an AR-weight curriculum w, see §12.4)
```

**v4.1.0 change**: `violation_penalty` used to be dead code (computed, never wired into the reward). It is now wired in. Because masking already structurally prevents almost all violations (campaign results: CapViol%=ConflictViol%=0.00% across lt/eq/gt), this fix only ever matters in the extremely rare forced-overflow-fallback case — **its effect on overall performance is negligible**, and pre/post numbers are essentially unchanged. `eq` has no forced-overflow-fallback branch at all, so this term doesn't apply there and that file was left untouched.

**Key point**: constraint handling happens **before** sampling, making this the only one of the 6 that can, in principle, guarantee zero violations at both training and deployment time. The cost is needing to enumerate the feasible action set, and the one-step lookahead is itself a heuristic approximation.

---

## Lagrangian PPO (P5, ppo_lagrangian) — no masking, pure-penalty constraints, ultimately reverted to "non-terminal reward = 0"

**Uses no action masking whatsoever.** `env.py` defines `action_masks()`, but it is dead code: `run_all.py` trains a plain `PrunedPPO` (a `stable_baselines3.PPO` subclass that only down-weights negative-advantage samples — unrelated to masking), never wrapped with `ActionMasker`/`MaskablePPO`. **This is by design, not a bug**: P5's entire methodological point is to substitute a penalty/dual-variable approach for masking; if it also hard-masked, there would be no distinction left between it and P4.

```
init: lambda <- LAMBDA_INIT, base_penalty <- 0.2, LAMBDA_LR, LAMBDA_TARGET, LAMBDA_MAX (=2.0 or 10.0 depending on scenario)

# inside the environment, every step:
for t = 1..M:
    a_t <- policy.sample(all N ECUs)               # no mask; the policy can freely land on a violating ECU
    cap_violated      <- remaining_vms[a_t] < svc[t].requirement
    conflict_violated <- has_conflict(a_t, t)
    c_t <- 1.0 if conflict_violated else 0.0        # conflict indicator (lambda tracks conflict only, not capacity)
    cap_penalty      <- -2.0 if cap_violated else 0.0
    lagrange_penalty <- -(lambda + base_penalty) * c_t
    match_gain       <- svc[t].requirement / initial_vms[a_t]
    total_ru += match_gain                          # violations still count toward the AR numerator (same convention as P3)
    place(a_t, svc[t])                               # no correction at all; executed as-is
    reward_t <- 0.0    # *** final decision: non-terminal reward is always 0, cap_penalty/lagrange_penalty/match_gain are NOT wired in
if t == M:
    reward <- §12.3-style graded terminal reward (violations==0 check includes both conflict and capacity counts)

# training callback (LagrangeCallback, runs dual ascent every LAMBDA_UPDATE_WINDOW episodes, unaffected by reward_t=0):
avg_viol <- moving average of conflict-violation rate over that window
lambda <- clip(lambda + LAMBDA_LR * (avg_viol - LAMBDA_TARGET), 0, LAMBDA_MAX)
env.set_lambda(lambda)
```

**⚠️ This is not "left unfinished" — it's a deliberate decision after three rounds of ablation** (full story in `paper_contents/v4.1.0_changelog.md`):

1. **v4.1.0**: wired `reward_t` to the full formula `match_gain + cap_penalty + lagrange_penalty` (including the positive utilisation term). Result: on lt, success_rate fell 0.81->0.52 and conflict_viol rose 0.18->0.47 — a clear regression; eq actually improved slightly, gt was roughly flat.
2. **v4.1.0.1**: hypothesised `match_gain` double-counted utilisation against the terminal AR term, so it was dropped, leaving `reward_t <- cap_penalty + lagrange_penalty`. Retested at 5M steps/3 seeds: lt moved to 0.52->0.54 and 0.47->0.45 — **essentially no change**. Hypothesis falsified.
3. **v4.1.0.2 (current)**: confirmed the problem isn't match_gain — it's that *any* non-zero per-step reward here destabilises PPO training on lt specifically (plausibly because the penalty magnitude — a fixed -2.0 plus an adaptively growing lambda term — can rival the terminal reward's own scale, and this environment has no structural safeguard the way P4's masking or P6's repair do, so it disturbs GAE advantage estimation). **Reverted to `reward_t = 0.0`, i.e. identical to v4.0.0's original behaviour.**

**Key point**: neither capacity nor conflict blocks execution. The dual-ascent update on lambda is, among the four constraint mechanisms, the only one that maintains a genuine dual variable updated by a violation-rate gradient — and **this mechanism's operation is completely independent of whether the per-step reward is zero**: lambda only looks at episode-level violation-rate statistics. Strictly speaking this is "no hard constraint at all, plus whatever the policy learns to avoid on its own, plus a dual-ascent knob that lives outside the reward" — it should not be described loosely as "Lagrangian handling of all constraints."

---

## Repair PPO (P6, ppo_opt, heuristic repair fallback) — the algorithm where the v4.1.0 fix had the clearest positive effect

The policy samples from a fully unconstrained action space; if the choice violates a constraint, the environment immediately **reassigns** it to a feasible ECU via a best-fit heuristic, then continues execution.

```
def best_fit_repair(svc_idx):
    svc <- services[svc_idx]
    valid <- {j : remaining_vms[j] >= svc.requirement AND svc_idx in ecu_allowed[j]}
    if valid is empty: return None
    return argmax_{j in valid} (svc.requirement / initial_vms[j])   # best-fit: highest remaining-utilisation feasible ECU

for t = 1..M:
    a_t <- policy.sample(all N ECUs)                # no mask; the policy can pick any ECU
    cap_violated      <- remaining_vms[a_t] < svc[t].requirement
    conflict_violated <- has_conflict(a_t, t)
    was_repaired <- False
    if cap_violated OR conflict_violated:
        a_repaired <- best_fit_repair(t)
        if a_repaired is None:
            return reward=-M, episode terminates immediately
        a_t <- a_repaired
        was_repaired <- True
        repairs += 1
    place(a_t, svc[t])
    reward_t <- -0.1 if was_repaired else 0.0        # *** wired in starting v4.1.0; used to be dead code (computed, never used)
if t == M:
    reward <- §12.3-style graded terminal reward (success = no repair fired the entire episode)
```

**Effect of the v4.1.0 change (consistent improvement across all 3 scenarios — the clearest positive result of this fix series)**:

| Scenario | conflict_viol_rate: v4.0.0 -> v4.1.0 | success_rate: v4.0.0 -> v4.1.0 |
|---|---|---|
| lt | 0.5425 -> 0.1450 (large drop) | 0.5005 -> 0.5408 |
| eq | 1.0000 -> 0.7208 (clear drop, still high) | 1.0000 -> 1.0000 |
| gt | 1.0000 -> 0.7692 (clear drop, still high) | 1.0000 -> 1.0000 |

**Key point**: the "zero violations" guarantee comes from the environment's built-in repair fallback, not from the policy learning to avoid violations on its own. Wiring in the -0.1 repair penalty visibly reduced how often the policy triggers a repair (especially on lt) — but **eq/gt still sit above 70%**, meaning the "policy exploits the repair mechanism" (reward-hacking) problem was **mitigated, not eliminated**: on eq/gt resources are abundant enough that success is nearly guaranteed regardless, and -0.1 is still too small relative to the terminal reward to make the policy give up leaning on repair. "The patch helped but wasn't sufficient" is a more accurate takeaway than either "there was no penalty at all" or "the penalty solved everything."

---

## DQN — the v4.1.0 fix had an equally clear positive effect

Standard off-policy Q-learning, same action space as the PPO family, no masking and no repair.

```
init: replay_buffer, q_net, q_net_target <- copy(q_net)
for env_step = 1..TOTAL_STEPS:
    a_t <- epsilon_greedy(q_net, obs_t)
    # --- inside the environment, every step ---
    cap_violated      <- remaining_vms[a_t] < svc[t].requirement
    conflict_violated <- has_conflict(a_t, t)
    cap_penalty      <- -2.0 if cap_violated else 0.0
    conflict_penalty <- -2.0 if conflict_violated else 0.0
    ru <- 0.0 if violated else svc[t].requirement/initial_vms[a_t]   # violations do NOT add to the AR numerator (unlike P3)
    place(a_t, svc[t])                                                # forced execution; remaining_vms can go negative
    r_t <- cap_penalty + conflict_penalty   # *** wired in starting v4.1.0; previously always 0 (dead code)
    if t == M: r_t <- §12.3-style graded terminal reward
    # --- gt scenario special case: a capacity violation is an immediate hard termination, r_t = conflict_penalty ---
    obs_{t+1}, done <- ...
    replay_buffer.add(obs_t, a_t, r_t, obs_{t+1}, done)
    if step % TRAIN_FREQ == 0:
        batch <- replay_buffer.sample(BATCH_SIZE)
        target_q <- r + (1-done) * gamma * max_a' q_net_target(obs_{t+1}, a')   # vanilla DQN: target net both selects and evaluates
        loss <- smooth_l1(q_net(obs_t, a_t), target_q)
        gradient step on q_net
    if step % TARGET_UPDATE == 0:
        q_net_target <- copy(q_net)
```

**Effect of the v4.1.0 change (consistent improvement across all 3 scenarios)**:

| Scenario | success_rate: v4.0.0 -> v4.1.0 | conflict_viol_rate: v4.0.0 -> v4.1.0 |
|---|---|---|
| lt | 0.361 -> 0.533 | 0.577 -> 0.427 |
| eq | 0.660 -> 0.763 | 0.332 -> 0.230 |
| gt | 0.836 -> 0.908 | 0.143 -> 0.076 |

**Key point**: DQN/DDQN are the two algorithms that benefited most directly from fixing this dead-code bug — with neither masking's structural protection nor repair's fallback, they had zero per-step feedback at all under v4.0.0 and could only learn from the one sparse signal at episode end. Once the dense per-step penalty was wired in, even though violation rates remain non-trivial (in the 20-43% range), Q-learning's Bellman bootstrap can propagate that signal back to earlier decisions reasonably effectively, and all three scenarios improved in the same direction.

---

## DDQN — same mechanism as DQN, similar effect

Subclasses `stable_baselines3.DQN`, overriding only the target-Q computation in `train()` (Double DQN, van Hasselt et al. 2016). The environment side (constraint handling, reward formula) is identical to DQN — the one deliberate difference under otherwise-controlled conditions.

```
class DDQN(DQN):
    def train(gradient_steps, batch_size):
        for _ in range(gradient_steps):
            batch <- replay_buffer.sample(batch_size)
            best_actions <- argmax_a' q_net(obs_{t+1}, a')          # online network picks the action
            next_q       <- q_net_target(obs_{t+1})[best_actions]   # target network only evaluates that action's value
            target_q <- r + (1-done) * gamma * next_q
            loss <- smooth_l1(q_net(obs_t, a_t), target_q)
            gradient step on q_net
        # environment/reward/exploration schedule/replay buffer are all shared with DQN; the v4.1.0
        # cap_penalty + conflict_penalty wiring is identical
```

**Effect of the v4.1.0 change**: also a consistent improvement across all 3 scenarios (success_rate: lt 0.351->0.465, eq 0.638->0.740, gt 0.881->0.925), similar magnitude to DQN — on gt, DDQN even edges out DQN after the fix.

**Key point**: DQN uses `q_net_target` for both selecting and evaluating the next action, which is prone to overestimation; DDQN decouples the two roles across the online/target networks to suppress that bias. That is the only code-level difference between them; everything else (constraint handling, reward, exploration) is shared.

---

## Six-Algorithm Constraint-Handling Overview (v4.1.0.2 final)

| Algorithm | Action space restricted? | When constraints are handled | Zero-violation guarantee (train/deploy) | v4.1.0 dead-code fix |
|---|---|---|---|---|
| PPO (P3) | No, free sampling over the full space | Never — recorded only | None | N/A (no such penalty was ever designed) |
| PPO+Mask (P4) | Yes, hard-masked before sampling (capacity+conflict+one-step lookahead) | Before sampling | Yes (structural, whenever the mask isn't all False) | Wired in; negligible effect (masking already prevented violations) |
| Lagrangian PPO (P5) | No, free sampling over the full space | Never — pure penalty in concept, but **the penalty was never wired into the reward** | None | **Wired in, measured a negative effect, ultimately reverted to reward=0** |
| Repair PPO (P6/ppo_opt) | No, free sampling over the full space | After the fact: environment-internal best-fit reassignment | Conditional (episode ends in failure if repair is impossible) | **Wired in, consistent improvement across all 3 scenarios (clearest positive effect)** |
| DQN | No, free sampling over the full space | Never — recorded + penalised | None | **Wired in, consistent improvement across all 3 scenarios** |
| DDQN | No, free sampling over the full space | Never — recorded + penalised (shares the environment with DQN) | None | **Wired in, consistent improvement across all 3 scenarios** |

This overview complements the formula/pipeline-level pseudocode in `论文概要3.md` §12/§13 and the full LaTeX formulas in `formula.md`. The complete investigation/ablation/revert trail is recorded in `paper_contents/v4.1.0_changelog.md`.
