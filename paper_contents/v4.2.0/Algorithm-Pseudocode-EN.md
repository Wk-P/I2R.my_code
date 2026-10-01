# Algorithm Pseudocode (v4.2.0)

Pseudocode for the six algorithms in the v4.2.0 best-model set. Each algorithm is written according to **the code that actually trained that cell's models** (training code version per cell: see the table at the top of `formula.md`). Scenario-specific differences are marked with `if scenario == ...`. Formulas: `formula.md`.

---

## 0. Shared parts

```
# One episode (training and evaluation)
function RUN_EPISODE(instance, policy):
    remaining capacity R[j] ← e_j ;  valid ← 0 ;  total_u ← 0
    service queue ← services of the instance (sorted by descending demand if this cell's model does so)
    for t = 0 .. M-1:
        s_t ← OBS()                                    # observation, formula.md 3.2
        a_t ← policy(s_t)                               # algorithm-specific, see below
        v_cap  ← (R[a_t] < n_t)
        v_conf ← (ECU a_t already hosts a service from a conflict set containing t)
        place: R[a_t] ← R[a_t] - n_t                    # executed even on capacity violation (except Repair-PPO)
        if legal: valid ← valid + 1
        total_u ← total_u + u_t                         # whether a violating placement counts is algorithm-specific
        AR ← total_u / (number of activated ECUs)
        r_t ← non-terminal reward (algorithm-specific)
    r_{M-1} ← R_term                                    # terminal reward replaces the last step's non-terminal reward
    return AR, valid, violation counts

# Base terminal reward
function R_TERM(AR, valid, any_violation):
    if not any_violation: return M * (2*AR - 1)
    else:                 return -M * (1 - valid / M)
```

---

## 1. PPO (unconstrained baseline)

Training code v4.0.0 (all scenarios). Services are processed in dataset order; the observation has no `svc_valid_ecus`.

```
for t = 0 .. M-1:
    a_t ← π_θ(s_t)                       # any of the N ECUs
    record v_cap, v_conf (no penalty, not prevented)
    place; u_t counts towards AR regardless of violations
    r_t ← 0
r_{M-1} ← R_TERM(AR, valid, any_violation)
```

Training: standard PPO-Clip (Section 7).

---

## 2. Mask-PPO (hard action masking)

Training code v4.0.0 (all scenarios). MaskablePPO with the environment wrapped by ActionMasker.

```
function ACTION_MASK(t):
    m[j] ← (R[j] >= n_t) AND (no conflict on j)          # capacity + conflict
    if scenario == LT AND m is not all zero:
        m'[j] ← m[j] AND NOT LETHAL_AFTER(j)             # one-step lookahead
        if m' is not all zero: m ← m'
    return m

function LETHAL_AFTER(j):                                # would placing on j strand a later service?
    hypothetically place service t on j
    for each later service i > t:
        if no ECU satisfies both capacity and conflict constraints: return True
    return False

for t = 0 .. M-1:
    m ← ACTION_MASK(t)
    if m is all zero:
        a_t ← uniform over all N ECUs (result of MaskablePPO setting every logit to -1e8)
        # this step necessarily violates; in evaluation the episode ends here as a failure
    else:
        a_t ← π_θ(s_t | ECUs with m = 1)                 # infeasible actions get probability 0
    place; a violating placement has u_t = 0
    r_t ← 0
if no violation:
    if scenario == LT: r_{M-1} ← M * [(1-w) + w*(2*AR - 1)]   # w: linear 0 → 1 over training
    else:              r_{M-1} ← M * (2*AR - 1)
else:
    r_{M-1} ← -M * (1 - valid / M)
```

Training: PPO-Clip on the masked distribution; on LT the entropy coefficient anneals linearly from 0.02 to 0.002.

---

## 3. Lagrange-PPO (no masking, penalties + dual variable)

Training code: LT v4.0.0, EQ v4.1.0, GT v4.1.0.1. Unrestricted action space; λ enters the observation as clip(λ/λ_max, 0, 1).

```
# Environment: every step
for t = 0 .. M-1:
    a_t ← π_θ(s_t)                                       # any of the N ECUs
    compute v_cap, v_conf; place; u_t counts towards AR
    if scenario == LT:
        r_t ← 0
    elif scenario == EQ:
        r_t ← u_t / (number of activated ECUs) - 2*v_cap - (λ + 0.2)*v_conf
    elif scenario == GT:
        r_t ← -2*v_cap - (λ + 0.2)*v_conf
r_{M-1} ← R_TERM(AR, valid, any_violation)

# Training callback: dual ascent (at the end of every episode)
ν_ep ← (capacity violations + conflict violations) / M
push ν_ep into a sliding window of length W
if episodes_done >= E_w AND W episodes since the last update:
    λ ← clip(λ + η * (window mean - 0), 0, λ_max)
    broadcast λ to all parallel environments
# (λ_0, η, λ_max, W, E_w):
#   LT (0, 3e-4, 2.0, 20, 20000)   EQ (0.5, 0.01, 5.0, 200, 0)   GT (0.1, 0.005, 5.0, 20, 5000)

# Evaluation: λ fixed at its end-of-training value
```

Training: PPO-Clip, entropy coefficient 0.005.

---

## 4. Repair-PPO (best-fit repair)

Training code v4.1.0 (all scenarios).

```
function BEST_FIT_REPAIR(t):
    V ← { j : R[j] >= n_t AND no conflict on j }
    if V is empty: return None
    return argmax_{j∈V} n_t / e_j                        # legal ECU with the highest resulting utilisation

repaired_any ← False
for t = 0 .. M-1:
    a_t ← π_θ(s_t)                                       # any of the N ECUs
    if v_cap OR v_conf:
        a' ← BEST_FIT_REPAIR(t)
        if a' is None:
            return reward -M, episode ends immediately     # repair failed
        a_t ← a' ;  repaired_any ← True ;  repair-trigger count +1
        r_t ← -0.1
    else:
        r_t ← 0
    place (always legal); valid ← valid + 1
if scenario == LT:
    r_{M-1} ← (repaired_any ? 0 : M * (2*AR - 1))
else:  # EQ, GT
    r_{M-1} ← M * (2*AR - 1)
```

Training: PPO-Clip, entropy coefficient 0.001.

---

## 5. DQN

Training code v4.1.0 (all scenarios). No masking, no repair.

```
# Environment: every step
for t = 0 .. M-1:
    a_t ← ε-greedy(Q_θ, s_t)                            # ε = 0 at evaluation
    if scenario == GT AND v_cap:
        return reward -M, episode ends immediately
    place; a violating placement has u_t = 0
    if scenario == GT: r_t ← -2*v_conf
    else:              r_t ← -2*v_cap - 2*v_conf
r_{M-1} ← R_TERM(AR, valid, any_violation)

# Training
initialise Q_θ, Q_θ⁻ ← Q_θ, replay buffer D
for env_step = 1 .. 5e6:
    ε decays linearly from 1 to 0 over the first fraction f of steps (f: LT/GT 0.5, EQ 0.1)
    interact, store (s, a, r, s', done) in D
    if past learning_starts AND env_step % 4 == 0:
        sample 64 transitions from D
        y ← r + γ(1-done) * max_a' Q_θ⁻(s', a')
        one gradient step on Huber(Q_θ(s,a) - y)
    every 500 steps: Q_θ⁻ ← Q_θ
```

---

## 6. DDQN

Training code v4.1.0 (all scenarios). Environment, reward, exploration and replay are identical to DQN; only the target differs:

```
a* ← argmax_a' Q_θ(s', a')                               # online network selects
y  ← r + γ(1-done) * Q_θ⁻(s', a*)                        # target network evaluates
one gradient step on Huber(Q_θ(s,a) - y)
```

---

## 7. PPO-family training loop

Shared by PPO, Mask-PPO, Lagrange-PPO and Repair-PPO:

```
for iteration = 1 .. 5e6 / (n_steps * n_envs):
    collect n_steps per parallel environment with π_θ (Mask-PPO samples under the mask)
    compute advantages Â_t and returns R̂_t with GAE
    for epoch = 1 .. n_epochs:
        for each minibatch:
            normalise Â within the minibatch
            ρ ← π_θ(a|s) / π_θ_old(a|s)
            L_clip ← mean( min(ρÂ, clip(ρ, 1-ε, 1+ε)Â) )
            L ← -L_clip + 0.5 * mean((V_θ(s) - R̂)²) - c_e * mean(entropy)
            gradient step (gradient-norm clipping 0.5)
    (Mask-PPO LT: update c_e and curriculum weight w with training progress; Lagrange-PPO: λ updated by the callback)
```

---

## 8. Evaluation (as used for the paper results)

```
for each model (one per seed):
    for each instance in that seed's test set (400):
        run RUN_EPISODE once with deterministic actions
        (Mask-PPO: stop when no legal ECU exists; Lagrange-PPO: λ fixed at end-of-training value)
        success ← (valid == M) and no executed placement violates
        record AR, success, capacity / conflict violation, (Repair-PPO) repair triggered, time
    average over the model's test instances
mean ± sample std across the seed models of the same cell
```

---

## 9. Constraint-handling overview

| Algorithm | When constraints are handled | Can an executed placement violate? | Non-terminal reward |
|---|---|---|---|
| PPO | not handled, only recorded | yes | 0 |
| Mask-PPO | mask before sampling (LT adds one-step lookahead) | only when no legal ECU exists | 0 |
| Lagrange-PPO | penalties + dual variable | yes | LT 0; EQ utilisation gain + capacity penalty + λ-scaled conflict penalty; GT capacity penalty + λ-scaled conflict penalty |
| Repair-PPO | repair before execution | no (episode ends early if repair fails) | -0.1 × repaired |
| DQN / DDQN | penalties | yes (on GT a capacity violation ends the episode) | -2 each for capacity and conflict (GT: conflict only) |
