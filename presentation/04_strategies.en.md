# 4. Four constraint strategies: what if the agent chooses wrong?

"Wrong" = an ECU without room or with a conflicting service. Four approaches:

| Variant | In plain words | Can it violate? |
|---|---|---|
| **Mask** | Never lets it choose wrong: illegal options are blocked | Never |
| **Repair** | Lets it choose; the environment fixes a wrong choice | Only at dead ends |
| **Lagrangian** | Lets it choose; violations cost points, and the price adjusts itself | Yes, less over training |
| **Unconstraint** | Does nothing; the agent only loses at the final score | Yes |

## Mask

- Each step computes the legal ECUs: enough room **and** no conflicting service. Illegal options get probability/score −∞ and cannot be chosen.
- At a dead end, EXIT is the only option; it ends the episode with the failure reward.
- So Mask **never violates by construction**, the best fit for "privacy is a hard constraint".

## Repair

- If the agent picks an illegal ECU, the environment replaces it with the legal ECU of largest e_m/c_n, i.e. capacity-based best-fit: it looks only at the total capacity c_n and picks the smallest-capacity legal ECU, ignoring current remaining capacity.
- Training data still stores the agent's own choice, so it must learn not to rely on the fix.
- At a dead end there is nothing to swap in, so it violates.

## Lagrangian

- Whatever is chosen is executed. Each step has a cost c_t: +1 for over capacity, +1 for conflict (0, 1 or 2).
- Step reward = objective − **λ × c_t**. λ is the "price per violation":
  - more violations → λ rises, harsher penalty;
  - fewer (below the target 0.001 per episode) → λ falls.
- λ is tuned automatically by Adam (file 7) and given to the network (last observation entry λ/(1+λ)).
- Its objective `ar_obj` differs: the mean load ratio over all ECUs used (violating placements included), **no failure penalty**; violations are punished only through λ.
- Final λ: PPO about 1–2.4, DQN/DDQN about 4.4–5.5.

## Unconstraint

- No blocking, no fixing, no extra penalty; only `ar_pen`: an episode with a violation fails and scores negative.
- The control group: how bad "doing nothing" is.

## Summary

```
Agent picks ECU a
   │
   ├─ Mask: a illegal? → cannot be picked (dead end: EXIT only)
   ├─ Repair: a illegal? → swapped for the smallest-capacity legal ECU (capacity-based best-fit) (dead end: violation)
   ├─ Lagrangian: placed anyway; a violation costs λ×c_t, λ self-adjusts
   └─ Unconstraint: placed anyway; a violation fails the episode
```
