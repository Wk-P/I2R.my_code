# 3. The environment: the problem as a step-by-step game

Code: `PlacementEnv` in `src/paper_rl/env.py`, shared by all 12 variants.

## Rules

- Services are first **sorted by demand, largest first** (big items first, as people pack).
- Step 1: choose an ECU for the largest service; step 2: the second largest; … M steps in total.
- One episode = all services placed.

## What the agent sees each step (observation / state)

The whole situation, unchanged, in 5 parts:

| Content | Length | Meaning |
|---|---|---|
| Capacity c of each ECU | N | Box sizes |
| Demand e of each service (placement order) | M | Item sizes |
| Conflict matrix | M × M | Which pairs cannot share a box |
| ECU of each service (0 = not placed) | M | Current layout |
| Current step t | 1 | Whose turn it is |

The Lagrangian variant adds one number, λ/(1+λ) (file 4).

- This is **full information**, nothing hidden. Values are raw, not normalized; normalization and "where does it fit" features are computed inside the network (file 6).
- There is no separate "observer" class: `_raw_obs()` builds the observation directly.
- Size: LT 266, EQ 131, GT 136 (Lagrangian +1 each).

## What the agent can do (action space)

- **Discrete**, not continuous: an action = an ECU index, 0 … N−1.
- The Mask variant has one extra **EXIT (give up)** action, allowed only when the current service fits nowhere.
- So the action space is Discrete(N), or Discrete(N+1) for Mask.

## Reward

**Every intermediate step gives 0; the score comes after the last placement**:

- All placed legally: reward = this episode's AR (0–1, fuller is higher).
- Otherwise (violation or EXIT): reward = −(share not legally placed), between −1 and 0.

This is `ar_pen`. Lagrangian uses a different one (file 4). Discount γ = 1, so an episode's return = that final score.

## Dead ends

If at some step the current service fits on no ECU (no room or conflicts everywhere), it is a **dead end**.
We **do not end the episode early** (full_episode): Mask can only choose EXIT; the other variants keep placing and record a violation.

## Inside one step

1. Mask chose EXIT → episode ends with the failure reward.
2. Repair chose an illegal ECU → the environment swaps in a legal one (file 4).
3. Check for violations (over capacity? conflict?), place the service, update remaining capacity.
4. Step +1; after the last service, compute the reward.
