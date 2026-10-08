# v4.3.8 EXIT action for Maskable

On top of v4.3.7 (unified reward, no early stop), only Maskable changes:

- The action space is [ECU1, …, ECUn, EXIT].
- While any ECU is feasible, EXIT is masked and the agent chooses among feasible ECUs only, so it **never violates a constraint**.
- When no ECU is feasible, EXIT is the only valid action; it ends the episode with the failure reward −(1 − valid/M) ∈ [−1, 0). The penalty propagates back to the earlier placements that were feasible at the time but led into the dead end, so the agent learns to avoid them.
- The mask always has at least one valid action: no all-False mask and no forced placement.
- Test: Maskable's capacity / privacy violation rates are 0 by construction; its EXIT rate (share of test instances that ended with EXIT) is reported; AR / ILP AR / AR gap are averaged over instances with no violation and no EXIT.
- All other mechanisms (none, Lagrangian, Repair) are identical to v4.3.7.

Run: `src/scripts/run_v4.3.8.py --pilot` (retrains only Maskable PPO / DQN × 3 scenarios × seed 1 × 1M steps; the other 4 models come from the v4.3.7 pilot), report `pilot_report.md`.

## State (observation)

v4.3.8 uses the base observation (`obs = base`, without the v4.3.4 conflict graph), the same as every version since v4.3.1.6; the EXIT action does not change it. Code: `_obs()` in `src/paper_rl/env.py`.

Notation: N ECUs, M services (sorted by descending demand; step t places service t), c_j the capacity of ECU j, r_j its remaining capacity, d_i the demand of service i, c_max = max_j c_j, current service i = t.

Dimension = 6 + 5N + 2M + 1: LT (N=10, M=15) 87, EQ (N=10, M=10) 77, GT (N=15, M=10) 102.

| # | Component | Length | Meaning |
|---|---|---|---|
| 1 | d_i / c_max | 1 | demand of the current service |
| 2 | AR | 1 | AR of the feasible placements so far |
| 3 | Σ_j max(r_j, 0) / Σ_j c_j | 1 | share of capacity still free over all ECUs |
| 4 | Σ_{k≥t} d_k / Σ_j c_j | 1 | total demand of the services not yet placed |
| 5 | feasible-ECU share | 1 | share of the N ECUs the current service can be placed on feasibly |
| 6 | (M − t) / M | 1 | share of services still to place |
| 7 | c_j / c_max | N | capacity of each ECU |
| 8 | clip(r_j / c_max, −1, 1) | N | remaining capacity of each ECU |
| 9 | conflict flag | N | 1 if ECU j already hosts a service in conflict with the current one |
| 10 | allowance | N | 1 − (services in conflict with something on ECU j and not placed on j) / M; 1 for an empty ECU |
| 11 | feasible flag | N | 1 if placing the current service on ECU j violates neither capacity nor privacy (the mask; MaskableDQN reads it here) |
| 12 | remaining demands | M | d_k / c_max if service k is not placed yet, else 0 |
| 13 | remaining feasibility | M | for an unplaced service k, share of ECUs it could be placed on feasibly now; 0 if placed |
| 14 | λ / λ_max | 1 | Lagrangian only (λ_max = 50), 0 for the other mechanisms |

In the terminal state (t = M, or after EXIT) components 1, 5, 9 and 11 are 0.

The EXIT mask has no dimension of its own: Maskable PPO takes [feasible flags, no feasible ECU] from `action_masks()`; MaskableDQN uses the feasible flags (11) and opens EXIT when they are all 0.
