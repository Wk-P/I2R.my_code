# v4.4.0 Action becomes (service, ECU)

Professor's comment 2 (placement order). On top of v4.3.8 (unified reward, no early stop, Maskable with EXIT, base observation):

- **Action**: no fixed descending-demand order; at each step the agent picks a service and an ECU together. Action k × N + j places service k on ECU j, M × N actions in total, plus EXIT (index M × N). LT 150+1, EQ 100+1, GT 150+1.
- **Mask**: placed services and (service, ECU) pairs that would exceed capacity or conflict are masked; EXIT is masked while any feasible pair exists and is the only valid action when none does, ending the episode with the failure reward −(1 − valid/M). Maskable never violates.
- **Scope**: Maskable (PPO / DQN) only for now. The other mechanisms first need a rule for picking an already placed service.
- The v4.3.9 F matrix is no longer an extra feature.

## State (observation)

There is no current service, so the components built around it are dropped and replaced by one group per service. Code: `_obs_joint()` in `src/paper_rl/env.py`. Dimension = 4 + 3N + 4M + MN + 1: LT 245, EQ 175, GT 240.

| # | Component | Length | Meaning |
|---|---|---|---|
| 1 | AR | 1 | AR of the feasible placements so far |
| 2 | Σ_j max(r_j, 0) / Σ_j c_j | 1 | share of capacity still free over all ECUs |
| 3 | Σ_{unplaced k} d_k / Σ_j c_j | 1 | total demand of the services not yet placed |
| 4 | unplaced services / M | 1 | share of services still to place |
| 5 | c_j / c_max | N | capacity of each ECU |
| 6 | clip(r_j / c_max, −1, 1) | N | remaining capacity of each ECU |
| 7 | allowance | N | 1 − (services in conflict with something on ECU j and not placed on j) / M; 1 for an empty ECU |
| 8 | service demand | M | d_k / c_max if service k is unplaced, else 0 |
| 9 | placed flag | M | 1 if service k is placed |
| 10 | feasible-ECU share | M | share of ECUs an unplaced service k can be placed on feasibly now; 0 if placed |
| 11 | conflicts | M | conflicts of unplaced service k with other unplaced services / M; 0 if placed |
| 12 | (service, ECU) feasible flags | M × N | the mask (read by MaskableDQN); required by the action structure, not an extra feature |
| 13 | λ / λ_max | 1 | Lagrangian only; always 0 for Maskable |

Services are still indexed by descending demand (only an index; it no longer fixes the placement order).

Pilot: `src/scripts/run_v4.4.0.py --pilot`, Maskable PPO / DQN × 3 scenarios × seed 1 × 1M steps; `pilot_report.md` puts each model next to the same model of the v4.3.8 pilot (ECU only).
