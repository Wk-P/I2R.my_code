# v4.3.8 EXIT action for Maskable

On top of v4.3.7 (unified reward, no early stop), only Maskable changes:

- The action space is [ECU1, …, ECUn, EXIT].
- While any ECU is feasible, EXIT is masked and the agent chooses among feasible ECUs only, so it **never violates a constraint**.
- When no ECU is feasible, EXIT is the only valid action; it ends the episode with the failure reward −(1 − valid/M) ∈ [−1, 0). The penalty propagates back to the earlier placements that were feasible at the time but led into the dead end, so the agent learns to avoid them.
- The mask always has at least one valid action: no all-False mask and no forced placement.
- Test: Maskable's capacity / privacy violation rates are 0 by construction; its EXIT rate (share of test instances that ended with EXIT) is reported; AR / ILP AR / AR gap are averaged over instances with no violation and no EXIT.
- All other mechanisms (none, Lagrangian, Repair) are identical to v4.3.7.

Run: `scripts/run_v4.3.8.py --pilot` (retrains only Maskable PPO / DQN × 3 scenarios × seed 1 × 1M steps; the other 4 models come from the v4.3.7 pilot), report `pilot_report.md`.
