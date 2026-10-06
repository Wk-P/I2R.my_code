# v4.3.7 — Unified reward + no early stop (professor's comment 1-②)

New minor version (tag `v4.3.7`): one change on top of v4.3.6 (no M, success AR ∈ (0, 1], failure −(1 − valid/M) ∈ [−1, 0), γ = 1).

## 1. Change: every episode places all M services

- Previously Mask / Repair ended the episode when "the next service has no feasible ECU" and called it a "dead end". That is in fact a violation that must happen (any ECU exceeds capacity or has a privacy conflict); stopping early hid it and did not say which constraint was violated.
- Now (`--full-episode`, `full_episode=True` in `paper_rl/env.py`): no early stop. Mask opens the mask when no ECU is feasible and executes the agent's own choice; Repair executes the original action when there is nothing to repair to. Violations are recorded separately as capacity / privacy.
- Reward unchanged (once, at the end): AR without violations, −(1 − valid/M) with violations, valid = number of legal placements.
- The greedy baseline follows the same protocol: with no feasible ECU it prefers an ECU without privacy conflict and with the most remaining capacity, otherwise the one with the most remaining capacity; violations are recorded.

## 2. Test metrics

- Capacity violation rate, privacy violation rate: share of test instances with at least one violation of that kind.
- AR, ILP AR, AR gap: averaged over violation-free test instances only.
- Success rate and dead-end rate are no longer reported.

## 3. Self-check

3 scenarios × 4 mechanisms × 150 random episodes: every episode runs M steps; terminal rewards match the definition with 0 mismatches; "violation-free" coincides with "all M services placed legally"; violations are recorded by type.

## 4. Experiment

`scripts/run_v4.3.7.py --pilot`: the four PPO variants + Mask-DQN + Repair-DQN × 3 scenarios × seed 1 × 1M (18 jobs), report `pilot_report.md`.
