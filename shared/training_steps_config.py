from __future__ import annotations

"""Centralized default training-step configuration for all algorithms.

Edit this file when you want to change the default TOTAL_STEPS used by each algorithm.
Command-line overrides such as ``--total-timesteps`` still take precedence at runtime.
"""

GLOBAL_TOTAL_STEPS = 2_000_000

# Optional per-algorithm overrides. If a key is absent, GLOBAL_TOTAL_STEPS is used.
PROBLEM_TOTAL_STEPS: dict[str, int] = {
    "ppo": GLOBAL_TOTAL_STEPS,
    "ppo_mask": GLOBAL_TOTAL_STEPS,
    "ppo_lagrangian": GLOBAL_TOTAL_STEPS,
    "ppo_opt": GLOBAL_TOTAL_STEPS,
    "dqn": 2_000_000,
    "ddqn": 2_000_000,
}

# Per-(scenario, algorithm) overrides — takes precedence over PROBLEM_TOTAL_STEPS.
#
# IMPORTANT: steps must stay uniform ACROSS ALGORITHMS WITHIN one scenario --
# a controlled cross-algorithm comparison table is meaningless if some
# algorithms in it got more training budget than others (you can no longer
# tell whether a gap is the algorithm or the step count).
#
# 2026-09-05: moved to fully uniform 5M across ALL THREE scenarios (lt/eq/gt)
# and all six algorithms, superseding the earlier lt-only-then-gt-only
# per-scenario rollout. gt/dqn and gt/ddqn were confirmed not converged at
# 2M (Episode AR still trending up, conflict violation rate still trending
# down at the cutoff); rather than keep chasing per-scenario exceptions
# (eq was "confirmed converged at 2M" right up until it needed rechecking
# too), the whole three-scenario x six-algorithm campaign now runs on one
# shared 5M budget so every cell in the final comparison table came from
# the same steps count, full stop -- no per-scenario judgment calls left
# to get wrong later.
#
# 2026-09-07 (a): dqn/ddqn briefly carved OUT of the uniform 5M bucket to
# 15M -- a convergence probe run with a NEW candidate hyperparameter set
# (net_arch=[64,64], larger replay buffers, from a since-rejected
# full_hparam_sweep.py search) showed that config nowhere near converged at
# 5M (success_rate ~0.05-0.16, still climbing to a ~0.35-0.50 plateau that
# only stabilizes from ~10M steps onward).
#
# 2026-09-07 (b): that 15M number turned out to be measuring the WRONG
# config -- the candidate hyperparameters were separately rejected (a 5M
# head-to-head across lt/eq/gt showed the OLD default config winning or
# tying in every case; see paper_contents/), so dqn/ddqn's kept
# hyperparameters were never actually the ones the 15M figure was based on.
# Rerunning the SAME convergence probe with the actual kept defaults
# (net_arch=[128,128], buffer_size=100_000, from scenarios/*/dqn|ddqn/
# config.py) shows a much faster climb that's already within a noisy-but-
# flat band by 5M (dqn: 0.40-0.52 across 5M-7M; ddqn: 0.24-0.38 across
# 5M-7M -- residual spread here reads as off-policy evaluation noise, not a
# still-rising trend, unlike the rejected candidate's clear multi-million-
# step climb). Reverted to the uniform 5M budget on that basis -- every
# cell in the six-algorithm comparison table is back to the same steps
# count, and this time the convergence check was actually run against the
# hyperparameters being kept.
SCENARIO_TOTAL_STEPS: dict[tuple[str, str], int] = {
    (scenario, algo): 5_000_000
    for scenario in ("lt", "eq", "gt")
    for algo in ("ppo_mask", "ppo_lagrangian", "ppo", "ppo_opt", "dqn", "ddqn")
}


def get_total_steps(problem_name: str, scenario: str | None = None) -> int:
    if scenario is not None and (scenario, problem_name) in SCENARIO_TOTAL_STEPS:
        return int(SCENARIO_TOTAL_STEPS[(scenario, problem_name)])
    return int(PROBLEM_TOTAL_STEPS.get(problem_name, GLOBAL_TOTAL_STEPS))