"""src/shared/reward_config.py — switchable reward (v4.3.1).

$REWARD_MODE selects the terminal reward of a violation-free episode:

    legacy  M * (2*AR - 1)        (default; v4.3.0 and earlier)
    ar      M * AR                in (0, M]
    ratio   M * AR / AR*          in (0, M], AR* = ILP optimum of this instance
    directional  per-step objective reward (paper_contents/v4.3.0/Reward-Discussion.md),
            replaces the whole reward -- see directional_step() below

The failure branch is unchanged in every mode: -M * (1 - valid_placed/M),
which lies in [-M, 0) because a failed episode always has valid_placed < M.
So under `ar` and `ratio` every success (> 0) strictly beats every failure
(< 0); under `legacy` a success with AR < 0.5 - 1/(2M) scores below the best
failure (-1).

Environments call success_quality(ar, ar_star) and multiply by M (Mask-PPO
LT keeps its curriculum weight w on top: M * ((1-w) + w*quality)).

AR* per instance comes from results/<branch>/<scenario>/ilp/ar_star.json,
built once by src/scripts/build_ar_star.py; instances are keyed by
scenario_key(caps, reqs, conflict_sets) on the ORIGINAL (unsorted) service
order, i.e. exactly the tuple an environment draws in reset().
"""

from __future__ import annotations

import hashlib
import json
import os

from shared.paths import results_dir

REWARD_MODE = os.environ.get("REWARD_MODE", "legacy")
assert REWARD_MODE in ("legacy", "ar", "ratio", "directional"), f"unknown REWARD_MODE={REWARD_MODE!r}"

_AR_STAR: dict[str, dict[str, float]] = {}


def scenario_key(caps, reqs, conflict_sets) -> str:
    return hashlib.sha1(repr((
        [float(c) for c in caps],
        [float(r) for r in reqs],
        [sorted(int(k) for k in cs) for cs in conflict_sets],
    )).encode()).hexdigest()


def ar_star_path(scenario: str):
    return results_dir(scenario, "ilp", "ar_star.json")


def lookup_ar_star(scenario: str, caps, reqs, conflict_sets) -> float | None:
    """ILP-optimal AR of this instance; None unless REWARD_MODE needs it."""
    if REWARD_MODE != "ratio":
        return None
    if scenario not in _AR_STAR:
        path = ar_star_path(scenario)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run src/scripts/build_ar_star.py first")
        _AR_STAR[scenario] = json.loads(path.read_text())
    return _AR_STAR[scenario][scenario_key(caps, reqs, conflict_sets)]


def success_quality(ar: float, ar_star: float | None) -> float:
    """Success-branch reward divided by M."""
    if REWARD_MODE == "ar":
        return ar
    if REWARD_MODE == "ratio":
        return ar / ar_star
    # legacy; also under directional, whose step decorator discards this value
    return 2.0 * ar - 1.0


# ── directional objective reward (REWARD_MODE=directional) ─────────────────
#
#   dAR over legally executed placements only (see directional_step)
#   r_obj = 1 + beta*dAR      if dAR >  eps
#           beta*dAR          if |dAR| <= eps
#           -lam + beta*dAR   if dAR < -eps          (dAR = AR_{t+1} - AR_t)
#
#   + B   on the step that completes all M services with no violation
#   = -C  (episode ends) on a dead end: Mask-* -- the next state has no legal
#         ECU; Repair-* -- the env found no ECU to repair to
#   - eta*c_t for Lagrange-PPO: eta = the env's dual variable lambda_val,
#         c_t = number of constraint violations of this step
#
# Unconstrained PPO / DQN / DDQN get r_obj (+B) only. Parameters via env vars.
DIR_BETA = float(os.environ.get("DIR_BETA", "10"))
DIR_LAM = float(os.environ.get("DIR_LAMBDA", "1"))
DIR_EPS = float(os.environ.get("DIR_EPS", "1e-3"))
_DIR_B = os.environ.get("DIR_B")      # default M
_DIR_C = os.environ.get("DIR_C")      # default M


def objective_reward(d_ar: float) -> float:
    if d_ar > DIR_EPS:
        return 1.0 + DIR_BETA * d_ar
    if d_ar < -DIR_EPS:
        return -DIR_LAM + DIR_BETA * d_ar
    return DIR_BETA * d_ar


def _viol_count(env) -> int:
    cap = getattr(env, "capacity_violations", getattr(env, "cap_violations", 0))
    return int(cap) + int(getattr(env, "conflict_violations", 0))


def directional_step(step_fn):
    """Decorator for an environment's step(): leaves it untouched unless
    REWARD_MODE=directional, otherwise replaces the returned reward.

    dAR is computed here, not from the env's own self.ar, on legally executed
    placements only (v4.3.1.2): the envs disagree on how a violating
    placement enters AR (DQN envs count 0, PPO / Lagrange envs count its
    over-capacity utilisation, which can push AR above 1), and counting it
    would reward violations. A violating step therefore has dAR = 0 and is
    handled by the method's constraint mechanism alone. For Repair-* the
    executed (repaired) placement is always legal."""
    if REWARD_MODE != "directional":
        return step_fn

    def step(self, action):
        if self._step == 0:                      # first step of an episode
            self._dir_ru, self._dir_active = 0.0, set()
        idx = self._step
        req = float(self.services[idx].requirement)
        before = [len(p) for p in self.ecu_placements]
        ar0 = self._dir_ru / len(self._dir_active) if self._dir_active else 0.0
        v0 = _viol_count(self)

        obs, _, done, truncated, info = step_fn(self, action)
        M = float(self.M)
        B = float(_DIR_B) if _DIR_B is not None else M
        C = float(_DIR_C) if _DIR_C is not None else M
        is_repair = hasattr(self, "repairs")
        c_t = _viol_count(self) - v0

        if is_repair and done and self._step < self.M:   # repair impossible -> dead end
            info["dead_end"] = True
            return obs, -C, done, truncated, info

        placed_on = next((j for j, p in enumerate(self.ecu_placements) if len(p) > before[j]), None)
        if placed_on is not None and (is_repair or c_t == 0):
            self._dir_ru += req / float(self.initial_vms[placed_on])
            self._dir_active.add(placed_on)
        ar1 = self._dir_ru / len(self._dir_active) if self._dir_active else 0.0
        info["dir_ar"] = ar1

        r = objective_reward(ar1 - ar0)
        if hasattr(self, "lambda_val"):
            r -= float(self.lambda_val) * c_t

        if done:
            clean = is_repair or _viol_count(self) == 0
            if clean and int(getattr(self, "valid_placed", 0)) == self.M:
                r += B
        elif hasattr(self, "action_masks") and not is_repair and not self.action_masks().any():
            info["dead_end"] = True
            return obs, -C, True, truncated, info
        return obs, r, done, truncated, info

    return step
