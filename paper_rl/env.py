"""Unified placement environment (v4.3.1.3).

One class for every model and scenario. `mechanism` selects the
constraint-handling mechanism; the reward is selected by `reward_mode`
(legacy | ar | directional). Nothing here depends on the scenario except
N (ECUs) and M (services), which come from the instance data.

Constraints (checked for the service being placed):
  capacity  -- containers already placed on the ECU + this service's demand
               must not exceed the ECU's container capacity;
  privacy   -- the ECU must not already host a service from a conflict set
               that also contains this service. Services sharing no conflict
               set may be co-located.

Episode: services are presented in descending order of demand; step t
places service t on ECU a_t, so an episode has M steps.

Mechanisms:
  none      action executed as chosen; violations are recorded.
  mask      action_masks() = ECUs satisfying both constraints. If no ECU is
            feasible the mask is all-True and the executed placement is a
            violation (under `directional` the episode ends first, see below).
  lagrange  action executed as chosen; the episode's constraint cost is
            charged once at the end (v4.3.1.5): -lambda * sum_t c_t, c_t =
            number of constraints placement t violates (v4.3.1.3-v4.3.1.4:
            -lambda * c_t at every step). lambda is a dual variable updated by
            the trainer and visible in the observation.
  repair    an infeasible action is replaced by the best-fit feasible ECU
            (argmax demand / capacity); if none exists the episode ends
            (dead end). Executed placements are therefore always feasible.
  (mask)    since v4.3.1.4 a dead end ends the episode before the next
            placement, so a masked agent never executes an infeasible one.

AR (average resource utilization) is computed over feasibly executed
placements only:  AR = sum_{legal (i, j)} n_i / e_j / |ECUs hosting a legal
placement|. On a violation-free episode this is the usual AR; a violating
placement adds nothing.

Dead end (v4.3.1.4): mask -- the next service has no feasible ECU; repair --
no ECU to repair to. In every reward mode the episode ends at once as a
failure (no infeasible placement is executed); the penalty is
-M(1 - valid/M), or -C under `directional`.

Rewards (lagrange additionally gets -lambda * sum_t c_t at the end of the episode):
  objective    (v4.3.1.4 default) the objective only: r_t = 0 for t < M-1,
               terminal M * AR_exec, where AR_exec counts every executed
               placement, feasible or not -- no penalty for violations;
               the mechanism alone handles the constraints. Dead end: as above.
  legacy       r_t = 0 for t < M-1; terminal  M(2AR-1) if all M services are
               placed feasibly, else -M(1 - valid/M)
  ar           same, success branch M * AR
  directional  r_t = obj(dAR_t) every step, +B on feasible completion,
               -C and termination at a dead end (mask: no feasible ECU for
               the next service; repair: no ECU to repair to);
               obj(d) = 1 + beta d (d > eps), beta d (|d| <= eps),
               -lam_d + beta d (d < -eps)
A repair dead end under legacy / ar ends the episode with -M(1 - valid/M).
"""
from __future__ import annotations

import random

import gymnasium as gym
import numpy as np

from paper_rl import config as C

MECHANISMS = ("none", "mask", "lagrange", "repair")


def obs_dim(n: int, m: int) -> int:
    return 6 + 5 * n + 2 * m + 1


def mask_slice(n: int) -> slice:
    """Position of the feasible-ECU flags in the observation (MaskableDQN)."""
    return slice(6 + 4 * n, 6 + 5 * n)


def objective_reward(d: float) -> float:
    if d > C.DIR_EPS:
        return 1.0 + C.DIR_BETA * d
    if d < -C.DIR_EPS:
        return -C.DIR_LAMBDA + C.DIR_BETA * d
    return C.DIR_BETA * d


class PlacementEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, instances: list[dict], mechanism: str = "none",
                 reward_mode: str = "legacy", lam: float = 0.0, rng_seed: int | None = None):
        super().__init__()
        assert mechanism in MECHANISMS, mechanism
        assert reward_mode in ("objective", "legacy", "ar", "directional"), reward_mode
        self.instances = instances
        self.mechanism = mechanism
        self.reward_mode = reward_mode
        self.lam = float(lam)
        self._rng = random.Random(rng_seed)
        self._fixed = None           # instance index forced by the next reset (evaluation)
        self.N = len(instances[0]["ECUs"])
        self.M = len(instances[0]["SVCs"])
        self.action_space = gym.spaces.Discrete(self.N)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (obs_dim(self.N, self.M),), np.float32)

    # ── setup ────────────────────────────────────────────────────────────────
    def set_lambda(self, lam: float) -> None:
        self.lam = float(lam)

    def use_instance(self, idx: int) -> None:
        self._fixed = idx

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        idx = self._fixed if self._fixed is not None else self._rng.randrange(len(self.instances))
        self._fixed = None
        inst = self.instances[idx]
        self.inst_idx = idx
        self.ar_star = float(inst.get("ar_star", 0.0))
        order = sorted(range(self.M), key=lambda i: -inst["SVCs"][i])   # descending demand
        pos = {old: new for new, old in enumerate(order)}
        self.cap = np.array(inst["ECUs"], dtype=np.float32)
        self.req = np.array([inst["SVCs"][i] for i in order], dtype=np.float32)
        self.sets = [frozenset(pos[i] for i in s) for s in inst["conflict_sets"]]
        self.partners = [set().union(*(s for s in self.sets if i in s)) - {i} if any(i in s for s in self.sets)
                         else set() for i in range(self.M)]
        self.max_cap = float(self.cap.max())
        self.remaining = self.cap.copy()
        self.hosted = [set() for _ in range(self.N)]       # every executed placement
        self.t = 0
        self.legal_ru = 0.0
        self.legal_ecus: set[int] = set()
        self.exec_ru = 0.0                                  # every executed placement (objective)
        self.exec_ecus: set[int] = set()
        self.valid_placed = 0
        self.cap_violations = 0
        self.conflict_violations = 0
        self.repairs = 0
        self.dead_end = False
        return self._obs(), {}

    # ── constraints ─────────────────────────────────────────────────────────
    def _conflict(self, j: int, i: int) -> bool:
        return bool(self.partners[i] & self.hosted[j])

    def _feasible(self, i: int) -> np.ndarray:
        return np.array([self.remaining[j] >= self.req[i] and not self._conflict(j, i)
                         for j in range(self.N)], dtype=bool)

    def action_masks(self) -> np.ndarray:
        if self.t >= self.M or self.mechanism != "mask":
            return np.ones(self.N, dtype=bool)
        f = self._feasible(self.t)
        return f if f.any() else np.ones(self.N, dtype=bool)

    @property
    def ar_exec(self) -> float:
        """AR over every executed placement, infeasible ones included (objective reward)."""
        return self.exec_ru / len(self.exec_ecus) if self.exec_ecus else 0.0

    @property
    def ar(self) -> float:
        return self.legal_ru / len(self.legal_ecus) if self.legal_ecus else 0.0

    # ── observation ─────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        n, m, mc = self.N, self.M, self.max_cap
        total_cap = float(self.cap.sum())
        if self.t < m:
            i = self.t
            feas = self._feasible(i)
            conflict = np.array([self._conflict(j, i) for j in range(n)], dtype=np.float32)
            head = [self.req[i] / mc, feas.mean()]
        else:
            feas = np.zeros(n, dtype=bool)
            conflict = np.zeros(n, dtype=np.float32)
            head = [0.0, 0.0]
        rem_svcs = np.where(np.arange(m) >= self.t, self.req / mc, 0.0)
        svc_valid = np.zeros(m, dtype=np.float32)
        for k in range(self.t, m):
            svc_valid[k] = self._feasible(k).mean()
        allowed = np.array([1.0 - len(set().union(*(self.partners[h] for h in self.hosted[j])) - self.hosted[j]) / m
                            if self.hosted[j] else 1.0 for j in range(n)], dtype=np.float32)
        lam_norm = min(self.lam / C.LAMBDA_MAX, 1.0) if self.mechanism == "lagrange" else 0.0
        return np.concatenate([
            [head[0], self.ar,
             np.clip(self.remaining, 0, None).sum() / total_cap,
             self.req[self.t:].sum() / total_cap,
             head[1], (m - self.t) / m],
            self.cap / mc,
            np.clip(self.remaining / mc, -1.0, 1.0),
            conflict,
            allowed,
            feas.astype(np.float32),        # mask_slice(n)
            rem_svcs,
            svc_valid,
            [lam_norm],
        ]).astype(np.float32)

    # ── step ────────────────────────────────────────────────────────────────
    def _fail_reward(self) -> float:
        return -self.M * (1.0 - self.valid_placed / self.M)

    def _success_reward(self) -> float:
        if self.reward_mode == "ar":
            return self.M * self.ar
        return self.M * (2.0 * self.ar - 1.0)

    def _info(self, cap_v=False, conf_v=False) -> dict:
        return {"ar": self.ar, "ar_star": self.ar_star, "valid_placed": self.valid_placed,
                "services_placed": self.t, "capacity_violations": self.cap_violations,
                "conflict_violations": self.conflict_violations, "repairs": self.repairs,
                "cap_violated": cap_v, "conflict_violated": conf_v, "dead_end": self.dead_end,
                "ecus_used": len(self.legal_ecus), "inst_idx": self.inst_idx,
                "viol_rate_ep": (self.cap_violations + self.conflict_violations) / self.M}

    def step(self, action: int):
        i, a = self.t, int(action)
        ar0 = self.ar
        directional = self.reward_mode == "directional"

        if self.mechanism == "repair":
            feas = self._feasible(i)
            if not feas[a]:
                if not feas.any():                       # nothing to repair to: dead end
                    self.dead_end = True
                    r = -C.DIR_C * self.M if directional else self._fail_reward()
                    return self._obs(), r, True, False, self._info()
                cand = np.flatnonzero(feas)
                a = int(cand[np.argmax(self.req[i] / self.cap[cand])])
                self.repairs += 1

        cap_v = bool(self.remaining[a] < self.req[i])
        conf_v = self._conflict(a, i)
        self.cap_violations += cap_v
        self.conflict_violations += conf_v
        self.remaining[a] -= self.req[i]
        self.hosted[a].add(i)
        self.exec_ru += float(self.req[i] / self.cap[a])
        self.exec_ecus.add(a)
        if not (cap_v or conf_v):
            self.valid_placed += 1
            self.legal_ru += float(self.req[i] / self.cap[a])
            self.legal_ecus.add(a)
        self.t += 1
        done = self.t >= self.M
        success = done and self.valid_placed == self.M
        c_t = int(cap_v) + int(conf_v)

        dead = (not done and self.mechanism in ("mask", "repair")
                and not self._feasible(self.t).any())
        if dead:                                      # failure: stop before an infeasible placement
            self.dead_end = True
            done = True
            r = -C.DIR_C * self.M if directional else self._fail_reward()
        elif directional:
            r = objective_reward(self.ar - ar0)
            if success:
                r += C.DIR_B * self.M
        elif self.reward_mode == "objective":
            r = self.M * self.ar_exec if done else 0.0
        else:
            r = (self._success_reward() if success else self._fail_reward()) if done else 0.0
        if self.mechanism == "lagrange" and done:          # terminal constraint cost (v4.3.1.5)
            r -= self.lam * (self.cap_violations + self.conflict_violations)
        return self._obs(), float(r), done, False, self._info(cap_v, conf_v)
