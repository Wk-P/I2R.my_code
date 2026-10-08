"""Unified placement environment (v4.3.1.3).

One class for every model and scenario. `mechanism` selects the
constraint-handling mechanism; the reward is selected by `reward_mode`
(succ_first | legacy | ar | objective | directional). Nothing here depends on the scenario except
N (ECUs) and M (services), which come from the instance data.
`obs_mode` (v4.3.4): "base" = the observation used up to v4.3.3; "conflict"
appends the raw conflict graph among the not-yet-placed services (fixed length
M(M-1)/2, see _conflict_obs). Everything else is identical.

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
  full_episode=True (v4.3.7): no early stop for any mechanism. Mask: at a dead end the
            mask is all-True and the chosen placement is executed; repair: with nothing to
            repair to, the chosen action is executed. Violations are recorded; the episode
            always has M steps. dead_end then means "a dead end occurred".

AR (average resource utilization) is computed over feasibly executed
placements only:  AR = sum_{legal (i, j)} n_i / e_j / |ECUs hosting a legal
placement|. On a violation-free episode this is the usual AR; a violating
placement adds nothing.

Dead end (v4.3.1.4): mask -- the next service has no feasible ECU; repair --
no ECU to repair to. In every reward mode the episode ends at once as a
failure (no infeasible placement is executed); the penalty is
-M(1 - valid/M), or -C under `directional`.

Rewards (lagrange additionally gets -lambda * sum_t c_t at the end of the episode):
  objective    (default in v4.3.1.4-v4.3.1.5) the objective only: r_t = 0 for t < M-1,
               terminal M * AR_exec, where AR_exec counts every executed
               placement, feasible or not -- no penalty for violations;
               the mechanism alone handles the constraints. Dead end: as above.
  succ_first   (default since v4.3.1.9, all 12 models) as legacy, but the
               success branch is M(1 + kappa AR), kappa = C.SUCC_KAPPA = 2,
               i.e. legacy + 2M: the same AR slope 2M, and every feasible
               completion (>= M) scores above every failure (<= 0)
  legacy       (default in v4.3.1.6 - v4.3.1.8)
               r_t = 0 for t < M-1; terminal  M(2AR-1) if all M services are
               placed feasibly, else -M(1 - valid/M)
  ar           same, success branch M * AR
  ar_raw       (v4.3.5) the objective itself: r_t = 0 for t < M-1, terminal
               AR if all M services are placed feasibly, else 0 (dead ends
               included); no scaling by M. Trained with gamma = 1, so the return
               is exactly AR * 1{feasible}.
  ar_pen       (v4.3.6, professor's spec) as ar_raw but a failure is penalised:
               terminal AR in (0, 1] if all M services are placed feasibly, else
               -(1 - valid/M) in [-1, 0) (dead ends included); no scaling by M,
               reward > 0 on success and < 0 on failure. gamma = 1.
  directional  r_t = obj(dAR_t) every step, +B on feasible completion,
               -C and termination at a dead end (mask: no feasible ECU for
               the next service; repair: no ECU to repair to);
               obj(d) = 1 + beta d (d > eps), beta d (|d| <= eps),
               -lam_d + beta d (d < -eps)
A repair dead end under succ_first / legacy / ar ends the episode with -M(1 - valid/M)
(ar_raw: 0; ar_pen: -(1 - valid/M)).
"""
from __future__ import annotations

import random

import gymnasium as gym
import numpy as np

from paper_rl import config as C

MECHANISMS = ("none", "mask", "lagrange", "repair")


OBS_MODES = ("base", "conflict", "feas", "raw")   # raw: v4.4.3, state for src/paper_rl/graph_net
REWARD_MODES = ("objective", "succ_first", "legacy", "ar", "ar_raw", "ar_pen", "directional")


def obs_dim(n: int, m: int, obs_mode: str = "base", action_mode: str = "ecu") -> int:
    if action_mode == "joint":
        return 4 + 3 * n + 4 * m + m * n + 1
    if obs_mode == "raw":                    # caps N | demands M | conflict graph M*M | assignment+1 M | t
        return n + m + m * m + m + 1
    base = 6 + 5 * n + 2 * m + 1
    return base + {"conflict": m * (m - 1) // 2, "feas": m * n}.get(obs_mode, 0)


def mask_slice(n: int, m: int = 0, action_mode: str = "ecu") -> slice:
    """Position of the feasible-ECU flags in the observation (MaskableDQN); for the joint
    action, the M x N (service, ECU) feasibility block."""
    if action_mode == "joint":
        return slice(4 + 3 * n + 4 * m, 4 + 3 * n + 4 * m + m * n)
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
                 reward_mode: str = "legacy", lam: float = 0.0, rng_seed: int | None = None,
                 obs_mode: str = "base", full_episode: bool = False, exit_action: bool = False,
                 action_mode: str = "ecu"):
        super().__init__()
        # v4.3.7 (professor's comment 1-②): never stop early -- every episode places all M
        # services; a mask / repair dead end executes the agent's own (infeasible) choice and
        # the violation is recorded instead of ending the episode.
        self.full_episode = full_episode
        # Mask gets an extra EXIT action (index N). It is masked while any ECU is
        # feasible and is the only valid action when none is, so Mask never violates; EXIT
        # ends the episode with the failure reward and the agent must learn to avoid it.
        self.exit_action = exit_action and mechanism == "mask"
        # v4.4.0: action_mode "joint" -- the agent picks (service, ECU), action = k * N + j
        # (+ EXIT = M * N); no fixed service order. Implemented for Mask (full episode).
        assert action_mode in ("ecu", "joint"), action_mode
        assert action_mode == "ecu" or (mechanism == "mask" and full_episode), "joint: Mask + full episode only"
        self.joint = action_mode == "joint"
        assert mechanism in MECHANISMS, mechanism
        assert obs_mode in OBS_MODES, obs_mode
        self.obs_mode = obs_mode
        assert reward_mode in REWARD_MODES, reward_mode
        self.instances = instances
        self.mechanism = mechanism
        self.reward_mode = reward_mode
        self.lam = float(lam)
        self._rng = random.Random(rng_seed)
        self._fixed = None           # instance index forced by the next reset (evaluation)
        self.N = len(instances[0]["ECUs"])
        self.M = len(instances[0]["SVCs"])
        self.action_space = gym.spaces.Discrete((self.M * self.N if self.joint else self.N) + self.exit_action)
        self.observation_space = gym.spaces.Box(-1.0, 1.0 if obs_mode != "raw" else 1e4,
                                                (obs_dim(self.N, self.M, obs_mode, action_mode),), np.float32)
        self._pairs = np.triu_indices(self.M, k=1)       # (i, j), i < j, in placement order

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
        adj = np.zeros((self.M, self.M), dtype=np.float32)
        for i, ps in enumerate(self.partners):
            adj[i, list(ps)] = 1.0
        self.conflict_pairs = adj[self._pairs]           # raw conflict graph, upper triangle
        self.remaining = self.cap.copy()
        self.hosted = [set() for _ in range(self.N)]       # every executed placement
        self.t = 0
        self.placed = np.zeros(self.M, dtype=bool)
        self.legal_ru = 0.0
        self.legal_ecus: set[int] = set()
        self.exec_ru = 0.0                                  # every executed placement (objective)
        self.exec_ecus: set[int] = set()
        self.valid_placed = 0
        self.cap_violations = 0
        self.conflict_violations = 0
        self.repairs = 0
        self.dead_end = False
        self.exited = False
        return self._obs(), {}

    # ── constraints ─────────────────────────────────────────────────────────
    def _conflict(self, j: int, i: int) -> bool:
        return bool(self.partners[i] & self.hosted[j])

    def _feasible(self, i: int) -> np.ndarray:
        return np.array([self.remaining[j] >= self.req[i] and not self._conflict(j, i)
                         for j in range(self.N)], dtype=bool)

    def action_masks(self) -> np.ndarray:
        if self.t >= self.M or self.mechanism != "mask":
            return np.ones(self.action_space.n, dtype=bool)
        if self.joint:
            f = self._joint_feasible().ravel()
            return np.append(f, not f.any()) if self.exit_action else f
        f = self._feasible(self.t)
        if self.exit_action:
            return np.append(f, not f.any())
        return f if f.any() else np.ones(self.N, dtype=bool)

    def _joint_feasible(self) -> np.ndarray:
        """M x N: unplaced service k can be placed on ECU j without violating anything."""
        f = np.zeros((self.M, self.N), dtype=bool)
        for k in np.flatnonzero(~self.placed):
            f[k] = self._feasible(k)
        return f

    @property
    def ar_exec(self) -> float:
        """AR over every executed placement, infeasible ones included (objective reward)."""
        return self.exec_ru / len(self.exec_ecus) if self.exec_ecus else 0.0

    @property
    def ar(self) -> float:
        return self.legal_ru / len(self.legal_ecus) if self.legal_ecus else 0.0

    # ── observation ─────────────────────────────────────────────────────────
    def _obs_joint(self) -> np.ndarray:
        """Joint action (v4.4.0): no current service. Global (4) | per ECU: capacity, remaining,
        allowance (3N) | per service: demand if unplaced, placed flag, feasible-ECU share,
        conflicts with other unplaced services / M (4M) | (service, ECU) feasibility = the
        action mask (M x N, read by MaskableDQN) | lambda (1)."""
        n, m, mc = self.N, self.M, self.max_cap
        total_cap = float(self.cap.sum())
        un = ~self.placed
        F = self._joint_feasible()
        allowed = np.array([1.0 - len(set().union(*(self.partners[h] for h in self.hosted[j])) - self.hosted[j]) / m
                            if self.hosted[j] else 1.0 for j in range(n)], dtype=np.float32)
        conf = np.array([len(self.partners[k] & set(np.flatnonzero(un))) / m if un[k] else 0.0
                         for k in range(m)], dtype=np.float32)
        lam_norm = min(self.lam / C.LAMBDA_MAX, 1.0) if self.mechanism == "lagrange" else 0.0
        return np.concatenate([
            [self.ar, np.clip(self.remaining, 0, None).sum() / total_cap,
             self.req[un].sum() / total_cap, un.sum() / m],
            self.cap / mc, np.clip(self.remaining / mc, -1.0, 1.0), allowed,
            np.where(un, self.req / mc, 0.0), self.placed.astype(np.float32), F.mean(axis=1), conf,
            F.ravel().astype(np.float32),
            [lam_norm],
        ]).astype(np.float32)

    def _obs(self) -> np.ndarray:
        if self.joint:
            return self._obs_joint()
        if self.obs_mode == "raw":
            return self._raw_obs()
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
            self._conflict_obs(),
            self._feas_obs(),
        ]).astype(np.float32)

    def _feas_obs(self) -> np.ndarray:
        """v4.3.9: remaining-service x ECU feasibility matrix F (M x N, row-major, placement
        order); F[k, j] = 1 iff service k is not placed yet and could be placed on ECU j now
        without violating capacity or privacy, else 0."""
        if self.obs_mode != "feas":
            return np.zeros(0, dtype=np.float32)
        f = np.zeros((self.M, self.N), dtype=np.float32)
        for k in range(self.t, self.M):
            f[k] = self._feasible(k)
        return f.ravel()

    def _raw_obs(self) -> np.ndarray:
        """v4.4.3: raw state for the structure-aware policy (src/paper_rl/graph_net.decode_raw):
        capacities (N), demands in placement order (M), conflict graph (M x M), ECU of every
        placed service + 1 (0 = not placed, M), current step t (1)."""
        adj = np.zeros((self.M, self.M), dtype=np.float32)
        for i, ps in enumerate(self.partners):
            adj[i, list(ps)] = 1.0
        assign = np.zeros(self.M, dtype=np.float32)
        for j, hs in enumerate(self.hosted):
            for i in hs:
                assign[i] = j + 1
        return np.concatenate([self.cap, self.req, adj.ravel(), assign, [self.t]]).astype(np.float32)

    def _conflict_obs(self) -> np.ndarray:
        """v4.3.4: raw conflict graph among the services not yet placed, fixed length
        M(M-1)/2 (upper triangle in placement order); a pair is 1 iff the two services
        conflict and both are still unplaced (current service included), else 0."""
        if self.obs_mode != "conflict":
            return np.zeros(0, dtype=np.float32)
        i, j = self._pairs
        return self.conflict_pairs * ((i >= self.t) & (j >= self.t))

    # ── step ────────────────────────────────────────────────────────────────
    def _fail_reward(self) -> float:
        if self.reward_mode == "ar_raw":
            return 0.0
        if self.reward_mode == "ar_pen":
            return -(1.0 - self.valid_placed / self.M)
        return -self.M * (1.0 - self.valid_placed / self.M)

    def _success_reward(self) -> float:
        if self.reward_mode in ("ar_raw", "ar_pen"):
            return self.ar
        if self.reward_mode == "ar":
            return self.M * self.ar
        if self.reward_mode == "succ_first":
            return self.M * (1.0 + C.SUCC_KAPPA * self.ar)
        return self.M * (2.0 * self.ar - 1.0)

    def _info(self, cap_v=False, conf_v=False) -> dict:
        return {"ar": self.ar, "ar_star": self.ar_star, "valid_placed": self.valid_placed,
                "services_placed": self.t, "capacity_violations": self.cap_violations,
                "conflict_violations": self.conflict_violations, "repairs": self.repairs,
                "cap_violated": cap_v, "conflict_violated": conf_v, "dead_end": self.dead_end, "exited": self.exited,
                "ecus_used": len(self.legal_ecus), "inst_idx": self.inst_idx,
                "viol_rate_ep": (self.cap_violations + self.conflict_violations) / self.M}

    def step(self, action: int):
        i, a = self.t, int(action)
        ar0 = self.ar
        directional = self.reward_mode == "directional"
        n_place = self.M * self.N if self.joint else self.N
        if self.exit_action and a == n_place:                # EXIT: no feasible ECU left
            self.exited = True
            return self._obs(), float(self._fail_reward()), True, False, self._info()
        if self.joint:
            i, a = divmod(a, self.N)
            assert not self.placed[i], "service already placed"

        if self.full_episode and self.mechanism == "mask" and not self._feasible(i).any():
            self.dead_end = True                          # all-True mask: placement will violate
        if self.mechanism == "repair":
            feas = self._feasible(i)
            if not feas[a]:
                if not feas.any() and self.full_episode:  # nothing to repair to: keep the action
                    self.dead_end = True
                elif not feas.any():                     # nothing to repair to: dead end
                    self.dead_end = True
                    r = -C.DIR_C * self.M if directional else self._fail_reward()
                    return self._obs(), r, True, False, self._info()
                if feas.any():
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
        self.placed[i] = True
        self.t += 1
        done = self.t >= self.M
        success = done and self.valid_placed == self.M
        # c_t = int(cap_v) + int(conf_v) # removed from v4.3.1.5+

        dead = (not done and not self.full_episode and self.mechanism in ("mask", "repair")
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
