"""
P5 Environment — Lagrangian Constraint Relaxation (no action masking).

N < M: each ECU hosts multiple services.

Neither constraint is action-masked here -- hard masking is exclusively P4
(ppo_mask)'s mechanism; P5's entire point is to compare a penalty/dual-ascent
approach against P4's structural guarantee, so masking capacity here would
collapse that distinction. action_masks() is defined below but is dead code
(never wrapped via ActionMasker/MaskablePPO in run_all.py -- P5 trains a
plain PrunedPPO), kept only as an unused interface.

Design:
    - Capacity violation → fixed penalty (-2.0 per step, conceptually -- see
      below); episode continues, remaining_vms may go negative.
    - Conflict violation → adaptive Lagrangian penalty (λ + base_penalty) * c_t;
      λ is updated externally by the training callback via dual ascent.

Per-step reward history (v4.1.0 -> v4.1.0.2 ablation, all at 5M steps/3 seeds,
see v4.1.0_changelog.md for the full lt/eq/gt numbers):
    - v4.0.0: non-terminal reward hardcoded 0.0 (lagrange_penalty/
      forced_overflow_penalty computed but never wired in -- dead code).
    - v4.1.0: wired in the full docstring formula (match_gain included). lt
      regressed hard: success_rate 0.81->0.52, conflict_viol 0.18->0.47.
    - v4.1.0.1: dropped match_gain (kept lagrange_penalty + forced_overflow_
      penalty only), hypothesising match_gain double-counted utilisation
      already in the terminal AR term. Did NOT recover lt (0.54/0.45 --
      statistically indistinguishable from v4.1.0). Hypothesis falsified.
    - v4.1.0.2 (current): reverted to v4.0.0 behaviour (reward=0.0). The
      regression isn't about match_gain -- it's that ANY non-zero per-step
      reward destabilises this env's PPO training on lt specifically (unlike
      ppo_opt, which has no structural violation guard here either, but whose
      repair_penalty is a much smaller -0.1 vs this env's -2.0/adaptive-λ
      penalties that can rival the terminal reward's scale). lagrange_penalty
      and forced_overflow_penalty are computed below but intentionally unused,
      same as v4.0.0 -- this is a data-backed decision, not an oversight.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import gymnasium as gym
import numpy as np
from ilp.objects import ECU, SVC


class LagrangeEnv(gym.Env):
    """
    Each episode assigns M services to N ECUs (N < M), one service per step.

    Observation (shape: 5N+7+2M):
        [0]          current service demand (normalised)
        [1]          current cumulative AR
        [2]          sum of remaining ECU capacity (normalised, clipped ≥ 0)
        [3]          sum of remaining service demand (normalised)
        [4]          fraction of ECUs with sufficient capacity
        [5]          fraction of services remaining
        [6:6+N]      initial capacity fraction per ECU
        [6+N:6+2N]   remaining capacity fraction per ECU
        [6+2N:6+3N]  conflict flag per ECU (1 = placing current svc here violates a conflict set)
        [6+3N:6+4N]  ECU allowed fraction (fraction of SVCs still placeable without conflict)
        [6+4N:6+5N]  valid-action flags (1 = sufficient capacity)
        [6+5N:6+5N+M] remaining service demands (sorted descending)
        [6+5N+M:6+5N+2M] valid ECU count per remaining service (normalised by N; 0 for placed)
        [6+5N+2M]    current λ value, normalised by λ_max

    Reward per step (v4.1.0.2, see module docstring for the full ablation history):
        r_t = 0.0  (reverted to v4.0.0 behaviour -- data-backed, see above)
        (terminal step instead uses the graded terminal formula)
    """

    metadata = {"render_modes": []}

    def __init__(self, ecus: list[ECU], services: list[SVC],
                 scenarios=None, lambda_init: float = 0.0,
                 lambda_max: float = 10.0):
        super().__init__()
        self._scenarios = scenarios
        self.ecus       = ecus
        self.services   = services
        self.N          = len(ecus)
        self.M          = len(services)
        self.lambda_val = float(lambda_init)
        self.lambda_max = max(float(lambda_max), 1e-8)

        self.action_space = gym.spaces.Discrete(self.N)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(5 * self.N + 7 + 2 * self.M,), dtype=np.float32,
        )

        self.initial_vms = np.array([e.capacity for e in ecus], dtype=np.float32)
        self.remaining_vms:       np.ndarray
        self.ecu_placements:      list[set]
        self.conflict_sets:       list[set]
        self.ecu_allowed:         list[set]
        self.ar:                  float
        self._step:               int
        self.episode_violations:  int
        self.cap_violations:      int
        self.conflict_violations: int
        self.reset()

    def set_lambda(self, val: float):
        self.lambda_val = float(val)

    # ── conflict helpers ─────────────────────────────────────────────────────
    def _init_conflict_sets(self, K: int = 10) -> list[set]:
        sets = []
        for _ in range(K):
            j = random.randint(2, self.M)
            sets.append(set(random.sample(range(self.M), j)))
        return sets

    def _has_conflict(self, ecu_idx: int, svc_idx: int) -> bool:
        return svc_idx not in self.ecu_allowed[ecu_idx]

    def _update_ecu_allowed(self, ecu_idx: int, svc_idx: int) -> None:
        for subset in self.conflict_sets:
            if svc_idx in subset:
                self.ecu_allowed[ecu_idx] -= (subset - {svc_idx})

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._scenarios is not None:
            caps, reqs, _cs = random.choice(self._scenarios)
            self.ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
            self.services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
            self.initial_vms = np.array([e.capacity for e in self.ecus], dtype=np.float32)
            self.conflict_sets = [set(cs) for cs in _cs]
        else:
            self.conflict_sets = self._init_conflict_sets()
        # Sort services descending and remap conflict_set indices to match the new order.
        sort_idx = sorted(range(self.M), key=lambda i: -self.services[i].requirement)
        self.services = [self.services[i] for i in sort_idx]
        inv_perm = [0] * self.M
        for new_i, old_i in enumerate(sort_idx):
            inv_perm[old_i] = new_i
        self.conflict_sets = [{inv_perm[k] for k in cs} for cs in self.conflict_sets]
        self.remaining_vms   = self.initial_vms.copy()
        self.ecu_placements  = [set() for _ in range(self.N)]
        self._req_arr = np.array([s.requirement for s in self.services], dtype=np.float32)
        self._n_active = 0
        self.ecu_allowed     = [set(range(self.M)) for _ in range(self.N)]
        self.ar              = 0.0
        self._total_ru       = 0.0
        self._step           = 0
        self.episode_violations  = 0
        self.cap_violations      = 0
        self.conflict_violations = 0
        self.valid_placed = 0
        self.episode_has_cap_violation      = False
        self.episode_has_conflict_violation = False
        return self._obs(), {}

    # ── action mask (capacity only) ──────────────────────────────────────────
    def action_masks(self) -> np.ndarray:
        if self._step >= self.M:
            return np.zeros(self.N, dtype=bool)
        svc = self.services[self._step]
        mask = (self.remaining_vms >= svc.requirement)
        if not np.any(mask):
            best = int(np.argmax(self.remaining_vms))
            mask = np.zeros(self.N, dtype=bool)
            mask[best] = True
        return mask

    # ── observation ───────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        max_cap   = float(np.max(self.initial_vms) + 1e-8)
        total_cap = float(np.sum(self.initial_vms) + 1e-8)

        if self._step >= self.M:
            service_demand_norm          = np.float32(0.0)
            conflict_flag                = np.zeros(self.N, dtype=np.float32)
            valid_flag                   = np.zeros(self.N, dtype=np.float32)
            remaining_service_demand_sum = np.float32(0.0)
            remaining_services_count     = np.float32(0.0)
            remaining_usable_ecu_count   = np.float32(0.0)
        else:
            svc = self.services[self._step]
            service_demand_norm = np.float32(svc.requirement / max_cap)
            conflict_flag = np.array(
                [float(self._has_conflict(j, self._step)) for j in range(self.N)],
                dtype=np.float32,
            )
            valid_flag = (self.remaining_vms >= svc.requirement).astype(np.float32)
            remaining_service_demand_sum = np.float32(
                float(np.sum(self._req_arr[self._step:])) / total_cap
            )
            remaining_services_count   = np.float32((self.M - self._step) / max(self.M, 1))
            remaining_usable_ecu_count = np.float32(
                np.sum(self.remaining_vms >= svc.requirement) / max(self.N, 1)
            )

        remaining_abs_norm = np.clip(self.remaining_vms, -max_cap, max_cap) / max_cap
        initial_cap_pct    = self.initial_vms / max_cap
        remaining_usable_capacity_sum = np.float32(
            np.sum(np.clip(self.remaining_vms, 0.0, None)) / total_cap
        )
        remaining_svcs = np.zeros(self.M, dtype=np.float32)
        if self._step < self.M:
            remaining_svcs[self._step:] = self._req_arr[self._step:] / max_cap
        lambda_norm = np.float32(np.clip(self.lambda_val / self.lambda_max, 0.0, 1.0))

        ecu_allowed_frac = np.array(
            [len(self.ecu_allowed[j]) / self.M for j in range(self.N)],
            dtype=np.float32,
        )

        svc_valid_ecus = np.zeros(self.M, dtype=np.float32)
        for i in range(self._step, self.M):
            svc_valid_ecus[i] = sum(
                1 for j in range(self.N)
                if self.remaining_vms[j] >= self.services[i].requirement
                and not self._has_conflict(j, i)
            ) / self.N

        return np.concatenate([
            [service_demand_norm],
            np.array([self.ar], dtype=np.float32),
            np.array([remaining_usable_capacity_sum], dtype=np.float32),
            np.array([remaining_service_demand_sum], dtype=np.float32),
            np.array([remaining_usable_ecu_count], dtype=np.float32),
            np.array([remaining_services_count], dtype=np.float32),
            initial_cap_pct,
            remaining_abs_norm,
            conflict_flag,
            ecu_allowed_frac,
            valid_flag,
            remaining_svcs,
            svc_valid_ecus,
            np.array([lambda_norm], dtype=np.float32),
        ]).astype(np.float32)

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: int):
        svc = self.services[self._step]

        cap_violated      = bool(self.remaining_vms[action] < svc.requirement)
        conflict_violated = self._has_conflict(action, self._step)

        if cap_violated:
            self.cap_violations += 1
            self.episode_has_cap_violation = True
            self.episode_violations += 1
        if conflict_violated:
            self.conflict_violations += 1
            self.episode_has_conflict_violation = True
            self.episode_violations += 1
        if not (cap_violated or conflict_violated):
            self.valid_placed += 1

        # Lagrangian penalty for conflict only; -2.0 for forced overflow (cap).
        # match_gain is always computed — conflict violation is penalized separately
        # via Lagrangian, not by zeroing out the utilisation reward (aligns with eq P5).
        base_penalty            = 0.2
        c_t                     = 1.0 if conflict_violated else 0.0
        lagrange_penalty        = -(self.lambda_val + base_penalty) * c_t
        forced_overflow_penalty = -2.0 if cap_violated else 0.0
        match_gain              = svc.requirement / (self.initial_vms[action] + 1e-8)

        self.remaining_vms[action] -= svc.requirement
        _was_empty = not self.ecu_placements[action]
        self.ecu_placements[action].add(self._step)
        if _was_empty:
            self._n_active += 1
        self._update_ecu_allowed(action, self._step)
        self._total_ru += match_gain  # always add (conflict penalized via λ, not ru=0)
        _active = self._n_active
        self.ar = self._total_ru / _active if _active > 0 else 0.0
        self._step += 1

        done = self._step >= self.M
        violated = cap_violated or conflict_violated
        if done:
            # Ported from scenarios/lt/ppo_mask/env.py (v2.6.0): graded
            # AR-quality success reward + graded-by-completion failure
            # penalty, instead of flat +M/-M.
            if self.episode_violations == 0:
                reward = float(self.M) * (2.0 * self.ar - 1.0)
            else:
                reward = -float(self.M) * (1.0 - self.valid_placed / float(self.M))
        else:
            # v4.1.0.2: reverted to v4.0.0 behaviour (reward=0.0 for non-terminal
            # steps). v4.1.0.1's penalty-only formula (reward = lagrange_penalty +
            # forced_overflow_penalty, no match_gain) was ALSO tested at 5M steps/
            # 3 seeds and did NOT recover lt (success_rate 0.54, conflict_viol 0.45
            # -- statistically indistinguishable from v4.1.0's 0.52/0.47 with
            # match_gain included). So the lt regression isn't about match_gain
            # double-counting; it's that ANY non-zero per-step reward here
            # destabilises this PPO variant's training on lt specifically (eq/gt
            # were roughly unaffected in all 3 variants) -- likely because, unlike
            # ppo_opt's small -0.1 repair_penalty in a repair-guarded environment,
            # this env has no structural violation guard at all and the penalty
            # magnitudes (-2.0, and an adaptively-growing lambda term) can rival
            # the terminal reward's scale, destabilising GAE/advantage estimation.
            # See v4.1.0_changelog.md for the full lt/eq/gt comparison across all
            # 3 variants. Kept as reward=0.0 -- data-backed, not an oversight.
            reward = 0.0
        return self._obs(), reward, done, False, {
            "ar":                  self.ar,
            "violated":            violated,
            "violations_ep":       self.episode_violations,
            "viol_rate_ep":        self.episode_violations / self._step,
            "cap_violations":      self.cap_violations,
            "conflict_violations": self.conflict_violations,
            "services_placed":     self._step,
            "valid_placed":        self.valid_placed,
            "ecus_used":          _active,
            "lambda":              self.lambda_val,
            "episode_has_cap_violation":      self.episode_has_cap_violation,
            "episode_has_conflict_violation": self.episode_has_conflict_violation,
        }

    # ── render ────────────────────────────────────────────────────────────────
    def render(self):
        if self._step < self.M:
            svc = self.services[self._step]
            print(f"  Step {self._step}/{self.M} | need {svc.requirement} VMs"
                  f" | AR={self.ar:.4f} | λ={self.lambda_val:.3f}"
                  f" | cap_viol={self.cap_violations} conflict_viol={self.conflict_violations}")
        else:
            print(f"  Done | AR={self.ar:.4f} | violations={self.episode_violations}/{self.M}")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    N, M = 7, 10
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(random.sample(range(50, 200, 5), N))]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(random.sample(range(10, 80, 5), M))]

    env = LagrangeEnv(ecus, services, lambda_init=1.0)
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}  (expected {5*N + 7 + 2*M})")

    done = False
    while not done:
        mask  = env.action_masks()
        valid = np.where(mask)[0]
        a = int(valid[0])
        obs, r, done, _, info = env.step(a)
        env.render()

    print(f"\nFinal AR: {info['ar']:.4f} | violations: {info['violations_ep']}")
