# v6.0.1-lagrangian: standard Lagrangian definition

Branch `v6.0.1-lagrangian` (commit c2cc5e6). Code: `src/paper_rl/env.py` (`ar_obj`, the `reward_mode == "ar_obj"` branch of `step`), `src/paper_rl/train.py` (`_dual`, `_step_lambda`). Flags: `--mechanism lagrange --reward ar_obj --lambda-target 0.001`.

The v6.0.0 form (ar_pen − λc_t) is renamed "ar_pen + adaptive violation penalty" and is no longer called Lagrangian; see `lagrangian_v6.0.0.en.md`.

## 1. Notation

| Symbol | Meaning |
|---|---|
| N, M | number of ECUs, services |
| c_n | capacity of ECU n |
| e_m | demand of service m |
| a_m | ECU that service m is placed on |
| c_t | violations at step t = 𝟙[capacity] + 𝟙[privacy] |
| 𝓔_exec | ECUs that received a service (violating placements included) |
| λ | dual variable (fixed within an episode, applied at the next reset) |
| d | cost limit, d = 0.001 |

## 2. CMDP

$$
\max_\pi\ J_R(\pi) \quad \text{s.t.}\quad J_C(\pi) \le d
$$

$$
J_R(\pi) = \mathbb{E}_\pi\big[\text{AR}_{\text{obj}}\big],\qquad
J_C(\pi) = \mathbb{E}_\pi\Big[\sum_{t=1}^{M} c_t\Big]
$$

J_R is the unconstrained objective with no violation penalty; violations appear only in J_C.

## 3. Objective AR_obj (violating placements included)

$$
\text{AR}_{\text{obj}} = \frac{1}{|\mathcal{E}_{\text{exec}}|} \sum_{n \in \mathcal{E}_{\text{exec}}} \min\!\Big(\frac{\sum_{m:\,a_m = n} e_m}{c_n},\ 1\Big)
$$

Each ECU's utilisation is clipped to 1, so overloading cannot push AR_obj above 1. On a successful episode (no violations) AR_obj = AR.

## 4. Per-step reward (γ = 1, full episode, always M steps)

$$
r_t = -\lambda\, c_t \quad (t < M),\qquad
r_M = \text{AR}_{\text{obj}} - \lambda\, c_M
$$

Episode return:

$$
G = \text{AR}_{\text{obj}} - \lambda \sum_{t=1}^{M} c_t
$$

i.e. the π-dependent part of the Lagrangian L(π, λ) = J_R − λ(J_C − d) (λd does not depend on π).

## 5. λ update (dual ascent)

$$
\mathcal{L}(\lambda) = -\lambda\,(J_C - d),\qquad \lambda \leftarrow \text{Adam}(\nabla_\lambda \mathcal{L}),\qquad \lambda \leftarrow \max(\lambda, 0)
$$

- J_C > d: λ increases; J_C < d: λ decreases (v6.0.0 had d = 0, so λ could only increase)
- PPO: one update per rollout, J_C = mean violations of the episodes finished in that rollout
- DQN / DDQN: one update every 1000 episodes, J_C = mean violations of those 1000 episodes
- Adam lr = 0.035, λ₀ = 0.001, λ uncapped, J_C from the behaviour policy; the observation carries λ/(1+λ)

## 6. Difference from v6.0.0

| Item | v6.0.0 | v6.0.1-lagrangian |
|---|---|---|
| Terminal reward | ar_pen: AR on success, −(1 − M_v/M) on failure | AR_obj (also on failure) |
| Violations penalised | twice (ar_pen + λc_t) | once (λc_t) |
| d | 0 | 0.001 |
| Can λ decrease | no | yes |
| Name | ar_pen + adaptive violation penalty | Lagrangian |

## 7. Open issue

The DQN/DDQN replay buffer stores rewards under the old λ and does not recompute them when λ changes (same as v6.0.0, not addressed).
