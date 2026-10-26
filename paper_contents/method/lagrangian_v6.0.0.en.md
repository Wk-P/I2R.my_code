# v6.0.0 Lagrangian: AR and reward formulas

> This form is not a standard Lagrangian and is renamed "ar_pen + adaptive violation penalty". The standard definition is in `lagrangian_v6.0.1.en.md`.

Code: `src/paper_rl/env.py` (`step`, `_success_reward`, `_fail_reward`), `src/paper_rl/train.py` (`_dual`, `_step_lambda`); configuration from `src/scripts/run_campaign.py` at tag v6.0.0 (cbf08e9). The Lagrangian code is unchanged up to the current HEAD.

## 1. Notation

| Symbol | Meaning |
|---|---|
| N, M | number of ECUs, services |
| c_n | capacity of ECU n |
| e_m | demand of service m |
| a_m | ECU that service m is placed on |
| c_t | violations at step t = 𝟙[capacity] + 𝟙[privacy] |
| M_v | number of legally placed services |
| λ | dual variable (fixed within an episode, applied at the next reset) |

## 2. AR accumulation (legal placements only)

Service m placed on ECU n at step t; if c_t = 0:

$$
\text{legal\_ru} \leftarrow \text{legal\_ru} + \frac{e_m}{c_n},\qquad \mathcal{E}_{\text{legal}} \leftarrow \mathcal{E}_{\text{legal}} \cup \{n\}
$$

$$
\text{AR} = \frac{\sum_{\text{legal}} e_m / c_{a_m}}{|\mathcal{E}_{\text{legal}}|}
$$

i.e. the mean utilisation of the used ECUs; violating placements count in neither numerator nor denominator.

## 3. Per-step reward (reward = ar_pen, γ = 1, full episode, always M steps)

$$
r_t = -\lambda\, c_t \quad (t < M)
$$

$$
r_M = R_{\text{term}} - \lambda\, c_M,\qquad
R_{\text{term}} =
\begin{cases}
\text{AR}, & M_v = M \\
-(1 - M_v / M), & M_v < M
\end{cases}
$$

## 4. Episode return

$$
G = R_{\text{term}} - \lambda \sum_{t=1}^{M} c_t
$$

- success: G = AR
- failure: G = −(1 − M_v/M) − λ·C_ep, with C_ep the episode's total violations

## 5. λ update (dual ascent)

v6.0.0 setting: `lambda_update=adam`, lr = 0.035, λ₀ = 0.001, d = 0, λ uncapped, J_C from the behaviour policy.

- PPO: one update per rollout, J_C = mean violations of the episodes finished in that rollout
- DQN / DDQN: one update every 1000 episodes, J_C = mean violations of those 1000 episodes

$$
\mathcal{L}(\lambda) = -\lambda\,(J_C - d),\qquad \lambda \leftarrow \text{Adam}(\nabla_\lambda \mathcal{L}),\qquad \lambda \leftarrow \max(\lambda, 0)
$$

The observation carries λ/(1+λ) (λ uncapped).

## 6. Assessment

| Item | Finding |
|---|---|
| Matches the agreed spec (full ar_pen − λc_t) | yes, no implementation bug |
| Standard Lagrangian (L = J_R − λJ_C, J_R unconstrained objective) | no: J_R is ar_pen, which already penalises violations; −λc_t penalises them a second time |
| Can λ decrease | no: d = 0 and J_C ≥ 0, so the gradient is always ≤ 0; 0 decreases in 201 logged λ values |
| Final λ | PPO 1.0–2.6, DQN/DDQN 4.8–5.5 |
| DQN/DDQN replay buffer | stores rewards with the old λ·c_t baked in, not recomputed when λ changes (suspected to hurt Lagrangian DQN, unverified) |

Standard alternative: r_M = AR_obj − λ·c_M (AR given on failed episodes too; `ar_obj` from v6.0.1, per-ECU utilisation clipped to 1) with d > 0. If the current form is kept, the paper should call it "ar_pen + adaptive violation penalty".
