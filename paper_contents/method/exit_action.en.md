# The EXIT action: definition and necessity

- Scope: Maskable (Mask PPO / Mask DQN / Mask DDQN); every Maskable experiment since v4.3.8 uses it; code `src/paper_rl/env.py` (`exit_action=True`).
- This file gives the definition, formulas, a counter-example and the experimental evidence only; the narrative text is written by the author.

## 1. Definition

At step t service $s_t$ is placed (descending demand order). The action space is the N ECUs plus EXIT:

$$\mathcal{A} = \{1, \dots, N\} \cup \{\text{EXIT}\}$$

ECU j is legal when both the capacity and the privacy constraint hold:

$$m_t(j) = \mathbb{1}\left[\, r_t(j) \ge d(s_t) \ \wedge\ \nexists\, s' \in H_t(j):\ (s_t, s') \in \mathcal{C} \,\right], \quad j = 1, \dots, N$$

where $r_t(j)$ is the remaining capacity of ECU j, $d(s_t)$ the demand of the service, $H_t(j)$ the services already on ECU j, and $\mathcal{C}$ the set of privacy-conflict pairs.

EXIT is legal only when no ECU is:

$$m_t(\text{EXIT}) = \mathbb{1}\left[\, \textstyle\sum_{j=1}^{N} m_t(j) = 0 \,\right]$$

So every step is exactly one of: (a) at least one ECU is legal and EXIT is masked; (b) no ECU is legal and EXIT is the only legal action.

Choosing EXIT ends the episode with the failure reward (ar_pen):

$$R = \begin{cases} \text{AR}, & M_v = M \ (\text{all services placed legally}) \\ -\left(1 - M_v / M\right), & \text{EXIT}\end{cases}$$

$M_v$ is the number of services placed legally before EXIT.

## 2. Necessity

### 2.1 Dead ends occur even on ILP-feasible instances

An instance having a feasible solution (every instance is solved to optimality by the ILP) does not guarantee that a legal action exists after an arbitrary partial placement. A minimal counter-example (no privacy conflicts, capacity only):

| | Capacity / demand |
|---|---|
| ECUs | $e_1 = 10$, $e_2 = 6$ |
| Services (descending) | $s_1 = 6$, $s_2 = 5$, $s_3 = 5$ |

- Feasible solution: $e_1 \leftarrow \{s_2, s_3\}$, $e_2 \leftarrow \{s_1\}$.
- A policy trajectory: $s_1 \to e_1$ (legal, 4 left); $s_2 \to e_2$ (legal, 1 left); $s_3$: $e_1$ has 4 and $e_2$ has 1 left, neither fits, $m_3(j) = 0\ \forall j$, a dead end.

Every action was legal, yet the earlier choices leave no way forward. This happens a lot during training (§3.1).

### 2.2 An all-zero mask cannot be given to a Maskable policy

The Maskable policy's action distribution is

$$\pi_\theta(a \mid s_t) = \frac{m_t(a)\, e^{z_a}}{\sum_{a'} m_t(a')\, e^{z_{a'}}}$$

(implemented by setting the logits of illegal actions to $-\infty$). If $m_t \equiv 0$ the denominator is 0 and the distribution is undefined: sampling, $\log \pi$ and the entropy are all NaN. Mask DQN's $\arg\max_{a: m_t(a)=1} Q(s_t, a)$ is likewise an argmax over the empty set. Every step must therefore have at least one legal action.

### 2.3 Three ways to handle a dead end

| Handling | At a dead end | Does the executed placement violate a constraint | Data that enters training |
|---|---|---|---|
| **EXIT (adopted)** | EXIT is the only legal action, the episode ends, $R = -(1 - M_v/M)$ | No: every placement Maskable executes is legal | Only the step with $\pi_\theta(\text{EXIT} \mid s_t) = 1$ |
| Mask fully opened (v4.3.7) | $m_t \equiv 1$, the policy picks one of all ECUs, it is executed, the episode continues to M steps | Yes: the executed placement violates capacity or privacy | The policy's choice among illegal actions only |
| Random placement by the environment | The policy is skipped, the environment places the service on a random ECU, continues to M steps | Yes | The dead-end step does not enter training |

The privacy constraint is a hard constraint in this work: the last two options actually execute a violating placement $x_{s_t, j} = 1$, so the resulting deployment does not satisfy the ILP's constraint set, which contradicts the hard-constraint setting. EXIT is the only option that executes no violating placement at a dead end; its meaning is "this deployment request does not fit; reject / escalate it".

### 2.4 EXIT does not contaminate the policy gradient and still gives the earlier steps a learning signal

At a dead end EXIT is the only legal action:

$$\pi_\theta(\text{EXIT} \mid s_t) = 1 \ \Rightarrow\ \nabla_\theta \log \pi_\theta(\text{EXIT} \mid s_t) = 0$$

This step contributes nothing to the policy gradient (its entropy is 0 as well) and pushes no probability toward any illegal action. The failure reward reaches the earlier steps through the value function and the advantages (γ = 1):

$$\hat{A}_k = \sum_{l \ge k} (\gamma\lambda)^{l-k} \delta_l, \qquad k < t$$

so the early choices that led to the dead end receive negative advantages. $-(1 - M_v/M)$ becomes milder as $M_v$ grows: the later the dead end, the smaller the penalty, so the reward still separates "almost placed everything" from "stuck early".

## 3. Experimental evidence

### 3.1 How often dead ends occur during training

v4.4.16 (structure-aware Mask PPO, 50k-instance pool, 10M steps, 3 seeds; training EXIT rate of the stochastic policy = 1 − success rate):

| Scenario | 0–0.1M steps | 0.1–1M | 1–5M | 5–10M |
|---|---|---|---|---|
| LT | 36.0–37.5% | 6.9–7.5% | 2.9–3.2% | 2.5% |
| EQ | 0.1% | 0.0% | 0.0% | 0.0% |
| GT | 0.0% | 0.0% | 0.0% | 0.0% |

Early in LT training more than a third of the episodes reach a dead end; handling it is not an edge case and some fallback is required (§2.2). Test EXIT rate (deterministic policy): LT 1.0–2.25%, EQ 0–0.25%, GT 0%.

### 3.2 EXIT vs "open the mask, violate and continue"

Same setting (MLP, 1M steps, seed 1), only the dead-end handling differs:

| Scenario | Handling | Capacity violation rate | Privacy violation rate | EXIT rate | AR gap |
|---|---|---|---|---|---|
| LT | Mask fully opened (v4.3.7) | 4.8% | 7.8% | — | 0.198 |
| LT | EXIT (v4.3.8) | 0% | 0% | 12.5% | 0.207 |
| EQ | Mask fully opened (v4.3.7) | 0.2% | 0% | — | 0.141 |
| EQ | EXIT (v4.3.8) | 0% | 0% | 0.2% | 0.143 |
| GT | Mask fully opened (v4.3.7) | 0% | 0% | — | 0.149 |
| GT | EXIT (v4.3.8) | 0% | 0% | 0% | 0.148 |

The AR gap is the same under both (within single-seed noise), but with the mask opened 4.8% / 7.8% of the LT test instances violate the capacity / privacy constraint; with EXIT both violation rates are always 0. That is, EXIT enforces the constraints strictly at no cost in AR.

## 4. Evaluation convention

- Maskable's capacity / privacy violation rates are always 0; the **EXIT rate** = share of test instances on which EXIT was chosen (deployment-failure rate) is reported instead.
- AR, ILP AR and the relative gap are computed only on the test instances without EXIT (the same instances for both).

Sources: `paper_contents/v4.3.7/`, `paper_contents/v4.3.8/pilot_report.md`, `training_curve.csv` and `results.json` of the runs in `logs/v4.4.16/manifest.json`.
