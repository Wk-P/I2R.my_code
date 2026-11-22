# 4 Proposed Safe Reinforcement Learning Framework

This chapter presents the safe reinforcement learning (Safe RL) framework that transfers the ILP of Chapter 3 into a sequential decision problem. Section 4.1 gives an overview of the framework, including the policy network and the learning algorithms. Section 4.2 formulates the Markov decision process (MDP). Section 4.3 describes the four constraint-handling strategies.

## 4.1 Framework Overview

The framework keeps the optimization problem of Chapter 3 unchanged and changes only how it is solved. Instead of searching over complete placements, an agent places the services one at a time, and the elements of the ILP are mapped to the MDP as shown in Table 4.1.

**Table 4.1** Correspondence between the ILP and the Safe RL formulation.

| ILP element | Safe RL counterpart |
|---|---|
| Objective AR (Section 3.2) | Terminal reward (Section 4.2.3) |
| Decision variables $x_{mn}$ | Action at each step: choose an ECU for the current service (Section 4.2.2) |
| (C1) each service placed once | Episode structure: one service per step, $M$ steps in total |
| (C2) capacity, (C4) privacy conflict | Per-step feasible action set $\mathcal{A}_t$, handled by a constraint-handling strategy (Section 4.3) |
| (C3) activation, $y_n$ | Implicit: an ECU becomes active when a service is placed on it |

The ILP evaluates the constraints once on a complete solution. The agent, in contrast, makes decisions step by step without seeing the final placement, so the constraints are checked at every step. The objective is transferred directly into the reward, while the constraints are enforced through one of four strategies, which range from a soft penalty to a hard guarantee (Section 4.3).

The ILP solver does not take part in training. Learning is model-free: the agent receives only the reward and the constraint signals produced by the environment and is never shown the optimal ILP placement. The ILP is used only for instance selection and evaluation (Section 3.2).

**Policy network.** A plain multilayer perceptron (MLP) flattens all inputs into one vector and cannot directly represent which service conflicts with which, or which service sits on which ECU. Such a network tends to learn a myopic, greedy placement rule. We therefore use a *relation-aware Transformer* \cite{vaswani2017attention, ying2021transformers} shared by all variants. The state is converted into $T = 1 + N + M$ tokens: one global token, one token per ECU, and one token per service. Each token carries normalized features such as remaining capacity, load, whether the ECU is feasible for the current service, and the number of remaining conflicts of a service. The attention logits of each head are augmented with learnable scalar biases for four relations: (1) a conflict between two unplaced services, (2) a service placed on an ECU, (3) a conflict between an unplaced service and a service already on an ECU, and (4) an unplaced service whose demand fits into the free capacity of an ECU:

$$
\alpha^{(h)}_{ij} = \operatorname{softmax}_j\!\Big( \frac{q^{(h)}_i \cdot k^{(h)}_j}{\sqrt{d/H}} + \sum_{r=1}^{4} w^{(\ell)}_{r,h}\, R_{r,ij} \Big).
$$

The network uses $L = 3$ pre-LN layers \cite{xiong2020layer}, model width $d = 128$, $H = 4$ heads, and a feed-forward size of 256, for a total of 450,227 parameters. An ECU head produces one score per ECU token, a separate head scores the EXIT action used by action masking (Section 4.3.3), and a value head reads the global token and the mean token. The network is permutation-equivariant over ECUs and services, and its parameter count does not depend on $N$ or $M$.

**Learning algorithms.** The framework is combined with three deep RL algorithms that represent the two main families. Proximal Policy Optimization (PPO) \cite{schulman2017proximal} is an on-policy actor-critic method that optimizes a clipped surrogate objective, here with generalized advantage estimation ($\lambda_{\text{GAE}} = 0.95$) and clip range $\epsilon = 0.1$. Deep Q-Network (DQN) \cite{mnih2015human} is an off-policy value-based method that learns $Q(s,a)$ from an experience replay buffer with a target network $\bar Q$:

$$
y_{\text{DQN}} = r + \gamma (1 - d) \max_{a'} \bar Q(s', a').
$$

Double DQN (DDQN) \cite{vanhasselt2016deep} reduces the overestimation bias of DQN by selecting the next action with the online network and evaluating it with the target network:

$$
y_{\text{DDQN}} = r + \gamma (1 - d)\, \bar Q\big(s', \arg\max_{a'} Q(s', a')\big).
$$

For PPO, the ECU scores are used as policy logits. For DQN and DDQN, they are used as Q-values. Combining the three algorithms with the four strategies of Section 4.3 gives the 12 variants listed in Table 1.1, which are denoted PPO-M, DQN-R, and so on.

<!-- TODO Fig. 4.1: framework flow chart (ILP → objective→reward, constraints→feasible set/strategies → MDP ↔ agent; ILP solver only to instance selection and evaluation). Fig. 4.2: network architecture. -->

## 4.2 Markov Decision Process Formulation

Services are sorted in non-increasing order of demand and renumbered so that $e_1 \ge e_2 \ge \dots \ge e_M$, with ties keeping their original order. At step $t = 1, \dots, M$, the agent places service $t$. Placing large services first is a standard heuristic for packing problems, and in a preliminary ablation it outperformed ascending and random orders. All returns are undiscounted ($\gamma = 1$).

### 4.2.1 State Space

The environment provides the raw state

$$
s_t = \big(\, c,\ e,\ A,\ z_t,\ t \,\big),
$$

where $c \in \mathbb{R}^N$ is the vector of ECU capacities, $e \in \mathbb{R}^M$ is the vector of service demands in placement order, $A \in \{0,1\}^{M\times M}$ is the conflict graph (Section 3.3), $z_t \in \{0, 1, \dots, N\}^M$ records the ECU of each service (0 for not yet placed), and $t$ is the current step. Under the Lagrangian strategy, the state also includes $\lambda/(1+\lambda)$, so that the policy can observe the current penalty weight (Section 4.3.2). The state therefore contains complete information about the instance and the partial placement. Its dimension is $N + M + M^2 + M + 1$, which is 266, 131, and 136 for the three scenarios of Section 5.1. Feature construction and normalization take place inside the policy network (Section 4.1).

### 4.2.2 Action Space

At each step, the agent selects the ECU for the current service:

$$
a_t \in \{1, \dots, N\}.
$$

Under action masking, the action space is extended with one additional EXIT action, giving $a_t \in \{1, \dots, N, \text{EXIT}\}$ (Section 4.3.3).

Let $H_n$ be the set of services placed on ECU $n$ before step $t$, and let $r_n = c_n - \sum_{m \in H_n} e_m$ be its remaining capacity. The *feasible action set* is

$$
\mathcal{A}_t = \{\, n \;:\; r_n \ge e_t \ \text{and}\ A_{tm} = 0\ \ \forall m \in H_n \,\}.
$$

If $\tilde a_t$ is the ECU on which service $t$ is actually placed (which may differ from $a_t$ under action repair), the capacity and privacy violation indicators of step $t$ are

$$
v^{\text{cap}}_t = \mathbb{1}\big[r_{\tilde a_t} < e_t\big],\qquad
v^{\text{priv}}_t = \mathbb{1}\big[\exists\, m \in H_{\tilde a_t}:\ A_{tm} = 1\big],\qquad
c_t = v^{\text{cap}}_t + v^{\text{priv}}_t .
$$

The per-step cost $c_t$ is the constraint signal used by the strategies in Section 4.3.

### 4.2.3 Reward Function

The reward is determined by the constraint-handling strategy and is the same for all three algorithms. Let $M_v = \sum_t \mathbb{1}[c_t = 0]$ be the number of valid placements. AR is computed over valid placements only:

$$
\text{AR} = \frac{\sum_{t:\, c_t = 0} e_t / c_{\tilde a_t}}{\big|\{\tilde a_t : c_t = 0\}\big|}.
$$

**Unconstrained, masking, and repair.** These three strategies use a sparse terminal reward that transfers the ILP objective directly:

$$
r_t = 0 \ \ (t < T),\qquad
r_T =
\begin{cases}
\text{AR}, & M_v = M,\\[2pt]
-\left(1 - \dfrac{M_v}{M}\right), & \text{otherwise (including EXIT)},
\end{cases}
$$

where $T$ is the last step of the episode. A complete, violation-free placement receives a positive reward in $(0, 1]$ equal to its AR. Any other outcome receives a negative reward in $[-1, 0)$ proportional to the fraction of services that were not placed validly. Every successful episode is therefore preferred to every failed one, and successful episodes are ranked by AR.

**Lagrangian.** The Lagrangian strategy separates the objective from the constraint cost:

$$
r_t = -\lambda\, c_t \ \ (t < M),\qquad
r_M = \text{AR}_{\text{obj}} - \lambda\, c_M,\qquad
\text{AR}_{\text{obj}} = \frac{1}{|\mathcal{E}|} \sum_{n \in \mathcal{E}} \min\!\Big(\frac{\sum_{t:\, \tilde a_t = n} e_t}{c_n},\ 1\Big),
$$

where $\mathcal{E}$ is the set of ECUs used in the episode. $\text{AR}_{\text{obj}}$ includes all executed placements, valid or not, and has no failure branch. Violations are penalized only through the per-step term $\lambda c_t$, whose weight $\lambda$ is adapted during training (Section 4.3.2).

### 4.2.4 Episode Termination

Each episode lasts $M$ steps, one per service. Violations do **not** terminate the episode early: a violating placement is executed and recorded, and the episode continues. As a result, the outcome of every episode is a complete placement whose number of violations can be counted.

A *dead end* occurs when the current service has no feasible ECU, $\mathcal{A}_t = \varnothing$. Under the unconstrained, repair, and Lagrangian strategies, the agent must still choose an ECU, so a violation is unavoidable. Under action masking, the only available action is EXIT, which ends the episode immediately with the failure reward $-(1 - M_v/M)$, where $M_v = t - 1$. This is the only case in which an episode ends before step $M$. Because every training and test instance is feasible by construction (Section 3.2), a dead end always results from the agent's own earlier decisions.

## 4.3 Constraint-Handling Strategies

The four strategies differ in how strongly the constraints are enforced, from no special treatment to a hard guarantee. Table 4.2 summarizes them.

**Table 4.2** Constraint-handling strategies.

| Strategy | Mechanism | Reward | Can violate? |
|---|---|---|---|
| Unconstrained (U) | None | Terminal AR / failure | Yes |
| Lagrangian (L) | Per-step penalty $\lambda c_t$, $\lambda$ by dual ascent | $\text{AR}_{\text{obj}} - \lambda c_t$ | Yes, decreasing during training |
| Action Masking (M) | Infeasible actions removed; EXIT at dead end | Terminal AR / failure | Never |
| Action Repair (R) | Infeasible action replaced by a heuristic | Terminal AR / failure | Only at dead ends |

### 4.3.1 Unconstrained Learning

The unconstrained strategy serves as the baseline. Actions are executed as chosen, and the constraints enter learning only through the failure branch of the terminal reward. A violating episode receives a negative reward instead of its AR. The agent is never told which step caused the violation, so it must infer the constraints from sparse episode-level feedback.

### 4.3.2 Lagrangian-Based Learning

The Lagrangian strategy treats the problem as a constrained MDP \cite{altman1999constrained}:

$$
\max_\pi\ J_R(\pi) = \mathbb{E}_\pi\big[\text{AR}_{\text{obj}}\big]
\qquad \text{s.t.} \qquad
J_C(\pi) = \mathbb{E}_\pi\Big[\sum_{t=1}^{M} c_t\Big] \le d,
$$

with cost budget $d = 0.001$ violations per episode. Following the reward-constrained policy optimization approach \cite{tessler2019reward}, the constraint is relaxed with a Lagrange multiplier:

$$
\mathcal{L}(\pi, \lambda) = J_R(\pi) - \lambda\,\big(J_C(\pi) - d\big),
$$

and the policy maximizes the penalized return $G = \text{AR}_{\text{obj}} - \lambda \sum_t c_t$ (Section 4.2.3). The multiplier is updated by projected dual ascent:

$$
\lambda \leftarrow \max\!\Big(0,\ \lambda + \eta_\lambda\, \big(\hat J_C - d\big)\Big),\qquad
\hat J_C = \frac{1}{|\mathcal{B}|}\sum_{\text{episodes} \in \mathcal{B}} \sum_t c_t,
$$

where $\mathcal{B}$ is a batch of recently completed episodes. In practice, the step is taken with the Adam optimizer (learning rate 0.035, $\lambda_0 = 0.001$, no upper bound). When the observed cost exceeds the budget, $\lambda$ increases and violations are penalized more heavily. When the cost falls below the budget, $\lambda$ decreases. For PPO, $\lambda$ is updated once per rollout. For DQN and DDQN, it is updated every 1000 episodes. Within an episode, $\lambda$ is fixed and is exposed to the policy as $\lambda / (1+\lambda)$.

This strategy turns the hard constraint into an adaptive soft penalty. It can drive violations toward the budget, but it does not guarantee zero violations at any finite point in training.

### 4.3.3 Action Masking

Action masking enforces the constraints by construction \cite{huang2022closer}. At each step, the environment provides the mask

$$
\text{mask}_t(n) = \mathbb{1}[n \in \mathcal{A}_t],\qquad
\text{mask}_t(\text{EXIT}) = \mathbb{1}[\mathcal{A}_t = \varnothing].
$$

If at least one feasible ECU exists, the agent can choose only among feasible ECUs. If none exists, the only available action is EXIT, which ends the episode with the failure reward (Section 4.2.4). A violating placement is therefore never executed, and the only form of failure is an early EXIT.

For PPO, the logits of masked actions are set to $-10^9$ before the softmax, so that sampling, the probability ratio, and the entropy are computed only over valid actions (MaskablePPO). For DQN and DDQN, the mask is applied to exploration, to greedy action selection, and to the bootstrapped target:

$$
y_{\text{M-DQN}} = r + \gamma (1 - d) \max_{a' \in \mathcal{A}(s')} \bar Q(s', a'),\qquad
y_{\text{M-DDQN}} = r + \gamma (1 - d)\, \bar Q\big(s',\ \arg\max_{a' \in \mathcal{A}(s')} Q(s', a')\big).
$$

The EXIT action is needed because a policy must always output a valid action. Without it, a dead end would leave the agent with an empty action set. Assigning EXIT the failure reward makes dead ends costly, so the agent is still motivated to avoid them earlier in the episode.

### 4.3.4 Action Repair

Action repair lets the agent choose freely and corrects infeasible choices with a heuristic before execution:

$$
\tilde a_t =
\begin{cases}
a_t, & a_t \in \mathcal{A}_t,\\[2pt]
\arg\min_{n \in \mathcal{A}_t} c_n, & a_t \notin \mathcal{A}_t,\ \mathcal{A}_t \ne \varnothing,\\[2pt]
a_t, & \mathcal{A}_t = \varnothing .
\end{cases}
$$

The repair heuristic is a capacity-based best fit: among the feasible ECUs, it selects the one with the smallest total capacity, breaking ties by index, which maximizes the utilization contribution $e_t / c_n$ of the current service. If no feasible ECU exists, nothing can be repaired, so the original action is executed and recorded as a violation. Repair therefore guarantees feasibility at every step except dead ends.

The learner stores and trains on the action $a_t$ chosen by the agent, not on the repaired action $\tilde a_t$. The agent thus receives the consequences of the repair through the reward but is not directly taught the heuristic's choice.

<!--
TODO
- Implementation notes from Framework/4.5, for an appendix or for 5.1 if needed: in each parallel environment, the first episode uses λ = 0.1; the DQN replay buffer stores r_t from sampling time and does not recompute it after λ changes.
- Decide whether inference complexity (old 4.6, propositions 2 and 3) goes into 5.5 Discussion. It is currently summarized there.
-->
