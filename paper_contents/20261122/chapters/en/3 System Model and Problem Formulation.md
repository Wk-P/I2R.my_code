# 3 System Model and Problem Formulation

This chapter defines the service placement problem studied in this thesis. Section 3.1 introduces the system model and notation, Section 3.2 gives the integer linear programming (ILP) formulation and the exact solution method used as the reference baseline, and Section 3.3 describes the privacy conflict constraint, which is the hard constraint that motivates the safe reinforcement learning framework of Chapter 4.

## 3.1 System Model

We consider an edge computing system consisting of $N$ edge computing units (ECUs) and $M$ services that must be placed on them. Each ECU $n \in \{1, \dots, N\}$ has a resource capacity $c_n$, and each service $m \in \{1, \dots, M\}$ has a resource demand $e_m$. Every service must be deployed on exactly one ECU, and an ECU is considered *active* once at least one service is deployed on it.

Some services must not share an ECU because co-locating them creates a privacy risk. \cite{xu2020trust} defines these relations as pairwise conflicts between services. We organize the pairwise conflicts into $K$ privacy conflict sets $C_1, \dots, C_K \subseteq \{1, \dots, M\}$: any two services in the same conflict set conflict with each other and must be placed on different ECUs.

The placement decision is described by two groups of binary variables:

$$
x_{mn} =
\begin{cases}
1, & \text{if service } m \text{ is deployed on ECU } n\\
0, & \text{otherwise}
\end{cases}
\qquad
y_n =
\begin{cases}
1, & \text{if ECU } n \text{ is active}\\
0, & \text{otherwise}
\end{cases}
$$

Table 3.1 summarizes the notation.

**Table 3.1** Notation.

| Symbol | Meaning |
|---|---|
| $N$, $M$ | Number of ECUs, number of services |
| $n \in \{1,\dots,N\}$ | ECU index |
| $m \in \{1,\dots,M\}$ | Service index |
| $c_n$ | Capacity of ECU $n$ |
| $e_m$ | Resource demand of service $m$ |
| $C_k$, $k = 1,\dots,K$ | Privacy conflict sets |
| $x_{mn} \in \{0,1\}$ | Service $m$ is deployed on ECU $n$ |
| $y_n \in \{0,1\}$ | ECU $n$ is active |

The goal is to use the edge resources efficiently. We adopt the *average resource utilization* (AR) of \cite{xu2020trust}, defined as the mean utilization over all active ECUs. In \cite{xu2020trust} all ECUs have the same capacity; we generalize to heterogeneous capacities $c_n$:

$$
\text{AR}(x, y) = \frac{F(x)}{G(y)},\qquad
F(x) = \sum_{m=1}^{M}\sum_{n=1}^{N} x_{mn}\,\frac{e_m}{c_n},\qquad
G(y) = \sum_{n=1}^{N} y_n .
$$

For example, if two ECUs are active with utilizations 0.9 and 0.5, then $\text{AR} = 0.7$. A higher AR means that fewer ECUs are activated and each of them is more fully loaded.

## 3.2 Integer Linear Programming Formulation

The service placement problem is formulated as the following program:

$$
\begin{aligned}
\max_{x,\,y}\quad & \text{AR}(x, y) = \frac{F(x)}{G(y)} \\
\text{s.t.}\quad
&\textstyle\sum_{n=1}^{N} x_{mn} = 1, && \forall m && \text{(C1)}\\
&\textstyle\sum_{m=1}^{M} e_m\, x_{mn} \le c_n\, y_n, && \forall n && \text{(C2)}\\
&x_{mn} \le y_n, && \forall m, n && \text{(C3)}\\
&\textstyle\sum_{m \in C_k} x_{mn} \le 1, && \forall k, \forall n && \text{(C4)}\\
&x_{mn},\, y_n \in \{0, 1\}, && \forall m, n && \text{(C5)}
\end{aligned}
$$

Constraint (C1) requires every service to be deployed exactly once. Constraint (C2) is the capacity constraint: the total demand of the services placed on an ECU cannot exceed its capacity, and no service can be placed on an inactive ECU. Constraint (C3) links placement and activation. Constraint (C4) is the privacy conflict constraint discussed in Section 3.3, and (C5) states that all variables are binary. Together, (C1) and (C2) also ensure that the total demand of the system can be served. The model contains $MN + N$ binary variables and $M + N + MN + KN$ constraints.

**Exact solution by Dinkelbach's method.** Because AR is a ratio of two linear functions, the program cannot be given to a linear solver directly. We solve it with Dinkelbach's method \cite{dinkelbach1967nonlinear}, which replaces the fractional objective by a sequence of parametric ILPs with a linear objective and the same constraints (C1)–(C5):

$$
\varphi(\rho) = \max_{(x,y)\ \text{feasible}} \big[\, F(x) - \rho\, G(y) \,\big].
$$

Let $\text{AR}_{\text{ILP}}$ denote the optimal AR. Since $G(y) \ge 1$ for every feasible solution, $F(x) - \rho G(y) > 0$ holds if and only if $\text{AR}(x,y) > \rho$. Hence $\varphi(\rho) > 0$ if and only if $\rho < \text{AR}_{\text{ILP}}$, and $\varphi(\rho) = 0$ if and only if $\rho = \text{AR}_{\text{ILP}}$. The iteration is:

1. Set $\rho_0 = 0$.
2. Solve $\varphi(\rho_t)$ to optimality and obtain $(x_t, y_t)$.
3. Set $\rho_{t+1} = F(x_t) / G(y_t)$.
4. Stop if $|\rho_{t+1} - \rho_t| < 10^{-6}$ and return $\text{AR}_{\text{ILP}} = \rho_{t+1}$; otherwise return to Step 2.

Every iterate is feasible and $\rho_t$ is non-decreasing. Because the number of feasible solutions is finite, the procedure terminates after a finite number of iterations. In our experiments, each parametric ILP is solved with CBC through PuLP, and the method converges in 3.06 iterations on average.

**Computational complexity.** We first show that the problem is strongly NP-hard.

**Proposition 1.** The decision version of the service placement problem, *"is there a feasible placement with $\text{AR}(x,y) \ge \rho$?"*, is strongly NP-complete. This holds even when $K = 0$ and all ECUs have the same capacity.

*Proof.* Membership in NP: given $(x, y)$, constraints (C1)–(C5) and the inequality $F(x) \ge \rho\, G(y)$ can be checked in $O(MN + KN)$ time.

Hardness: we reduce from bin packing, which is strongly NP-complete \cite{garey1979computers}. A bin packing instance consists of item sizes $s_1, \dots, s_M \in \mathbb{Z}_{>0}$, a bin capacity $B \ge \max_m s_m$, and an integer $k$, and asks whether the items fit into at most $k$ bins. We construct a placement instance with $N = M$ ECUs of equal capacity $c_n = B$, demands $e_m = s_m$, no conflict sets ($K = 0$), and threshold $\rho = S / (kB)$, where $S = \sum_m s_m$. The construction takes polynomial time and copies all numbers unchanged, so it preserves strong NP-hardness.

With equal capacities, $F(x) = \sum_m e_m / B = S / B$ for every placement that satisfies (C1), so

$$
\text{AR}(x, y) = \frac{S}{B\, G(y)},
$$

which depends only on the number of active ECUs. Hence $\text{AR}(x,y) \ge \rho \iff G(y) \le k$. A packing into at most $k$ bins gives a feasible placement with $G(y) \le k$: place the items of bin $j$ on ECU $j$ and activate exactly the non-empty bins ($N = M$ ECUs always suffice). Conversely, the services on each active ECU of a feasible placement with $G(y) \le k$ form a valid bin by (C2). Therefore, the bin packing instance is a yes-instance if and only if the placement instance is. $\square$

As a consequence, no exact algorithm with polynomial or pseudo-polynomial running time exists for the optimization problem unless P = NP. Privacy conflicts ($K > 0$) only add constraints, and with $K > 0$ even deciding whether a feasible placement exists contains graph coloring as a special case (all $c_n$ large, one conflict set per edge). In the worst case, the search space of the ILP grows as $N^M$, and it must be explored in each Dinkelbach iteration. This cost motivates replacing the online exact solver with a learned policy.

**Role of the ILP in this thesis.** The ILP is not used to train the reinforcement learning agents and provides no supervision labels. It is used only to (i) select problem instances, keeping only instances for which an optimal solution is found, so that every instance is guaranteed to be feasible, and (ii) evaluate the learned policies, with $\text{AR}_{\text{ILP}}$ serving as the reference for the optimality gap (Section 5.2).

## 3.3 Privacy Conflict Constraint

In edge computing, services that process sensitive data may expose each other's information when they run on the same physical node, for example through shared memory, storage, or side channels. Privacy leakage cannot be compensated by better resource utilization, so it must be treated as a *hard* constraint: a placement that violates it is unacceptable regardless of its AR.

\cite{xu2020trust} describes the privacy relations as a set of conflicting service pairs and treats them as a hard constraint. We represent the same pairwise relations with the conflict sets $C_k$, each of which corresponds to a clique of the conflict graph. Constraint (C4),

$$
\sum_{m \in C_k} x_{mn} \le 1, \qquad \forall k,\ \forall n,
$$

states that at most one service from each conflict set may be deployed on any ECU. Equivalently, two services that belong to a common conflict set must never be co-located. For the sequential decision process in Chapter 4, it is convenient to represent the conflict sets as a symmetric conflict graph $A \in \{0,1\}^{M \times M}$, where $A_{mk} = 1$ if and only if $m \ne k$ and services $m$ and $k$ belong to a common conflict set.

The privacy conflict constraint makes the problem considerably harder than pure capacity-constrained packing. Capacity violations depend only on the total demand on an ECU, whereas a conflict excludes an ECU for a service regardless of how much free capacity it has. In our experimental setting ($K = 10$ conflict sets, pairwise conflict probability $p = 0.6$; see Section 5.1), conflicts are dense. A sequence of individually reasonable placements can therefore leave a later service with no ECU that is both large enough and conflict-free. We call such a state a *dead end*. Since all test instances are feasible by construction, a dead end is always caused by earlier placement decisions. How a learning agent avoids, or reacts to, dead ends is the central difference between the constraint-handling strategies compared in this thesis.

<!--
TODO
- List the changes this thesis makes to the model of xu2020trust (facts checked against the original; see notes/reading_questions.en.md [1]):
  - Objective: [1] has three objectives, max AR, min load-balance variance AL, min energy TE (SPEA2 + TOPSIS); we keep only AR.
  - AR: same denominator (active ECUs); [1] uses one capacity c, we use heterogeneous c_n.
  - Constraints: [1] has only the privacy conflict (Eq. 17) and execution time t_m ≤ TO (Eq. 18); capacity and exactly-once are not written explicitly. We state C1–C3 explicitly and drop the time constraint.
  - Conflicts: [1] uses pairwise conflicting service pairs; we organize them into K conflict sets (cliques).
  - Solution: [1] computes no optimum; we solve the ILP exactly (Dinkelbach) and approximate it with Safe RL.
  - Parameters: [1] uses c=7, e_m∈[1,6], M=50–250; none of our parameters come from [1].
- Optional: Fig. 3.1 (figs/fig_ilp_vs_saferl.pdf / .eps).
-->
