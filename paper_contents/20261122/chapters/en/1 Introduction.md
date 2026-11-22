# 1 Introduction (material)

> The prose of this chapter is written by the author. Below are the facts it can use. Numbers match `experiments/experimental_results.en.md`; section references follow the final Chapters 3–5.

## Motivation

- The ILP (Dinkelbach + CBC) gives the optimal AR; all 3×5000 test instances were solved to optimality.
- The problem is strongly NP-hard (Section 3.2, Proposition 1). Measured mean time per instance is 1500 ms, 3285 ms on LT (median 1301 ms), with a long tail (Section 5.3, Table 5.5).
- Privacy conflicts are a hard constraint, so an approximate method must either guarantee the constraints or report its violation rate.

## Transfer idea

- The ILP objective (AR) becomes the terminal RL reward; the ILP constraints (capacity, privacy conflicts) become step-wise rules handled by different constraint-handling strategies (Section 4.1).
- The ILP is not used in training, only for filtering feasible instances and for evaluating the gap, so the method is model-free, not supervised.

## Contributions (facts; wording by the author)

1. An ILP→Safe RL transfer framework: objective → reward, constraints → four constraint-handling strategies; 4 strategies × 3 algorithms = 12 variants compared on three scenarios.
2. A relation-aware Transformer policy network (four relation biases, permutation-equivariant over ECUs and services, the same network for different sizes) (Section 4.1).
3. PPO-M has zero violations, a three-scenario mean relative AR gap of 4.36% (on violation-free instances), and is 28× faster than the ILP (47.5× on LT). Exact solution is strongly NP-hard (Proposition 1, Section 3.2), while online inference is polynomial (Section 5.5).

## Table 1.1  The 12 algorithm–strategy combinations

Row order follows Section 4.3 and Table 5.3: U → L → M → R.

| Strategy \ Algorithm | PPO | DQN | DDQN |
|---|---|---|---|
| Unconstrained (U) | PPO-U | DQN-U | DDQN-U |
| Lagrangian (L) | PPO-L | DQN-L | DDQN-L |
| Action Masking (M) | PPO-M | DQN-M | DDQN-M |
| Action Repair (R) | PPO-R | DQN-R | DDQN-R |

These labels denote the combinations throughout the thesis. Strategies are described in Section 4.3 and algorithms in Section 4.1. EXIT belongs to all masking variants (PPO-M, DQN-M, DDQN-M) and exists in both training and evaluation (PPO-M ends with EXIT on 1.38% of LT test instances).

## Thesis organization

| Ch. | Title | Content |
|---|---|---|
| 1 | Introduction | Motivation, transfer idea, contributions, the 12 labels |
| 2 | Background and Related Work | Edge service placement, privacy, ILP, RL, Safe RL, research gap |
| 3 | System Model and Problem Formulation | 3.1 System model (notation, AR); 3.2 ILP, Dinkelbach exact solution, strong NP-hardness (Proposition 1); 3.3 Privacy conflict constraint |
| 4 | Proposed Safe Reinforcement Learning Framework | 4.1 Overview (relation-aware Transformer, PPO/DQN/DDQN); 4.2 MDP; 4.3 Four constraint-handling strategies |
| 5 | Experimental Results and Analysis | 5.1 Setup; 5.2 Metrics; 5.3 Performance comparison (vs. ILP, strategies, violations, time); 5.4 Per-scenario analysis; 5.5 Discussion (inference complexity, offline vs. online cost) |
| 6 | Conclusion and Future Work | |
