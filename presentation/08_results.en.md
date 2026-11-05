# 8. Results

Numbers from `paper_contents/v6.0.0-merge/`; 5,000 test instances per scenario, averaged over 3 seeds.

## Gap of the 12 variants (mean of the three scenarios, smaller is better)

| Strategy \ Algorithm | PPO | DQN | DDQN | Mean |
|---|---|---|---|---|
| Mask | **4.38%** | 7.17% | 6.44% | 6.00% |
| Repair | 7.15% | 6.50% | 7.37% | 7.01% |
| Lagrangian | 6.24% | 12.51% | 13.84% | 10.86% |
| Unconstraint | 6.12% | 24.97% | 22.59% | 17.90% |

## Reading the table

1. **Mask PPO is best**: only 4.38% below ILP on average, with a **violation rate of 0** (only Mask achieves 0).
   Per scenario: LT 6.22%, GT 3.28%, EQ 3.64%; the hardest, LT, has the largest gap.
2. **PPO is the most stable overall**: 4–7% under every strategy. DQN/DDQN degrade clearly under Lagrangian and Unconstraint.
3. **Stronger constraint handling is better**: Mask > Repair > Lagrangian > Unconstraint.
   Unconstraint DQN/DDQN violate in over 30% of episodes; they never learned the rules.
4. Note: every variant except Mask has violations (0.6–34%), so their gaps are not fully comparable with Mask's. Violating placements score nothing, but those variants still break the rules.

## Speed

| Method | Time per instance (mean of three scenarios) |
|---|---|
| ILP (Dinkelbach) | about 1,515 ms (LT about 3.2 s) |
| 12 RL variants | about 49–57 ms |

RL is about **24–27× faster** than ILP (one CPU thread).

> Note: in this benchmark RL and ILP ran together in one process pool, which slowed RL by about 30%. Measured alone on LT, DQN takes about 49 ms and Mask PPO about 56 ms, so the true speed-up is somewhat higher. PPO is a bit slower because it also computes a value head.

## Conclusion

With "mask + PPO + relation Transformer", RL gets within about 4% of the optimum **without violating any constraint**, over 20× faster.
