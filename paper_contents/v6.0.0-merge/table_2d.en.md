# v6.0.0-merge 2-D summary: mechanism × algorithm (gap averaged over three scenarios)

- Generated 2026-10-29 16:01:08; script `src/scripts/merge_2d_table.py`; no training, no re-evaluation.
- Test set: the fixed v6.0.4 test set, 5000 ILP-optimal instances per scenario.
- Sources: Mask PPO = v6.0.5-PPO-hyparam (n_epochs=6, `v6.0.5-PPO-hyparam-test-50k`); Lagrangian = v6.0.1-lagrangian (standard Lagrangian, `v6.0.1-lagrangian-test-50k`); Unconstraint seed 1 = v6.0.1-unconstraint (`v6.0.1-unconstraint-test-50k`); others = v6.0.0 (`v6.0.4-test-50k`).
- **gap** = 1 − mean(AR) / mean(ILP AR), AR over legal placements; per scenario averaged over seeds, each cell = plain mean of the LT, GT, EQ gaps; row / column means are plain means of the cells.

## 1. Gap averaged over LT / GT / EQ

| Mechanism \ Algorithm | PPO | DQN | Double DQN | **Row mean** |
|---|---|---|---|---|
| Mask | 4.38% | 7.17% | 6.44% | **6.00%** |
| Repair | 7.15% | 6.50% | 7.37% | **7.01%** |
| Lagrangian | 6.24% | 12.51% | 13.84% | **10.86%** |
| Unconstraint | 6.12% | 24.97% | 22.59% | **17.90%** |
| **Column mean** | **5.97%** | **12.79%** | **12.56%** | **10.44%** |

![table_2d](table_2d.png)

## 2. Per-scenario breakdown

| Mechanism | Algorithm | Seeds | LT | GT | EQ | Mean | Violation rate (cap or privacy) |
|---|---|---|---|---|---|---|---|
| Mask | PPO | 1,2,3 | 6.22% | 3.28% | 3.64% | 4.38% | 0.00% |
| Mask | DQN | 1,2,3 | 13.57% | 4.41% | 3.54% | 7.17% | 0.00% |
| Mask | Double DQN | 1,2,3 | 10.62% | 5.05% | 3.65% | 6.44% | 0.00% |
| Repair | PPO | 1,2,3 | 8.56% | 6.04% | 6.85% | 7.15% | 0.85% |
| Repair | DQN | 1,2,3 | 12.49% | 3.41% | 3.61% | 6.50% | 1.88% |
| Repair | Double DQN | 1,2,3 | 15.55% | 2.94% | 3.61% | 7.37% | 2.20% |
| Lagrangian | PPO | 1,2,3 | 6.32% | 6.86% | 5.55% | 6.24% | 0.64% |
| Lagrangian | DQN | 1,2,3 | 18.37% | 8.27% | 10.87% | 12.51% | 1.20% |
| Lagrangian | Double DQN | 1,2,3 | 22.87% | 5.03% | 13.61% | 13.84% | 2.11% |
| Unconstraint | PPO | 1,2,3 | 7.27% | 5.10% | 5.99% | 6.12% | 0.75% |
| Unconstraint | DQN | 1,2,3 | 29.88% | 15.37% | 29.67% | 24.97% | 30.44% |
| Unconstraint | Double DQN | 1,2,3 | 27.42% | 19.08% | 21.27% | 22.59% | 33.89% |

- Violation rate = 1 − violation-free − EXIT, averaged over scenarios and seeds; only Mask is 0, so other mechanisms' gaps are not directly equivalent to Mask's.
