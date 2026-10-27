# v6.0.0-merge 2-D summary: algorithm × mechanism (gap averaged over three scenarios)

- Generated 2026-10-27 15:57:18; script `src/scripts/merge_2d_table.py`; no training, no re-evaluation.
- Test set: the fixed v6.0.4 test set, 5000 ILP-optimal instances per scenario.
- Sources: Mask PPO = v6.0.0.2 (n_epochs=4, `v6.0.0.2-test-50k`); Lagrangian = v6.0.1-lagrangian (standard Lagrangian, `v6.0.1-lagrangian-test-50k`); others = v6.0.0 (`v6.0.4-test-50k`).
- **gap** = 1 − mean(AR) / mean(ILP AR), AR over legal placements; per scenario averaged over seeds, each cell = plain mean of the LT, GT, EQ gaps; row / column means are plain means of the cells.

## 1. Gap averaged over LT / GT / EQ

| Algorithm \ Mechanism | Mask | Repair | Lagrangian | Unconstraint | **Row mean** |
|---|---|---|---|---|---|
| PPO | 4.59% | 7.15% | 6.24% | 6.01% | **6.00%** |
| DQN | 7.17% | 6.50% | 12.51% | 24.38% | **12.64%** |
| Double DQN | 6.44% | 7.37% | 13.84% | 21.23% | **12.22%** |
| **Column mean** | **6.07%** | **7.01%** | **10.86%** | **17.21%** | **10.29%** |

## 2. Per-scenario breakdown

| Mechanism | Algorithm | Seeds | LT | GT | EQ | Mean | Violation rate (cap or privacy) |
|---|---|---|---|---|---|---|---|
| Mask | PPO | 1,2,3 | 6.55% | 3.47% | 3.74% | 4.59% | 0.00% |
| Mask | DQN | 1,2,3 | 13.57% | 4.41% | 3.54% | 7.17% | 0.00% |
| Mask | Double DQN | 1,2,3 | 10.62% | 5.05% | 3.65% | 6.44% | 0.00% |
| Repair | PPO | 1,2,3 | 8.56% | 6.04% | 6.85% | 7.15% | 0.85% |
| Repair | DQN | 1,2,3 | 12.49% | 3.41% | 3.61% | 6.50% | 1.88% |
| Repair | Double DQN | 1,2,3 | 15.55% | 2.94% | 3.61% | 7.37% | 2.20% |
| Lagrangian | PPO | 1,2,3 | 6.32% | 6.86% | 5.55% | 6.24% | 0.64% |
| Lagrangian | DQN | 1,2,3 | 18.37% | 8.27% | 10.87% | 12.51% | 1.20% |
| Lagrangian | Double DQN | 1,2,3 | 22.87% | 5.03% | 13.61% | 13.84% | 2.11% |
| Unconstraint | PPO | 2,3 | 7.36% | 4.66% | 6.01% | 6.01% | 0.78% |
| Unconstraint | DQN | 2,3 | 30.92% | 13.72% | 28.49% | 24.38% | 28.12% |
| Unconstraint | Double DQN | 2,3 | 25.52% | 20.81% | 17.35% | 21.23% | 30.58% |

- Violation rate = 1 − violation-free − EXIT, averaged over scenarios and seeds; only Mask is 0, so other mechanisms' gaps are not directly equivalent to Mask's.
