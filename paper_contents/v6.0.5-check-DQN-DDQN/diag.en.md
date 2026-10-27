# v6.0.5-check-DQN/DDQN: why Repair helps DQN / DDQN

- Generated 2026-10-27 16:15:09, 12.1 min; script `src/scripts/diag_repair_v6_0_5.py`; evaluation only, no training.
- Test set: the fixed v6.0.4 test set, 5000 ILP-optimal instances per scenario; deterministic; seed means.
- Models: Repair = v6.0.0 (seeds 1–3); Mask DQN/DDQN = v6.0.0; Mask PPO = v6.0.0.2 (n_epochs=4).
- Repair rule: an infeasible choice is replaced by the feasible ECU with argmax e_m / c_n (best-fit).

## 1. Repair: repair rate and the no-repair counterfactual

- **Repair rate**: share of steps with a feasible ECU whose choice is infeasible and repaired; early / mid / late = thirds of the episode.
- **Best-fit share**: share of steps with a feasible ECU whose executed ECU is the best-fit ECU (repaired steps included).
- **Q margin / infeasible mass**: median over repaired steps. DQN/DDQN = max Q over all ECUs − max Q over feasible ECUs (absolute); PPO = probability mass on infeasible ECUs.
- **gap (feasible only)**: the same model, argmax over the feasible ECUs only, so Repair never fires.

| Scen | Algo | gap (Repair) | gap (feasible only) | Δ | Repair rate | early | mid | late | best-fit share | Q margin / infeasible mass |
|---|---|---|---|---|---|---|---|---|---|---|
| LT | PPO | 8.56% | 8.78% | +0.22pp | 27.90% | 17.11% | 24.41% | 42.26% | 42.91% | 91.90% |
| LT | DQN | 12.49% | 12.59% | +0.09pp | 13.22% | 11.30% | 15.83% | 12.50% | 45.12% | 0.0062 |
| LT | Double DQN | 15.55% | 15.47% | -0.08pp | 14.57% | 12.90% | 15.22% | 15.64% | 41.84% | 0.0033 |
| GT | PPO | 6.04% | 6.33% | +0.28pp | 31.34% | 29.82% | 34.74% | 29.96% | 45.92% | 93.28% |
| GT | DQN | 3.41% | 3.44% | +0.02pp | 8.68% | 10.91% | 8.59% | 5.78% | 41.99% | 0.0026 |
| GT | Double DQN | 2.94% | 2.95% | +0.01pp | 11.20% | 14.61% | 11.67% | 6.17% | 41.65% | 0.0026 |
| EQ | PPO | 6.85% | 6.99% | +0.14pp | 34.59% | 28.22% | 36.03% | 41.63% | 52.34% | 92.99% |
| EQ | DQN | 3.61% | 3.61% | +0.01pp | 6.52% | 5.03% | 7.17% | 7.87% | 45.88% | 0.0026 |
| EQ | Double DQN | 3.61% | 3.61% | -0.00pp | 10.63% | 7.08% | 8.98% | 17.02% | 47.59% | 0.0025 |

## 2. Best-fit share by mechanism

| Scen | Mask PPO | Mask DQN | Mask Double DQN | Repair PPO | Repair DQN | Repair Double DQN |
|---|---|---|---|---|---|---|
| LT | 38.94% (gap 6.55%) | 34.55% (gap 13.57%) | 41.20% (gap 10.62%) | 42.91% (gap 8.56%) | 45.12% (gap 12.49%) | 41.84% (gap 15.55%) |
| GT | 40.08% (gap 3.47%) | 41.42% (gap 4.41%) | 43.25% (gap 5.05%) | 45.92% (gap 6.04%) | 41.99% (gap 3.41%) | 41.65% (gap 2.94%) |
| EQ | 48.05% (gap 3.74%) | 45.54% (gap 3.54%) | 45.59% (gap 3.65%) | 52.34% (gap 6.85%) | 45.88% (gap 3.61%) | 47.59% (gap 3.61%) |

## 3. Per run

| exp_id | Mech | Algo | Scen | Seed | gap | gap (feasible only) | Repair rate | best-fit |
|---|---|---|---|---|---|---|---|---|
| bbbdae7d | Mask | Double DQN | EQ | 1 | 4.13% | — | — | 44.93% |
| a81e4b73 | Mask | Double DQN | EQ | 2 | 3.42% | — | — | 44.95% |
| ef858f5a | Mask | Double DQN | EQ | 3 | 3.40% | — | — | 46.90% |
| 4e410c33 | Mask | DQN | EQ | 1 | 3.37% | — | — | 45.66% |
| 248187c7 | Mask | DQN | EQ | 2 | 3.98% | — | — | 45.86% |
| f7d702da | Mask | DQN | EQ | 3 | 3.25% | — | — | 45.11% |
| 6edb712f | Mask | PPO | EQ | 1 | 3.81% | — | — | 48.65% |
| 07185a24 | Mask | PPO | EQ | 2 | 3.62% | — | — | 46.85% |
| e735fbbe | Mask | PPO | EQ | 3 | 3.80% | — | — | 48.64% |
| e121eb1d | Mask | Double DQN | GT | 1 | 2.52% | — | — | 40.68% |
| 2f06241c | Mask | Double DQN | GT | 2 | 9.35% | — | — | 51.27% |
| 94aad994 | Mask | Double DQN | GT | 3 | 3.29% | — | — | 37.81% |
| 16b09659 | Mask | DQN | GT | 1 | 2.73% | — | — | 38.92% |
| 0da380c2 | Mask | DQN | GT | 2 | 3.80% | — | — | 39.12% |
| a14d9ac1 | Mask | DQN | GT | 3 | 6.70% | — | — | 46.22% |
| 7073eee7 | Mask | PPO | GT | 1 | 3.24% | — | — | 40.15% |
| 16434836 | Mask | PPO | GT | 2 | 3.00% | — | — | 39.78% |
| 497326d4 | Mask | PPO | GT | 3 | 4.17% | — | — | 40.31% |
| 18e7a36c | Mask | Double DQN | LT | 1 | 11.06% | — | — | 31.67% |
| 6e6d36fd | Mask | Double DQN | LT | 2 | 10.63% | — | — | 45.66% |
| 7c4da36e | Mask | Double DQN | LT | 3 | 10.16% | — | — | 46.26% |
| f5dc3448 | Mask | DQN | LT | 1 | 12.14% | — | — | 33.70% |
| aa67b4f4 | Mask | DQN | LT | 2 | 18.39% | — | — | 21.28% |
| ccdb97ac | Mask | DQN | LT | 3 | 10.20% | — | — | 48.67% |
| e1f78a4c | Mask | PPO | LT | 1 | 6.50% | — | — | 38.10% |
| 77ffb38a | Mask | PPO | LT | 2 | 6.46% | — | — | 39.91% |
| 521ec2d3 | Mask | PPO | LT | 3 | 6.70% | — | — | 38.82% |
| 7a039725 | Repair | Double DQN | EQ | 1 | 3.77% | 3.76% | 13.18% | 46.60% |
| d78f35d3 | Repair | Double DQN | EQ | 2 | 3.88% | 3.88% | 13.87% | 51.34% |
| 8053b1df | Repair | Double DQN | EQ | 3 | 3.18% | 3.19% | 4.83% | 44.82% |
| 9f514bbc | Repair | DQN | EQ | 1 | 3.58% | 3.58% | 4.07% | 46.23% |
| b22fbbb1 | Repair | DQN | EQ | 2 | 3.76% | 3.76% | 10.97% | 47.46% |
| 289f66d2 | Repair | DQN | EQ | 3 | 3.48% | 3.49% | 4.53% | 43.97% |
| 9d059434 | Repair | PPO | EQ | 1 | 6.78% | 6.89% | 32.19% | 51.31% |
| 270f4f72 | Repair | PPO | EQ | 2 | 6.74% | 6.76% | 35.41% | 53.53% |
| a0169b83 | Repair | PPO | EQ | 3 | 7.03% | 7.32% | 36.16% | 52.17% |
| 32730a2c | Repair | Double DQN | GT | 1 | 2.79% | 2.80% | 4.94% | 41.47% |
| 37e290f4 | Repair | Double DQN | GT | 2 | 3.21% | 3.21% | 13.21% | 41.41% |
| 5e33ba62 | Repair | Double DQN | GT | 3 | 2.83% | 2.84% | 15.44% | 42.07% |
| 2bfce454 | Repair | DQN | GT | 1 | 3.67% | 3.68% | 4.22% | 41.08% |
| d8ca43f5 | Repair | DQN | GT | 2 | 3.65% | 3.71% | 12.36% | 43.46% |
| c4d64c4e | Repair | DQN | GT | 3 | 2.92% | 2.92% | 9.44% | 41.45% |
| b19556e3 | Repair | PPO | GT | 1 | 6.21% | 6.54% | 31.31% | 46.04% |
| 4665d5cc | Repair | PPO | GT | 2 | 6.22% | 6.28% | 29.35% | 46.42% |
| 786fc4d6 | Repair | PPO | GT | 3 | 5.69% | 6.15% | 33.36% | 45.29% |
| 61d0f4d6 | Repair | Double DQN | LT | 1 | 13.50% | 13.18% | 18.06% | 51.83% |
| a715e721 | Repair | Double DQN | LT | 2 | 18.97% | 18.97% | 16.61% | 39.81% |
| 5b4abc6e | Repair | Double DQN | LT | 3 | 14.20% | 14.27% | 9.04% | 33.86% |
| 4bf4eb83 | Repair | DQN | LT | 1 | 11.23% | 11.27% | 4.84% | 40.06% |
| 956a9876 | Repair | DQN | LT | 2 | 14.00% | 14.16% | 27.38% | 53.82% |
| 2cc0c321 | Repair | DQN | LT | 3 | 12.25% | 12.33% | 7.44% | 41.47% |
| 0c9a096d | Repair | PPO | LT | 1 | 8.03% | 8.32% | 31.49% | 43.39% |
| a42246fe | Repair | PPO | LT | 2 | 9.43% | 9.68% | 25.60% | 41.47% |
| 523edcd0 | Repair | PPO | LT | 3 | 8.21% | 8.34% | 26.61% | 43.86% |
