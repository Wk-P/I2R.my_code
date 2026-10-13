# v5.0.3 λ_max sweep (DQN)

- lagrange_dqn, seed 1, 3M steps, λ_max ∈ {1, 2, 5}; deterministic evaluation on the 400 test instances.
- Effective gap: a violating instance counts as AR = 0, 1 − mean AR / mean ILP AR.
- Rule (violations first): highest mean violation-free share over the scenarios; within 0.5 pp of it is a tie, broken by the lower mean effective gap. A 3M-step sweep says little about the gap; violations weigh more.
- **Chosen λ_max = 1**

| λ_max | LT eff. gap / violation-free / gap / final λ | EQ eff. gap / violation-free / gap / final λ | GT eff. gap / violation-free / gap / final λ | mean eff. gap | mean violation-free |
|---|---|---|---|---|---|
| 1 | 22.9% / 94.5% / 18.4% / 1.000 | 19.4% / 99.8% / 19.1% / 0.998 | 10.1% / 100.0% / 10.1% / 0.926 | 17.5% | 98.1% |
| 2 | 41.3% / 77.8% / 25.3% / 2.000 | 15.7% / 99.8% / 15.5% / 1.998 | 15.6% / 100.0% / 15.6% / 1.942 | 24.2% | 92.5% |
| 5 | 34.8% / 85.0% / 23.7% / 5.000 | 17.6% / 99.8% / 17.4% / 4.998 | 16.0% / 100.0% / 16.0% / 4.967 | 22.8% | 94.9% |
