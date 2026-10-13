# v5.0.3 λ_max sweep (DDQN)

- lagrange_ddqn, seed 1, 3M steps, λ_max ∈ {1, 2, 5}; deterministic evaluation on the 400 test instances.
- Effective gap: a violating instance counts as AR = 0, 1 − mean AR / mean ILP AR.
- Rule (violations first): highest mean violation-free share over the scenarios; within 0.5 pp of it is a tie, broken by the lower mean effective gap. A 3M-step sweep says little about the gap; violations weigh more.
- **Chosen λ_max = 1**

| λ_max | LT eff. gap / violation-free / gap / final λ | EQ eff. gap / violation-free / gap / final λ | GT eff. gap / violation-free / gap / final λ | mean eff. gap | mean violation-free |
|---|---|---|---|---|---|
| 1 | 26.5% / 93.8% / 21.5% / 1.000 | 16.0% / 99.5% / 15.6% / 1.000 | 8.9% / 100.0% / 8.9% / 0.930 | 17.1% | 97.8% |
| 2 | 32.5% / 90.2% / 25.5% / 2.000 | 17.1% / 99.8% / 16.8% / 1.998 | 16.7% / 100.0% / 16.7% / 1.985 | 22.1% | 96.7% |
| 5 | 37.6% / 90.5% / 31.2% / 5.000 | 17.7% / 99.8% / 17.4% / 4.992 | 16.2% / 100.0% / 16.2% / 4.956 | 23.8% | 96.7% |
