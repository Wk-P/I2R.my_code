# v5.0.3 λ_max sweep (PPO)

- lagrange_ppo, seed 1, 3M steps, λ_max ∈ {1, 2, 5}; deterministic evaluation on the 400 test instances.
- Effective gap: a violating instance counts as AR = 0, 1 − mean AR / mean ILP AR.
- Rule (violations first): highest mean violation-free share over the scenarios; within 0.5 pp of it is a tie, broken by the lower mean effective gap. A 3M-step sweep says little about the gap; violations weigh more.
- **Chosen λ_max = 2**

| λ_max | LT eff. gap / violation-free / gap / final λ | EQ eff. gap / violation-free / gap / final λ | GT eff. gap / violation-free / gap / final λ | mean eff. gap | mean violation-free |
|---|---|---|---|---|---|
| 1 | 16.8% / 94.5% / 12.0% / 1.000 | 11.6% / 98.2% / 10.0% / 1.000 | 10.8% / 99.8% / 10.5% / 0.999 | 13.1% | 97.5% |
| 2 | 16.7% / 96.2% / 13.5% / 2.000 | 11.1% / 98.8% / 10.1% / 1.999 | 11.6% / 100.0% / 11.6% / 1.997 | 13.1% | 98.3% |
| 5 | 22.3% / 95.0% / 18.1% / 5.000 | 13.6% / 99.8% / 13.3% / 4.998 | 11.0% / 100.0% / 11.0% / 4.995 | 15.6% | 98.2% |
