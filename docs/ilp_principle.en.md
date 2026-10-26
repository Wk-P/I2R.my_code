# How ILP AR is computed

This note explains where the **ILP AR** in the reports comes from: the optimisation problem, the exact solution method (Dinkelbach iteration), instance generation and filtering, and how ILP AR and the gap are computed in the reports. Implementation: `src/shared/ilp_utils.py` (`solve_ilp_max_ar`) and `src/paper_rl/data.py`.

## 1. Problem

Place M services on N ECUs; every service goes to exactly one ECU.

| Symbol | Meaning |
|---|---|
| c_n | capacity of ECU n (containers), from {50, 55, …, 195} |
| e_m | demand of service m (containers), from {10, 15, …, 95} |
| C_k | k-th privacy conflict set (K = 10); services of one set may not share an ECU |
| x_mn ∈ {0,1} | service m is placed on ECU n |
| y_n ∈ {0,1} | ECU n is used |

Scenario sizes: EQ N=10, M=10; GT N=15, M=10; LT N=10, M=15.

**Objective AR (average ECU utilisation)**:

    AR = F(x) / G(y),  F(x) = Σ_m Σ_n x_mn · e_m / c_n,  G(y) = Σ_n y_n

i.e. the sum of the utilisations of the used ECUs divided by the number of used ECUs.

**Constraints**:

1. Every service placed exactly once: Σ_n x_mn = 1, ∀m
2. Capacity: Σ_m x_mn · e_m ≤ c_n · y_n, ∀n (an unused ECU hosts nothing)
3. Activation: x_mn ≤ y_n, ∀m, n
4. Privacy conflict: Σ_{m∈C_k} x_mn ≤ 1, ∀k, ∀n

Constraint 4 allows at most one member of each conflict set per ECU, i.e. no two services of the same set share an ECU.

## 2. Why a single ILP is not enough

AR is a ratio of two linear expressions (a fractional objective), so it cannot be handed to an ILP solver directly.

The old `solve_ilp` maximised only the numerator F (total utilisation) and divided by the number of used ECUs afterwards. Since every service must be placed, F is driven by putting services on small ECUs and does not try to use fewer ECUs, so the result is not the AR optimum; on the v4.3.1.4 data it is 0.07–0.12 below it. Since v4.3.1.7 only `solve_ilp_max_ar` is used, and the ILP AR field of the data sets (`ar_star`) was recomputed with `src/scripts/recompute_ar_star.py`.

## 3. Dinkelbach iteration (exact maximum AR)

For a parameter λ define the parametric ILP

    φ(λ) = max over feasible (x, y) of  F(x) − λ · G(y)

Key property, with λ_opt the optimal AR:

- φ(λ) > 0 for λ < λ_opt, φ(λ_opt) = 0, φ(λ) < 0 for λ > λ_opt.
- Since G ≥ 1, φ is strictly decreasing in λ.

Iteration:

1. λ₀ = 0.
2. Solve φ(λ_t) with CBC (PuLP's default MILP solver) to get a feasible (x_t, y_t).
3. Set λ_{t+1} = F(x_t) / G(y_t), the actual AR of that solution.
4. Stop when |λ_{t+1} − λ_t| < 10⁻⁶ and return λ_{t+1} with its placement; otherwise go to 2 (at most 30 rounds).

Every round yields a feasible solution, so λ_t never decreases; it stops changing only when no feasible solution has a higher AR, where φ(λ) = 0 and λ is the global optimum AR. The number of rounds is finite (there are finitely many feasible solutions) and in practice small. Each subproblem is solved to proven optimality by CBC (status `Optimal`), so the result is the exact optimum, not a heuristic.

## 4. Instance generation and filtering

- ECU capacities and service demands are drawn independently and uniformly from the sets above.
- Conflict sets: K = 10 sets, every service joins each set independently with probability q, sets with fewer than 2 members are dropped. Two services conflict with probability p = 1 − (1 − q²)^K; the main experiments use p = 0.6.
- Every instance is solved with `solve_ilp_max_ar`; if the status is not `Optimal` (no feasible placement) the whole instance is redrawn. Hence **every instance in the data has a placement satisfying all constraints, and its optimal AR is known**.

Data used:

| Use | Source | Count |
|---|---|---|
| Training | `results/sup_diag/<scenario>.json` (same generator, p = 0.6, all solved to Optimal by the ILP) | 50,000 per scenario |
| Test | 2000 instances of `data/v4.3.1.4/<scenario>.yaml`, shuffled by the seed, last 20% | 400 per seed |

Statistics over all 2000 instances of `data/v4.3.1.4`:

| Scenario | Mean ILP AR | Mean ECUs used by the optimum |
|---|---|---|
| EQ | 0.858 | 5.88 |
| GT | 0.910 | 5.91 |
| LT | 0.880 | 7.96 |

The test split depends on the seed, so ILP AR differs slightly between seeds of one scenario (e.g. EQ: 0.860 for seed 2, 0.858 for seed 3).

## 5. ILP AR and gap in the reports

For the 400 test instances of a run:

    ILP AR = mean optimal AR over the 400 instances
    AR     = mean AR of the RL policy on the same instances
    gap    = 1 − AR / ILP AR

- RL's AR counts legal placements only: a placement that exceeds capacity or breaks a privacy conflict adds no utilisation and does not count as a used ECU; services left unplaced after a Mask EXIT are not counted either. RL's AR and ILP AR are therefore measured under the same constraints and are directly comparable.
- The report tables give mean ± std of the gap over seeds.
- ILP AR is the maximum reachable on each instance under all constraints, so a violation-free RL policy cannot exceed it on any single instance.

## 6. The unconstrained objective AR_obj is not comparable with ILP AR

The v6.0.1 unconstrained reward AR_obj also counts violating placements and clips each ECU's utilisation at 1. Without constraints, packing services onto a few ECUs saturates every ECU and AR_obj → 1. It is not bound by constraints 2 and 4 of Section 1, so it is a different optimisation problem and an AR_obj above ILP AR means nothing.
