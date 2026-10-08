# v4.4.1 Random conflict density

Same as v4.3.8 (unified reward, full episode, Mask with EXIT, base observation, descending-demand order, the agent picks only the ECU); **only the data changes**. The v4.4.0 (service, ECU) action is dropped.

- **Conflict density is no longer fixed**: every instance draws its own p ~ U[0,1) and is then generated as before (K = 10 conflict sets, each service joins each set with probability q, p = 1 − (1 − q²)^K). Each instance stores its `p` and the measured fraction of conflicting service pairs `conflict_ratio` (ρ); all binning uses ρ.
- **Only ILP-feasible instances are kept**: an infeasible draw is redrawn together with p. In LT most instances with ρ ≥ 0.8 are infeasible, so LT's p is skewed low (mean 0.420); EQ / GT are almost always feasible (mean ≈ 0.50). Feasibility against ρ is in `density_report.md` (10,000 extra draws, feasibility only).
- **ILP AR**: generated directly with Dinkelbach (`solve_ilp_max_ar`), the true AR optimum; `src/paper_rl/data.py` no longer calls the old `solve_ilp`. Regenerating the first 40 instances at p = 0.6 reproduces `data/v4.3.1.4` (recomputed) exactly.
- 2000 instances per scenario, `data/v4.4.1/{lt,eq,gt}.yaml`; split (80% train) and seeds as before. The data set is selected by the `DATA_VERSION` environment variable, default still v4.3.1.4.
- **Evaluation**: every instance is ILP-feasible, so the EXIT rate is the share of instances the policy turned into a dead end. AR and ILP AR are averaged over the same instances without EXIT; **absolute gap = ILP AR − AR and relative gap = (ILP AR − AR) / ILP AR are both reported**, overall and in 10 bins of ρ.

Run: `src/scripts/run_v4.4.1.py --pilot` (Mask PPO × 3 scenarios × seed 1 × 1M steps), report `pilot_report.md`; `src/scripts/density_v4.4.1.py` writes `density_report.md`.
