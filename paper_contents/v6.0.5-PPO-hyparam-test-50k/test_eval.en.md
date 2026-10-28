# v6.0.5-PPO-hyparam-test-50k v6.0.5-PPO-hyparam models on the fixed v6.0.4 test set

- Generated 2026-10-28 23:27:52; script `src/scripts/eval_testset_v6_0_4.py`; took 6.1 min; CPU only, no training.
- **Models: the v6.0.5-PPO-hyparam final models (20M steps)**, all 36 runs (Mask seeds 1, 2, 3, as in v6.0.5-PPO-hyparam itself), not retrained.
- Test set: `data/testset_v6.0.4/<scenario>.json` (v6.0.4-testset), 5000 ILP-optimal instances per scenario, shared by every seed; deterministic evaluation with each run's final-evaluation settings (Lagrangian: the λ and λ_max at the end of training).
- **gap** = 1 − mean(AR) / mean(ILP AR) over all test instances, AR over legal placements (as the panel and results.json); **own final eval** = the run's own final evaluation on the fixed test set `data/testset_v6.0.4/{scen}.json` (5000 instances, shared by every seed, the same instances as here) (same definition).
- **Violation-free instances**: ILP AR − AR (absolute) and (ILP AR − AR) / ILP AR (relative), over the instances completed without violation. EXIT exists for Mask only.
- Mean ± std over seeds.

## EQ

| Mechanism | Algo | gap (5000 instances) | gap (own final eval) | violation-free | violation-free abs. gap | violation-free rel. gap | cap viol | privacy viol | EXIT |
|---|---|---|---|---|---|---|---|---|---|
| Mask | PPO | 4.873% ± 1.034% | 4.873% ± 1.034% | 99.993% ± 0.010% | 4.202% ± 0.892% | 4.872% ± 1.035% | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.007% ± 0.010% |

## GT

| Mechanism | Algo | gap (5000 instances) | gap (own final eval) | violation-free | violation-free abs. gap | violation-free rel. gap | cap viol | privacy viol | EXIT |
|---|---|---|---|---|---|---|---|---|---|
| Mask | PPO | 4.502% ± 1.076% | 4.502% ± 1.076% | 100.000% ± 0.000% | 4.076% ± 0.974% | 4.502% ± 1.076% | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.000% ± 0.000% |

## LT

| Mechanism | Algo | gap (5000 instances) | gap (own final eval) | violation-free | violation-free abs. gap | violation-free rel. gap | cap viol | privacy viol | EXIT |
|---|---|---|---|---|---|---|---|---|---|
| Mask | PPO | 7.506% ± 1.001% | 7.506% ± 1.001% | 98.427% ± 0.244% | 6.595% ± 0.893% | 7.483% ± 1.013% | 0.000% ± 0.000% | 0.000% ± 0.000% | 1.573% ± 0.244% |
