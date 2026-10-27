# v6.0.0.2-test-50k v6.0.0.2 models on the fixed v6.0.4 test set

- Generated 2026-10-27 15:47:16; script `src/scripts/eval_testset_v6_0_4.py`; took 1.9 min; CPU only, no training.
- **Models: the v6.0.0.2 final models (20M steps)**, all 9 runs (Mask seeds 1, 2, 3, as in v6.0.0.2 itself), not retrained.
- Test set: `data/testset_v6.0.4/<scenario>.json` (v6.0.4-testset), 5000 ILP-optimal instances per scenario, shared by every seed; deterministic evaluation with each run's final-evaluation settings (Lagrangian: the λ and λ_max at the end of training).
- **gap** = 1 − mean(AR) / mean(ILP AR) over all test instances, AR over legal placements (as the panel and results.json); **own final eval** = the run's own final evaluation on its seed's 400 test instances (same definition).
- **Violation-free instances**: ILP AR − AR (absolute) and (ILP AR − AR) / ILP AR (relative), over the instances completed without violation. EXIT exists for Mask only.
- Mean ± std over seeds.

## EQ

| Mechanism | Algo | gap (5000 instances) | gap (own final eval) | violation-free | violation-free abs. gap | violation-free rel. gap | cap viol | privacy viol | EXIT |
|---|---|---|---|---|---|---|---|---|---|
| Mask | PPO | 3.741% ± 0.106% | 3.587% ± 0.168% | 99.987% ± 0.012% | 3.225% ± 0.090% | 3.740% ± 0.105% | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.013% ± 0.012% |

## GT

| Mechanism | Algo | gap (5000 instances) | gap (own final eval) | violation-free | violation-free abs. gap | violation-free rel. gap | cap viol | privacy viol | EXIT |
|---|---|---|---|---|---|---|---|---|---|
| Mask | PPO | 3.469% ± 0.620% | 3.360% ± 0.766% | 100.000% ± 0.000% | 3.141% ± 0.562% | 3.469% ± 0.620% | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.000% ± 0.000% |

## LT

| Mechanism | Algo | gap (5000 instances) | gap (own final eval) | violation-free | violation-free abs. gap | violation-free rel. gap | cap viol | privacy viol | EXIT |
|---|---|---|---|---|---|---|---|---|---|
| Mask | PPO | 6.553% ± 0.126% | 6.665% ± 0.126% | 98.640% ± 0.191% | 5.747% ± 0.115% | 6.521% ± 0.132% | 0.000% ± 0.000% | 0.000% ± 0.000% | 1.360% ± 0.191% |
