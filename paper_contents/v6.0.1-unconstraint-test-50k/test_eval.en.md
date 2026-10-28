# v6.0.1-unconstraint-test-50k v6.0.1-unconstraint models on the fixed v6.0.4 test set

- Generated 2026-10-29 00:02:21; script `src/scripts/eval_testset_v6_0_4.py`; took 1.6 min; CPU only, no training.
- **Models: the v6.0.1-unconstraint final models (20M steps)**, all 9 runs (Penalty seeds 1, as in v6.0.1-unconstraint itself), not retrained.
- Test set: `data/testset_v6.0.4/<scenario>.json` (v6.0.4-testset), 5000 ILP-optimal instances per scenario, shared by every seed; deterministic evaluation with each run's final-evaluation settings (Lagrangian: the λ and λ_max at the end of training).
- **gap** = 1 − mean(AR) / mean(ILP AR) over all test instances, AR over legal placements (as the panel and results.json); **own final eval** = the run's own final evaluation on its seed's 400 test instances (same definition).
- **Violation-free instances**: ILP AR − AR (absolute) and (ILP AR − AR) / ILP AR (relative), over the instances completed without violation. EXIT exists for Mask only.
- Mean ± std over seeds.

## EQ

| Mechanism | Algo | gap (5000 instances) | gap (own final eval) | violation-free | violation-free abs. gap | violation-free rel. gap | cap viol | privacy viol | EXIT |
|---|---|---|---|---|---|---|---|---|---|
| Penalty | PPO | 5.943% | 5.552% | 99.940% | 5.110% | 5.926% | 0.060% | 0.000% | — |
| Penalty | DQN | 32.022% | 31.889% | 98.640% | 27.620% | 32.001% | 0.620% | 1.000% | — |
| Penalty | DDQN | 29.113% | 29.141% | 99.580% | 25.132% | 29.128% | 0.000% | 0.420% | — |

## GT

| Mechanism | Algo | gap (5000 instances) | gap (own final eval) | violation-free | violation-free abs. gap | violation-free rel. gap | cap viol | privacy viol | EXIT |
|---|---|---|---|---|---|---|---|---|---|
| Penalty | PPO | 5.990% | 6.293% | 100.000% | 5.423% | 5.990% | 0.000% | 0.000% | — |
| Penalty | DQN | 18.670% | 18.910% | 88.300% | 17.222% | 18.786% | 0.040% | 11.700% | — |
| Penalty | DDQN | 15.610% | 16.170% | 64.540% | 14.183% | 15.134% | 0.720% | 35.380% | — |

## LT

| Mechanism | Algo | gap (5000 instances) | gap (own final eval) | violation-free | violation-free abs. gap | violation-free rel. gap | cap viol | privacy viol | EXIT |
|---|---|---|---|---|---|---|---|---|---|
| Penalty | PPO | 7.101% | 7.404% | 97.960% | 6.254% | 7.094% | 0.780% | 1.720% | — |
| Penalty | DQN | 27.806% | 28.276% | 7.780% | 24.006% | 25.748% | 4.480% | 92.140% | — |
| Penalty | DDQN | 31.232% | 31.577% | 14.360% | 23.953% | 26.297% | 7.200% | 85.300% | — |
