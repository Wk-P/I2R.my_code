# v6.0.1-lagrangian-test-50k v6.0.1-lagrangian models on the fixed v6.0.4 test set

- Generated 2026-10-27 15:47:16; script `src/scripts/eval_testset_v6_0_4.py`; took 4.4 min; CPU only, no training.
- **Models: the v6.0.1-lagrangian final models (20M steps)**, all 27 runs (Lagrangian seeds 1, 2, 3, as in v6.0.1-lagrangian itself), not retrained.
- Test set: `data/testset_v6.0.4/<scenario>.json` (v6.0.4-testset), 5000 ILP-optimal instances per scenario, shared by every seed; deterministic evaluation with each run's final-evaluation settings (Lagrangian: the λ and λ_max at the end of training).
- **gap** = 1 − mean(AR) / mean(ILP AR) over all test instances, AR over legal placements (as the panel and results.json); **own final eval** = the run's own final evaluation on the fixed test set `data/testset_v6.0.4/{scen}.json` (5000 instances, shared by every seed, the same instances as here) (same definition).
- **Violation-free instances**: ILP AR − AR (absolute) and (ILP AR − AR) / ILP AR (relative), over the instances completed without violation. EXIT exists for Mask only.
- Mean ± std over seeds.

## EQ

| Mechanism | Algo | gap (5000 instances) | gap (own final eval) | violation-free | violation-free abs. gap | violation-free rel. gap | cap viol | privacy viol | EXIT |
|---|---|---|---|---|---|---|---|---|---|
| Lagrangian | PPO | 5.547% ± 0.507% | 5.547% ± 0.507% | 99.993% ± 0.012% | 4.784% ± 0.437% | 5.547% ± 0.507% | 0.007% ± 0.012% | 0.000% ± 0.000% | — |
| Lagrangian | DQN | 10.871% ± 5.184% | 10.870% ± 5.183% | 99.993% ± 0.012% | 9.374% ± 4.470% | 10.870% ± 5.183% | 0.007% ± 0.012% | 0.000% ± 0.000% | — |
| Lagrangian | DDQN | 13.614% ± 1.495% | 13.614% ± 1.497% | 99.980% ± 0.000% | 11.740% ± 1.290% | 13.613% ± 1.496% | 0.013% ± 0.012% | 0.007% ± 0.012% | — |

## GT

| Mechanism | Algo | gap (5000 instances) | gap (own final eval) | violation-free | violation-free abs. gap | violation-free rel. gap | cap viol | privacy viol | EXIT |
|---|---|---|---|---|---|---|---|---|---|
| Lagrangian | PPO | 6.859% ± 2.266% | 6.859% ± 2.266% | 99.953% ± 0.031% | 6.211% ± 2.051% | 6.860% ± 2.265% | 0.027% ± 0.046% | 0.020% ± 0.020% | — |
| Lagrangian | DQN | 8.275% ± 4.932% | 8.275% ± 4.933% | 100.000% ± 0.000% | 7.491% ± 4.465% | 8.275% ± 4.932% | 0.000% ± 0.000% | 0.000% ± 0.000% | — |
| Lagrangian | DDQN | 5.029% ± 0.845% | 5.029% ± 0.845% | 100.000% ± 0.000% | 4.552% ± 0.765% | 5.029% ± 0.845% | 0.000% ± 0.000% | 0.000% ± 0.000% | — |

## LT

| Mechanism | Algo | gap (5000 instances) | gap (own final eval) | violation-free | violation-free abs. gap | violation-free rel. gap | cap viol | privacy viol | EXIT |
|---|---|---|---|---|---|---|---|---|---|
| Lagrangian | PPO | 6.322% ± 0.360% | 6.322% ± 0.360% | 98.127% ± 0.142% | 5.580% ± 0.322% | 6.332% ± 0.365% | 1.153% ± 0.153% | 0.913% ± 0.129% | — |
| Lagrangian | DQN | 18.371% ± 2.302% | 18.371% ± 2.300% | 96.393% ± 0.863% | 16.434% ± 2.117% | 18.649% ± 2.410% | 1.973% ± 0.713% | 1.853% ± 0.253% | — |
| Lagrangian | DDQN | 22.875% ± 2.317% | 22.874% ± 2.318% | 93.693% ± 1.681% | 20.797% ± 2.303% | 23.568% ± 2.591% | 1.707% ± 0.867% | 5.133% ± 2.264% | — |
