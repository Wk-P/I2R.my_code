# v4.2.0 — Collection of the best models (final paper results)

v4.2.0 does not retrain. It picks, for each (scenario, algorithm) cell, the batch of trained models with the best results, forms a mixed collection, re-evaluates it under one evaluation protocol, and measures ILP vs RL solving time.

## Evaluation protocol

- Each test instance is run once with deterministic actions (`EVAL_BEST_OF_N = 1`, `deterministic=True`). best-of-8 is fully disabled (see "Unified evaluation protocol" in `../v4.1.0_changelog.md`).
- Every model is evaluated on the test set of its own seed (400 instances); table values are the mean ± sample std over seeds (ddof=1).
- Training-code versions differ only in the reward; the observations and transitions used in evaluation are identical, so everything is evaluated with the current `scenarios/` code.

## Model selection (manifest.json)

| Algorithm | LT | EQ | GT |
|---|---|---|---|
| PPO | v4.0.0 (5 seeds) | v4.0.0 (5) | v4.0.0 (5) |
| Mask-PPO | v4.0.0 (5) | v4.0.0 (5) | v4.0.0 (5) |
| Lagrange-PPO | v4.0.0 (5) | v4.1.0 (3) | v4.1.0.1 (3) |
| Repair-PPO | v4.1.0 (3) | v4.1.0 (3) | v4.1.0 (3) |
| DQN | v4.1.0 (3) | v4.1.0 (3) | v4.1.0 (3) |
| DDQN | v4.1.0 (3) | v4.1.0 (3) | v4.1.0 (3) |

Cells with only one 5M-step batch use that batch; cells with several candidates (Mask-PPO, Lagrange-PPO) take the batch with the highest success rate under the unified protocol. Each run's exp_id, seed, source directory and training-code version are in `manifest.json`; model copies are in `results/v4.2.0/<scen>/<algo>/<exp_id>/` (git-ignored, kept locally).

## Test-set results (`final_summary_data.json`)

| Scenario | Algorithm | Success rate | AR | Capacity violation | Conflict violation |
|---|---|---|---|---|---|
| LT | PPO | 0.3505±0.0198 | 0.6781 | 0.266 | 0.553 |
| LT | Mask-PPO | 0.6190±0.0291 | 0.6290 | 0 | 0 |
| LT | Lagrange-PPO | 0.5645±0.0257 | 0.6514 | 0.059 | 0.410 |
| LT | Repair-PPO | 0.5408±0.0146 | 0.6347 | 0 | 0 |
| LT | DQN | 0.5333±0.0138 | 0.6408 | 0.120 | 0.427 |
| LT | DDQN | 0.4650±0.0325 | 0.6464 | 0.113 | 0.489 |
| EQ | PPO | 0.8045±0.0227 | 0.5296 | 0.049 | 0.152 |
| EQ | Mask-PPO | 1.0000 | 0.5419 | 0 | 0 |
| EQ | Lagrange-PPO | 0.9292±0.0153 | 0.5438 | 0.013 | 0.058 |
| EQ | Repair-PPO | 1.0000 | 0.5388 | 0 | 0 |
| EQ | DQN | 0.7625±0.0246 | 0.5056 | 0.059 | 0.230 |
| EQ | DDQN | 0.7400±0.0434 | 0.5041 | 0.066 | 0.255 |
| GT | PPO | 0.9350±0.0138 | 0.5863 | 0.043 | 0.022 |
| GT | Mask-PPO | 1.0000 | 0.5889 | 0 | 0 |
| GT | Lagrange-PPO | 0.9617±0.0080 | 0.5842 | 0.022 | 0.017 |
| GT | Repair-PPO | 1.0000 | 0.5764 | 0 | 0 |
| GT | DQN | 0.9075±0.0373 | 0.5684 | 0.017 | 0.076 |
| GT | DDQN | 0.9250±0.0338 | 0.5659 | 0.017 | 0.059 |

ILP-optimal AR (seed-1 test set): LT 0.7412, EQ 0.5461, GT 0.6237. Violation columns are "the share of test episodes with ≥1 violation among the executed placements"; Repair-PPO is 0 by construction, and its repair-trigger rate is stored separately in the `repair_trigger_*` fields.

## Solving time (ms / test instance, single-thread CPU)

| Method | LT (M=15) | EQ (M=10) | GT (M=10) |
|---|---|---|---|
| ILP (Dinkelbach + CBC) | 180.7 ± 213.6 | 45.0 ± 7.3 | 60.3 ± 4.6 |
| PPO | 8.86 (20.4×) | 6.06 (7.4×) | 5.98 (10.1×) |
| Mask-PPO | 17.89 (10.1×) | 10.70 (4.2×) | 10.10 (6.0×) |
| Lagrange-PPO | 10.12 (17.9×) | 6.29 (7.2×) | 6.56 (9.2×) |
| Repair-PPO | 10.02 (18.0×) | 6.79 (6.6×) | 7.13 (8.5×) |
| DQN | 6.67 (27.1×) | 4.39 (10.3×) | 4.49 (13.4×) |
| DDQN | 6.70 (27.0×) | 4.44 (10.2×) | 4.46 (13.5×) |

Numbers in parentheses are speed-ups relative to the ILP. ILP: the 400 seed-1 test instances solved one by one, mean ± std across instances. RL: each seed's model runs its own 400 test instances to get the mean time per episode, then averaged over seeds; timing includes environment computation (reset/step/action masks) and network inference, starting after 5 warm-up episodes.

## Files

- `formula.md`: MDP, reward, learning algorithm, hyper-parameters, evaluation and timing formulas for every cell of v4.2.0
- `各算法伪代码.md` / `Algorithm-Pseudocode-EN.md`: pseudocode in Chinese and English
- `manifest.json`: model list
- `final_summary_data.json`: final data (including timing)
- `figs/`: `{ar,success,violation,timing}_{lt,eq,gt}.png`, `summary_table.png`, `timing_table.png`
- `tools/`: `build_v4_2_0.py` (pick models + copy + list), `run_eval.py` (evaluation + timing + summary), `eval_model.py`, `ilp_timing.py`, `make_figs.py`

Reproduce:

```
python paper_contents/v4.2.0/tools/build_v4_2_0.py
python paper_contents/v4.2.0/tools/run_eval.py
python paper_contents/v4.2.0/tools/make_figs.py
```

Success rate / AR / violation rates are deterministic and reproduce bit for bit; timing fluctuates by a few percent with machine load.

## Main differences from the v4.1.0.2 docs

The formulas and pseudocode in `../v4.1.0.2/` describe the final v4.1.0.2 code, not the models of v4.2.0. The v4.2.0 docs take precedence; main differences:

- Lagrange-PPO's EQ/GT models had λ in the step-wise reward during training (EQ: utilization gain + capacity penalty + λ conflict penalty; GT: capacity penalty + λ conflict penalty); only LT has zero step-wise reward.
- λ is updated from the combined capacity and conflict violation rate, with different hyper-parameters in the three scenarios.
- Repair-PPO LT model: if a repair was triggered the terminal reward is 0; EQ/GT always run the whole episode with M(2AR−1).
- Mask-PPO's one-step look-ahead and curriculum weight w exist only on LT; with no legal ECU the mask is all 0 and the action is uniform over the N ECUs.
- PrunedPPO's negative-advantage weight is 1, i.e. standard PPO.

## To do: retrain DQN/DDQN on GT

The formula is designed as a single-step penalty $r_t = R_t - P_t$, $P_t = 2v^{\text{cap}}_t + 2v^{\text{conf}}_t$, identical in all three scenarios. But the current GT DQN/DDQN models in v4.2.0 (v4.1.0 code) terminate the episode with reward −M on a capacity violation, which is inconsistent with the formula. `step()` in `scenarios/gt/{dqn,ddqn}/env.py` must be changed to the same single-step penalty logic as eq, then retrained (3 seeds × 5M steps × 2 algorithms) to replace the data of these two GT cells.
