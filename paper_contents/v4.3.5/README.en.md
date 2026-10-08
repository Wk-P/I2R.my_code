# v4.3.5 — AR directly as the reward

New minor version (tag `v4.3.5`).

## 1. Motivation

The legacy reward used since v4.3.1.6 is not strictly consistent with the optimization objective (maximize AR under hard constraints, i.e. the ILP's objective):

- success branch M(2AR − 1): negative when AR < 0.5, possibly below the −1 of "failing at the last step";
- failure branch −M(1 − valid/M) gives partial credit for "how many services were placed", which has no counterpart in the ILP objective;
- γ = 0.99: dead-end episodes are shorter and discounted less, so the return does not equal the objective value.

## 2. Change (only variable: the reward)

$$r_t = 0\ (t<T),\qquad r_T = AR\cdot\mathbb{1}\{\text{all } M \text{ services placed legally}\},\qquad \gamma = 1$$

The episode return is exactly $AR\cdot\mathbb{1}\{\text{feasible}\}$, $J(\pi)=\mathbb{E}[AR\cdot\mathbb{1}\{\text{feasible}\}]$.

- `src/paper_rl/env.py`: new `reward_mode="ar_raw"`. Success gets AR, failure (violation or dead end, including Repair with no ECU to repair to) gets 0; no scaling by M. Lagrangian still subtracts $\lambda\sum_t c_t$ at the end.
- A failed episode cannot get the raw AR: the AR of what was placed before a dead end is often high, which would encourage walking into dead ends early (the same problem as the stacking in v4.3.1.4).
- `src/paper_rl/train.py`: new `--gamma` (default PPO_GAMMA / DQN_GAMMA from config, i.e. the old behaviour), written to the `gamma` field of `results.json`; all learners use γ = 1 in this version.
- Everything else exactly as in v4.3.1.6: base observation, services in descending demand, 12 models, other hyper-parameters, 5M steps, seeds 1–3, p = 0.6 data and splits, evaluation (one deterministic output per test instance).
- `src/scripts/run_v4.3.5.py`: 108 jobs; the report compares legacy (v4.3.1.6, re-evaluated in v4.3.1.7), AR (this version) and greedy.

## 3. Historical evidence and known risks

- Intermediate steps always 0: the step-wise shaping of v0.3.1 and v4.1.0.0–.4 was all ineffective or harmful.
- The closest earlier variant is `ar` of v4.3.1.3 (success M·AR, failure −M(1−valid/M), p = 0.3, single seed, 1M steps), slightly better than legacy but not adopted.
- Without M the reward lies in [0, 1]: PPO normalizes advantages and is barely affected by the scale; the DQN family regresses absolute returns with a Huber threshold of 1, and in v4.3.1.8 dividing by M made Mask / Repair DQN worse (LT Repair success rate 0.80 → 0.60); this may repeat here and will be reported as is.
- Failure is uniformly 0 with no partial credit: the unconstrained / Lagrangian DQN models may find it harder to learn feasibility.
- Expectation: every earlier reward variant stopped at a similar AR; the main point of this version is that the training objective matches the paper's objective by definition. The service placement order (2026-10-06 greedy diagnosis: the order changes AR by 0.02–0.11) is left to a later version.

## 4. Self-check

- 3 scenarios × 4 mechanisms × 150 random episodes (1800 in total, 756 successful): all intermediate rewards are 0; terminal rewards match $AR\cdot\mathbb{1}\{\text{feasible}\}-\lambda\sum c_t$ (Lagrangian only) with 0 mismatches.
- On EQ, Mask-PPO / Lagrange-PPO / Mask-DQN / Repair-DDQN each ran 30k steps through training, evaluation and saving; `results.json` contains `reward_mode=ar_raw`, `gamma=1.0`.
- ILP benchmark re-check (2026-10-06): an independent model with scipy milp (HiGHS) (enumerating the number of active ECUs k, AR* = max_k obj_k / k), without the project's ILP code, 60 random instances per scenario; difference from the stored AR* ≤ 5e-7 (stored values rounded to 6 digits), and every solution checked feasible by independent code.

## 5. Results

**Not completed**: stopped on 2026-10-06 15:03 by the user's decision (about 1 hour, 0 jobs finished). Reason: giving 0 on failure does not meet the professor's comment "a penalty must be negative"; superseded by v4.3.6 (failure −(1−valid/M)).
