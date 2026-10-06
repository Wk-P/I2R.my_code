# v4.3.4 — Conflict-aware observation (state ablation)

New minor version (tag `v4.3.4`).

## 1. Motivation

- The greedy baseline and v4.3.3's best-of-K show: RL clearly raises the success rate on LT, but the AR of successful solutions is on par with the heuristic; on EQ/GT, with the same time budget, randomized greedy reaches a higher AR.
- The observation of v4.3.1.x (87 / 77 / 102 dimensions) does not contain the conflict relations between services. Conflict pairs occur with probability p = 0.6: about 63/105 pairs conflict on LT and 27/45 on EQ/GT; optimizing AR is largely "how to group the conflict graph"; the ILP sees the full conflict graph, while RL only sees a compressed summary (whether the current service conflicts with each ECU, the share of services each ECU can still accept, the share of feasible ECUs for each remaining service).
- Two instances can have the same current observation but different future conflict structures and different optimal actions, so the problem is effectively a POMDP. This version tests: **does the missing raw conflict information limit the quality of the RL policy?**

## 2. Changes (only variable: the observation)

- `paper_rl/env.py`: new `obs_mode` (`base` = the observation of v4.3.3 and earlier, `conflict` = this version). `conflict` appends the conflict graph among not-yet-placed services to the original observation: the upper triangle in placement order, fixed length M(M−1)/2 (LT 105, EQ/GT 45 dimensions); 1 if a pair conflicts and both are unplaced (including the current service), else 0. No hand-crafted features (conflict degree, blocking score, etc.) and no choice of any heuristic.
- The original part of the observation is unchanged at every step (checked at every step of 30 instances per scenario); the position of the action mask in the observation is unchanged (MaskableDQN depends on it).
- `paper_rl/train.py`: new `--obs {base, conflict}` (default `base`, as before), written to the `obs` field of `results.json`; the evaluation log now prints AR, ILP AR and AR gap over successful instances.
- Everything else exactly as in v4.3.1.6: shared base reward (legacy), 12 models, hyper-parameters, 5M steps, seeds 1–3, data and splits, evaluation (one deterministic output per test instance).
- `scripts/run_v4.3.4.py`: 108 jobs; the report compares the "base observation" (v4.3.1.6, re-evaluated in v4.3.1.7), the "conflict observation" (this version) and greedy.

## 3. Schedule

Started after v4.3.1.10 (10M) finished, with the machine to itself. While v4.3.1.10 was running the code was kept as a patch (`scripts/pending/v4.3.4_obs_conflict.patch`), so that all v4.3.1.10 jobs ran on their own commit; `scripts/pending/after_v4.3.1.10.sh` applied the patch after v4.3.1.10 finished, committed and tagged, regenerated the v4.3.1.10 report, then started this version.

## 4. Read together with v4.3.1.10

| v4.3.1.10 (10M) | v4.3.4 (conflict observation) | Meaning |
|---|---|---|
| AR up | AR up | both training amount and observation are bottlenecks |
| AR up | flat | mainly undertrained |
| flat | AR up | missing observation information is the main bottleneck |
| flat | flat | look at reward (credit assignment) / network / optimization |

## 5. Results

See `report.md` (generated when finished).
