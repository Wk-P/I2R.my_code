# Experimental Results and Analysis

> Basis: main table v6.0.0-merge (`experiments/table_2d.json`, original in `paper_contents/v6.0.0-merge/`), fixed test set `data/testset_v6.0.4` (5000 instances per scenario), 3 seeds. Timing from `experiments/time_full.json`, original in `paper_contents/ilp_vs_saferl/` (2026-11-07, all 3×5000 instances timed). Violation breakdowns are computed from the same runs selected by `src/scripts/merge_2d_table.py`. This file gives the setup, the data and the analysis points; the prose is left to the author.

## 1. Experimental Setup

### Table 1  Problem and data

| Item | Setting |
|---|---|
| Scenarios (N ECUs, M services) | LT (10, 15); EQ (10, 10); GT (15, 10) |
| Capacity c_n / demand e_m | {50, 55, …, 195} / {10, 15, …, 95}, uniform |
| Privacy conflicts | K = 10 conflict sets; each service joins each set independently with probability q, two services share at least one set with probability p = 0.6 |
| Feasibility | every instance is solved to optimality by the ILP (Optimal) |
| Train / test | 50,000 / 5,000 instances per scenario, disjoint; the test set is fixed and shared by all methods |
| Baseline | ILP: Dinkelbach iteration (tolerance 1e-6, 3.06 iterations on average), PuLP + CBC, single thread |

### Table 2  Compared methods (4 mechanisms × 3 learners = 12 variants)

| Mechanism | Label | Constraint handling |
|---|---|---|
| Unconstraint | -U | none; violations only show up in the terminal reward |
| Mask | -M | infeasible ECUs are masked; at a dead end only EXIT is allowed |
| Repair | -R | an infeasible action is replaced by capacity-based best-fit (the feasible ECU with the smallest total capacity) |
| Lagrangian | -L | each step's reward is reduced by λ·c_t; λ is tuned by dual ascent |

The learners are PPO, DQN and DDQN; variants are labelled PPO-M, DQN-R, and so on.

### Table 3  MDP and network

| Item | Setting |
|---|---|
| Placement order | services in descending order of demand; an episode has M steps |
| Action | Discrete(N); Mask adds EXIT, Discrete(N+1) |
| Observation | c, e, the M×M conflict matrix, the current assignment and step t (Lagrangian also gets λ/(1+λ)) |
| Reward (U / M / R: ar_pen) | 0 at intermediate steps; at the end, the AR if every service was placed legally, otherwise −(fraction of services not placed legally) |
| Reward (L: ar_obj − λ·c_t) | the terminal reward ar_obj is the mean load ratio over the ECUs of all executed placements, violating ones included, with no failure penalty; violations are penalised only by −λ·c_t at each step, where c_t = capacity + privacy violations of that step |
| Episode | γ = 1, no early termination, a dead end counts as a violation |
| Network | relation-aware Transformer: d = 128, 4 heads, 3 layers, FFN 256, Pre-LN, 4 relation biases, 450,227 parameters |

### Table 4  Training hyperparameters

| | PPO | DQN / DDQN |
|---|---|---|
| Parallel envs | 40 | 12 |
| Learning rate | 3e-4 | 1e-3 |
| Batch | 20,480 steps per update, minibatch 256 | replay buffer 100k, batch 64 |
| Updates | n_epochs: 6 for PPO-M, 10 for the others | one gradient step per 48 new transitions |
| Other | GAE 0.95, clip 0.1, entropy coefficient 0.005 | ε decays linearly from 1 to 0 over the first 2M steps; hard target sync about every 492 steps |
| Common | 20M steps, seeds 1/2/3, each episode draws a random training instance | |
| Lagrangian | λ₀ = 0.001, Adam lr 0.035, target d = 0.001 violations per episode, λ ≥ 0 with no upper bound; updated once per rollout for PPO and once every 1000 episodes for DQN | |
| Software / hardware | Python 3.12, PyTorch 2.5.1, SB3 / sb3-contrib 2.7.1, Gymnasium 1.2.3; training on GTX 1080 Ti, timing on Xeon E5-2660 v4 | |

### Table 4b  Provenance (traced back from the main table through each run's run.json)

| Variant | Training batch | Commit | Seeds |
|---|---|---|---|
| PPO-M | v6.0.5-PPO-hyparam (`--n-epochs 6`) | 1a344ea | 1–3 |
| DQN-M / DDQN-M | v6.0.0 | cbf08e9 | 1–3 |
| PPO/DQN/DDQN-R | v6.0.0 | cbf08e9 (seed 1) / 4445cf9 (seeds 2, 3) | 1–3 |
| PPO/DQN/DDQN-L | v6.0.1-lagrangian (`--reward ar_obj`) | 7c165b5 | 1–3 |
| PPO/DQN/DDQN-U | v6.0.0 (seeds 2, 3) + v6.0.1-unconstraint (seed 1) | 4445cf9 / f56abd3, 1a344ea | 1–3 |

108 runs in total, all with `--gamma 1.0 --full-episode --steps 20000000 --net graph` and training pool `results/sup_diag/{scen}.json`; every DQN/DDQN run uses `--eps-steps 2000000`. The commits of the Unconstraint seed-1 runs are all after f56abd3 ("Unconstraint back to the v6.0.0 definition, full ar_pen"), so the definition matches seeds 2 and 3.

## 2. Evaluation Metrics

- **Test mode**: deterministic inference (argmax at every step) with the final 20M-step model on the fixed test set, 5000 instances per scenario. Each scenario is first averaged over the 3 seeds; the "mean" column is the arithmetic mean over the three scenarios; ± is the standard deviation across seeds.
- **AR** counts legally placed services only.
- **AR gap**: absolute gap = ILP AR − AR; relative gap = 1 − mean(AR) / mean(ILP AR), **computed only on instances completed without violation** (and without EXIT for Mask), with the ILP AR of the same instances. Failed instances are reported separately by the violation and EXIT rates. "Gap" below means the relative gap unless stated otherwise.
- **Violation rates**: capacity and privacy, each reported as the share of instances with at least one violation of that type. Both types can occur in the same instance, so the total violation rate is at most their sum.
- **EXIT rate** (early termination): Mask only; the share of instances where the policy chose EXIT at a dead end.
- **Time cost**: every one of the 3×5000 test instances is timed individually, single CPU thread (24 processes in parallel). ILP time includes model construction and all Dinkelbach iterations; RL time is one full deterministic episode (reset, masking, forward pass, step), using each variant's seed-1 model. Speed-up = mean ILP time / mean RL time.

## 3. Performance Comparison with ILP

### Table 5  AR of PPO-M vs ILP

| Scenario | ILP AR | PPO-M AR | Absolute gap | Relative gap |
|---|---|---|---|---|
| LT | 0.8814 | 0.8270 | 0.0545 | 6.18% ± 0.32% |
| EQ | 0.8624 | 0.8310 | 0.0314 | 3.64% ± 0.21% |
| GT | 0.9053 | 0.8756 | 0.0297 | 3.28% ± 0.08% |
| Mean | 0.8830 | 0.8445 | 0.0385 | 4.36% ± 0.18% |

### Table 6  Time per instance (ms, mean; ILP median in parentheses)

| Method | LT | EQ | GT | Mean | Speed-up |
|---|---|---|---|---|---|
| ILP | 3285 (1301) | 515 (286) | 701 (417) | 1500 | 1× |
| PPO-M | 69.2 | 44.4 | 46.0 | 53.2 | 28.2× |
| DQN-M | 61.2 | 38.8 | 40.6 | 46.9 | 32.0× |
| DDQN-M | 59.9 | 38.8 | 40.9 | 46.5 | 32.2× |
| PPO-R | 65.1 | 41.8 | 43.1 | 50.0 | 30.0× |
| DQN-R | 61.7 | 38.4 | 41.0 | 47.0 | 31.9× |
| DDQN-R | 61.1 | 38.7 | 43.5 | 47.7 | 31.4× |
| PPO-L | 64.7 | 41.6 | 42.4 | 49.6 | 30.3× |
| DQN-L | 61.7 | 39.0 | 41.5 | 47.4 | 31.6× |
| DDQN-L | 61.2 | 38.9 | 41.0 | 47.0 | 31.9× |
| PPO-U | 64.5 | 41.7 | 43.3 | 49.8 | 30.1× |
| DQN-U | 62.4 | 38.7 | 40.8 | 47.3 | 31.7× |
| DDQN-U | 60.9 | 41.6 | 41.4 | 47.9 | 31.3× |

Per scenario, PPO-M's speed-up is 47.5× on LT, 11.6× on EQ and 15.3× on GT.

**Analysis points**
1. PPO-M has a three-scenario mean gap of 4.36% (seed std 0.18 pp), the lowest of the 12 variants, with zero capacity and zero privacy violations.
2. RL takes 46–53 ms per instance against 1500 ms for the ILP, a speed-up of 28–32×. The ILP reached Optimal on all 15,000 test instances.
3. ILP time grows sharply with the number of services M: on LT (M = 15) the mean is 6.4× that of EQ and 4.7× that of GT (LT and GT have the same N·M), with a heavy tail (mean 3285 ms vs median 1301 ms). RL time scales only with the number of steps M and tokens, so LT is only about 1.5× EQ, and the speed-up is largest on LT (about 48–55×).
4. Time differences between variants are small (≤ 7 ms); PPO is slightly slower because it also evaluates the value head.

## 4. Comparison of Constraint Handling Strategies

### Table 7  Three-scenario mean relative gap (mean ± seed std, lower is better)

| Mechanism \ Learner | PPO | DQN | DDQN | Row mean |
|---|---|---|---|---|
| Mask | **4.36% ± 0.18%** | 7.20% ± 1.40% | **6.42% ± 1.18%** | **5.99%** |
| Repair | 7.17% ± 0.28% | **6.55% ± 0.57%** | 7.45% ± 1.19% | 7.06% |
| Lagrangian | 6.25% ± 1.02% | 12.60% ± 1.39% | 14.07% ± 0.59% | 10.97% |
| Unconstraint | 6.11% ± 0.27% | 24.42% ± 4.80% | 22.07% ± 1.43% | 17.53% |
| Column mean | 5.97% | 12.69% | 12.50% | 10.39% |

(Bold marks the best value in each column.)

**Analysis points**
1. By row mean, the mechanisms rank Mask (5.99%) < Repair (7.06%) < Lagrangian (10.97%) < Unconstraint (17.53%). Only Mask has zero violations.
2. Best mechanism within each learner: Mask for PPO (4.36%), Mask for DDQN (6.42%), and Repair for DQN (6.55%, against 7.20% for Mask).
3. PPO is the least sensitive to the mechanism: all four lie between 4.4% and 7.2%, and PPO-U (6.11%) and PPO-L (6.25%) are both slightly better than PPO-R (7.17%). DQN/DDQN degrade sharply under Lagrangian and Unconstraint (12.6–24.4%): value-based learners handle constraints poorly when the only signal is the reward.
4. For DQN/DDQN, the hard mechanisms (Mask, Repair) have gaps about 5–18 pp lower than the soft ones (Lagrangian, Unconstraint).
5. Why Repair beats Mask on DQN is not yet settled (an open question in the current line of work).
6. Seed variance: PPO-M is the most stable (0.18 pp); DQN-U varies the most (4.80 pp).

## 5. Performance Across Placement Scenarios

### Table 8  Relative gap per scenario (violation-free instances; except DQN-U and DDQN-U, the ILP AR is 0.881–0.883 on LT, 0.862 on EQ, 0.905 on GT)

| Variant | LT | EQ | GT |
|---|---|---|---|
| PPO-M | **6.18%** | 3.64% | 3.28% |
| DQN-M | 13.64% | **3.54%** | 4.41% |
| DDQN-M | 10.57% | 3.65% | 5.05% |
| PPO-R | 8.61% | 6.85% | 6.04% |
| DQN-R | 12.64% | 3.61% | 3.41% |
| DDQN-R | 15.80% | 3.61% | **2.94%** |
| PPO-L | 6.33% | 5.55% | 6.86% |
| DQN-L | 18.65% | 10.87% | 8.27% |
| DDQN-L | 23.57% | 13.61% | 5.03% |
| PPO-U | 7.24% | 5.98% | 5.11% |
| DQN-U | 27.98% | 29.83% | 15.45% |
| DDQN-U | 25.45% | 21.69% | 19.08% |
| Mean of 12 variants | 14.72% | 9.37% | 7.08% |

**Analysis points**
1. Difficulty ranks LT > EQ > GT: the mean gap over the 12 variants is 14.72%, 9.37% and 7.08%. LT has more services than ECUs (M > N), so both capacity and conflicts are tightest there.
2. Each scenario has a different best variant: PPO-M on LT (6.18%), DQN-M on EQ (3.54%), DDQN-R on GT (2.94%). On EQ and GT the leading variants are within 1 pp of each other (EQ 3.54–3.65%, GT 2.94–3.41%).
3. LT is where the methods separate: PPO-M is 0.15 pp below the runner-up PPO-L and 7.46 pp below DQN-M, and every DQN/DDQN variant is above 10% on LT. PPO-M ranks first on the three-scenario mean mainly because of its clear lead on LT.
4. LT is also where the ILP is slowest (Table 6), so it is both the hardest scenario for RL and the one where the speed-up pays off most.

## 6. Constraint Violation and Early Termination Analysis

### Table 9  Violations and early termination on LT (share of instances, mean of 3 seeds)

| Variant | Completed without violation | Capacity viol. | Privacy viol. | EXIT |
|---|---|---|---|---|
| PPO-M | 98.62% | 0 | 0 | 1.38% |
| DQN-M | 96.13% | 0 | 0 | 3.87% |
| DDQN-M | 96.95% | 0 | 0 | 3.05% |
| PPO-R | 97.46% | 0.99% | 2.37% | — |
| DQN-R | 94.35% | 1.32% | 5.37% | — |
| DDQN-R | 93.41% | 1.82% | 6.21% | — |
| PPO-L | 98.13% | 1.15% | 0.91% | — |
| DQN-L | 96.39% | 1.97% | 1.85% | — |
| DDQN-L | 93.69% | 1.71% | 5.13% | — |
| PPO-U | 97.82% | 1.03% | 1.87% | — |
| DQN-U | 20.33% | 8.31% | 79.35% | — |
| DDQN-U | 22.59% | 13.83% | 76.56% | — |

### Table 10  EQ / GT (violation rate = 1 − completed without violation − EXIT)

| Variant | EQ viol. | EQ EXIT | GT viol. | GT EXIT |
|---|---|---|---|---|
| Mask (PPO/DQN/DDQN) | 0 | 0.01% | 0 | 0 |
| Repair (PPO/DQN/DDQN) | 0.01% / 0 / 0 | — | 0 | — |
| Lagrangian (PPO/DQN/DDQN) | 0.01% / 0.01% / 0.02% | — | 0.05% / 0 / 0 | — |
| PPO-U | 0.05% | — | 0.03% | — |
| DQN-U | 7.09% | — | 4.58% | — |
| DDQN-U | 8.51% | — | 15.74% | — |

### Table 11  Three-scenario mean violation rate (capacity or privacy)

| Mechanism \ Learner | PPO | DQN | DDQN |
|---|---|---|---|
| Mask | 0.00% | 0.00% | 0.00% |
| Repair | 0.85% | 1.88% | 2.20% |
| Lagrangian | 0.64% | 1.20% | 2.11% |
| Unconstraint | 0.75% | 30.44% | 33.89% |

**Analysis points**
1. Mask cannot violate by construction: of the 12 variants, only the three Mask variants have zero capacity and zero privacy violations, which is what treating privacy as a hard constraint requires. Their only failure mode is EXIT at a dead end, and it occurs almost only on LT (1.38–3.87%).
2. Every test instance is proven feasible by the ILP, so an EXIT or a violation means the policy led itself into a dead end, not that the instance has no solution.
3. On LT, PPO-M's EXIT rate (1.38%) is lower than the violation rates of PPO-R, PPO-L and PPO-U (1.87–2.54%). Adding the mask does not cause more failures.
4. Repair violates only at dead ends (when no feasible ECU is left to swap in), and its violations are mostly privacy violations (2.4–6.2% on LT). With the high conflict density p = 0.6, dead ends are mainly caused by conflicts.
5. Lagrangian keeps violations fairly low (0.6–2.1%), but at 20M steps J_C has still not reached the target 0.001, so violations are not zero.
6. Unconstraint DQN/DDQN essentially fail to learn the constraints: about 80% of LT instances are violated, almost all on privacy. PPO-U's violation rate is only 0.75%, so PPO learns most of the constraints from the negative terminal reward of ar_pen alone.
7. The gap is computed on violation-free instances only and measures solution quality; the violation rate measures constraint satisfaction, and the two must be read together. DQN-U and DDQN-U complete only about 20% of LT instances without violation, mostly the easier ones (ILP AR 0.898–0.899 vs 0.883–0.884 for the others), so their gaps rest on far fewer instances.

## 7. Discussion

1. **Main result**: PPO-M (Mask + PPO + relation-aware Transformer) reaches a three-scenario mean gap of 4.36% with zero violations and runs 28× faster than the ILP (47.5× on LT), meeting the target (gap ≤ 5% with Mask best).
2. **Mechanisms**: only Mask strictly guarantees the constraints; Repair comes next (it violates only at dead ends); Lagrangian and Unconstraint reduce violations only gradually through the reward and work poorly with value-based learners.
3. **Learners**: PPO is stable under all four mechanisms (4.4–7.2%); DQN/DDQN are competitive only with the hard mechanisms.
4. **Scenarios**: the gap is largest on LT (M > N). On EQ/GT several variants are at 3–4%, so LT is the scenario that separates the methods.
5. **Limitations**:
   - RL has no optimality guarantee, so the comparison with the ILP is not like-for-like;
   - Mask still has 1.38% EXIT on LT;
   - why Repair beats Mask on DQN remains to be explained;
   - timing uses only the seed-1 models, measured on a single CPU thread with 24 processes in parallel.
