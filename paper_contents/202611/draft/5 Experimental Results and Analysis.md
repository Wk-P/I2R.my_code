# 5 Experimental Results and Analysis

This chapter evaluates the 12 algorithm–strategy variants against the optimal ILP solution. Section 5.1 describes the experimental setup, Section 5.2 defines the evaluation metrics, and Section 5.3 compares solution quality, constraint satisfaction, and computation time. Section 5.4 analyzes the results by placement scenario, and Section 5.5 discusses the findings and limitations.

## 5.1 Experimental Setup

**Scenarios and instances.** We consider three placement scenarios with different ratios between services and ECUs: LT ($N = 10$, $M = 15$; fewer ECUs than services), EQ ($N = 10$, $M = 10$), and GT ($N = 15$, $M = 10$; more ECUs than services). ECU capacities are drawn uniformly from $\{50, 55, \dots, 195\}$ and service demands from $\{10, 15, \dots, 95\}$. There are $K = 10$ privacy conflict sets, and any two services conflict with probability $p = 0.6$. For each scenario, we generate 50,000 training instances and a fixed set of 5,000 test instances that does not overlap with the training set. Every instance is solved to optimality by the ILP of Section 3.2, so all instances are known to be feasible and $\text{AR}_{\text{ILP}}$ is available for each test instance. All methods use the same test set.

**Training.** At the start of each episode, a training instance is drawn at random from the training pool. Every variant uses the policy network of Section 4.1 and is trained for $2 \times 10^7$ environment steps with $\gamma = 1$ and full-length episodes, using three random seeds (1, 2, 3). In total, 108 runs were trained (12 variants × 3 scenarios × 3 seeds). Table 5.1 lists the hyperparameters.

**Table 5.1** Training hyperparameters.

| | PPO | DQN / DDQN |
|---|---|---|
| Parallel environments | 40 | 12 |
| Learning rate (Adam, constant) | $3\times10^{-4}$ | $10^{-3}$ |
| Batch | 20,480 steps per update, minibatch 256 | Replay buffer 100k, batch 64 |
| Updates | 10 epochs (6 for PPO-M) | One gradient step per 48 new transitions |
| Other | GAE 0.95, clip 0.1, entropy coef. 0.005, value coef. 0.5 | $\varepsilon$: 1 → 0 linearly over the first $2\times10^6$ steps; hard target update every ~492 steps |
| Lagrangian | $\lambda_0 = 0.001$, Adam lr 0.035, $d = 0.001$; per rollout | Same; every 1000 episodes |

**Baseline and environment.** The ILP baseline uses Dinkelbach's method (tolerance $10^{-6}$) with PuLP and the CBC solver in single-threaded mode. The implementation uses Python 3.12, PyTorch 2.5.1, Stable-Baselines3 / sb3-contrib 2.7.1, and Gymnasium 1.2.3. Training ran on an NVIDIA GTX 1080 Ti, and all timing measurements were taken on an Intel Xeon E5-2660 v4 CPU.

## 5.2 Evaluation Metrics

Each trained model is evaluated on the 5,000 test instances of its scenario with deterministic inference: at every step, the action with the highest score is selected. Results are first averaged over the three seeds; ± denotes the standard deviation across seeds. Unless stated otherwise, the reported value for a variant is the arithmetic mean over the three scenarios.

**Optimality gap.** Let $\mathcal{T}$ be the test set. AR counts only valid placements (Section 4.2.3), and failed episodes remain in the average. We report

$$
\text{absolute gap} = \overline{\text{AR}_{\text{ILP}}} - \overline{\text{AR}},\qquad
\text{relative gap} = 1 - \frac{\frac{1}{|\mathcal{T}|}\sum_{i \in \mathcal{T}} \text{AR}^{(i)}}{\frac{1}{|\mathcal{T}|}\sum_{i \in \mathcal{T}} \text{AR}^{(i)}_{\text{ILP}}}.
$$

"Gap" refers to the relative gap unless stated otherwise.

**Constraint violation.** The capacity violation rate and the privacy violation rate are the fractions of test instances with at least one violation of each type:

$$
\text{capacity violation rate} = \frac{1}{|\mathcal{T}|}\sum_{i} \mathbb{1}\Big[\textstyle\sum_t v^{\text{cap}(i)}_t > 0\Big],\qquad
\text{privacy violation rate} = \frac{1}{|\mathcal{T}|}\sum_{i} \mathbb{1}\Big[\textstyle\sum_t v^{\text{priv}(i)}_t > 0\Big].
$$

Both types can occur in the same instance, so the total violation rate is at most their sum. For action masking, we also report the *EXIT rate*, the fraction of instances that end with EXIT at a dead end.

**Computation time.** We time each of the 3 × 5,000 test instances individually on a single CPU thread. ILP time includes model construction and all Dinkelbach iterations. RL time covers one complete deterministic episode, including reset, mask computation, forward passes, and environment steps, using the seed-1 model of each variant. The speed-up is the mean ILP time divided by the mean RL time.

## 5.3 Performance Comparison

**Solution quality.** Table 5.2 compares the 12 variants with the ILP optimum.

**Table 5.2** AR and optimality gap, averaged over the three scenarios (ILP AR = 0.8829).

| Variant | AR | Absolute gap | Relative gap |
|---|---|---|---|
| PPO-U | 0.8290 | 0.0540 | 6.12% ± 0.28% |
| DQN-U | 0.6635 | 0.2194 | 24.97% ± 4.13% |
| DDQN-U | 0.6837 | 0.1993 | 22.59% ± 2.37% |
| PPO-L | 0.8277 | 0.0552 | 6.24% ± 1.01% |
| DQN-L | 0.7728 | 0.1102 | 12.51% ± 1.36% |
| DDQN-L | 0.7614 | 0.1215 | 13.84% ± 0.50% |
| PPO-M | **0.8443** | **0.0386** | **4.38% ± 0.17%** |
| DQN-M | 0.8196 | 0.0633 | 7.17% ± 1.38% |
| DDQN-M | 0.8260 | 0.0569 | 6.44% ± 1.19% |
| PPO-R | 0.8199 | 0.0631 | 7.15% ± 0.27% |
| DQN-R | 0.8256 | 0.0574 | 6.50% ± 0.55% |
| DDQN-R | 0.8180 | 0.0649 | 7.37% ± 1.14% |

PPO-M achieves the lowest gap of all variants, 4.38%, with the smallest variance across seeds (0.17 percentage points). Nine of the 12 variants reach gaps between 4.4% and 7.4%. The exceptions are DQN and DDQN under the Lagrangian and unconstrained strategies, whose gaps range from 12.5% to 25.0%.

**Comparison of constraint-handling strategies.** Table 5.3 arranges the same gaps by strategy and algorithm.

**Table 5.3** Relative gap by strategy and algorithm (best per column in bold).

| Strategy \ Algorithm | PPO | DQN | DDQN | Row mean |
|---|---|---|---|---|
| Unconstrained | 6.12% | 24.97% | 22.59% | 17.90% |
| Lagrangian | 6.24% | 12.51% | 13.84% | 10.86% |
| Action Masking | **4.38%** | 7.17% | **6.44%** | **6.00%** |
| Action Repair | 7.15% | **6.50%** | 7.37% | 7.01% |
| Column mean | 5.97% | 12.79% | 12.56% | 10.44% |

On average, the strategies rank as masking (6.00%) < repair (7.01%) < Lagrangian (10.86%) < unconstrained (17.90%). The two hard strategies, masking and repair, act on the action itself and clearly outperform the two soft strategies, which act only through the reward. The difference is largest for the value-based algorithms: for DQN and DDQN, the hard strategies reduce the gap by roughly 5 to 18 percentage points. PPO is much less sensitive to the strategy, staying between 4.4% and 7.2% under all four. PPO-U (6.12%) and PPO-L (6.24%) even slightly outperform PPO-R (7.15%). Masking is the best strategy for PPO and DDQN, whereas DQN does slightly better with repair (6.50%) than with masking (7.17%). We have not yet determined the reason for this exception.

**Constraint satisfaction.** Table 5.4 reports the average violation rates.

**Table 5.4** Violation rate (capacity or privacy), averaged over the three scenarios.

| Strategy \ Algorithm | PPO | DQN | DDQN |
|---|---|---|---|
| Unconstrained | 0.75% | 30.44% | 33.89% |
| Lagrangian | 0.64% | 1.20% | 2.11% |
| Action Masking | 0.00% | 0.00% | 0.00% |
| Action Repair | 0.85% | 1.88% | 2.20% |

Only action masking satisfies both constraints on every test instance, which meets the requirement that privacy is a hard constraint. Its only failure mode is EXIT at a dead end (Section 5.4). Repair violates constraints only at dead ends, where no feasible ECU is available for substitution. The Lagrangian strategy keeps violations low (0.6–2.1%), but after $2\times10^7$ steps the expected cost has not yet reached the budget $d = 0.001$. Without constraint handling, DQN and DDQN fail to learn the constraints, with about one third of instances violating at least one. PPO-U, by contrast, violates on only 0.75% of instances, which shows that PPO can learn most of the constraints from the sparse failure reward alone. Because violating placements do not count toward AR, the gaps of the non-masking variants already include the AR lost to violations. Those variants were nevertheless allowed to violate, so their gaps are not strictly comparable with that of masking.

**Computation time.** Table 5.5 compares the time needed to solve one instance.

**Table 5.5** Mean time per instance (ms); ILP median in parentheses.

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

All learned policies solve an instance in 46–53 ms on average, compared with 1500 ms for the ILP, a speed-up of 28–32×. The speed-up is computed from the pooled means over all three scenarios and is dominated by LT. Averaging the per-scenario speed-ups gives about 24.8× for PPO-M. Differences among variants are at most 7 ms. PPO is slightly slower because it also evaluates the value head. ILP time grows sharply with problem size and has a heavy tail: on LT, the mean is 6.4 times that on EQ, and the mean (3285 ms) is 2.5 times the median (1301 ms). RL inference performs a fixed $M$ forward passes, each costing $O(L(T^2 d + T d^2))$, so its time grows only about 1.5× from EQ to LT and does not depend on how hard an instance is.

<!-- Figures: figs/fig_time_bar_speedup.svg (main text), figs/fig_time_bar_ms.svg; gap bar chart not yet in SVG (v6.0.0-merge/fig_gap_bars.{pdf,png}). -->

## 5.4 In-depth Analysis Across Placement Scenarios

**Table 5.6** Relative gap by scenario.

| Variant | LT | EQ | GT |
|---|---|---|---|
| PPO-U | 7.27% | 5.99% | 5.10% |
| DQN-U | 29.88% | 29.67% | 15.37% |
| DDQN-U | 27.42% | 21.27% | 19.08% |
| PPO-L | 6.32% | 5.55% | 6.86% |
| DQN-L | 18.37% | 10.87% | 8.27% |
| DDQN-L | 22.87% | 13.61% | 5.03% |
| PPO-M | **6.22%** | 3.64% | 3.28% |
| DQN-M | 13.57% | **3.54%** | 4.41% |
| DDQN-M | 10.62% | 3.65% | 5.05% |
| PPO-R | 8.56% | 6.85% | 6.04% |
| DQN-R | 12.49% | 3.61% | 3.41% |
| DDQN-R | 15.55% | 3.61% | **2.94%** |
| Mean of 12 variants | 14.93% | 9.32% | 7.07% |
| ILP AR | 0.8811 | 0.8624 | 0.9053 |

**Scenario difficulty.** The scenarios rank in difficulty as LT > EQ > GT, with mean gaps of 14.93%, 9.32%, and 7.07%. LT has more services than ECUs, so both capacity and conflicts are tightest there, and early decisions leave the least room for later services. GT has more ECUs than services, so a conflict-free ECU is usually available.

**Where the variants differ.** On EQ and GT, several variants perform almost equally well. On EQ, five variants lie within 0.11 percentage points of each other (3.54–3.65%), and on GT the best three lie within 0.47 percentage points (2.94–3.41%), and different variants are best in each scenario (DQN-M on EQ, DDQN-R on GT). The differences arise on LT. There, PPO-M is best (6.22%), only slightly ahead of PPO-L (6.32%) but 7.35 percentage points ahead of DQN-M, and every DQN and DDQN variant exceeds a 10% gap. PPO-M ranks first on the three-scenario average mainly because of its lead on LT. LT is also the scenario where the ILP is slowest (Table 5.5), so the hardest scenario for RL is also the one with the largest speed-up: 47.5× for PPO-M, compared with 11.6× on EQ and 15.3× on GT.

**Violations and dead ends.** Table 5.7 shows how each variant fails on LT. On EQ and GT, all variants except DQN-U and DDQN-U violate or exit on at most 0.05% of instances, so we omit those scenarios here.

**Table 5.7** Outcomes on LT (fraction of instances, mean of three seeds).

| Variant | Violation-free | Capacity viol. | Privacy viol. | EXIT |
|---|---|---|---|---|
| PPO-U | 97.82% | 1.03% | 1.87% | — |
| DQN-U | 20.33% | 8.31% | 79.35% | — |
| DDQN-U | 22.59% | 13.83% | 76.56% | — |
| PPO-L | 98.13% | 1.15% | 0.91% | — |
| DQN-L | 96.39% | 1.97% | 1.85% | — |
| DDQN-L | 93.69% | 1.71% | 5.13% | — |
| PPO-M | 98.62% | 0 | 0 | 1.38% |
| DQN-M | 96.13% | 0 | 0 | 3.87% |
| DDQN-M | 96.95% | 0 | 0 | 3.05% |
| PPO-R | 97.46% | 0.99% | 2.37% | — |
| DQN-R | 94.35% | 1.32% | 5.37% | — |
| DDQN-R | 93.41% | 1.82% | 6.21% | — |

Under masking, failures occur almost exclusively on LT, as EXIT at a dead end in 1.38–3.87% of instances. Masking does not increase the failure rate: PPO-M exits on 1.38% of LT instances, fewer than the 1.87–2.54% of instances on which PPO-R, PPO-L, and PPO-U violate a constraint. For repair, violations are mostly privacy violations (2.4–6.2% on LT). Repair violates only when no feasible ECU exists, so most dead ends in this setting are caused by conflicts rather than by capacity, which is expected given the high conflict density ($p = 0.6$). Without constraint handling, DQN and DDQN violate on about 80% of LT instances, almost all of them privacy violations.

A diagnostic study of the EXIT episodes of the masking variants on LT shows that the episode can no longer be completed, as determined by the ILP, from step 4.3–4.5 on average, whereas EXIT occurs only at step 11.9–12.1. The dead end is thus created about 7.5 steps before it becomes visible. The failures come from early grouping decisions rather than from the final placement step, and a one-step feasibility check cannot detect them in time.

<!-- Source of the EXIT diagnosis: paper_contents/v6.0.2/exit_diag.md (v6.0.0 Mask models, 400 instances per seed). The models differ slightly from the final ones. State this or rerun the diagnosis on the final models. -->

## 5.5 Discussion

**Main findings.** Combining action masking, PPO, and the relation-aware Transformer (PPO-M) gives a policy that never violates the capacity or privacy constraints, has an average optimality gap of 4.38%, and solves instances 28× faster than the ILP (47.5× on the largest scenario). Among the strategies, only action masking guarantees the constraints. Repair fails only at dead ends, while the Lagrangian and unconstrained strategies reduce violations gradually through the reward and work poorly with value-based algorithms. Among the algorithms, PPO performs consistently under all strategies, whereas DQN and DDQN are competitive only with the hard strategies. Most of the remaining gap is concentrated in the scenario with more services than ECUs.

**Why hard strategies help value-based learning.** Under the soft strategies, a violation affects only the return, and for the unconstrained strategy only the terminal return. DQN and DDQN must therefore propagate a sparse, delayed signal back to the step that caused the violation through bootstrapped targets, while the replay buffer also contains transitions collected under older penalty weights. Masking and repair remove infeasible actions from execution, and masking also removes them from the TD target, so the value function only has to rank feasible actions. PPO's on-policy updates with Monte Carlo-style advantages ($\gamma = 1$, GAE 0.95) appear to handle sparse terminal penalties more robustly.

**Offline design cost versus online cost.** The Safe RL approach does not require less design effort than the ILP. The MDP, reward, constraint-handling strategy, network, and hyperparameters must all be designed, and training takes $2\times10^7$ steps per run. This cost, however, is paid once and offline. Online, a trained policy needs only $M$ forward passes per instance, whose cost is polynomial in $N$ and $M$ and independent of instance hardness. Exact solution of the strongly NP-hard problem has worst-case exponential cost in every Dinkelbach iteration. If $T_{\text{ILP}}$ and $T_{\text{RL}}$ are the per-instance times and $C_{\text{train}}$ is the training cost, RL becomes cheaper overall after $I > C_{\text{train}} / (T_{\text{ILP}} - T_{\text{RL}})$ instances, and this break-even point decreases as the problem size grows. Measured on our hardware (three runs sharing one GTX 1080 Ti), training PPO-M takes 3.6 h on LT, 3.2 h on EQ, and 3.6 h on GT per run. With the per-instance times of Table 5.5, the break-even point of a single PPO-M model is about $1.3\times10^4\,\text{s} / 3.22\,\text{s} \approx 4.1\times10^3$ instances on LT, $2.5\times10^4$ on EQ, and $2.0\times10^4$ on GT. The training cost is therefore recovered fastest on the scenario where the ILP is slowest. These numbers compare GPU training time with CPU inference time and ignore hyperparameter tuning, so they are an indication of scale rather than an exact cost comparison.

**Sources of the gap.** The remaining gap has two components. First, on LT, failed episodes (EXIT or violation) contribute little or no AR. Second, successful episodes tend to open more lightly loaded ECUs than the optimum. An earlier diagnosis with a previous network showed that almost all newly opened ECUs were opened because the current service fit on no active ECU, not by the agent's free choice. This also points to earlier grouping decisions as the cause. Together with the EXIT analysis in Section 5.4, these results suggest that further improvement requires better long-horizon planning rather than better single-step decisions.

**Limitations.** (i) The learned policies do not guarantee optimality, and on the hardest scenario masking still exits on 1.38% of instances. In a deployment, such cases would need a fallback such as the ILP or a heuristic. (ii) Only three problem sizes with $N, M \in [10, 15]$ were studied, and each scenario was trained separately. Although the network itself does not depend on $N$ and $M$, generalization to larger or unseen sizes has not been evaluated. (iii) The reason repair outperforms masking for DQN remains open. (iv) Timing was measured on CPU with the seed-1 model of each variant. The three seeds share the same architecture and per-step computation, so their inference times are expected to be the same.

<!--
TODO
- The gap-source diagnosis (opened ECUs) used the old MLP network and was not repeated on the final model. Either rerun it or keep the wording "earlier diagnosis with a previous network".
- If the descending-order ablation is cited in 4.2, add its numbers: 10.1% / 12.6% / 13.4% for descending / ascending / random, from v4.4.7 with the old network.
-->
