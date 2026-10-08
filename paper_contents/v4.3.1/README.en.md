# v4.3.1 — Reward design exploration (switchable rewards) + experiment panel redesign

Incremental version after v4.3.0 (DQN family switched to the PPO reward; Mask/Repair-DQN/DDQN added). **Default behaviour is identical to v4.3.0**: every new reward is selected through the environment variable `REWARD_MODE`, and `legacy` is used when it is not set.

## 1. Switchable reward `REWARD_MODE` (`src/shared/reward_config.py`)

| Mode | Terminal reward, violation-free episode | Terminal reward, episode with violations | Intermediate steps |
|---|---|---|---|
| `legacy` (default) | $M(2AR-1)\in(-M,M]$ | $-M(1-\text{valid}/M)\in[-M,0)$ | 0 |
| `ar` | $M\cdot AR\in(0,M]$ | same | 0 |
| `ratio` | $M\cdot AR/AR^*\in(0,M]$ | same | 0 |
| `directional` | see Section 2 (step-wise reward, replaces everything) | | |

- AR = average resource utilization (the objective); $AR^*$ is the instance's ILP-optimal AR.
- Problem with `legacy`: when $AR<0.5-\frac{1}{2M}$ a violation-free episode scores below the $-1$ of "violating only at the last step" (about 1/10 of EQ instances are affected, see "Known issues" in `../v4.3.0/README.md`). `ar` / `ratio` guarantee that every violation-free episode (>0) is strictly better than every violating one (<0).
- $AR^*$ is computed by `src/scripts/build_ar_star.py`, solving the ILP for 2000 instances per scenario in parallel, stored in `results/<branch>/<scen>/ilp/ar_star.json` (indexed by a hash of the instance), about 80 seconds.
- Environments wired in: PPO, Mask-PPO, DQN, DDQN (Mask-DQN/DDQN reuse the Mask-PPO environment). The curriculum weight $w$ of Mask-PPO LT is kept: the success branch is $M[(1-w)+w\cdot q]$, where $q$ is the quality term in the table above.

## 2. directional reward (based on `../v4.3.0/Reward-Discussion.md`)

All methods share the same objective reward; methods differ only in how they handle constraints. $\Delta AR_t=AR_{t+1}-AR_t$:

$$
r_t^{\text{obj}}=\begin{cases}1+\beta\Delta AR_t,&\Delta AR_t>\epsilon\\ \beta\Delta AR_t,&|\Delta AR_t|\le\epsilon\\ -\lambda_d+\beta\Delta AR_t,&\Delta AR_t<-\epsilon\end{cases}
$$

| Case | Reward | Applies to |
|---|---|---|
| Every step | $r_t^{\text{obj}}$ | all |
| All $M$ services placed legally (last step) | $+B$ | all; placements executed by Repair-* are legal by construction, so running to the end counts |
| Dead end: services remain but the next state has no legal ECU | $-C$, episode ends | Mask-PPO / Mask-DQN / Mask-DDQN |
| Dead end: repair finds no legal ECU | $-C$, episode ends | Repair-PPO / Repair-DQN / Repair-DDQN |
| Constraint cost | $-\eta\,c_t$, $\eta$ = the dual variable $\lambda$ in the environment, $c_t$ = violations in this step | Lagrange-PPO |
| No constraint handling | only $r_t^{\text{obj}}$ ($+B$) | PPO / DQN / DDQN |

Default parameters (environment variables): `DIR_BETA` $\beta=10$, `DIR_LAMBDA` $\lambda_d=1$, `DIR_EPS` $\epsilon=10^{-3}$, `DIR_B` $B=M$, `DIR_C` $C=M$.

Implementation: a `directional_step()` decorator wraps `step()` of 18 environments (6 kinds × 3 scenarios), reads the environment's own AR, violation counts, `action_masks()` and repair-failure termination, and replaces the reward entirely; when `REWARD_MODE≠directional` the decorator returns the original `step` and changes nothing. Under directional, Repair-PPO's single-step −0.1 repair penalty and Lagrange-PPO's step-wise penalties on EQ/GT are no longer used.

### Open questions

1. **Are the discrete terms consistent with the objective?** The continuous term $\beta\sum_t\Delta AR_t=\beta\,AR_{\text{final}}$ matches the objective exactly; the discrete term $+1/0/-\lambda_d$ counts "how many steps went up / down", is not monotone in the final AR, and may push the policy towards many small improvements instead of the highest final AR. The larger $\beta$, the closer to a pure AR objective.
2. **Violations of unconstrained PPO/DQN have no direct cost**: a violating placement still earns its $\Delta AR$ reward; the only cost is missing $B$. With a random policy, EQ PPO's mean return is about 8.8 with success_rate 0.
3. **Lagrange-PPO depends on $\lambda$ growing**: on LT, $\lambda$ warms up for 20000 episodes with step size $3\times10^{-4}$, so it is almost 0 within 1M steps.
4. Lagrange-DQN / Lagrange-DDQN mentioned in Reward-Discussion are not implemented yet.

## 3. 1M-step reward pilot (`legacy` / `ar` / `ratio`)

- Script: `src/scripts/run_v4.3.1_reward_pilot.py`; 3 rewards × {PPO, Mask-PPO, DQN} × 3 scenarios × seed 1 = 27 jobs.
- Report: `reward_pilot_report.md` (this directory); manifest: `logs/v4.3.1_reward_pilot/manifest.json`.
- Scheduling weights follow measured CPU usage (PPO 7 cores, DQN 3 cores); `--resume` supported.

Results (success_rate; single seed, 1M steps, AR is the mean over all test episodes):

| Scenario | Algorithm | legacy | ar | ratio |
|---|---|---|---|---|
| LT | Mask-PPO | 0.608 | 0.665 | 0.643 |
| LT | PPO | 0.355 | 0.345 | 0.353 |
| LT | DQN | 0.135 | 0.068 | 0.068 |
| EQ | Mask-PPO | 1.000 | 1.000 | 1.000 |
| EQ | PPO | 0.733 | 0.990 | 1.000 |
| EQ | DQN | 0.573 | 0.975 | 0.978 |
| GT | Mask-PPO | 1.000 | 1.000 | 1.000 |
| GT | PPO | 0.970 | 0.998 | 0.998 |
| GT | DQN | 0.740 | 0.875 | 0.918 |

- Consistent with the analysis of the legacy loophole: on EQ, the most affected scenario, unconstrained PPO / DQN success_rate rises from 0.73 / 0.57 to 0.99–1.00 / 0.98 and the conflict-violation rate drops from 25% / 40% to 0–2%; GT also improves; LT (not affected by the loophole) is basically unchanged, and DQN even drops (single seed, to be re-checked).
- Unconstrained algorithms have slightly lower AR under ar / ratio (EQ PPO 0.506 → 0.489), but legacy's AR includes many violating episodes, so they are not fully comparable; Mask-PPO's AR is about the same under all three rewards.
- ar and ratio differ very little; ratio has a slightly higher success_rate on GT DQN.

## 4. Repair-* no longer records the repair-trigger rate

`scenarios/*/ppo_opt/run_all.py` and `src/shared/dqn_variant_runner.py`: the violation columns count only executed placements (0 for Repair-* by construction), and result files no longer output the repair-trigger rate. The environment still counts repairs internally because the `legacy` reward uses them.

## 5. git branch cleanup and result directories

- The former 5 branches (main → pretrain → paper-verfication → add_states → final_paper_experiments) were research stages on one straight line, all contained in the latest commit. Only **main** is kept as the main line (fast-forwarded to the latest), and versions are marked with tags.
- The old branches became archive tags and were deleted (locally and on GitHub): `archive/stage1-main`, `archive/stage2-pretrain`, `archive/stage3-paper-verification` (including 2 previously unpushed local commits), `archive/stage4-add-states`, `archive/stage5-final-paper-experiments`. To view an old stage: `git checkout archive/stage4-add-states`.
- Where results are written is decoupled from the branch name: `src/shared/version_config.RESULTS_SPACE = "final_paper_experiments"` (overridable with `$RESULTS_SPACE`); new training keeps writing to `results/final_paper_experiments/`; existing `results/<old branch name>/` directories and every path reference in the docs stay unchanged. The panel shows them as "data spaces", labelled with stage name and version range (`app/backend/result_spaces.json`).

## 6. Experiment panel (`app/`) redesign

- Layout: left navigation (Overview / Monitor / Batches / Results / Versions / Paper Draft), hierarchy version → batch → scenario → algorithm → run.
- Results: version→batch tree + scenario/algorithm/variant filters + keywords; details (sorting, paging) and summary comparison (algorithm × scenario pivot, selectable metrics); filters are written to the URL.
- Global branch switch in the top bar (lists only branches with results, remembers the choice), global search; drawer navigation and responsive layout on phones.
- Terms: success_rate, AR (average resource utilization), AR gap = ILP AR − AR.
- Backend fixes: version list returned 500 on non-numeric tags; manifest batch detection; result-branch inference for old batches; progress of concurrent jobs read from their own logs.
