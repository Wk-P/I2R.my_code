// 后端数据里的中文（数据空间名、version/VERSION.md 摘要）的英文，整句匹配。新增版本摘要时在这里补一条。
export const DATA = {
  "v4.2.0 · 论文最优模型合集": "v4.2.0 · best paper models", "v4.2.0（模型副本）": "v4.2.0 (model copy)", "v4.3.1.3 起": "since v4.3.1.3",
  "阶段1 · 初始实现": "Stage 1 · initial implementation", "阶段2 · BC 预训练": "Stage 2 · BC pre-training", "阶段3 · 论文验证": "Stage 3 · paper validation",
  "阶段4 · 状态设计与奖励探索": "Stage 4 · state design and reward exploration", "阶段5 · 论文最终实验": "Stage 5 · final paper experiments",
  "阶段6 · 统一实现": "Stage 6 · unified implementation",
  "`add_states`分支：reward-engineering路线收尾——5轮独立多seed实验(bottleneck shaping/权重对等+熵退火/梯度分级失败/FFD可行性shaping)全部零效应，success_rate 80~90%、AR 0.60~0.65是当前决策结构下的真实能力边界；否决了求解器masking方案(会让success_rate失去研究意义)；lt/eq/gt最终兜底数据完整跑一次，三场景数据严格分开存放":
    "`add_states` branch: wrap-up of the reward-engineering line — 5 independent multi-seed experiments (bottleneck shaping / equal weights + entropy annealing / graded-gradient failure / FFD-feasibility shaping) all had zero effect; success_rate 80~90% and AR 0.60~0.65 are the real limit of the current decision structure; rejected solver-based masking (it would make success_rate meaningless); one full fallback run of lt/eq/gt, data of the three scenarios kept strictly apart",
  "`add_states`分支：bottleneck_risk特征推广到eq/gt/lt；发现eq/gt仍用v1.1.0旧二元reward(从未获得AR梯度信号)，移植v2.2.0的M*AR reward后AR gap从0.03~0.17压到0.004~0.05——目前影响最大的单一改动；同时发现一次\"lt 5M步无特征对照\"因stash/sleep时序问题被污染，需重新验证":
    "`add_states` branch: bottleneck_risk feature extended to eq/gt/lt; found eq/gt still used the old binary reward of v1.1.0 (never got an AR gradient); porting the M*AR reward of v2.2.0 cut the AR gap from 0.03~0.17 to 0.004~0.05 — the largest single change so far; also found the \"lt 5M no-feature control\" was contaminated by a stash/sleep timing issue and must be re-run",
  "**BC预训练永久停用**：隔离实验发现lt/ppo_mask去掉BC后success_rate 47.5%→70.0%(+22.5pp,同样5M步)，是目前发现的最大单一负面因素；state设计(bottleneck_risk特征)贡献较小(+5pp,仅500k步单次验证)，仍需继续研究，另开`add_states`分支跟进":
    "**BC pre-training permanently disabled**: an isolated experiment showed lt/ppo_mask success_rate 47.5%→70.0% without BC (+22.5pp, same 5M steps), the largest single negative factor found; state design (bottleneck_risk feature) contributes less (+5pp, one 500k-step run), still under study on the new `add_states` branch",
  "**AR正式成为优化目标**：终局reward从二元(±M)改为零违反时M*AR、违反时-M，让梯度真正区分成功放置的质量高低；违规仍是硬约束，数学上保证\"任何成功优于任何违规\"；需要完整重训5M步，结果待补":
    "**AR becomes the training objective**: terminal reward changed from binary (±M) to M*AR without violations and -M with violations, so the gradient distinguishes the quality of successful placements; violations stay a hard constraint, guaranteeing \"any success beats any violation\"; needs a full 5M-step retrain, results pending",
  "Self-Imitation/Expert Iteration微调（纯BC + BC+RL恢复两版）：核心假设不成立，6轮下来N=1单次成功率始终低于基线，负结果":
    "Self-Imitation / Expert Iteration fine-tuning (pure BC and BC+RL recovery): the core hypothesis failed; after 6 rounds the N=1 success rate stayed below baseline, negative result",
  "**重大发现**：best-of-N的N=8从未调过参，N敏感性扫描(N=8→1024)显示success_rate从52.5%涨到95%，推翻此前\"lt success_rate系统性瓶颈≈65%\"的结论；40场景逐一定位2个真正卡死的高利用率难例":
    "**Major finding**: N=8 of best-of-N had never been tuned; an N sweep (8→1024) raised success_rate from 52.5% to 95%, overturning the earlier \"lt success_rate is capped at ≈65%\" conclusion; going through the 40 scenarios located 2 truly stuck high-utilization hard cases",
  "新增实验进程看门狗；lt/ppo_mask多seed(42/1/2/3/4)诊断实验，success_rate均值≈0.640(区间0.575~0.675)，系统性瓶颈非seed方差；修复lt/ppo/ppo_opt/dqn/ddqn的TOTAL_STEPS未传scenario参数，SCENARIO_TOTAL_STEPS覆盖表对齐":
    "Added an experiment-process watchdog; multi-seed (42/1/2/3/4) diagnosis of lt/ppo_mask: success_rate mean ≈0.640 (range 0.575~0.675), a systematic bottleneck rather than seed variance; fixed TOTAL_STEPS not receiving the scenario argument in lt/ppo/ppo_opt/dqn/ddqn, aligned the SCENARIO_TOTAL_STEPS override table",
  "训练步数改为按(场景,算法)覆盖；lt场景valid_placed曲线2M步未收敛，恢复5M步，eq/gt维持2M":
    "Training steps now overridden per (scenario, algorithm); the lt valid_placed curve had not converged at 2M, so back to 5M; eq/gt stay at 2M",
  "评估阶段加best-of-N随机重采样（N=8），缓解在线不可回溯策略的成功率上限；lt/ppo_mask验证 success_rate 42.5%→65.0%":
    "Evaluation adds best-of-N random resampling (N=8) to ease the success-rate ceiling of an online, non-backtracking policy; lt/ppo_mask success_rate 42.5%→65.0%",
  "AR指标排除失败episode，只在success episode上算ar_mean/ar_std（此前混合成功失败平均，与ILP的AR不是同一统计量）":
    "AR excludes failed episodes: ar_mean/ar_std over successful episodes only (previously averaged over both, not the same statistic as the ILP AR)",
  "训练步数5M→2M；ppo_mask训练曲线补充valid_placed指标（原Services Placed曲线因env强制补齐机制恒等于M，不反映真实进展）":
    "Training steps 5M→2M; ppo_mask training curve adds valid_placed (the old Services Placed curve always equalled M because the env force-completed episodes, so it showed no real progress)",
  "reward改为纯终局稀疏奖励，GAE负责反向信用分配；仅ppo_mask(部分场景ppo_lagrangian)在此奖励下success_rate=1.0，其余算法掉到0":
    "Reward changed to a purely terminal sparse reward, with GAE doing the credit assignment; only ppo_mask (and ppo_lagrangian in some scenarios) reach success_rate=1.0 under it, the other algorithms drop to 0",
  "剪枝力度调优（权重0.3，更温和）exp_id=0c80d082": "Pruning strength tuning (weight 0.3, milder) exp_id=0c80d082",
  "负优势样本剪枝（CMA-ES风格，权重0.1）exp_id=ac669003": "Negative-advantage sample pruning (CMA-ES style, weight 0.1) exp_id=ac669003",
  "熵系数调度（前高后低，缓解PPO探索过早收敛）exp_id=9a75a253": "Entropy coefficient schedule (high then low, against premature PPO convergence) exp_id=9a75a253",
  "熵系数bug修复（静态0.005对齐，无调度）exp_id=12ea950c": "Entropy coefficient bug fix (static 0.005, no schedule) exp_id=12ea950c",
  "回滚v0.3.1的feasibility-shaping实验，退回v0.2.1的env baseline": "Rolled back the feasibility-shaping experiment of v0.3.1 to the env baseline of v0.2.1",
  "eq/gt/lt全场景feasibility-shaping实验 — lt场景success_rate上是负结果": "Feasibility-shaping experiment on all of eq/gt/lt — negative result on lt success_rate",
  "部署可靠性修复（backend --reload、deploy.sh）": "Deployment reliability fixes (backend --reload, deploy.sh)",
  "场景/算法目录重构为 scenarios/ + shared/ 架构": "Scenario/algorithm directories restructured into scenarios/ + shared/",
  "dashboard (FastAPI+Vue) + N_ENVS统一 + shared/整合，在v0.1.0核心算法代码基础上": "Dashboard (FastAPI+Vue) + unified N_ENVS + shared/ consolidation, on top of the v0.1.0 core algorithms",
  "推理时采样 K 个解取最好（best-of-K）：v4.3.1.6 的 PPO 系列（5M）vs 随机贪心，不重新训练":
    "Inference-time best-of-K sampling: the v4.3.1.6 PPO models (5M) vs randomized greedy, no retraining",
  "v4.3.1.7（CBC，单线程，16 进程并行）": "v4.3.1.7 (CBC, single thread, 16 parallel processes)",
  "v4.3.1.6 的 108 个模型按真正的 AR 最优（ILP，Dinkelbach）重新评估，每个测试实例 1 次确定性输出；另测 ILP 耗时":
    "The 108 models of v4.3.1.6 re-evaluated against the true AR optimum (ILP, Dinkelbach), one deterministic output per test instance; ILP time also measured",
  "核心算法代码定型": "Core algorithm code finalized", "初始版本": "Initial version",
};
