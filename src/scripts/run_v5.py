#!/usr/bin/env python3
"""v5.0.x campaigns (code version 5.0.0): structure-aware network on GPU (MLP on CPU), 20M steps with checkpoints
at 1 / 3 / 5 / 7.5 / 10 / 15M, 3 seeds x 3 scenarios, ar_pen, gamma 1, full episode, EXIT (Mask), descending demand;
PPO / DQN hyper-parameters from paper_rl/config.py, the same for both networks.

    5.0.1  Mask-DQN / Mask-DDQN x {MLP, Transformer}, the 1600-instance split (the v4.5.3 experiment, redone with the
           v5.0.0 code); the 50k half is v4.5.2
    5.0.2  Transformer x {unconstrained, Lagrangian, Repair} x {PPO, DQN, DDQN} + Maskable PPO, 50k instances (90 runs);
           Maskable DQN / DDQN from v4.5.2 (same setting, one continuous 20M run each; the v5.0.0 code trains them
           bit-identically, so re-running them would reproduce the same models). Maskable PPO is re-trained here because
           v4.4.15-17 reached 20M by resuming twice (5M -> 10M -> 20M).
    5.0.3  the lambda_max sweep for the new Lagrangian (27 runs): Lagrangian PPO / DQN / DDQN x lambda_max {1, 2, 5}
           x 3 scenarios, seed 1, 3M steps. New Lagrangian: per-step cost lambda * c_t, lambda adapted per episode by
           dual ascent and fixed within an episode, J_C = violations per episode, d = 0.001, window 1000, eta 0.5;
           lambda_max is only its cap. Each learner takes the value with the lowest mean effective gap over the
           scenarios (violating instances count as AR = 0); written to the manifest as "lambda_max".
    5.0.4  seed 1 only: every learner x mechanism ({PPO, DQN, DDQN} x {unconstrained, Lagrangian, Maskable, Repair},
           36 runs, 20M steps); each learner's Lagrangian run uses the lambda_max that v5.0.3 chose for it (the driver
           refuses to start without all three).

Speed-ups that leave training unchanged (v5.0.0, checked by src/scripts/check_v5.py: training is bit-identical to the
v4.5 code on CPU and GPU): Mask DQN / DDQN build the state features once for Q-values and mask (1.20x total throughput
with 18 concurrent structure-aware Mask-DQN runs); at most `parallel` runs at a time, each GPU run on the GPU with the
fewest active runs. Optional, off by default because they did not help in that check: V5_NUMA=1 binds every run
(numactl) to the NUMA node of its GPU / the less loaded node (0.99x), V5_MPS=1 shares each GPU through NVIDIA MPS
with this campaign's own pipe directory (0.97x).
Report: learning curve (test gap / failure rate on each seed's 400 test instances, training-set gap on 1600 training
instances), the 20M models with violation rates and (Maskable only) the per-step regret; Chinese + English, email.

    nohup .venv/bin/python src/scripts/run_v5.py --version 5.0.1 > logs/run_v5.0.1_driver.log 2>&1 &
    .venv/bin/python src/scripts/run_v5.py --version 5.0.1 --report
    .venv/bin/python src/scripts/run_v5.py --version 5.0.1 --rerun 1 gt dqn_graph_1600
    5.0.4  same plan as 5.0.3 (36 runs + per-learner lambda_max sweep); new version because the code and the panel
           changed after the 5.0.3 tag (per-learner sweep, report_data.json / lam_sweep_<learner>.json, panel v2).
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))
PY, PY_GPU = str(ROOT / ".venv" / "bin" / "python"), str(ROOT / ".venv-gpu" / "bin" / "python")
DATA, POOL = "v4.3.1.4", "results/sup_diag/{scen}.json"
if os.environ.get("V5_SMOKE") == "1":              # docs/data_spec.md §3.6: every smoke test shares results/smoke/
    os.environ.setdefault("RESULTS_SPACE", "smoke")   # (inherited by the training processes)
RESULTS = ROOT / "results" / os.environ.get("RESULTS_SPACE", "unified")
SCENARIOS, SEEDS = ["lt", "eq", "gt"], [1, 2, 3]
STEPS = 20_000_000
CKPTS = [1_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000, 15_000_000]
SMOKE = os.environ.get("V5_SMOKE") == "1"          # end-to-end test: tiny budget, seed 1, results in RESULTS_SPACE
if SMOKE:                                            # V5_SMOKE_STEPS: smoke budget (default 41k)
    STEPS = int(os.environ.get("V5_SMOKE_STEPS", 41_000))
    SEEDS, CKPTS = [1], [max(2_048, STEPS // 2 // 2_048 * 2_048)]
EARLY = 5

# variant -> (algo, net, 50k pool?, label zh, label en)
REF_452 = {"dqn_mlp": ("mask_dqn", "mlp", True, "MLP Mask-DQN（5 万）", "MLP Mask-DQN (50k)"),
                           "dqn_graph": ("mask_dqn", "graph", True, "Transformer Mask-DQN（5 万）", "Transformer Mask-DQN (50k)"),
                           "ddqn_mlp": ("mask_ddqn", "mlp", True, "MLP Mask-DDQN（5 万）", "MLP Mask-DDQN (50k)"),
                           "ddqn_graph": ("mask_ddqn", "graph", True, "Transformer Mask-DDQN（5 万）", "Transformer Mask-DDQN (50k)")}   # v4.5.2: the 50k references

CONFIG = {
    "5.0.1": {"variants": {"dqn_mlp_1600": ("mask_dqn", "mlp", False, "MLP Mask-DQN（1600）", "MLP Mask-DQN (1600)"),
                           "dqn_graph_1600": ("mask_dqn", "graph", False, "Transformer Mask-DQN（1600）", "Transformer Mask-DQN (1600)"),
                           "ddqn_mlp_1600": ("mask_ddqn", "mlp", False, "MLP Mask-DDQN（1600）", "MLP Mask-DDQN (1600)"),
                           "ddqn_graph_1600": ("mask_ddqn", "graph", False, "Transformer Mask-DDQN（1600）", "Transformer Mask-DDQN (1600)")},
              "title": ("Mask-DQN / Mask-DDQN：训练实例数（1600 vs 5 万）× 步数", "Mask-DQN / Mask-DDQN: training instances (1600 vs 50k) x steps"),
              "parallel": 36, "gpu_round_robin": True},
    "5.0.2": {"variants": {
                           "ppo_mask": ("mask_ppo", "graph", True, "Maskable PPO", "Maskable PPO"),   # one continuous 20M run
                           "ppo_none": ("ppo", "graph", True, "无约束 PPO", "Unconstrained PPO"),
                           "ppo_lagrange": ("lagrange_ppo", "graph", True, "Lagrangian PPO", "Lagrangian PPO"),
                           "ppo_repair": ("repair_ppo", "graph", True, "Repair PPO", "Repair PPO"),
                           "dqn_none": ("dqn", "graph", True, "无约束 DQN", "Unconstrained DQN"),
                           "dqn_lagrange": ("lagrange_dqn", "graph", True, "Lagrangian DQN", "Lagrangian DQN"),
                           "dqn_repair": ("repair_dqn", "graph", True, "Repair DQN", "Repair DQN"),
                           "ddqn_none": ("ddqn", "graph", True, "无约束 DDQN", "Unconstrained DDQN"),
                           "ddqn_lagrange": ("lagrange_ddqn", "graph", True, "Lagrangian DDQN", "Lagrangian DDQN"),
                           "ddqn_repair": ("repair_ddqn", "graph", True, "Repair DDQN", "Repair DDQN"),
                           },
              "title": ("结构感知网络：无约束 / Lagrangian / Maskable / Repair × PPO / DQN / DDQN（5 万实例、20M）",
                        "Structure-aware network: unconstrained / Lagrangian / Maskable / Repair x PPO / DQN / DDQN (50k instances, 20M)"),
              "parallel": 36, "gpu_round_robin": True, "time_bench": True,
              # v5.0.2 (second launch): one learner family at a time, so they do not crowd each other out on the GPUs.
              # DQN / DDQN first (CPU-bound, CUDA graphs: 1.46x), then PPO (GPU-bound: MPS 1.56x with 12 runs,
              # check_v5 speed_mask_ppo.md); an interim report after each stage but the last.
              "stages": [{"name": "DQN / DDQN", "learners": ["dqn", "ddqn"], "parallel": 27, "mps": False},   # DQN: MPS no gain (speed_mask_dqn_ABCD)
                         {"name": "PPO", "learners": ["ppo"], "parallel": 18, "mps": True}]},  # Legacy MPS (Pascal 1080 Ti): 27 clients fail with "Failed to create/open SHM segment", 18 ran fine
    "5.0.3": {"variants": {},                      # the lambda_max sweep only; its choice is used by 5.0.4
              "title": ("λ_max 扫描：Lagrangian PPO / DQN / DDQN × λ_max {1, 2, 5}（种子 1、3M、5 万实例）",
                        "λ_max sweep: Lagrangian PPO / DQN / DDQN x λ_max {1, 2, 5} (seed 1, 3M, 50k instances)"),
              "seeds": [1], "parallel": 18, "mps": True,
              "lam_sweep": {"algos": {"ppo": "lagrange_ppo", "dqn": "lagrange_dqn", "ddqn": "lagrange_ddqn"},
                            "values": [1.0, 2.0, 5.0], "steps": 3_000_000, "ckpts": [1_000_000]}},
    # 5.0.3.1 (sub-experiment, 1M): the Lagrangian dual step without the warm-up -- linear (v5.0.3) vs normalized error;
    # lambda_max 5 for all so that the cap does not hide where lambda settles. A 6th element = the run's own params.
    "5.0.3.1": {"variants": {f"{l}_lagrangian_{u}": (f"lagrange_{l}", "graph", True, f"{l.upper()} · lagrangian · {zu}",
                                                     f"{l.upper()} · lagrangian · {u}",
                                                     {"lambda_max": 5.0, "lambda_update": u, "lambda_lr": lr})
                             for l in ("ppo", "dqn", "ddqn")
                             for u, zu, lr in (("linear", "现行（η = 0.5）", 0.5), ("normalized", "归一化（η = 0.05）", 0.05))},
                "title": ("λ 更新方式子实验：现行 λ + 0.5(J_C − d) vs 归一化 λ + 0.05(J_C − d)/max(J_C, d)，无预热、λ_max = 5（种子 1、1M、5 万实例）",
                          "λ update sub-experiment: linear λ + 0.5(J_C − d) vs normalized λ + 0.05(J_C − d)/max(J_C, d), no warm-up, λ_max = 5 (seed 1, 1M, 50k instances)"),
                "seeds": [1], "steps": 1_000_000, "ckpts": [500_000], "parallel": 18, "mps": True},
    # 5.0.3.2: the same two dual steps as 5.0.3.1 without a cap on lambda (lambda_max "none"; the observation is then
    # lambda / (1 + lambda)), 20M with the default checkpoints, to see the curve lambda follows on its own
    "5.0.3.2": {"variants": {f"{l}_lagrangian_{u}": (f"lagrange_{l}", "graph", True, f"{l.upper()} · lagrangian · {zu}",
                                                     f"{l.upper()} · lagrangian · {u}",
                                                     {"lambda_max": "none", "lambda_update": u, "lambda_lr": lr})
                             for l in ("ppo", "dqn", "ddqn")
                             for u, zu, lr in (("linear", "现行（η = 0.5）", 0.5), ("normalized", "归一化（η = 0.05）", 0.05))},
                "title": ("取消 λ_max：现行 λ + 0.5(J_C − d) vs 归一化 λ + 0.05(J_C − d)/max(J_C, d)，λ 不设上限、观测为 λ/(1+λ)，无预热（种子 1、20M、5 万实例）",
                          "No λ_max: linear λ + 0.5(J_C − d) vs normalized λ + 0.05(J_C − d)/max(J_C, d), λ uncapped, observed as λ/(1+λ), no warm-up (seed 1, 20M, 50k instances)"),
                "seeds": [1], "parallel": 18, "mps": True},
    # 5.0.3.3: DQN / DDQN only, exploration decoupled from the dual update: epsilon falls 1 -> 0 over a fixed 2M steps
    # (not half of the run), J_C from 100 greedy (epsilon = 0) episodes per update, normalized step, no cap; 3 scenarios, 5M
    "5.0.3.3": {"variants": {f"{l}_lagrangian_greedy": (f"lagrange_{l}", "graph", True, f"{l.upper()} · lagrangian · greedy J_C",
                                                       f"{l.upper()} · lagrangian · greedy J_C",
                                                       {"lambda_max": "none", "lambda_update": "normalized", "lambda_lr": 0.05,
                                                        "lambda_cost": "greedy", "lambda_eval_episodes": 100, "eps_steps": 2_000_000})
                             for l in ("dqn", "ddqn")},
                "title": ("探索与 λ 更新解耦：ε 固定 2M 步降到 0、J_C 取 ε = 0 贪心策略（每次更新 100 局）、归一化更新 η = 0.05、λ 不设上限（种子 1、5M、5 万实例）",
                          "Exploration decoupled from the λ update: ε reaches 0 at a fixed 2M steps, J_C of the greedy (ε = 0) policy (100 episodes per update), "
                          "normalized update η = 0.05, λ uncapped (seed 1, 5M, 50k instances)"),
                "seeds": [1], "steps": 5_000_000, "ckpts": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
                "parallel": 18, "mps": True},
    # 5.0.3.4: 5.0.3.3 with lambda capped again (lambda_max 5) and a smaller normalized step (eta 0.025)
    "5.0.3.4": {"variants": {f"{l}_lagrangian_lambda_research": (f"lagrange_{l}", "graph", True, f"{l.upper()} · lagrangian · λ research",
                                                                f"{l.upper()} · lagrangian · λ research",
                                                                {"lambda_max": 5.0, "lambda_update": "normalized", "lambda_lr": 0.025,
                                                                 "lambda_cost": "greedy", "lambda_eval_episodes": 100, "eps_steps": 2_000_000})
                             for l in ("dqn", "ddqn")},
                "title": ("DQN / DDQN lagrangian λ research：λ_max = 5、归一化更新 η = 0.025，ε 从 1.0 固定 2M 步降到 0、J_C 取 ε = 0 贪心策略（每次更新 100 局）（种子 1、5M、5 万实例）",
                          "DQN / DDQN lagrangian λ research: λ_max = 5, normalized update η = 0.025, ε from 1.0 to 0 over a fixed 2M steps, "
                          "J_C of the greedy (ε = 0) policy (100 episodes per update) (seed 1, 5M, 50k instances)"),
                "seeds": [1], "steps": 5_000_000, "ckpts": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
                "parallel": 18, "mps": True},
    "5.0.3.5": {"variants": {f"{l}_lagrangian_dqn_series_20m": (f"lagrange_{l}", "graph", True, f"{l.upper()} · lagrangian · DQN series 20M",
                                                                f"{l.upper()} · lagrangian · DQN series 20M",
                                                                {"lambda_max": 5.0, "lambda_update": "normalized", "lambda_lr": 0.025,
                                                                 "lambda_cost": "greedy", "lambda_eval_episodes": 100, "eps_steps": 2_000_000})
                             for l in ("dqn", "ddqn")},
                "title": ("DQN series 20M：λ_max = 5、归一化更新 η = 0.025，ε 从 1.0 固定 2M 步降到 0、J_C 取 ε = 0 贪心策略（每次更新 100 局）（种子 1、20M、5 万实例）",
                          "DQN series 20M: λ_max = 5, normalized update η = 0.025, ε from 1.0 to 0 over a fixed 2M steps, "
                          "J_C of the greedy (ε = 0) policy (100 episodes per update) (seed 1, 20M, 50k instances)"),
                "seeds": [1], "steps": 20_000_000, "ckpts": [5_000_000, 10_000_000, 15_000_000],
                "parallel": 18, "mps": True},
    # 5.0.3.6: option (ii) -- PPO moves to the same dual step as DQN / DDQN (normalized, eta 0.025, lambda_max 5,
    # greedy J_C over 100 episodes); epsilon schedule (2M) for DQN / DDQN only; 5M, seed 1
    "5.0.3.6": {"variants": {f"{l}_lagrangian_normalized_eta": (f"lagrange_{l}", "graph", True, f"{l.upper()} · lagrangian · 归一化 η = 0.025",
                                                               f"{l.upper()} · lagrangian · normalized η = 0.025",
                                                               {"lambda_max": 5.0, "lambda_update": "normalized", "lambda_lr": 0.025,
                                                                "lambda_cost": "greedy", "lambda_eval_episodes": 100,
                                                                **({} if l == "ppo" else {"eps_steps": 2_000_000})})
                             for l in ("ppo", "dqn", "ddqn")},
                "title": ("PPO / DQN / DDQN lagrangian 统一 λ 更新：λ_max = 5、归一化更新 η = 0.025、J_C 取贪心策略（每次更新 100 局）；DQN / DDQN 的 ε 从 1.0 固定 2M 步降到 0（种子 1、5M、5 万实例）",
                          "PPO / DQN / DDQN lagrangian with one λ update: λ_max = 5, normalized update η = 0.025, J_C of the greedy policy "
                          "(100 episodes per update); ε for DQN / DDQN from 1.0 to 0 over a fixed 2M steps (seed 1, 5M, 50k instances)"),
                "seeds": [1], "steps": 5_000_000, "ckpts": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
                "parallel": 18, "mps": True},
    # 5.0.3.7: hyper-params lagrangian -- the usual Safe RL setting (OmniSafe PPOLag): lambda_init 0.001, Adam on
    # -lambda (J_C - d) with lr 0.035, lambda >= 0 without a cap, J_C of the behaviour policy, d = 0; PPO updates once per
    # rollout, DQN / DDQN every 1000 episodes (epsilon 1 -> 0 over 2M steps); 5M, seed 1
    "5.0.3.7": {"variants": {f"{l}_lagrangian_hyper_params": (f"lagrange_{l}", "graph", True, f"{l.upper()} · lagrangian · hyper-params",
                                                             f"{l.upper()} · lagrangian · hyper-params",
                                                             {"lambda_max": "none", "lambda_update": "adam", "lambda_lr": 0.035,
                                                              "lambda_init": 0.001, "lambda_target": 0.0, "lambda_cost": "behavior",
                                                              **({"lambda_window": "rollout"} if l == "ppo" else {"eps_steps": 2_000_000})})
                             for l in ("ppo", "dqn", "ddqn")},
                "title": ("hyper-params lagrangian：常规 Safe RL 设置（OmniSafe PPOLag）——λ 初值 0.001、Adam（lr 0.035）更新 λ、λ ≥ 0 不设上限、J_C 取行为策略、d = 0；PPO 每个 rollout 更新一次，DQN / DDQN 每 1000 局更新一次（ε 1.0→0 固定 2M 步）（种子 1、5M、5 万实例）",
                          "hyper-params lagrangian: the usual Safe RL setting (OmniSafe PPOLag) -- λ init 0.001, Adam (lr 0.035) on λ, λ ≥ 0 uncapped, "
                          "J_C of the behaviour policy, d = 0; PPO updates once per rollout, DQN / DDQN every 1000 episodes (ε 1.0→0 over a fixed 2M steps) (seed 1, 5M, 50k instances)"),
                "seeds": [1], "steps": 5_000_000, "ckpts": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
                "parallel": 18, "mps": True},
    "5.0.4": {"variants": {                         # every RL_algorithm x strategy (docs/data_spec.md), trained in this batch
                           "ppo_unconstraint": ("ppo", "graph", True, "PPO · unconstraint", "PPO · unconstraint"),
                           "ppo_lagrangian": ("lagrange_ppo", "graph", True, "PPO · lagrangian", "PPO · lagrangian"),
                           "ppo_maskable": ("mask_ppo", "graph", True, "PPO · maskable", "PPO · maskable"),
                           "ppo_repair": ("repair_ppo", "graph", True, "PPO · repair", "PPO · repair"),
                           "dqn_unconstraint": ("dqn", "graph", True, "DQN · unconstraint", "DQN · unconstraint"),
                           "dqn_lagrangian": ("lagrange_dqn", "graph", True, "DQN · lagrangian", "DQN · lagrangian"),
                           "dqn_maskable": ("mask_dqn", "graph", True, "DQN · maskable", "DQN · maskable"),
                           "dqn_repair": ("repair_dqn", "graph", True, "DQN · repair", "DQN · repair"),
                           "ddqn_unconstraint": ("ddqn", "graph", True, "DDQN · unconstraint", "DDQN · unconstraint"),
                           "ddqn_lagrangian": ("lagrange_ddqn", "graph", True, "DDQN · lagrangian", "DDQN · lagrangian"),
                           "ddqn_maskable": ("mask_ddqn", "graph", True, "DDQN · maskable", "DDQN · maskable"),
                           "ddqn_repair": ("repair_ddqn", "graph", True, "DDQN · repair", "DDQN · repair"),
                           },
              "title": ("RL_algorithm PPO / DQN / DDQN × strategies unconstraint / lagrangian / maskable / repair（种子 1、20M、5 万实例；λ_max 来自 v5.0.3）",
                        "RL_algorithm PPO / DQN / DDQN x strategies unconstraint / lagrangian / maskable / repair (seed 1, 20M, 50k instances; λ_max from v5.0.3)"),
              "seeds": [1], "parallel": 18, "mps": True, "time_bench": True,   # 18: Legacy MPS limit on the Pascal GPUs
              # lambda_max per RL_algorithm: chosen by the v5.0.3 sweep (its manifest's lambda_max); no sweep here
              "lam_max_from": "5.0.3"},
    # 5.1.0: lagrangian 20M version -- v5.0.3.7 hyper-params lagrangian unchanged, extended to 20M (checkpoints 5M / 10M / 15M); seed 1
    "5.1.0": {"variants": {f"{l}_lagrangian_20m_version": (f"lagrange_{l}", "graph", True, f"{l.upper()} · lagrangian 20M version",
                                                                 f"{l.upper()} · lagrangian 20M version",
                                                                 {"lambda_max": "none", "lambda_update": "adam", "lambda_lr": 0.035,
                                                                  "lambda_init": 0.001, "lambda_target": 0.0, "lambda_cost": "behavior",
                                                                  **({"lambda_window": "rollout"} if l == "ppo" else {"eps_steps": 2_000_000})})
                             for l in ("ppo", "dqn", "ddqn")},
                "title": ("lagrangian 20M version：沿用 v5.0.3.7 配置（λ 初值 0.001、Adam lr 0.035、λ ≥ 0 不设上限、J_C 取行为策略、d = 0；PPO 每个 rollout 更新一次，DQN / DDQN 每 1000 局、ε 1.0→0 固定 2M 步），扩展到 20M（种子 1、20M、5 万实例）",
                          "lagrangian 20M version: the v5.0.3.7 setting (λ init 0.001, Adam lr 0.035, λ ≥ 0 uncapped, J_C of the behaviour policy, d = 0; "
                          "PPO updates once per rollout, DQN / DDQN every 1000 episodes, ε 1.0→0 over a fixed 2M steps), extended to 20M (seed 1, 20M, 50k instances)"),
                "seeds": [1], "steps": 20_000_000, "ckpts": [5_000_000, 10_000_000, 15_000_000],
                "parallel": 18, "mps": True},
    # 5.3.1 / 5.3.2: BFT Repair / unconstraint -- two batches with the same training data (50k pool, seed 1) and the same
    # setting as 5.1.0: 20M (checkpoints 5M / 10M / 15M); DQN / DDQN ε 1.0→0 over a fixed 2M steps
    **{v: {"variants": {f"{l}_{key}": (algo(l), "graph", True, f"{l.upper()} · {lab}", f"{l.upper()} · {lab}",
                                       {} if l == "ppo" else {"eps_steps": 2_000_000}) for l in ("ppo", "dqn", "ddqn")},
           "title": (f"{lab}：PPO / DQN / DDQN，DQN / DDQN 的 ε 1.0→0 固定 2M 步（种子 1、20M、5 万实例）",
                     f"{lab}: PPO / DQN / DDQN, DQN / DDQN ε 1.0→0 over a fixed 2M steps (seed 1, 20M, 50k instances)"),
           "seeds": [1], "steps": 20_000_000, "ckpts": [5_000_000, 10_000_000, 15_000_000],
           "parallel": 18, "mps": True}
       for v, key, lab, algo in (("5.3.1", "bft_repair", "BFT Repair", lambda l: f"repair_{l}"),
                                 ("5.3.2", "unconstraint", "unconstraint", lambda l: l))},
    # 5.4.0: verication un/lag -- objective reward (terminal M·AR of the executed placement, violations ignored):
    # unconstraint (no mechanism) and Lagrangian (objective − λc_t, the 5.1.0 λ setting); else as 5.3.x
    "5.4.0": {"variants": {
        **{f"{l}_unconstraint_objective": (l, "graph", True, f"{l.upper()} · unconstraint · objective", f"{l.upper()} · unconstraint · objective",
                                           {"reward": "objective", **({} if l == "ppo" else {"eps_steps": 2_000_000})})
           for l in ("ppo", "dqn", "ddqn")},
        **{f"{l}_lagrangian_objective": (f"lagrange_{l}", "graph", True, f"{l.upper()} · lagrangian · objective", f"{l.upper()} · lagrangian · objective",
                                         {"reward": "objective", "lambda_max": "none", "lambda_update": "adam", "lambda_lr": 0.035,
                                          "lambda_init": 0.001, "lambda_target": 0.0, "lambda_cost": "behavior",
                                          **({"lambda_window": "rollout"} if l == "ppo" else {"eps_steps": 2_000_000})})
           for l in ("ppo", "dqn", "ddqn")}},
              "title": ("verication un/lag：objective 奖励（终局 M·AR，不管违约）下的 unconstraint 与 Lagrangian（objective − λc_t，λ 设置同 v5.1.0）；"
                        "DQN / DDQN 的 ε 1.0→0 固定 2M 步（种子 1、20M、5 万实例）",
                        "verication un/lag: unconstraint and Lagrangian (objective − λc_t, λ setting of v5.1.0) under the objective reward "
                        "(terminal M·AR, violations ignored); DQN / DDQN ε 1.0→0 over a fixed 2M steps (seed 1, 20M, 50k instances)"),
              "seeds": [1], "steps": 20_000_000, "ckpts": [5_000_000, 10_000_000, 15_000_000],
              "parallel": 18, "mps": True},
}

def references(version):
    """(label zh, label en, algo, [(seed, scen, exp_id, pattern, step)]) of the curves to compare against."""
    refs = []
    if version == "5.0.2-ppo-chain":                # not used: Maskable PPO is trained in 5.0.2 itself (one continuous run): v4.4.15 (1/3/5M), v4.4.16 (7.5/10M), v4.4.17 (15/20M)
        pts = []
        for ver, key, plan in (("v4.4.15", "mask_ppo_pool5m", [("ckpt_1000000_*", 1e6), ("ckpt_3000000_*", 3e6), ("model_*", 5e6)]),
                               ("v4.4.16", "mask_ppo_pool10m", [("ckpt_7500000_*", 7.5e6), ("model_*", 10e6)]),
                               ("v4.4.17", "mask_ppo_pool20m", [("ckpt_15000000_*", 15e6), ("model_*", 20e6)])):
            man = ROOT / "logs" / ver / "manifest.json"
            if not man.exists():
                continue
            ids = json.loads(man.read_text())["exp_ids"]
            pts += [(sd, s, ids[str(sd)][s][key], pat, int(st)) for sd in SEEDS for s in SCENARIOS for pat, st in plan
                    if ids.get(str(sd), {}).get(s, {}).get(key)]
        refs.append(("Maskable PPO（v4.4.15–17）", "Maskable PPO (v4.4.15-17)", "mask_ppo", True, pts))
    if version == "5.0.3-old-refs":                 # not used since the full 36-run plan; was: v5.0.2 (same setting, seed 1): DQN / DDQN unconstrained / Repair / old Lagrangian
        man = ROOT / "logs" / "v5.0.2" / "manifest.json"
        if man.exists():
            ids = json.loads(man.read_text())["exp_ids"]
            for key, (algo, _, _, lz, le) in CONFIG["5.0.2"]["variants"].items():
                if not key.startswith(("dqn_", "ddqn_")):
                    continue
                old = key.endswith("_lagrange")
                pts = [(sd, s, ids[str(sd)][s][key], pat, st) for sd in SEEDS for s in SCENARIOS
                       for pat, st in [(f"ckpt_{c}_*", c) for c in CKPTS] + [("model_*", STEPS)] if ids.get(str(sd), {}).get(s, {}).get(key)]
                refs.append((f"{lz}（{'旧，' if old else ''}v5.0.2）", f"{le} ({'old, ' if old else ''}v5.0.2)", algo, True, pts))
    if version == "5.0.2":                          # Maskable DQN / DDQN with the structure-aware network: v4.5.2
        man = ROOT / "logs" / "v4.5.2" / "manifest.json"
        if man.exists():
            ids = json.loads(man.read_text())["exp_ids"]
            for key, lz, le in (("dqn_graph", "Maskable DQN（v4.5.2）", "Maskable DQN (v4.5.2)"),
                                ("ddqn_graph", "Maskable DDQN（v4.5.2）", "Maskable DDQN (v4.5.2)")):
                algo = REF_452[key][0]
                pts = [(sd, s, ids[str(sd)][s][key], pat, st) for sd in SEEDS for s in SCENARIOS
                       for pat, st in [(f"ckpt_{c}_*", c) for c in CKPTS] + [("model_*", STEPS)] if ids.get(str(sd), {}).get(s, {}).get(key)]
                refs.append((lz, le, algo, True, pts))
    if version == "5.0.1":                          # the 50k half: v4.5.2
        man = ROOT / "logs" / "v4.5.2" / "manifest.json"
        if man.exists():
            ids = json.loads(man.read_text())["exp_ids"]
            for key, (algo, net, _, lz, le) in REF_452.items():
                pts = [(sd, s, ids[str(sd)][s][key], pat, st) for sd in SEEDS for s in SCENARIOS
                       for pat, st in [(f"ckpt_{c}_*", c) for c in CKPTS] + [("model_*", STEPS)] if ids.get(str(sd), {}).get(s, {}).get(key)]
                refs.append((lz, le, algo, True, pts))
    return refs


# ── placement (v5.0.0): GPU, NUMA node, MPS -- none of it changes what a run computes ──────────────────
def gpu_nodes() -> dict[int, int]:
    """CUDA index (PCI bus order, as nvidia-smi) -> NUMA node of the GPU (0 if unknown)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"], capture_output=True, text=True).stdout
    nodes = {}
    for line in out.strip().splitlines():
        m, bus = [x.strip() for x in line.split(",")]
        try:
            n = int(Path(f"/sys/bus/pci/devices/{bus[4:].lower()}/numa_node").read_text())
        except (OSError, ValueError):
            n = 0
        nodes[int(m)] = max(n, 0)
    return nodes


def mps_env() -> dict:
    d = ROOT / "logs" / "mps"
    return {"CUDA_MPS_PIPE_DIRECTORY": str(d / "pipe"), "CUDA_MPS_LOG_DIRECTORY": str(d / "log")}


def mps_start() -> None:
    """Start this project's MPS control daemon (own pipe directory: only processes given mps_env() use it)."""
    env = mps_env()
    for v in env.values():
        Path(v).mkdir(parents=True, exist_ok=True)
    if (Path(env["CUDA_MPS_PIPE_DIRECTORY"]) / "control").exists() and subprocess.run(
            ["pgrep", "-u", str(os.getuid()), "-x", "nvidia-cuda-mps"], capture_output=True).returncode == 0:
        print("[mps] daemon already running", flush=True)
        return
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env={**os.environ, **env}, check=True)
    print(f"[mps] daemon started (pipe {env['CUDA_MPS_PIPE_DIRECTORY']})", flush=True)


def mps_stop() -> None:
    subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env={**os.environ, **mps_env()})
    print("[mps] daemon stopped", flush=True)


def place(V, gpu: bool) -> tuple[int | None, int | None]:
    """(GPU, NUMA node) of a new run: a GPU run goes to the GPU with the fewest active runs and is bound to that
    GPU's node; a CPU run to the node with the fewest active runs."""
    act = V["active_res"].values()
    if gpu:
        g = min(V["gpu_node"], key=lambda m: (sum(1 for x in act if x[0] == m), m))
        return g, V["gpu_node"][g]
    nodes = sorted(set(V["gpu_node"].values()) | {0})
    return None, min(nodes, key=lambda n: (sum(1 for x in act if x[1] == n), n))


def new_exp_id():
    return subprocess.run([PY, "-c", "from shared.paths import new_exp_id; print(new_exp_id())"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sweep_key(key):
    """lam_sweep_<learner>_<value> -> (learner, value); None for any other key."""
    if not key.startswith("lam_sweep_"):
        return None
    _, _, learner, v = key.split("_")
    return learner, float(v)


def lagrange_learner(V, key):
    """The learner of a Lagrangian variant key (whose lambda_max comes from that learner's sweep), else None."""
    v = V["cfg"]["variants"].get(key)
    per_learner = V["cfg"].get("lam_sweep") or V["cfg"].get("lam_max_from")
    return v[0].split("_")[-1] if per_learner and v and v[0].startswith("lagrange_") else None


def spec(V, key):
    """(algo, net, 50k pool?, steps, checkpoints, extra train args) of a run key; v5.0.3: the lambda_max sweep keys
    lam_sweep_<learner>_<value>, and every Lagrangian run gets --lambda-max (its learner's sweep choice)."""
    sw = V["cfg"].get("lam_sweep")
    if sweep_key(key):
        learner, v = sweep_key(key)
        return (sw["algos"][learner], "graph", True, STEPS if SMOKE else sw["steps"], CKPTS if SMOKE else sw["ckpts"],
                ["--lambda-max", f"{v:g}"])
    algo, net, pool, *rest = V["cfg"]["variants"][key]
    own = rest[2] if len(rest) > 2 else {}           # 5.0.3.1: the variant's own params
    learner = lagrange_learner(V, key)
    extra = ["--lambda-max", f"{V['lam_max'][learner]:g}"] if learner else []
    flag = {"lambda_max": "--lambda-max", "lambda_update": "--lambda-update", "lambda_lr": "--lambda-lr",
            "lambda_cost": "--lambda-cost", "lambda_eval_episodes": "--lambda-eval-episodes", "eps_steps": "--eps-steps",   # 5.0.3.3
            "lambda_init": "--lambda-init", "lambda_target": "--lambda-target", "lambda_window": "--lambda-window",   # 5.0.3.7
            "reward": "--reward"}   # 5.4.0: overrides the default ar_pen (argparse keeps the last --reward)
    for k, v in own.items():
        extra += [flag[k], f"{v:g}" if isinstance(v, float) else str(v)]
    return algo, net, pool, STEPS, CKPTS, extra


def launch(V, seed, scen, key, exp_ids):
    algo, net, pool, steps, ckpts, extra = spec(V, key)
    exp_id = new_exp_id()
    log = open(V["log_dir"] / f"seed{seed}_{scen}_{key}_{exp_id}.log", "w")
    e = plan_entry(V, seed, scen, key)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(seed),
           "PAPER_VERSION": V["version"], "DATA_VERSION": DATA,
           # v5.0.3: run.json (docs/data_spec.md §2)
           "RUN_BATCH": V["log_dir"].name, "RUN_KEY": key, "RUN_ROLE": e["role"], "RUN_GROUP": e["group"] or "",
           "RUN_PARAMS": json.dumps(e["params"])}
    gpu = net == "graph"
    g, node = place(V, gpu)
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        if V["mps"]:
            env.update(mps_env())
    pre = ["numactl", f"--cpunodebind={node}", f"--preferred={node}"] if V["numa"] else []
    cmd = pre + [PY_GPU if gpu else PY, "-u", "-m", "paper_rl.train", "--scen", scen, "--algo", algo, "--net", net,
           "--reward", "ar_pen", "--reward-norm", "none", "--obs", "base", "--gamma", "1.0", "--full-episode",
           "--steps", str(steps), "--checkpoints", ",".join(map(str, ckpts)), "--seed", str(seed)] + extra
    cmd += ["--exit-action"] if algo.startswith("mask_") else []   # EXIT exists for Maskable only (PlacementEnv ignores it otherwise)
    cmd += ["--device", "cuda"] if gpu else []
    cmd += ["--train-pool", POOL] if pool else []
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    V["active_res"][proc.pid] = (g, node)
    exp_ids.setdefault(str(seed), {}).setdefault(scen, {})[key] = exp_id
    write_manifest(V, exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched seed{seed} {scen} {key} exp_id={exp_id} pid={proc.pid}"
          f" | {'GPU ' + str(g) if gpu else 'CPU'}{' MPS' if gpu and V['mps'] else ''}{f' node {node}' if V['numa'] else ''}", flush=True)
    return proc, log


STRATEGY = {"none": "unconstraint", "lagrange": "lagrangian", "mask": "maskable", "repair": "repair"}   # docs/data_spec.md §1


def manifest_lam_max(man: dict) -> dict:
    """{learner (lower case): lambda_max} of a manifest: schema 3 params.lambda_max {RL_algorithm: v}, schema 2 lambda_max."""
    lm = (man.get("params") or {}).get("lambda_max") if man.get("schema") == 3 else man.get("lambda_max")
    if lm is None:
        return {}
    return {k.lower(): v for k, v in lm.items()} if isinstance(lm, dict) else {l: lm for l in ("ppo", "dqn", "ddqn")}


def plan_entry(V, seed, scen, key):
    """One planned run of the batch (docs/data_spec.md §2, schema 3): RL_algorithm, strategy, seed, scenario, the
    params that differ from the batch's (lambda_max, steps of a sweep run), role / group / depends_on, dir.
    The lambda_max sweep runs are role "sweep", group "lambda_max_sweep_<RL_algorithm>"; with a sweep in the same
    batch, a Lagrangian run waits for its algorithm's group and gets params.lambda_max = its choice (null until then)."""
    from paper_rl.train import split_algo
    sw = V["cfg"].get("lam_sweep")
    if sweep_key(key):
        algo, steps = sw["algos"][sweep_key(key)[0]], STEPS if SMOKE else sw["steps"]
    else:
        algo, steps = V["cfg"]["variants"][key][0], STEPS
    mech, learner = split_algo(algo)
    rl = learner.upper()
    e = {"key": key, "role": "main", "group": None, "depends_on": [], "RL_algorithm": rl, "strategy": STRATEGY[mech],
         "seed": seed, "scenario": scen, "params": {}, "dir": algo}
    if steps != V["batch_steps"]:
        e["params"]["steps"] = steps
    if sweep_key(key):
        e.update(role="sweep", group=f"lambda_max_sweep_{rl}")
        e["params"]["lambda_max"] = sweep_key(key)[1]
    elif sw and mech == "lagrange":
        lm = V["lam_max"].get(learner)
        e["depends_on"] = [f"lambda_max_sweep_{rl}"]
        e["params"]["lambda_max"] = lm if lm is not None and lm == lm else None
    elif V["cfg"].get("lam_max_from") and mech == "lagrange":       # 5.0.4: fixed by an earlier sweep batch
        e["params"]["lambda_max"] = V["lam_max"][learner]
    if not sweep_key(key) and len(V["cfg"]["variants"][key]) > 5:   # 5.0.3.1: the variant's own params
        e["params"].update(V["cfg"]["variants"][key][5])
    return e


def ref_lam_max(run):
    """lambda_max of a finished Lagrangian run (runs before v5.0.3 do not record it: 50)."""
    try:
        return json.loads((run / "results.json").read_text())["training"].get("lambda_max", 50.0)
    except (OSError, KeyError, json.JSONDecodeError):
        return 50.0


def manifest_references(V):
    """The report's reference series (other batches' 20M models) as manifest v2 references; they always live in the
    default data space (formal batches), also when this batch is a smoke test."""
    from paper_rl.train import split_algo
    from shared.version_config import RESULTS_SPACE as REF_SPACE
    out = []
    for lz, le, algo, pool, pts in references(V["version"]):
        mech, learner = split_algo(algo)
        out += [{"label": {"zh": lz, "en": le}, "results_space": REF_SPACE, "scenario": s, "seed": sd, "algo": algo,
                 "learner": learner, "mechanism": mech, "net": "graph", "steps": st, "exp_id": eid,
                 "params": {"lambda_max": ref_lam_max(ROOT / "results" / REF_SPACE / s / algo / eid)} if mech == "lagrange" else {}}
                for sd, s, eid, pat, st in pts if pat == "model_*"]
    return out


def write_manifest(V, exp_ids):
    """manifest, schema 3 (docs/data_spec.md §2): RL_algorithm / strategies / seeds / scenarios and the shared params
    at the top, the whole plan up front (exp_id filled in at launch). "exp_ids" (nested {seed: {scen: {key: id}}}) is
    the driver's own index for --report / --rerun."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    runs = []
    for sd, s, k in V["plan_keys"]:
        e = plan_entry(V, sd, s, k)
        e["exp_id"] = exp_ids.get(str(sd), {}).get(s, {}).get(k)
        runs.append(e)
    if "refs" not in V:
        V["refs"] = manifest_references(V)
    sw = V["cfg"].get("lam_sweep")
    params = {"steps": V["batch_steps"], "checkpoints": V["batch_ckpts"], "net": "graph", "reward": "ar_pen", "gamma": 1.0,
              "full_episode": True, "exit_action": "maskable only", "train_pool": POOL, "data": DATA}
    if sw:
        params["lambda_max_sweep"] = {"values": sw["values"], "select": "lowest mean effective gap over the scenarios"}
    if sw or V.get("lam_max_from"):
        params["lambda_max"] = {l.upper(): v for l, v in V["lam_max"].items()}
    if V.get("lam_max_from"):
        params["lambda_max_from"] = V["lam_max_from"]
    tmp = V["manifest"].with_suffix(".tmp")
    tmp.write_text(json.dumps({"schema": 3, "batch": V["log_dir"].name, "version": V["version"],
                               "purpose": "smoke" if SMOKE else "main", "commit": commit, "results_space": RESULTS.name,
                               "driver_pid": os.getpid(), "created_at": V.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
                               "title": {"zh": V["cfg"]["title"][0], "en": V["cfg"]["title"][1]},
                               "RL_algorithm": [x for x in ("PPO", "DQN", "DDQN") if any(r["RL_algorithm"] == x for r in runs)],
                               "strategies": [x for x in ("unconstraint", "lagrangian", "maskable", "repair") if any(r["strategy"] == x for r in runs)],
                               "seeds": SEEDS, "scenarios": SCENARIOS, "params": params,
                               "runs": runs, "references": V["refs"], "exp_ids": exp_ids}, indent=2))
    tmp.replace(V["manifest"])


# ── evaluation ──────────────────────────────────────────────────────────────
def _gap(model, insts, learner, mech="mask", lam=0.0, lam_max=50.0):
    """Relative gap on the violation-free instances (Mask: not EXITed -- Mask never violates), the share of
    instances that are not (EXIT / any violation), and the capacity / privacy violation rates (v5.0.0)."""
    from paper_rl.train import evaluate
    obs = "raw" if model.policy.__class__.__name__.startswith("Graph") else "base"
    ev = evaluate(model, insts, mech, "ar_pen", lam, learner, obs, True, True, lam_max=lam_max)
    ok = [e for e in ev if not e["exited"] and e["cap_v"] == 0 and e["conf_v"] == 0]
    ar, ilp = (np.mean([e["ar"] for e in ok]), np.mean([e["ar_star"] for e in ok])) if ok else (np.nan, np.nan)
    return (float((ilp - ar) / ilp), 1 - len(ok) / len(ev),
            float(np.mean([e["cap_v"] > 0 for e in ev])), float(np.mean([e["conf_v"] > 0 for e in ev])))


def lagrange_lambda(run, mech, pat="model_*", step=None):
    """(lambda the policy was trained at, lambda_max = its scale in the observation) of a saved model (v5.0.0);
    runs before v5.0.3 do not record lambda_max: 50."""
    if mech != "lagrange":
        return 0.0, 50.0
    cap = lambda v: float("inf") if v == "none" else v     # v5.0.3.2: no cap ("none"); the env then observes lambda / (1 + lambda)
    if pat.startswith("ckpt_"):
        d = json.loads((run / "ckpt_lambda.json").read_text())
        return d[str(step)], cap(d.get("lambda_max", 50.0))
    t = json.loads((run / "results.json").read_text())["training"]
    return t["final_lambda"], cap(t.get("lambda_max", 50.0))


def ckpt_eval(job):
    """Test gap (own seed's 400 test instances), training-set gap (1600: a fixed subset of the 50k pool, or the
    1600-instance split itself) and test EXIT of one saved model."""
    import torch
    torch.set_num_threads(1)
    from paper_rl.train import model_class, split_algo, split_instances
    tag, algo, pool, sd, s, eid, pat, step = job
    files = list((RESULTS / s / algo / eid).glob(pat))
    if not files:
        return job, None
    mech, learner = split_algo(algo)
    _, train, test = split_instances(s, sd)
    if pool:
        p = json.loads((ROOT / POOL.format(scen=s)).read_text())
        train = [p[m] for m in sorted(random.Random(0).sample(range(len(p)), 1600))]
    model = model_class(learner, mech).load(str(files[0]), device="cpu")
    lam, lam_max = lagrange_lambda(RESULTS / s / algo / eid, mech, pat, step)
    (te, ex, cap, conf), (tr, ex_tr, *_) = _gap(model, test, learner, mech, lam, lam_max), _gap(model, train, learner, mech, lam, lam_max)
    return job, (te, tr, ex, cap, conf, ex_tr)


def regret(V, key, algo, seed):
    md = V["out"] / f"regret_s{seed}_{key}.md"
    js = md.with_suffix(".json")
    if not js.exists():
        lz, le = V["cfg"]["variants"][key][3:5]
        subprocess.run([PY, str(ROOT / "src" / "scripts" / "diag_regret.py"), "--out", str(md), "--seed", str(seed), "--algo", algo,
                        *(["--n", "25", "--workers", "24"] if SMOKE else []),
                        "--manifest", str(V["manifest"].relative_to(ROOT)), "--key", key,
                        "--label", f"v{V['version']} {lz}（种子 {seed}、{STEPS / 1e6:g}M）", "--label-en", f"v{V['version']} {le} (seed {seed}, {STEPS / 1e6:g}M)"],
                       cwd=ROOT, check=True)
    return json.loads(js.read_text())


# ── per-instance time: every 20M model vs the ILP (v5.0.2) ────────────────────────────────────────────────
ILP_WORKERS, TIME_WORKERS, WARMUP = 12, 24, 5


def test_index(scen, seed):
    """Indices (into the scenario's instance file) of a seed's test split, as paper_rl.train.split_instances."""
    from paper_rl.data import load
    n = len(load(scen)["instances"])
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx[int(0.8 * n):]


def policy_time(job):
    """One 20M model on its seed's test split, one CPU thread: full deterministic episode (reset, mask,
    forward, step) and the forward passes alone, per instance."""
    import torch
    torch.set_num_threads(1)
    from paper_rl.env import PlacementEnv
    from paper_rl.train import model_class, split_algo, split_instances
    m, algo, sd, s, eid = job
    files = list((RESULTS / s / algo / eid).glob("model_*"))
    if not files:
        return job, None
    mech, learner = split_algo(algo)
    test = split_instances(s, sd)[2]
    model = model_class(learner, mech).load(str(files[0]), device="cpu")
    lam, lam_max = lagrange_lambda(RESULTS / s / algo / eid, mech)
    obs_mode = "raw" if model.policy.__class__.__name__.startswith("Graph") else "base"
    env = PlacementEnv(test, mech, "ar_pen", lam=lam, obs_mode=obs_mode, full_episode=True, exit_action=True, lam_max=lam_max)
    masked_ppo = learner == "ppo" and mech == "mask"

    def episode(k):
        env.use_instance(k)
        obs, _ = env.reset()
        fwd, done = 0.0, False
        while not done:
            t = time.perf_counter()
            a, _ = model.predict(obs, deterministic=True, **({"action_masks": env.action_masks()} if masked_ppo else {}))
            fwd += time.perf_counter() - t
            obs, _, done, _, _ = env.step(int(a))
        return fwd
    for k in range(WARMUP):
        episode(k)
    rows = []
    for k in range(len(test)):
        t0 = time.perf_counter()
        fwd = episode(k)
        rows.append(((time.perf_counter() - t0) * 1000, fwd * 1000))
    return job, rows


def ilp_time(args):
    from paper_rl.data import load
    from shared.ilp_utils import solve_ilp_max_ar
    s, gi = args
    x = load(s)["instances"][gi]
    t0 = time.perf_counter()
    solve_ilp_max_ar(x["ECUs"], x["SVCs"], x["conflict_sets"])
    return s, gi, (time.perf_counter() - t0) * 1000


def time_bench(series):
    """{(series idx, scen): (model ms list, forward ms list, ILP ms list on the same instances)}; the ILP is solved
    once per instance of a scenario (union of the seeds' test splits), single-threaded CBC, model building included."""
    tidx = {(s, sd): test_index(s, sd) for s in SCENARIOS for sd in SEEDS}
    jobs = [(m, algo, sd, s, eid) for m, (_, _, algo, _, pts) in enumerate(series) for sd, s, eid, pat, st in pts if pat == "model_*"]
    print(f"[report] timing {len(jobs)} models (1 thread each)", flush=True)
    with Pool(min(len(jobs), TIME_WORKERS)) as pool:
        pol = pool.map(policy_time, jobs)
    todo = sorted({(s, gi) for (s, sd), ix in tidx.items() for gi in ix})
    print(f"[report] timing the ILP on {len(todo)} instances ({ILP_WORKERS} single-thread solvers)", flush=True)
    with Pool(ILP_WORKERS) as pool:
        ilp = {(s, gi): ms for s, gi, ms in pool.map(ilp_time, todo, chunksize=4)}
    out = {}
    for (m, algo, sd, s, eid), rows in pol:
        if rows is None:
            continue
        m_, f, l = out.setdefault((m, s), ([], [], []))
        m_ += [r[0] for r in rows]
        f += [r[1] for r in rows]
        l += [ilp[(s, gi)] for gi in tidx[(s, sd)]]
    return out


def lam_dynamics(run, cap):
    """How lambda moved during one Lagrangian run (training_curve.csv, one row per episode): final / max lambda, the
    first episode at the cap, share of episodes at the cap, number of decreases and the total fall from the peak, and
    J_C (violations per episode) over the first and the last 1000 episodes."""
    import pandas as pd
    p = run / "training_curve.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["lambda", "episode_cap_violated", "episode_conflict_violated"])
    lam = df["lambda"].to_numpy(dtype=float)
    v = (df["episode_cap_violated"] + df["episode_conflict_violated"]).to_numpy(dtype=float)
    at = lam >= cap * 0.999
    d = np.diff(lam)
    return {"episodes": len(lam), "final": float(lam[-1]), "max": float(lam.max()),
            "first_cap": int(np.argmax(at)) if at.any() else None, "at_cap": float(at.mean()),
            "decreases": int((d < -1e-9).sum()), "fall": float(lam.max() - lam[-1]),
            "jc_first": float(v[:1000].mean()), "jc_last": float(v[-1000:].mean())}


def pd_upd(run):
    """lambda_updates.csv of a Lagrangian run (v5.0.3.3) as an array (episode, timestep, J_C behaviour, J_C used, lambda), else None."""
    p = run / "lambda_updates.csv"
    if not p.exists():
        return None
    a = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
    return a if a.size else None


def write_report_data(V, series, curve, reg, times, interim):
    """report_data.json next to report.md (docs/data_spec.md §3 rule 7): the report's numbers per run, for the panel.
    curves: one point per (run, checkpoint); timing: per (series, scenario) over the seeds; ILP once per scenario."""
    keys = list(V["cfg"]["variants"])
    nan = lambda x: None if x is None or x != x else float(x)
    pts = []
    for m, (lz, le, algo, _, runs) in enumerate(series):
        for sd, s, eid, pat, st in runs:
            c = curve.get((m, s, sd, st))
            if c is None:
                continue
            te, tr, ex, cap, conf, ex_tr = c
            r = reg.get((keys[m], s, sd)) if m < len(keys) else None
            pts.append({"series": m, "key": keys[m] if m < len(keys) else None, "label": {"zh": lz, "en": le},
                        "algo": algo, "scenario": s, "seed": sd, "exp_id": eid, "step": st,
                        "test_gap": nan(te), "train_gap": nan(tr), "test_fail": nan(ex), "train_fail": nan(ex_tr),
                        "cap_viol": nan(cap), "priv_viol": nan(conf),
                        "regret_early": nan(r[0]) if r and st == STEPS else None,
                        "opt_rate_early": nan(r[1]) if r and st == STEPS else None})
    timing, ilp = [], {}
    for (m, s), (m_, f, l) in times.items():
        lz, le, algo = series[m][:3]
        timing.append({"series": m, "key": keys[m] if m < len(keys) else None, "label": {"zh": lz, "en": le}, "algo": algo,
                       "scenario": s, "mean_ms": float(np.mean(m_)), "median_ms": float(np.median(m_)),
                       "p95_ms": float(np.percentile(m_, 95)), "forward_mean_ms": float(np.mean(f)),
                       "speedup_mean": float(np.mean(l) / np.mean(m_)), "speedup_median": float(np.median(l) / np.median(m_))})
        ilp.setdefault(s, {"mean_ms": float(np.mean(l)), "median_ms": float(np.median(l)), "p95_ms": float(np.percentile(l, 95))})
    out = {"schema": 1, "batch": V["log_dir"].name, "version": V["version"], "interim": interim,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": CKPTS + [STEPS], "early_steps": EARLY,
           "curves": pts, "timing": timing, "ilp_timing": ilp}
    (V["out"] / "report_data.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


def write_sweep_report(V):
    """report.md (+ .en.md) of a sweep-only batch (5.0.3): the chosen lambda_max per learner, then each learner's
    lam_sweep_<learner>.md as written by choose_lam_max."""
    learners = list(V["cfg"]["lam_sweep"]["algos"])
    for zh in (True, False):
        T = lambda z, e: z if zh else e
        L = [f"# v{V['version']} {T(*V['cfg']['title'])}", "",
             T(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/run_v5.py --version {V['version']}`；"
               f"manifest：`{V['manifest'].relative_to(ROOT)}`。",
               f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; script `src/scripts/run_v5.py --version {V['version']}`; "
               f"manifest: `{V['manifest'].relative_to(ROOT)}`."),
             T("- 选定的 λ_max 由下一版本（正式实验）读取：manifest 的 `lambda_max`。",
               "- The chosen λ_max is read by the next version (the main experiment): `lambda_max` in the manifest."), "",
             T("| Learner | 选定 λ_max |", "| Learner | Chosen λ_max |"), "|---|---|"]
        L += [f"| {l.upper()} | {V['lam_max'].get(l, float('nan')):g} |" for l in learners]
        for l in learners:
            f = V["out"] / (f"lam_sweep_{l}.md" if zh else f"lam_sweep_{l}.en.md")
            if f.exists():
                L += ["", "#" + f.read_text().split("\n", 1)[0], *f.read_text().split("\n")[1:]]
        p = V["report"] if zh else V["report"].with_suffix(".en.md")
        p.write_text("\n".join(L) + "\n")
        print(f"report -> {p}", flush=True)


def write_report(V, exp_ids, interim: bool = False):
    if not V["cfg"]["variants"]:                     # 5.0.3: the lambda_max sweep only
        return write_sweep_report(V)
    vs = V["cfg"]["variants"]
    series = [(lz, le, algo, pool, [(sd, s, exp_ids[str(sd)][s][k], pat, st) for sd in SEEDS for s in SCENARIOS
                                   for pat, st in [(f"ckpt_{c}_*", c) for c in CKPTS] + [("model_*", STEPS)]
                                   if exp_ids.get(str(sd), {}).get(s, {}).get(k)])
              for k, (algo, net, pool, lz, le, *_) in vs.items()] + references(V["version"])
    jobs = [(m, algo, pool, *p) for m, (_, _, algo, pool, pts) in enumerate(series) for p in pts]
    print(f"[report] learning-curve evaluation ({len(jobs)} models)", flush=True)
    with Pool(min(len(jobs), 36)) as pool:
        out = pool.map(ckpt_eval, jobs)
    curve = {}                                       # (series idx, scen, seed, step) -> (test, train, exit)
    for job, r in out:
        if r is not None:
            curve[(job[0], job[4], job[3], job[7])] = r
    reg = {}
    # per-step regret is defined over legal ECUs, i.e. for Maskable policies only (v5.0.0)
    # ... and only for runs that have finished in every scenario (an interim report comes before later stages)
    done = lambda k, algo, sd: all((RESULTS / s / algo / exp_ids.get(str(sd), {}).get(s, {}).get(k, "-") / "results.json").exists()
                                   for s in SCENARIOS)
    todo = [(k, v[0], sd) for k, v in vs.items() for sd in SEEDS if v[0].startswith("mask_") and done(k, v[0], sd)]
    for m, (k, algo, sd) in enumerate(todo, 1):
        print(f"[report] regret {m}/{len(todo)} | seed {sd} {k}", flush=True)
        g = regret(V, k, algo, sd)
        for s in SCENARIOS:
            reg[(k, s, sd)] = (sum(g[s]["regret_t"][:EARLY]), float(np.mean(g[s]["opt_rate_t"][:EARLY])))
    times = time_bench(series) if V["cfg"].get("time_bench") and not interim else {}
    steps = CKPTS + [STEPS]
    write_report_data(V, series, curve, reg, times, interim)
    # lambda dynamics of every Lagrangian run of this batch (v5.0.3.1)
    lamdyn = []
    for k, v in vs.items():
        if not v[0].startswith("lagrange_"):
            continue
        cap = (v[5] if len(v) > 5 else {}).get("lambda_max") or V["lam_max"].get(v[0].split("_")[-1]) or 50.0
        cap = float("inf") if cap == "none" else cap     # v5.0.3.2: no cap
        for sd in SEEDS:
            for s in SCENARIOS:
                eid = exp_ids.get(str(sd), {}).get(s, {}).get(k)
                dyn = lam_dynamics(RESULTS / s / v[0] / eid, cap) if eid else None
                if dyn:
                    lamdyn.append((k, sd, s, cap, dyn))
    ms = lambda v, f: (lambda v: (f(np.mean(v)) + (f" ± {f(np.std(v, ddof=1))}" if len(v) > 1 else "")) if v else "—")(
        [x for x in v if x == x])                    # NaN (no violation-free instance) is left out
    pc = lambda v: "—" if v != v else f"{100 * v:.1f}%"
    sl = lambda st: f"{st / 1e6:g}M"
    for lang in ("zh", "en"):
        zh = lang == "zh"
        T = lambda z, e: z if zh else e
        lab = lambda m: series[m][0] if zh else series[m][1]
        L = [f"# v{V['version']} {T(*V['cfg']['title'])}", "",
             *([T("> **中间报告**：只包含已完成的阶段；其余组显示为 —，全部阶段完成后会被最终报告覆盖。",
                  "> **Interim report**: only the stages finished so far; the other groups show —. The final report replaces it."), ""]
               if interim else []),
             T(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `src/scripts/run_v5.py --version {V['version']}`。",
               f"- Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; script `src/scripts/run_v5.py --version {V['version']}`."),
             T(f"- 每个变体 {len(SEEDS)} 个种子（{', '.join(map(str, SEEDS))}）× {len(SCENARIOS)} 场景（{' / '.join(x.upper() for x in SCENARIOS)}）、各训练 {STEPS / 1e6:g}M 步，同一次训练在 "
               f"{' / '.join(f'{c / 1e6:g}M' for c in CKPTS) or '—'} 存 checkpoint；ar_pen 奖励、γ = 1、不提前结束、EXIT 只用于 Maskable、需求降序；"
               "PPO / DQN 超参取自 `paper_rl/config.py`；结构感知网络（Transformer）在 GPU 上训练。5 万实例 = 经 ILP 验证的实例池（`results/sup_diag/<场景>.json`）。",
               f"- Every variant: {len(SEEDS)} seed(s) ({', '.join(map(str, SEEDS))}) × {len(SCENARIOS)} scenario(s) ({' / '.join(x.upper() for x in SCENARIOS)}), {STEPS / 1e6:g}M steps each, checkpoints of the same run at "
               f"{' / '.join(f'{c / 1e6:g}M' for c in CKPTS) or '—'}; ar_pen reward, γ = 1, full episode, EXIT for Maskable only, descending demand; "
               "PPO / DQN hyper-parameters from `paper_rl/config.py`; structure-aware network (Transformer) trained on GPU. "
               "50k instances = the ILP-verified pool (`results/sup_diag/<scenario>.json`)."),
             T("- 测试集 gap：各种子自己的 400 个测试实例；训练集 gap：1600 个训练实例（5 万实例池中按固定种子抽取，或 1600 实例划分本身）；均为确定性回放；均值 ± 样本标准差。",
               "- Test gap: each seed's own 400 test instances; training-set gap: 1600 training instances (a fixed subset of the 50k pool, or the "
               "1600-instance split itself); deterministic rollouts; mean ± sample std."),
             T(f"- manifest：`{V['manifest'].relative_to(ROOT)}`。", f"- Manifest: `{V['manifest'].relative_to(ROOT)}`."), ""]
        if lamdyn:
            d = "/".join(sorted({f"{(vs[k][5] if len(vs[k]) > 5 else {}).get('lambda_target', 0.001):g}" for k, *_ in lamdyn}))   # 5.0.3.7: d from config
            L += [T("## λ 动态（训练过程，来自 training_curve.csv）", "## λ dynamics (during training, from training_curve.csv)"), "",
                  T(f"J_C = 每局违约次数（容量 + 隐私）；目标 d = {d}。「到上限」= 第一次 λ ≥ λ_max 的局；「降幅」= 最高点到训练结束的下降。",
                    f"J_C = violations per episode (capacity + privacy); target d = {d}. 'First at cap' = first episode with λ ≥ λ_max; "
                    "'fall' = from the peak to the end of training."), "",
                  T("| 场景 | 模型 | λ_max | 末 λ | 最高 λ | 到上限（局） | 上限占比 | 下降次数 | 降幅 | 前 1000 局 J_C | 末 1000 局 J_C |",
                    "| Scenario | Model | λ_max | Final λ | Max λ | First at cap (episode) | Share at cap | Decreases | Fall | J_C first 1000 | J_C last 1000 |"),
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for s in SCENARIOS:
                for k, sd, s2, cap, dy in lamdyn:
                    if s2 != s:
                        continue
                    lz, le = vs[k][3:5]
                    L.append(f"| {s.upper()} | {lz if zh else le} | {T('无', 'none') if cap == float('inf') else f'{cap:g}'} | {dy['final']:.3f} | {dy['max']:.3f} | "
                             f"{dy['first_cap'] if dy['first_cap'] is not None else '—'} | {pc(dy['at_cap'])} | {dy['decreases']} | "
                             f"{dy['fall']:.3f} | {dy['jc_first']:.3f} | {dy['jc_last']:.4f} |")
            L.append("")
            upd = [(k, s, pd_upd(RESULTS / s / vs[k][0] / exp_ids[str(sd)][s][k])) for k, sd, s, cap, dy in lamdyn]
            upd = [(k, s, u) for k, s, u in upd if u is not None]
            if upd:                                  # 5.0.3.3: λ and both J_C at each checkpoint (lambda_updates.csv)
                L += [T("### λ 与 J_C（来自 lambda_updates.csv，各步数前最后一次更新）", "### λ and J_C (from lambda_updates.csv, last update before each step)"), "",
                      T("每格：λ / 行为策略 J_C / 用于更新的 J_C（贪心时为 ε = 0 的局）。", "Each cell: λ / behaviour-policy J_C / J_C used for the update (greedy: ε = 0 episodes)."), "",
                      "| " + T("场景", "Scenario") + " | " + T("模型", "Model") + " | " + " | ".join(sl(st) for st in steps) + " |",
                      "|---|---|" + "---|" * len(steps)]
                for k, s, u in upd:
                    cells = []
                    for st in steps:
                        r = u[u[:, 1] <= st]
                        cells.append("—" if not len(r) else f"{r[-1, 4]:.2f} / {r[-1, 2]:.3f} / {r[-1, 3]:.3f}")
                    L.append(f"| {s.upper()} | {vs[k][3] if zh else vs[k][4]} | " + " | ".join(cells) + " |")
                L.append("")
        L += [T("## 学习曲线：测试集相对 gap", "## Learning curve: test relative gap"), ""]
        for s in SCENARIOS:
            L += [f"### {s.upper()}", "", "| " + T("步数", "Steps") + " | " + " | ".join(lab(m) for m in range(len(series))) + " |",
                  "|---|" + "---|" * len(series)]
            for st in steps:
                L.append(f"| {sl(st)} | " + " | ".join(ms([curve[(m, s, sd, st)][0] for sd in SEEDS if (m, s, sd, st) in curve], pc)
                                                        for m in range(len(series))) + " |")
            L.append("")
        L += [T("## 学习曲线：三场景平均（每个种子先对三场景取平均）", "## Learning curve: mean over the three scenarios (per seed first)"), "",
              "| " + T("步数", "Steps") + " | " + " | ".join(lab(m) for m in range(len(series))) + " |", "|---|" + "---|" * len(series)]
        for st in steps:
            cells = []
            for m in range(len(series)):
                v = [np.mean([curve[(m, s, sd, st)][0] for s in SCENARIOS]) for sd in SEEDS if all((m, s, sd, st) in curve for s in SCENARIOS)]
                cells.append(ms(v, pc))
            L.append(f"| {sl(st)} | " + " | ".join(cells) + " |")
        L += ["", T("## 学习曲线：测试集未通过率（EXIT 或任一违约）", "## Learning curve: test failure rate (EXIT or any violation)"), "",
              T("gap 只在未通过率之外的实例（无 EXIT、无违约）上计算；未通过率高时，gap 只反映容易的那部分实例。",
                "The gap is computed on the remaining instances (no EXIT, no violation); with a high failure rate it covers only the easier instances."), ""]
        for s in SCENARIOS:
            L += [f"### {s.upper()}", "", "| " + T("步数", "Steps") + " | " + " | ".join(lab(m) for m in range(len(series))) + " |",
                  "|---|" + "---|" * len(series)]
            for st in steps:
                L.append(f"| {sl(st)} | " + " | ".join(ms([curve[(m, s, sd, st)][2] for sd in SEEDS if (m, s, sd, st) in curve], pc)
                                                        for m in range(len(series))) + " |")
            L.append("")
        L += ["", T(f"## {sl(STEPS)}：测试集与训练集 gap、违约、前几步决策", f"## {sl(STEPS)}: test and training-set gap, violations, early decisions"), "",
              T(f"| 场景 | 模型 | 测试集相对 gap | 训练集相对 gap | 测试未通过率（EXIT / 违约） | 训练集未通过率 | 容量违约率 | 隐私违约率 | 第 1–{EARLY} 步 regret | 第 1–{EARLY} 步选中最优 |",
                f"| Scenario | Model | Test relative gap | Train relative gap | Test failure rate (EXIT / violation) | Train failure rate | Capacity violation rate | Privacy violation rate | Steps 1–{EARLY} regret | Steps 1–{EARLY} optimal-action rate |"),
              "|---|---|---|---|---|---|---|---|---|---|"]
        keys = list(vs)
        for s in SCENARIOS:
            for m in range(len(series)):
                c = [curve[(m, s, sd, STEPS)] for sd in SEEDS if (m, s, sd, STEPS) in curve]
                r = [reg[(keys[m], s, sd)] for sd in SEEDS if m < len(keys) and (keys[m], s, sd) in reg]
                L.append(f"| {s.upper()} | {lab(m)} | {ms([x[0] for x in c], pc)} | {ms([x[1] for x in c], pc)} | {ms([x[2] for x in c], pc)} | "
                         f"{ms([x[5] for x in c], pc)} | {ms([x[3] for x in c], pc)} | {ms([x[4] for x in c], pc)} | "
                         f"{ms([x[0] for x in r], lambda v: f'{v:.4f}')} | {ms([x[1] for x in r], pc)} |")
        L += ["", T("regret 按合法 ECU 之间的最优差距定义，只适用于 Maskable；其余机制可能执行违约动作，记为 —。",
                    "Regret is defined over the legal ECUs, i.e. for Maskable only; the other mechanisms may execute violating actions (—)."), ""]
        L += ["", T(f"## 逐种子测试集 gap（{sl(STEPS)}）", f"## Test gap per seed ({sl(STEPS)})"), "",   # columns = this batch's seeds
              T("| 场景 | 模型 | ", "| Scenario | Model | ") + " | ".join(T(f"种子 {sd}", f"Seed {sd}") for sd in SEEDS) + " |",
              "|---|---|" + "---|" * len(SEEDS)]
        for s in SCENARIOS:
            for m in range(len(series)):
                L.append(f"| {s.upper()} | {lab(m)} | " + " | ".join(pc(curve[(m, s, sd, STEPS)][0]) if (m, s, sd, STEPS) in curve else "—"
                                                                    for sd in SEEDS) + " |")
        L.append("")
        if times:
            q = lambda v, p_: float(np.percentile(v, p_))
            L += [T(f"## 求解时间：每个 {sl(STEPS)} 模型 vs ILP（逐实例）", f"## Solve time: every {sl(STEPS)} model vs the ILP (per instance)"), "",
                  T("单线程 CPU。模型：完整确定性 episode（reset、掩码、前向、step），另列仅前向；ILP：Dinkelbach + CBC 单线程，含建模，"
                    "每个场景对三个种子测试集的并集各求一次。加速比 = ILP 时间 / 模型时间，在同一批实例上计算（每个种子 400 个，三个种子合并）。",
                    "One CPU thread. Model: full deterministic episode (reset, mask, forward, step), and the forward passes alone; ILP: "
                    "Dinkelbach + single-thread CBC, model building included, solved once per instance of the union of the three seeds' test "
                    "splits. Speed-up = ILP time / model time on the same instances (400 per seed, three seeds pooled)."), ""]
            for s in SCENARIOS:
                L += [f"### {s.upper()}", "",
                      T("| 方法 | 平均 ms | 中位数 ms | P95 ms | 仅前向 平均 ms | 加速比（平均） | 加速比（中位数） |",
                        "| Method | Mean ms | Median ms | P95 ms | Forward only, mean ms | Speed-up (mean) | Speed-up (median) |"),
                      "|---|---|---|---|---|---|---|"]
                ilp_all = next((v[2] for (m, s2), v in times.items() if s2 == s), None)
                if ilp_all:
                    L.append(f"| ILP | {np.mean(ilp_all):.1f} | {q(ilp_all, 50):.1f} | {q(ilp_all, 95):.1f} | — | — | — |")
                for m in range(len(series)):
                    if (m, s) not in times:
                        continue
                    m_, f, l = times[(m, s)]
                    L.append(f"| {lab(m)} | {np.mean(m_):.1f} | {q(m_, 50):.1f} | {q(m_, 95):.1f} | {np.mean(f):.1f} | "
                             f"{np.mean(l) / np.mean(m_):.1f}× | {q(l, 50) / q(m_, 50):.1f}× |")
                L.append("")
        p = V["report"] if zh else V["report"].with_suffix(".en.md")
        p.write_text("\n".join(L))
        print(f"report -> {p}", flush=True)


# ── lambda_max sweep (v5.0.3) ───────────────────────────────────────────────
def sweep_eval(job):
    """Effective gap (violating instances count as AR = 0), violation-free share and the gap on the violation-free
    instances of one sweep model on its seed's test split."""
    import torch
    torch.set_num_threads(1)
    from paper_rl.train import evaluate, model_class, split_algo, split_instances
    algo, sd, s, eid = job
    run = RESULTS / s / algo / eid
    mech, learner = split_algo(algo)
    model = model_class(learner, mech).load(str(next(run.glob("model_*"))), device="cpu")
    lam, lam_max = lagrange_lambda(run, mech)
    ev = evaluate(model, split_instances(s, sd)[2], mech, "ar_pen", lam, learner, "raw", True, True, lam_max=lam_max)
    ok = [e["cap_v"] == 0 and e["conf_v"] == 0 for e in ev]
    ilp = np.mean([e["ar_star"] for e in ev])
    eff = 1 - np.mean([e["ar"] * o for e, o in zip(ev, ok)]) / ilp
    g = [e for e, o in zip(ev, ok) if o]
    gap = (np.mean([e["ar_star"] for e in g]) - np.mean([e["ar"] for e in g])) / np.mean([e["ar_star"] for e in g]) if g else float("nan")
    return job, (float(eff), float(np.mean(ok)), float(gap), lam)


LAM_TIE = 0.005     # mean violation-free shares within 0.5 pp of the highest are a tie (one seed, 400 test instances per scenario)


def select_lam_max(values, eff, free) -> float:
    """The selection rule (2026-10-13, user: a 3M sweep says little about the gap, violations weigh more): highest
    mean violation-free share over the scenarios; values within LAM_TIE of it are a tie, broken by the lower mean
    effective gap. eff / free: {value: mean over the scenarios}."""
    hi = max(free[v] for v in values)
    tied = [v for v in values if hi - free[v] <= LAM_TIE + 1e-12]
    return min(tied, key=lambda v: eff[v])


def write_lam_sweep(V, learner, rows, override=None) -> float:
    """Apply select_lam_max to one learner's evaluated sweep and write lam_sweep_<learner>.md / .en.md / .json.
    rows: [{lambda_max, scenario, seed, exp_id, eff_gap, viol_free, gap, final_lambda}] (one seed). Returns the choice."""
    sw = dict(V["cfg"]["lam_sweep"], algo=V["cfg"]["lam_sweep"]["algos"][learner])
    values = sw["values"]
    cell = {(r["lambda_max"], r["scenario"]): r for r in rows}
    nn = lambda x: float("nan") if x is None else x
    eff = {v: float(np.mean([nn(cell[(v, s)]["eff_gap"]) for s in SCENARIOS])) for v in values}
    free = {v: float(np.mean([nn(cell[(v, s)]["viol_free"]) for s in SCENARIOS])) for v in values}
    rule = select_lam_max(values, eff, free)
    best = float(override) if override else rule
    pc = lambda x: "—" if x is None or x != x else f"{100 * x:.1f}%"
    for zh in (True, False):
        T = lambda z, e: z if zh else e
        L = [f"# v{V['version']} " + T(f"λ_max 扫描（{learner.upper()}）", f"λ_max sweep ({learner.upper()})"), "",
             T(f"- {sw['algo']}、种子 {SEEDS[0]}、{sw['steps'] / 1e6:g}M 步，λ_max ∈ {{{', '.join(f'{v:g}' for v in values)}}}；"
               "测试集 400 实例确定性评估。",
               f"- {sw['algo']}, seed {SEEDS[0]}, {sw['steps'] / 1e6:g}M steps, λ_max ∈ {{{', '.join(f'{v:g}' for v in values)}}}; "
               "deterministic evaluation on the 400 test instances."),
             T("- 有效 gap：有违约的实例按 AR = 0 计，1 − 平均 AR / 平均 ILP AR。",
               "- Effective gap: a violating instance counts as AR = 0, 1 − mean AR / mean ILP AR."),
             T(f"- 选定规则（违约优先）：三场景平均无违约比例最高；与最高值相差 {100 * LAM_TIE:g} 个百分点以内视为持平，持平时选三场景平均有效 gap 更低的。"
               "3M 步的扫描对 gap 说明有限，违约的权重更高。",
               f"- Rule (violations first): highest mean violation-free share over the scenarios; within {100 * LAM_TIE:g} pp of it is a tie, "
               "broken by the lower mean effective gap. A 3M-step sweep says little about the gap; violations weigh more."),
             T(f"- **选定 λ_max = {best:g}**", f"- **Chosen λ_max = {best:g}**")
             + (T(f"（手动指定；规则给出 {rule:g}）", f" (set by hand; the rule gives {rule:g})") if override else ""), "",
             "| λ_max | " + " | ".join(f"{s.upper()} " + T("有效 gap / 无违约 / gap / 末 λ", "eff. gap / violation-free / gap / final λ") for s in SCENARIOS)
             + " | " + T("平均有效 gap", "mean eff. gap") + " | " + T("平均无违约", "mean violation-free") + " |",
             "|---" * (len(SCENARIOS) + 3) + "|"]
        for v in values:
            L.append(f"| {v:g} | " + " | ".join(f"{pc(cell[(v, s)]['eff_gap'])} / {pc(cell[(v, s)]['viol_free'])} / {pc(cell[(v, s)]['gap'])} / "
                                               f"{nn(cell[(v, s)]['final_lambda']):.3f}" for s in SCENARIOS)
                     + f" | {pc(eff[v])} | {pc(free[v])} |")
        (V["out"] / (f"lam_sweep_{learner}.md" if zh else f"lam_sweep_{learner}.en.md")).write_text("\n".join(L) + "\n")
    nan = lambda x: None if x is None or x != x else float(x)
    (V["out"] / f"lam_sweep_{learner}.json").write_text(json.dumps(   # for the panel (docs/data_spec.md §2 rule 6)
        {"schema": 2, "learner": learner, "RL_algorithm": learner.upper(), "algo": sw["algo"], "steps": sw["steps"], "values": values,
         "chosen": best, "rule_choice": rule, "override": bool(override), "tie_pp": 100 * LAM_TIE,
         "score": {f"{v:g}": nan(x) for v, x in eff.items()}, "viol_free_mean": {f"{v:g}": nan(x) for v, x in free.items()},
         "rows": rows}, indent=1))
    print(f"[{time.strftime('%H:%M:%S')}] lambda_max sweep {learner}: " + ", ".join(f"{v:g} -> {100 * eff[v]:.2f}% / free {100 * free[v]:.1f}%" for v in values)
          + f" | chosen {best:g}", flush=True)
    return best


def choose_lam_max(V, exp_ids, learner) -> float:
    """Evaluate one learner's sweep models (sweep_eval) and choose its lambda_max (write_lam_sweep)."""
    sw = dict(V["cfg"]["lam_sweep"], algo=V["cfg"]["lam_sweep"]["algos"][learner])
    keys = {v: f"lam_sweep_{learner}_{v:g}" for v in sw["values"]}
    jobs = [(sw["algo"], sd, s, exp_ids[str(sd)][s][k]) for k in keys.values() for sd in SEEDS for s in SCENARIOS]
    with Pool(min(len(jobs), 18)) as pool:
        res = dict(pool.map(sweep_eval, jobs))
    nan = lambda x: None if x != x else float(x)
    rows = []
    for v, k in keys.items():
        for sd in SEEDS:
            for s in SCENARIOS:
                eid = exp_ids[str(sd)][s][k]
                e, f, g, lam = res[(sw["algo"], sd, s, eid)]
                rows.append({"lambda_max": v, "scenario": s, "seed": sd, "exp_id": eid, "eff_gap": nan(e), "viol_free": nan(f),
                             "gap": nan(g), "final_lambda": nan(lam)})
    override = os.environ.get(f"V5_LAM_MAX_{learner.upper()}") or os.environ.get("V5_LAM_MAX")   # per learner, or all
    return write_lam_sweep(V, learner, rows, override)


def run_batch(V, runs, limit, exp_ids) -> list[str]:
    """Train `runs` with at most `limit` at a time (inside MPS when V["mps"]); returns the failed labels.
    v5.0.3: with a lambda_max sweep, a learner's Lagrangian runs wait until that learner's sweep runs have all
    finished and its lambda_max is chosen."""
    failed = []
    if V["mps"]:
        mps_start()
    pending, active = list(runs), []
    sweeping = lambda l: (any((sweep_key(r[2]) or ("",))[0] == l for r in pending)
                          or any((sweep_key(a[2].split()[-1]) or ("",))[0] == l for a in active))
    gated = lambda r: lagrange_learner(V, r[2]) is not None and V["lam_max"].get(lagrange_learner(V, r[2])) is None
    def choose_ready():
        """Pick lambda_max for every learner whose sweep has just finished (or drop its Lagrangian runs)."""
        nonlocal pending
        for l in (V["cfg"].get("lam_sweep") or {}).get("algos", {}):
            if V["lam_max"].get(l) is not None or sweeping(l):
                continue
            if any(f"lam_sweep_{l}_" in f for f in failed):       # no lambda_max: drop this learner's Lagrangian runs
                dropped = [r for r in pending if lagrange_learner(V, r[2]) == l]
                pending = [r for r in pending if lagrange_learner(V, r[2]) != l]
                failed.extend(f"seed{sd} {s} {k} (not started: sweep failed)" for sd, s, k in dropped)
                V["lam_max"][l] = float("nan")
            else:
                V["lam_max"][l] = choose_lam_max(V, exp_ids, l)
            write_manifest(V, exp_ids)

    while pending or active:
        choose_ready()
        while len(active) < limit:
            nxt = next((r for r in pending if not gated(r)), None)
            if nxt is None:
                break
            pending.remove(nxt)
            sd, s, k = nxt
            active.append((*launch(V, sd, s, k, exp_ids), f"seed{sd} {s} {k}"))
        if not active:
            continue
        running = []
        for p, f, label in active:
            if p.poll() is None:
                running.append((p, f, label))
                continue
            f.close()
            V["active_res"].pop(p.pid, None)
            print(f"[{time.strftime('%H:%M:%S')}] finished {label} ({'OK' if p.returncode == 0 else f'EXIT={p.returncode}'})", flush=True)
            if p.returncode != 0:
                failed.append(label)
        if len(running) == len(active):             # nothing finished: wait before polling again
            time.sleep(2 if SMOKE else 30)
        active = running
    choose_ready()                                  # the sweep that finished last (5.0.3: nothing runs after it)
    if V["mps"]:
        mps_stop()
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=list(CONFIG))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rerun", nargs=3, metavar=("SEED", "SCEN", "KEY"), action="append",
                    help="re-train one failed run under a new exp_id, replacing it in the manifest; report + email at the end")
    a = ap.parse_args()
    global SEEDS, SCENARIOS
    SCENARIOS = CONFIG[a.version].get("scenarios", SCENARIOS)   # a version's own scenarios (default all three)
    SEEDS = [1] if SMOKE else CONFIG[a.version].get("seeds", SEEDS)   # v5.0.3: seed 1 only
    global STEPS, CKPTS
    if not SMOKE and CONFIG[a.version].get("steps"):  # 5.0.3.1: a version's own training length (default 20M)
        STEPS, CKPTS = CONFIG[a.version]["steps"], CONFIG[a.version].get("ckpts", [])
    sfx = "_smoke" if SMOKE else ""
    V = {"version": a.version, "cfg": CONFIG[a.version], "log_dir": ROOT / "logs" / f"v{a.version}{sfx}",
         # a smoke test's report stays with its batch (logs/<batch>/report/), not as a fake space under results/
         "out": ROOT / "logs" / f"v{a.version}{sfx}" / "report" if SMOKE else ROOT / "paper_contents" / f"v{a.version}"}
    V["manifest"], V["report"] = V["log_dir"] / "manifest.json", V["out"] / "report.md"
    V["active_res"], V["gpu_node"] = {}, gpu_nodes()
    # {learner: lambda_max} (v5.0.3); an older manifest's single value applies to every learner
    V["lam_max"] = (manifest_lam_max(json.loads(V["manifest"].read_text())) if (a.rerun or a.report) and V["manifest"].exists() else {})
    sw = CONFIG[a.version].get("lam_sweep")
    sweep_only = sw and not CONFIG[a.version]["variants"]       # a sweep batch: its runs' steps are the batch's
    V["batch_steps"] = sw["steps"] if sweep_only and not SMOKE else STEPS
    V["batch_ckpts"] = sw["ckpts"] if sweep_only and not SMOKE else CKPTS
    src = V["cfg"].get("lam_max_from")
    if src:                                          # 5.0.4: lambda_max per learner from the sweep batch (v5.0.3)
        V["lam_max_from"] = f"v{src}{sfx}"
        p = ROOT / "logs" / V["lam_max_from"] / "manifest.json"
        lm = manifest_lam_max(json.loads(p.read_text())) if p.exists() else {}
        need = sorted({lagrange_learner(V, k) for k in V["cfg"]["variants"]} - {None})
        bad = [l for l in need if not isinstance(lm.get(l), (int, float)) or lm[l] != lm[l]]
        if bad:
            sys.exit(f"v{a.version}: lambda_max of {', '.join(bad)} not chosen in {p.relative_to(ROOT)} -- run v{src} first")
        V["lam_max"] = {l: float(lm[l]) for l in need}
        print(f"lambda_max from {V['lam_max_from']}: " + ", ".join(f"{l} {v:g}" for l, v in V["lam_max"].items()), flush=True)
    V["numa"] = os.environ.get("V5_NUMA", "0") == "1"      # off by default: no gain in check_v5 speed (0.99x)
    V["mps"] = os.environ.get("V5_MPS", "0") == "1" and any(v[1] == "graph" for v in V["cfg"]["variants"].values())
    V["log_dir"].mkdir(parents=True, exist_ok=True)
    V["out"].mkdir(parents=True, exist_ok=True)
    # every planned run (manifest v2): the lambda_max sweep first (v5.0.3: the Lagrangian runs wait for it), then the
    # variants, longest first (v5.0.0: PPO, then DQN / DDQN) so that the slowest runs do not start last
    keys = sorted(V["cfg"]["variants"], key=lambda k: V["cfg"]["variants"][k][0].split("_")[-1] != "ppo")
    sw = V["cfg"].get("lam_sweep")
    V["plan_keys"] = ([(sd, s, f"lam_sweep_{l}_{v:g}") for l in sw["algos"] for v in sw["values"] for sd in SEEDS for s in SCENARIOS]
                      if sw else []) + [(sd, s, k) for sd in SEEDS for k in keys for s in SCENARIOS]
    if a.report:
        write_report(V, json.loads(V["manifest"].read_text())["exp_ids"])
        return
    if a.rerun:                                     # replace failed runs; the old exp_id stays only in its log file name
        exp_ids = json.loads(V["manifest"].read_text())["exp_ids"]
        runs = [(int(sd), s, k) for sd, s, k in a.rerun]
        for sd, s, k in runs:
            assert sd in SEEDS and s in SCENARIOS and (k in V["cfg"]["variants"] or sweep_key(k)), (sd, s, k)
            print(f"=== v{a.version} rerun seed{sd} {s} {k} (was {exp_ids.get(str(sd), {}).get(s, {}).get(k)}) "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    else:
        exp_ids = {}
        write_manifest(V, exp_ids)
        print(f"=== v{a.version} start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
        runs = list(V["plan_keys"])
    env_par = int(os.environ.get("V5_PARALLEL", 0))
    stages = V["cfg"].get("stages") if not a.rerun else None
    if stages:                                      # v5.0.2: one stage after the other
        learner = lambda r: V["cfg"]["variants"][r[2]][0].split("_")[-1]
        plan = [(st, [r for r in runs if learner(r) in st["learners"]]) for st in stages]
        assert sum(len(x) for _, x in plan) == len(runs), "every run must belong to a stage"
    else:
        plan = [({"name": "all", "parallel": V["cfg"].get("parallel"), "mps": V["cfg"].get("mps", V["mps"])}, runs)]
    failed = []
    for k_st, (st, st_runs) in enumerate(plan):
        if os.environ.get("V5_MPS") is None:        # a stage's own MPS setting unless V5_MPS is given explicitly
            V["mps"] = bool(st.get("mps"))
        limit = env_par or st.get("parallel") or len(st_runs)
        print(f"=== stage {k_st + 1}/{len(plan)}: {st['name']} | {len(st_runs)} runs | at most {limit} at a time | "
              f"MPS {'on' if V['mps'] else 'off'} | {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
        failed += run_batch(V, st_runs, limit, exp_ids)
        if k_st + 1 < len(plan) and not failed:       # interim report: the stages done so far
            print(f"[report] interim report after stage {k_st + 1}", flush=True)
            write_report(V, exp_ids, interim=True)
    print(f"=== done {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(failed)} failed ===", flush=True)
    from shared.report_mail import send_failure, send_report
    if SMOKE:                                       # no email for the end-to-end test
        send_failure = send_report = lambda *a_, **k_: None
    if failed:
        send_failure(f"v{a.version}", failed, log_dir=V["log_dir"].relative_to(ROOT), manifest=V["manifest"].relative_to(ROOT))
        sys.exit(1)                                 # non-zero: the queue stops instead of starting the next item
    write_report(V, exp_ids)
    send_report(f"v{a.version}", V["report"], manifest=V["manifest"].relative_to(ROOT), driver_log=f"logs/run_v{a.version}_driver.log")


if __name__ == "__main__":
    main()
