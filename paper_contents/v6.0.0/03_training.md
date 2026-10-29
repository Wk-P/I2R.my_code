# 3. 库、超参数与回调函数

## 3.1 库与版本（训练环境 `.venv-gpu`）

| 库 | 版本 | 用途 |
|---|---|---|
| Python | 3.12.3 |  |
| torch | 2.5.1+cu121 | 网络与训练（CUDA） |
| stable-baselines3 | 2.7.1 | `PPO`、`DQN`（DoubleDQN / MaskableDQN / MaskableDDQN 的基类）、`Monitor`、`DummyVecEnv`、回调基类 |
| sb3-contrib | 2.7.1 | `MaskablePPO`、`MaskableActorCriticPolicy`（GraphMaskablePolicy 的基类） |
| gymnasium | 1.2.3 | 环境接口（`PlacementEnv`） |
| numpy | 2.4.2 |  |

| RL_algorithm | strategy | 类（来源） |
|---|---|---|
| PPO | maskable | `MaskablePPO` (sb3-contrib) |
| PPO | repair / lagrangian / unconstraint | `PPO` (stable-baselines3) |
| DQN | maskable | `MaskableDQN`（本项目，继承 SB3 `DQN`） |
| DQN | repair / lagrangian / unconstraint | `DQN` (stable-baselines3) |
| DDQN | maskable | `MaskableDDQN`（本项目，继承 MaskableDQN） |
| DDQN | repair / lagrangian / unconstraint | `DoubleDQN`（本项目，继承 SB3 `DQN`） |

策略 / Q 网络：PPO 全部为 `GraphMaskablePolicy`（本项目），DQN / DDQN 全部为 `GraphDQNPolicy`（本项目），见 02_network。

## 3.2 PPO 超参数

| 参数 | 值 | 来源 |
|---|---|---|
| 并行环境数 n_envs | 40 | config.PPO_N_ENVS |
| n_steps | 512（每次更新 20,480 条转移） | config.PPO_N_STEPS |
| batch_size | 256（每轮 80 个小批量） | config.PPO_BATCH_SIZE |
| n_epochs | maskable：6；其余：10 | maskable：`--n-epochs 6`（v6.0.5-PPO-hyparam）；其余：config.PPO_N_EPOCHS |
| learning_rate | 0.0003（常数） | config.PPO_LR |
| gamma | 1.0 | `--gamma 1.0` |
| gae_lambda | 0.95 | config.PPO_GAE_LAMBDA |
| clip_range | 0.1 | config.PPO_CLIP_RANGE |
| ent_coef | 0.005（常数） | config.PPO_ENT_COEF |
| vf_coef | 0.5 | 库默认 |
| max_grad_norm | 0.5 | 库默认 |
| normalize_advantage | True | 库默认 |
| clip_range_vf | None | 库默认（不裁剪价值） |
| target_kl | None | 未设置 |
| 优化器 | Adam, eps = 1e-5 | MaskableActorCriticPolicy 默认 |
| torch 线程 / 设备 | 6 / cuda | config.PPO_TORCH_THREADS, `--device cuda` |

## 3.3 DQN / DDQN 超参数（两者相同）

| 参数 | 值 | 来源 |
|---|---|---|
| 并行环境数 n_envs | 12 | config.DQN_N_ENVS |
| learning_rate | 0.001（常数） | config.DQN_LR |
| buffer_size | 100,000 | config.DQN_BUFFER_SIZE |
| learning_starts | 2,000 条转移（之前随机动作，maskable 只在可行动作中随机） | config.DQN_LEARNING_STARTS |
| batch_size | 64 | config.DQN_BATCH_SIZE |
| gamma | 1.0 | `--gamma 1.0` |
| train_freq / gradient_steps | 每 4 次向量步（48 条转移）做 1 次梯度更新 | config.DQN_TRAIN_FREQ / DQN_GRADIENT_STEPS |
| target_update_interval / tau | 500 / 1.0：每 41 次向量步（492 条转移）把在线网络整体复制到目标网络 | config；SB3 按 interval // n_envs 计次 |
| 探索 ε | 1.0 → 0.0，在前 2,000,000 步内线性下降，之后为 0 | `--eps-steps 2000000`（exploration_fraction = 0.1） |
| 损失 | Smooth L1 (Huber) | SB3 `DQN.train` / 本项目 `_td_update`，相同 |
| max_grad_norm | 10 | 库默认 |
| n_steps (TD) | 1 | 库默认（单步 TD） |
| 优化器 | Adam (torch default, eps = 1e-8) | DQNPolicy 默认（optimizer_kwargs 为空） |
| torch 线程 / 设备 | 2 / cuda | config.DQN_TORCH_THREADS, `--device cuda` |

DQN 与 DDQN 的唯一区别是 TD 目标（01_algorithms §1.4）。

## 3.4 Lagrangian 对偶更新（只用于 lagrangian）

| 参数 | 值 |
|---|---|
| 更新方式 | 把 λ 当作参数，损失 −λ·(J_C − d)，Adam（默认 β），更新后截断到 λ ≥ 0 |
| 学习率 | 0.035 |
| 初值 λ_0 | 0.001 |
| 目标 d | 0.001 |
| 上限 | 无（`--lambda-max none`） |
| J_C | 每个 episode 的违约次数 Σ_t c_t 的平均（行为策略的训练 episode） |
| 更新频率 | PPO：每个 rollout（20,480 条转移）一次，用该 rollout 内结束的 episode；DQN / DDQN：每 1000 个 episode 一次（不重叠的窗口） |
| 生效时机 | 下一次 reset（episode 内 λ 固定） |
| 观测中的 λ | λ / (1 + λ)（无上限时） |

## 3.5 共同的训练设置

| 项 | 值 |
|---|---|
| 总步数 | 20,000,000 |
| 种子 | 1, 2, 3 |
| 训练实例 | `results/sup_diag/<场景>.json`，每场景 50,000 个 ILP 验证实例 |
| 检查点 | 5M、10M、15M（另存 20M 最终模型） |
| 奖励缩放 | 无（`--reward-norm none`） |
| CUDA Graphs | 开启（默认 `V5_CUDA_GRAPHS=1`）；只影响速度，数值与普通执行一致（src/scripts/check_v5.py 逐位核对） |

## 3.6 本项目的回调函数（`src/paper_rl/train.py`）

训练时用 `CallbackList([主回调, 检查点回调])`：

**1. 主回调 `make_callback` → `Cb`（继承 SB3 `BaseCallback`）**

- `_on_training_start`：lagrangian 时把初始 λ 写入所有训练环境（`set_lambda`）。
- `_on_step`：对每个结束的 episode 记录一行 (timestep, episode 奖励, AR, 是否无违约完成, 合法放置数, 是否容量违约, 是否隐私违约, λ, 是否 EXIT)，训练结束写入 `training_curve.csv`（episode 奖励来自 SB3 `Monitor`，是原始奖励之和，lagrangian 含 −λ·c_t）；lagrangian + DQN / DDQN 时累计 episode 违约数，每 1000 个 episode 更新一次 λ；每 100,000 步打印进度并写 `.progress.json`（看板用）。
- `_on_rollout_end`：lagrangian + PPO 时，用本 rollout 内结束的 episode 的平均违约数更新一次 λ。
- λ 的每次更新记录在 `lambda_updates.csv`（列：episode, timestep, jc_behavior, jc_used, lambda；这些 run 中 jc_used = jc_behavior）。

**2. 检查点回调 `checkpoint_callback` → `Ckpt`**

- 步数第一次达到 5M / 10M / 15M 时保存 `ckpt_<步数>_<exp_id>.zip`；lagrangian 同时把当时的 λ 写入 `ckpt_lambda.json`（评估检查点时要用同一个 λ）。

这些 run 未使用的回调：`val_callback`（v6.1.0 调参用）、`GradDiagCallback`（梯度诊断）。

**其他自定义组件（不是回调）**

- `FastDummyVecEnv`（`src/paper_rl/vec_env.py`）：SB3 `DummyVecEnv` 去掉每步对 info 的深拷贝（改为一层拷贝，内容相同），只为提速。
- 每个训练环境的包装：`PlacementEnv` → SB3 `Monitor`（记录 episode 奖励）→ `TransformReward`（×1，即不缩放）。
- `DoubleDQN` / `MaskableDQN` / `MaskableDDQN`：覆盖 `train()`（TD 目标）；Maskable 版本还覆盖 `predict()` 与 `_sample_action()`（掩码下的 ε-贪心）。
