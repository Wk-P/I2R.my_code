# 3. Libraries, hyper-parameters and callbacks

## 3.1 Libraries and versions (training environment `.venv-gpu`)

| library | version | use |
|---|---|---|
| Python | 3.12.3 |  |
| torch | 2.5.1+cu121 | networks and training (CUDA) |
| stable-baselines3 | 2.7.1 | `PPO`, `DQN` (base of DoubleDQN / MaskableDQN / MaskableDDQN), `Monitor`, `DummyVecEnv`, callback base |
| sb3-contrib | 2.7.1 | `MaskablePPO`, `MaskableActorCriticPolicy` (base of GraphMaskablePolicy) |
| gymnasium | 1.2.3 | environment interface (`PlacementEnv`) |
| numpy | 2.4.2 |  |

| RL_algorithm | strategy | class (source) |
|---|---|---|
| PPO | maskable | `MaskablePPO` (sb3-contrib) |
| PPO | repair / lagrangian / unconstraint | `PPO` (stable-baselines3) |
| DQN | maskable | `MaskableDQN` (ours, subclass of SB3 `DQN`) |
| DQN | repair / lagrangian / unconstraint | `DQN` (stable-baselines3) |
| DDQN | maskable | `MaskableDDQN` (ours, subclass of MaskableDQN) |
| DDQN | repair / lagrangian / unconstraint | `DoubleDQN` (ours, subclass of SB3 `DQN`) |

Policy / Q-network: every PPO uses `GraphMaskablePolicy` (ours), every DQN / DDQN `GraphDQNPolicy` (ours); see 02_network.

## 3.2 PPO hyper-parameters

| parameter | value | source |
|---|---|---|
| parallel envs n_envs | 40 | config.PPO_N_ENVS |
| n_steps | 512 (20,480 transitions per update) | config.PPO_N_STEPS |
| batch_size | 256 (80 minibatches per epoch) | config.PPO_BATCH_SIZE |
| n_epochs | maskable: 6; others: 10 | maskable: `--n-epochs 6` (v6.0.5-PPO-hyparam); others: config.PPO_N_EPOCHS |
| learning_rate | 0.0003 (constant) | config.PPO_LR |
| gamma | 1.0 | `--gamma 1.0` |
| gae_lambda | 0.95 | config.PPO_GAE_LAMBDA |
| clip_range | 0.1 | config.PPO_CLIP_RANGE |
| ent_coef | 0.005 (constant) | config.PPO_ENT_COEF |
| vf_coef | 0.5 | library default |
| max_grad_norm | 0.5 | library default |
| normalize_advantage | True | library default |
| clip_range_vf | None | library default (no value clipping) |
| target_kl | None | not set |
| optimizer | Adam, eps = 1e-5 | MaskableActorCriticPolicy default |
| torch threads / device | 6 / cuda | config.PPO_TORCH_THREADS, `--device cuda` |

## 3.3 DQN / DDQN hyper-parameters (identical)

| parameter | value | source |
|---|---|---|
| parallel envs n_envs | 12 | config.DQN_N_ENVS |
| learning_rate | 0.001 (constant) | config.DQN_LR |
| buffer_size | 100,000 | config.DQN_BUFFER_SIZE |
| learning_starts | 2,000 transitions (random actions before; maskable: among valid actions only) | config.DQN_LEARNING_STARTS |
| batch_size | 64 | config.DQN_BATCH_SIZE |
| gamma | 1.0 | `--gamma 1.0` |
| train_freq / gradient_steps | 1 gradient step every 4 vector steps (48 transitions) | config.DQN_TRAIN_FREQ / DQN_GRADIENT_STEPS |
| target_update_interval / tau | 500 / 1.0: the online network is copied to the target every 41 vector steps (492 transitions) | config; SB3 counts interval // n_envs calls |
| exploration ε | 1.0 → 0.0, linear over the first 2,000,000 steps, 0 afterwards | `--eps-steps 2000000` (exploration_fraction = 0.1) |
| loss | Smooth L1 (Huber) | SB3 `DQN.train` / our `_td_update`, the same |
| max_grad_norm | 10 | library default |
| n_steps (TD) | 1 | library default (one-step TD) |
| optimizer | Adam (torch default, eps = 1e-8) | DQNPolicy default (empty optimizer_kwargs) |
| torch threads / device | 2 / cuda | config.DQN_TORCH_THREADS, `--device cuda` |

DQN and DDQN differ only in the TD target (01_algorithms §1.4).

## 3.4 Lagrangian dual update (lagrangian only)

| parameter | value |
|---|---|
| update | λ as a parameter, loss −λ·(J_C − d), Adam (default betas), clamped to λ ≥ 0 after each step |
| learning rate | 0.035 |
| initial λ_0 | 0.001 |
| target d | 0.001 |
| upper bound | none (`--lambda-max none`) |
| J_C | mean over episodes of the episode's violations Σ_t c_t (training episodes of the behaviour policy) |
| frequency | PPO: once per rollout (20,480 transitions) from the episodes that ended in it; DQN / DDQN: every 1000 episodes (non-overlapping windows) |
| takes effect | at the next reset (λ fixed within an episode) |
| λ in the observation | λ / (1 + λ) (no upper bound) |

## 3.5 Common training settings

| item | value |
|---|---|
| total steps | 20,000,000 |
| seeds | 1, 2, 3 |
| training instances | `results/sup_diag/<scenario>.json`, 50,000 ILP-verified instances per scenario |
| checkpoints | 5M, 10M, 15M (+ the 20M final model) |
| reward scaling | none (`--reward-norm none`) |
| CUDA Graphs | on (default `V5_CUDA_GRAPHS=1`); speed only, the same values as eager execution (checked bit for bit by src/scripts/check_v5.py) |

## 3.6 Our callbacks (`src/paper_rl/train.py`)

Training uses `CallbackList([main callback, checkpoint callback])`:

**1. Main callback `make_callback` → `Cb` (subclass of SB3 `BaseCallback`)**

- `_on_training_start`: for lagrangian, writes the initial λ into every training environment (`set_lambda`).
- `_on_step`: for every finished episode records (timestep, episode reward, AR, completed without violation, legal placements, capacity violated, privacy violated, λ, EXIT), written to `training_curve.csv` at the end (the episode reward comes from SB3 `Monitor`: the raw reward sum, −λ·c_t included for lagrangian); for lagrangian + DQN / DDQN accumulates the episode violations and updates λ every 1000 episodes; every 100,000 steps prints progress and writes `.progress.json` (for the panel).
- `_on_rollout_end`: for lagrangian + PPO, updates λ once from the mean violations of the episodes that ended in this rollout.
- Every λ update is logged in `lambda_updates.csv` (columns episode, timestep, jc_behavior, jc_used, lambda; jc_used = jc_behavior in these runs).

**2. Checkpoint callback `checkpoint_callback` → `Ckpt`**

- Saves `ckpt_<step>_<exp_id>.zip` the first time the step count reaches 5M / 10M / 15M; for lagrangian also writes the λ at that moment to `ckpt_lambda.json` (a checkpoint is evaluated with the same λ).

Callbacks not used by these runs: `val_callback` (v6.1.0 tuning), `GradDiagCallback` (gradient diagnostic).

**Other custom components (not callbacks)**

- `FastDummyVecEnv` (`src/paper_rl/vec_env.py`): SB3 `DummyVecEnv` without the per-step deep copy of the infos (a one-level copy with the same contents), for speed only.
- Each training environment is wrapped `PlacementEnv` → SB3 `Monitor` (episode reward) → `TransformReward` (×1, i.e. no scaling).
- `DoubleDQN` / `MaskableDQN` / `MaskableDDQN`: override `train()` (TD target); the Maskable ones also override `predict()` and `_sample_action()` (masked ε-greedy).
