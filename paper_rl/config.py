"""Single configuration for every model and scenario (v4.3.1.3).

Nothing here depends on the scenario; LT / EQ / GT differ only in N and M
(paper_rl/data.py). Values marked "from v4.x" are the setting most of the
previous per-scenario configs agreed on.
"""

# ── data / split ──────────────────────────────────────────────────────────
DATA_VERSION = "v4.3.1.4"             # data/<version>/{lt,eq,gt}.yaml; v4.3.1.3: p=0.3, v4.3.1.4: p=0.6
TRAIN_FRACTION = 0.8                  # of the 2000 instances; split seeded by the run seed

# ── PPO family (PPO, Mask-PPO, Lagrange-PPO, Repair-PPO) ───────────────────
PPO_N_ENVS = 40
PPO_LR = 3e-4
PPO_N_STEPS = 512                     # per env -> 20480 transitions per update
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.1                  # v4.3.1.6 (was 0.2): probability ratio clipped to [0.9, 1.1]
PPO_ENT_COEF = 0.005                  # constant (no annealing)
PPO_NET_ARCH = dict(pi=[256, 256], vf=[256, 256])
PPO_TORCH_THREADS = 6

# ── DQN family (DQN, DDQN and their Mask / Lagrange / Repair versions) ─────
DQN_N_ENVS = 12
DQN_LR = 1e-3
DQN_BUFFER_SIZE = 100_000
DQN_LEARNING_STARTS = 2_000
DQN_BATCH_SIZE = 64
DQN_TAU = 1.0
DQN_GAMMA = 0.99
DQN_TRAIN_FREQ = 4
DQN_GRADIENT_STEPS = 1
DQN_TARGET_UPDATE = 500
DQN_EXPLORATION_FRACTION = 0.5
DQN_EXPLORATION_FINAL_EPS = 0.0
DQN_NET_ARCH = [128, 128]
DQN_TORCH_THREADS = 2

# ── Lagrangian dual ascent (Lagrange-PPO / -DQN / -DDQN) ───────────────────
# Cost charged at the end of the episode: -lambda * (capacity + privacy violations).
# Every LAMBDA_UPDATE_WINDOW episodes after LAMBDA_WARMUP_EPISODES:
#   lambda <- clip(lambda + LAMBDA_LR * (mean violation rate - LAMBDA_TARGET), 0, LAMBDA_MAX)
# violation rate of an episode = (capacity + privacy violations) / M.
LAMBDA_INIT = 0.1
LAMBDA_LR = 0.005
LAMBDA_TARGET = 0.0
LAMBDA_MAX = 50.0                     # v4.3.1.5 (was 5.0): violating "stack everything on one ECU"
                                      # earns M*AR_exec ~ 175 in LT, so lambda must be able to exceed ~6
LAMBDA_UPDATE_WINDOW = 20
LAMBDA_WARMUP_EPISODES = 5000

# ── directional reward (REWARD_MODE=directional) ──────────────────────────
DIR_BETA = 10.0
DIR_LAMBDA = 1.0                      # discrete penalty for an AR decrease
DIR_EPS = 1e-3
DIR_B = 1.0                           # completion bonus = DIR_B * M
DIR_C = 1.0                           # dead-end penalty = DIR_C * M

# ── logging / evaluation ──────────────────────────────────────────────────
PROGRESS_LOG_EVERY_STEPS = 100_000
SMOOTH_W = 1000
