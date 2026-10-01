"""
Hyperparameter & path configuration for repair_dqn (v4.3.0).
DQN on the Repair-PPO environment (ppo_opt/env.py::P6Env) + best-fit repair.
Hyperparameters are identical to gt/dqn so the DQN family differs only in
the constraint mechanism. DQN_* names are kept for shared/dqn_variant_runner.py.
"""

from pathlib import Path
import sys
import os

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from shared.training_steps_config import get_total_steps

# ── Scenario source (same YAML config file as problem2_ilp) ──────────────
YAML_CONFIG  = ROOT / ".." / "ilp" / "config" / "config_ecu_gt_svc.yaml"
SCENARIO_IDX = 0   # 0-indexed: 0 = Scenario 1, 1 = Scenario 2 ...

with open(YAML_CONFIG) as f:
    import yaml
    cfg = yaml.safe_load(f)
    _all = cfg["scenarios"]
    N = len(_all[0]["ECUs"])
    M = len(_all[0]["SVCs"])

    SCENARIOS = [
        (
            [ecu["capacity"] for ecu in sc["ECUs"]],
            [svc["requirement"] for svc in sc["SVCs"]],
            sc.get("conflict_sets", []),
        )
        for sc in _all
    ]
    VMS_POOL = SCENARIOS[SCENARIO_IDX][0]
    REQ_POOL = SCENARIOS[SCENARIO_IDX][1]

# ── Training ──────────────────────────────────────────────────────────────────
TOTAL_STEPS = get_total_steps("repair_dqn", scenario=ROOT.parent.name)
SEED        = int(os.environ.get("TRAIN_SEED", "42"))
# ── Train / Test split (80/20, deterministic) ────────────────────────────────
import random as _random
_rng = _random.Random(SEED)
_idxs = list(range(len(SCENARIOS)))
_rng.shuffle(_idxs)
_n_train = int(0.8 * len(SCENARIOS))
TRAIN_SCENARIOS = [SCENARIOS[i] for i in _idxs[:_n_train]]
TEST_SCENARIOS  = [SCENARIOS[i] for i in _idxs[_n_train:]]

DEVICE      = "auto"
N_ENVS      = 12
TORCH_NUM_THREADS = 5        # shared host: keep total concurrent demand ~30 cores
PROGRESS_LOG_EVERY_STEPS = 200_000

# ── DQN hyperparameters ──────────────────────────────────────────────────────────────────
DQN_LR                    = 1e-3
DQN_BUFFER_SIZE           = 100_000
DQN_LEARNING_STARTS       = 2_000
DQN_BATCH_SIZE            = 64
DQN_TAU                   = 1.0
DQN_GAMMA                 = 0.99
DQN_TRAIN_FREQ            = 4
DQN_GRADIENT_STEPS        = 1
DQN_TARGET_UPDATE         = 500
DQN_EXPLORATION_FRACTION  = 0.5
DQN_EXPLORATION_FINAL_EPS = 0.0
DQN_NET_ARCH              = [128, 128]


# ── Evaluation ──────────────────────────────────────────────────────────────────
EVAL_EPS  = len(TEST_SCENARIOS)
SMOOTH_W  = 1000

# ── Paths ───────────────────────────────────────────────────────────────────────
from shared.paths import results_dir
OUTDIR     = results_dir(ROOT.parent.name, "repair_dqn")
