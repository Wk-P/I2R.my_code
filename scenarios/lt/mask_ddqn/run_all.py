"""
run_all.py — mask_ddqn (Double DQN on the Mask-PPO environment (ppo_mask/env.py::P4Env) + hard action mask), v4.3.0.

The pipeline lives in src/shared/dqn_variant_runner.py (ILP -> train -> evaluate
-> results.json / summary.csv / training_curve.* / comparison.png).

Run:
    python scenarios/lt/mask_ddqn/run_all.py [--total-timesteps N]
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "src"))  # src/ for shared

import config as C
from shared.dqn_variant_runner import make_runner
from shared.ilp_utils import load_scenario  # noqa: F401  (used by paper eval tooling)

R = make_runner(C, "mask", double=True)
run_episodes = R.run_episodes
MODEL_CLS = R.MODEL_CLS

if __name__ == "__main__":
    R.main()
