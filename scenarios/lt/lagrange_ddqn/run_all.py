"""
run_all.py — lagrange_ddqn (Double DQN on the Lagrange-PPO environment (ppo_lagrangian/env.py::LagrangeEnv) + dual ascent on lambda), v4.3.1.2.

The pipeline lives in src/shared/dqn_variant_runner.py (ILP -> train -> evaluate
-> results.json / summary.csv / training_curve.* / comparison.png).

Run:
    python scenarios/lt/lagrange_ddqn/run_all.py [--total-timesteps N]
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

R = make_runner(C, "lagrange", double=True)
run_episodes = R.run_episodes
MODEL_CLS = R.MODEL_CLS

if __name__ == "__main__":
    R.main()
