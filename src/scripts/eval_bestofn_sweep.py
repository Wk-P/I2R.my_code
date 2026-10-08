"""eval_bestofn_sweep.py — N-sensitivity sweep for best-of-N eval (v1.2.2's
EVAL_BEST_OF_N=8 was never tuned; this checks whether success_rate keeps
climbing past N=8 or has already plateaued).

Loads an already-trained model.zip and re-runs the same held-out
TEST_SCENARIOS eval at several N values — no retraining, so this is
minutes not hours. TRAIN_SEED must match the seed the model was trained
with (it also determines the train/test split), otherwise TEST_SCENARIOS
won't line up with what the model actually never saw during training.

Usage:
    TRAIN_SEED=42 python src/scripts/eval_bestofn_sweep.py \
        results/paper-verfication/lt/ppo_mask/0e1a36bb_bc/model_0e1a36bb_bc_v1.2.4 \
        8 16 32 64
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scenarios" / "lt" / "ppo_mask"))

import numpy as np
from sb3_contrib import MaskablePPO

import config as C
from shared.ilp_utils import load_scenario
import run_all as RA


def main():
    model_path = sys.argv[1]
    ns = [int(x) for x in sys.argv[2:]] or [8, 16, 32, 64]

    ecus, services, _, _ = load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
    print(f"TRAIN_SEED={C.SEED}  test_scenarios={len(C.TEST_SCENARIOS)}  model={model_path}")

    model = MaskablePPO.load(model_path)

    def ppo_policy(obs, mask):
        action, _ = model.predict(obs, deterministic=False, action_masks=mask)
        return int(action)

    print(f"{'N':>4} {'success_rate':>14} {'ar_mean':>10} {'ar_std':>10} {'avg_attempts':>14}")
    for n in ns:
        res = RA.run_episodes(ecus, services, ppo_policy, n_samples=n)
        success_rate = float(np.mean(res["success"]))
        ars = res["ars"][res["success"]]
        ar_mean = float(np.mean(ars)) if len(ars) else float("nan")
        ar_std = float(np.std(ars)) if len(ars) else float("nan")
        avg_attempts = float(np.mean(res["attempts"]))
        print(f"{n:>4} {success_rate:>14.4f} {ar_mean:>10.4f} {ar_std:>10.4f} {avg_attempts:>14.2f}")


if __name__ == "__main__":
    main()
