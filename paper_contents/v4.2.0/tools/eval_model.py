"""Evaluate one saved model on its own seed's test set and time inference (v4.2.0).

Timing covers the whole episode: env reset/steps/action masks + model.predict,
on 1 torch thread, after a 5-episode warm-up.

usage: worker.py <scen> <algo> <run_dir> <seed> <n_samples> <deterministic 0/1>
prints one JSON line.
"""
import json, os, sys, time
from pathlib import Path

scen, algo, run_dir, seed, n_samples, det = sys.argv[1:7]
n_samples, det = int(n_samples), bool(int(det))
os.environ["TRAIN_SEED"] = seed
ROOT = Path("/home/soar009/github/my_code")
here = ROOT / "scenarios" / scen / algo
for p in (str(ROOT), str(here.parent), str(here)):
    sys.path.insert(0, p)
os.chdir(here)

import numpy as np
import torch
torch.set_num_threads(1)
import run_all as R
C = R.C

run_dir = Path(run_dir)
model_file = next(run_dir.glob("model_*"))
res = json.loads((run_dir / "results.json").read_text())

if algo == "ppo_mask":
    from sb3_contrib import MaskablePPO as Cls
elif algo == "dqn":
    from stable_baselines3 import DQN as Cls
elif algo == "ddqn":
    Cls = R.DDQN
else:
    from stable_baselines3 import PPO as Cls
model = Cls.load(str(model_file), device="cpu")

if algo == "ppo_mask":
    def policy(obs, mask):
        a, _ = model.predict(obs, deterministic=det, action_masks=mask)
        return int(a)
else:
    def policy(obs):
        a, _ = model.predict(obs, deterministic=det)
        return int(a)

import inspect
kw = {}
_ns = "n_samples" in inspect.signature(R.run_episodes).parameters
if algo == "ppo_lagrangian":
    kw["lambda_eval"] = float(res["training"].get("eval_lambda", res["training"].get("final_lambda", 0.0)))

ecus, services, *_ = R.load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
# warm-up (first predict call carries torch init overhead)
first = C.TEST_SCENARIOS[:5]
_saved = C.TEST_SCENARIOS
C.TEST_SCENARIOS = first
R.run_episodes(ecus, services, policy, **({"n_samples": 1} if _ns else {}), **kw)
C.TEST_SCENARIOS = _saved

t0 = time.perf_counter()
out = R.run_episodes(ecus, services, policy, **({"n_samples": n_samples} if _ns else {}), **kw)
dt = time.perf_counter() - t0
n = len(C.TEST_SCENARIOS)
algo_key = [k for k in res if isinstance(res[k], dict) and "success_rate" in res[k]][0]

def g(k):
    return np.asarray(out[k]) if k in out else None

cap = g("cap_viols") if "cap_viols" in out else g("cap_violations")
conf = g("conflict_viols") if "conflict_viols" in out else g("conflict_violations")
print(json.dumps({
    "scen": scen, "algo": algo, "run": run_dir.name, "seed": int(seed),
    "n_samples": n_samples, "deterministic": det, "n_test": n,
    "success": float(np.mean(out["success"])),
    "ar": float(np.mean(out["ars"])),
    "cap_viol": None if cap is None else float(np.mean(cap > 0)),
    "conflict_viol": None if conf is None else float(np.mean(conf > 0)),
    "ms_per_episode": dt / n * 1000,
    "stored_success": res[algo_key]["success_rate"],
    "stored_ilp_ar": res["ilp"]["ar"],
}))
