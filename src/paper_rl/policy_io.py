"""Load a trained policy for the diagnostics (v4.4.3): one predict(obs, mask) -> action interface.

  saved SB3 model   : results/<space>/<scen>/<algo>/<exp_id>/<pattern>; observation mode read from
                      results.json ("obs", default base); Mask PPO gets the action mask.
  supervised ref    : exp_id "ref:<path>" = a GraphPolicyNet state dict (scripts/diag_arch.py),
                      raw observation, argmax over the legal actions.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "unified"


def load_policy(scen: str, algo: str, exp_id: str, pattern: str = "model_*", n: int = 0, m: int = 0):
    """-> (predict(obs, mask) -> int, obs_mode)"""
    import torch
    if exp_id.startswith("ref:"):
        from paper_rl.graph_net import GraphPolicyNet, decode_raw
        net = GraphPolicyNet()
        net.load_state_dict(torch.load(exp_id[4:], map_location="cpu"))
        net.eval()

        def predict(obs, mask):
            with torch.no_grad():
                logits, *_ = net(*decode_raw(torch.as_tensor(obs, dtype=torch.float32)[None], n, m))
            return int(logits.argmax(-1))
        return predict, "raw"
    from paper_rl.train import model_class, split_algo
    run_dir = RESULTS / scen / algo / exp_id
    mech, learner = split_algo(algo)
    model = model_class(learner, mech).load(str(next(run_dir.glob(pattern))), device="cpu")
    obs_mode = json.loads((run_dir / "results.json").read_text()).get("obs", "base")

    def predict(obs, mask):
        if learner == "ppo" and mech == "mask":
            a, _ = model.predict(obs, deterministic=True, action_masks=mask)
        else:
            a, _ = model.predict(obs, deterministic=True)
        return int(a)
    return predict, obs_mode
