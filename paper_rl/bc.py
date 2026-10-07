"""v4.4.2: ILP demonstrations -> behaviour-cloning warm start for Mask PPO.

Expert: the ILP optimum (solve_ilp_max_ar, Dinkelbach) of every training instance, replayed in
the environment's order (descending demand). Every prefix of a feasible assignment is feasible,
so the expert never hits EXIT. Label ambiguity: when the expert puts a service on an empty ECU,
every legal empty ECU of the same capacity is an equally good label (swapping them changes
nothing), so the loss is -log sum_{a in labels} pi(a | s) over the masked policy.
The value head is fitted to the expert return, which with gamma = 1 and reward ar_pen is the
ILP AR at every step.

    python -m paper_rl.bc v4.3.1.4        # cache ILP allocations -> data/<version>/ilp_alloc_<scen>.json
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper_rl.data import data_dir, load  # noqa: E402

BC_EPOCHS = 30
BC_LR = 1e-3
BC_BATCH = 256
BC_VALUE_COEF = 0.5


def _alloc_one(x):
    from shared.ilp_utils import solve_ilp_max_ar
    r = solve_ilp_max_ar(x["ECUs"], x["SVCs"], x["conflict_sets"])
    assert r["status"] == "Optimal", r["status"]
    svc_ecu = [None] * len(x["SVCs"])
    for j, svcs in r["allocation"].items():
        for i in svcs:
            svc_ecu[i] = int(j)
    return svc_ecu, float(r["avg_utilization"])


def alloc_path(scen: str, version: str | None = None) -> Path:
    return data_dir(version) / f"ilp_alloc_{scen}.json"


def build_allocs(version: str, workers: int = 48) -> None:
    for scen in ("lt", "eq", "gt"):
        inst = load(scen, version)["instances"]
        with Pool(workers) as pool:
            res = pool.map(_alloc_one, inst, chunksize=4)
        diff = max(abs(ar - x["ar_star"]) for (_, ar), x in zip(res, inst))
        assert diff < 1e-5, f"{scen}: ILP AR differs from stored ar_star by {diff}"
        alloc_path(scen, version).write_text(json.dumps([a for a, _ in res]))
        print(f"{scen}: {len(inst)} allocations, max |AR - ar_star| = {diff:.2e}", flush=True)


def expert_dataset(env_kwargs: dict, instances: list, allocs: list):
    """Replay the ILP allocation of every instance; returns obs, masks, label sets (bool), returns."""
    from paper_rl.env import PlacementEnv
    env = PlacementEnv(instances, **env_kwargs)
    obs_l, mask_l, lab_l, ret_l = [], [], [], []
    for k, inst in enumerate(instances):
        order = sorted(range(len(inst["SVCs"])), key=lambda i: -inst["SVCs"][i])   # as PlacementEnv.reset
        env.use_instance(k)
        obs, _ = env.reset()
        done, t = False, 0
        while not done:
            mask = env.action_masks()
            j = allocs[k][order[t]]
            assert mask[j], "expert action masked"
            lab = np.zeros_like(mask)
            if env.hosted[j]:
                lab[j] = True
            else:                                        # empty ECU: every legal empty ECU of equal capacity
                for q in range(env.N):
                    lab[q] = mask[q] and not env.hosted[q] and env.cap[q] == env.cap[j]
            obs_l.append(obs)
            mask_l.append(mask)
            lab_l.append(lab)
            ret_l.append(inst["ar_star"])
            obs, _, done, _, info = env.step(j)
            t += 1
        assert info["valid_placed"] == env.M and abs(info["ar"] - inst["ar_star"]) < 1e-4, "expert replay mismatch"
    return (np.array(obs_l, dtype=np.float32), np.array(mask_l), np.array(lab_l), np.array(ret_l, dtype=np.float32))


def pretrain(model, data, epochs: int = BC_EPOCHS, seed: int = 0) -> dict:
    """Fit the MaskablePPO policy to the expert labels and its value head to the expert return."""
    import torch
    obs, mask, lab, ret = (torch.as_tensor(x) for x in data)
    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=BC_LR)
    g = torch.Generator().manual_seed(seed)
    n = len(obs)
    hist = []

    def forward(o, mk):
        feats = pol.extract_features(o)
        fp, fv = feats if isinstance(feats, tuple) else (feats, feats)
        lat_pi, lat_vf = pol.mlp_extractor.forward_actor(fp), pol.mlp_extractor.forward_critic(fv)
        logits = pol.action_net(lat_pi).masked_fill(~mk, -1e9)
        return torch.log_softmax(logits, -1), pol.value_net(lat_vf).squeeze(-1)

    pol.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for s in range(0, n, BC_BATCH):
            b = perm[s:s + BC_BATCH]
            logp, v = forward(obs[b], mask[b])
            loss_pi = -torch.logsumexp(logp.masked_fill(~lab[b], -1e9), -1).mean()
            loss_v = torch.nn.functional.mse_loss(v, ret[b])
            loss = loss_pi + BC_VALUE_COEF * loss_v
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss_pi.detach()) * len(b)
        hist.append(tot / n)
    pol.eval()
    with torch.no_grad():
        logp, v = forward(obs, mask)
        acc = float(lab[torch.arange(n), logp.argmax(-1)].float().mean())
        v_mse = float(torch.nn.functional.mse_loss(v, ret))
    return {"bc_samples": n, "bc_epochs": epochs, "bc_lr": BC_LR, "bc_loss_first": round(hist[0], 6),
            "bc_loss_last": round(hist[-1], 6), "bc_train_acc": round(acc, 6), "bc_value_mse": round(v_mse, 6)}


if __name__ == "__main__":
    build_allocs(sys.argv[1])
