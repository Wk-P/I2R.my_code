"""v4.4.11: same-instance multi-trajectory Mask PPO with an instance-wise baseline (training only).

The n_envs environments are split into groups of K. At every (re)start all K environments of a group
take the same training instance (drawn uniformly, seeded) and run one stochastic episode each with
the current policy; environments that finish early wait for the rest of their group. When the K
episodes are complete, every step of trajectory k gets the advantage
    A = G_k - b_x,    b_x = (1/K) sum_k G_k      (gamma = 1: G_k is the episode return)
-- no critic and no GAE in the advantage. The critic is still trained, with target G_k (the PPO loss
is unchanged), but does not enter the advantage. A rollout collects complete groups until
n_steps * n_envs transitions are available; groups still running at that point are discarded, so
every trajectory used is complete and sampled from the current policy (the discarded steps still
count towards the step budget). Evaluation is unchanged: one deterministic rollout per instance.
"""
from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import obs_as_tensor


class GroupMaskablePPO(MaskablePPO):
    def __init__(self, *args, group_k: int = 4, group_seed: int = 0, **kwargs):
        self.group_k = group_k
        self.group_seed = group_seed
        super().__init__(*args, **kwargs)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int, use_masking: bool = True) -> bool:
        envs, K = env.envs, self.group_k
        assert len(envs) % K == 0, f"n_envs = {len(envs)} is not a multiple of K = {K}"
        if not hasattr(self, "_group_rng"):
            self._group_rng = np.random.default_rng(self.group_seed)
        n_inst = len(envs[0].unwrapped.instances)
        target = n_rollout_steps * len(envs)
        self.policy.set_training_mode(False)
        callback.on_rollout_start()

        obs = [None] * len(envs)
        active = [False] * len(envs)
        traj = [[] for _ in envs]                       # per env: (obs, action, log_prob, value, mask, reward)

        def start(g):
            k = int(self._group_rng.integers(n_inst))
            for i in range(g * K, (g + 1) * K):
                envs[i].unwrapped.use_instance(k)
                obs[i], _ = envs[i].reset()
                active[i], traj[i] = True, []

        for g in range(len(envs) // K):
            start(g)
        done_steps = []                                 # (obs, action, log_prob, value, mask, adv, ret)
        self.last_groups = []                           # per completed group: instances, returns, lengths (checks)
        while len(done_steps) < target:
            idx = [i for i in range(len(envs)) if active[i]]
            masks = np.stack([envs[i].unwrapped.action_masks() for i in idx])
            with th.no_grad():
                acts, vals, logps = self.policy(obs_as_tensor(np.stack([obs[i] for i in idx]), self.device),
                                                action_masks=masks)
            acts, vals, logps = acts.cpu().numpy(), vals.cpu().numpy().flatten(), logps.cpu().numpy()
            infos = []
            for n, i in enumerate(idx):
                o2, r, term, trunc, info = envs[i].step(int(acts[n]))
                traj[i].append((obs[i], acts[n], logps[n], vals[n], masks[n], r))
                obs[i] = o2
                infos.append(info)
                if term or trunc:
                    active[i] = False
            self.num_timesteps += len(idx)
            callback.update_locals({"infos": infos})
            if not callback.on_step():
                return False
            for g in range(len(envs) // K):
                members = range(g * K, (g + 1) * K)
                if any(active[i] for i in members):
                    continue
                G = np.array([sum(s[5] for s in traj[i]) for i in members])
                b = G.mean()
                self.last_groups.append({"inst": [envs[i].unwrapped.inst_idx for i in members], "G": G.tolist(),
                                         "len": [len(traj[i]) for i in members]})
                for i, Gk in zip(members, G):
                    done_steps += [(s[0], s[1], s[2], s[3], s[4], Gk - b, Gk) for s in traj[i]]
                start(g)

        steps = done_steps[:target]
        buf = rollout_buffer
        buf.reset()
        shape = (buf.buffer_size, buf.n_envs)
        buf.observations[:] = np.stack([s[0] for s in steps]).reshape(*shape, *buf.obs_shape)
        buf.actions[:] = np.array([s[1] for s in steps]).reshape(*shape, buf.action_dim)
        buf.log_probs[:] = np.array([s[2] for s in steps]).reshape(shape)
        buf.values[:] = np.array([s[3] for s in steps]).reshape(shape)
        buf.action_masks[:] = np.stack([s[4] for s in steps]).reshape(*shape, buf.mask_dims)
        buf.advantages[:] = np.array([s[5] for s in steps]).reshape(shape)
        buf.returns[:] = np.array([s[6] for s in steps]).reshape(shape)
        buf.pos, buf.full = buf.buffer_size, True
        callback.update_locals({"infos": []})
        callback.on_rollout_end()
        return True
