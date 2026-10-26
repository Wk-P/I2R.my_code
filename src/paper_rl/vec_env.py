"""v5.0.0: DummyVecEnv without the per-step deepcopy of the info dicts.

stable_baselines3 2.7.1 DummyVecEnv.step_wait() returns deepcopy(self.buf_infos) on every vector step; with 12-40
environments and ~16 keys per info dict that recursive copy is ~6% of a training run. Every entry of buf_infos is
replaced by the dict the environment returns on each step, PlacementEnv builds a new dict (numbers / booleans) and
a new observation array on every call, and Monitor a new "episode" dict per episode, so copying each dict one
level deep returns exactly the same contents. Everything else is DummyVecEnv's own code (same order of calls).
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn


class FastDummyVecEnv(DummyVecEnv):
    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(  # type: ignore[assignment]
                self.actions[env_idx]
            )
            self.buf_dones[env_idx] = terminated or truncated
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated
            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), [dict(m) for m in self.buf_infos])
