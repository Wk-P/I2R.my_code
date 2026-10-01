"""shared/dqn_variants.py — DQN-family learners added in v4.3.0.

    DoubleDQN     — Double DQN target (same as scenarios/*/ddqn/run_all.py::DDQN)
    MaskableDQN   — DQN with a hard action mask
    MaskableDDQN  — Double DQN with a hard action mask

The action mask is read straight from the observation instead of being
stored separately in the replay buffer: the Mask-PPO environment
(scenarios/*/ppo_mask/env.py::P4Env) writes action_masks() verbatim into the
`valid_flag` slice of its observation, so for any state s the mask is
obs[mask_start : mask_start+N] > 0.5. That gives the mask of s_{t+1} for the
TD target for free (replay_data.next_observations) without a custom buffer.

The mask is applied everywhere the DQN family picks an action:
    - epsilon-greedy exploration and the warm-up phase sample uniformly
      among valid ECUs only;
    - the greedy action is argmax over valid ECUs;
    - the TD target maxes (DQN) / argmaxes (DDQN) over the valid ECUs of
      s_{t+1}.
If a state has no valid ECU at all, its mask falls back to all-True (every
ECU allowed), so the agent still has to pick something and the step is
recorded as a violation by the environment -- the DQN counterpart of
MaskablePPO's behaviour on an all-False mask.
"""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN


def _td_update(model: DQN, gradient_steps: int, batch_size: int, next_value_fn) -> None:
    """Shared body of DQN.train(); `next_value_fn(next_obs) -> (B,1)` is the
    only part that differs between the variants."""
    model.policy.set_training_mode(True)
    model._update_learning_rate(model.policy.optimizer)
    losses = []
    for _ in range(gradient_steps):
        replay_data = model.replay_buffer.sample(batch_size, env=model._vec_normalize_env)
        discounts = replay_data.discounts if replay_data.discounts is not None else model.gamma
        with th.no_grad():
            next_q = next_value_fn(replay_data.next_observations)
            target_q = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q
        current_q = th.gather(model.q_net(replay_data.observations), dim=1,
                              index=replay_data.actions.long())
        loss = F.smooth_l1_loss(current_q, target_q)
        losses.append(loss.item())
        model.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
        model.policy.optimizer.step()
    model._n_updates += gradient_steps
    model.logger.record("train/n_updates", model._n_updates, exclude="tensorboard")
    model.logger.record("train/loss", float(np.mean(losses)))


class DoubleDQN(DQN):
    """Double DQN: online network selects the next action, target network
    evaluates it (van Hasselt et al., 2016)."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        def next_value(next_obs):
            best = self.q_net(next_obs).argmax(dim=1, keepdim=True)
            return self.q_net_target(next_obs).gather(1, best)
        _td_update(self, gradient_steps, batch_size, next_value)


class MaskableDQN(DQN):
    """DQN with a hard action mask read from obs[mask_start : mask_start+N]."""

    def __init__(self, *args, mask_start: int | None = None, **kwargs):
        # mask_start defaults to None only so that DQN.load() (which calls
        # __init__ without it and then restores attributes from the zip)
        # works; training always passes it explicitly.
        self.mask_start = mask_start
        super().__init__(*args, **kwargs)

    # ── mask helpers ─────────────────────────────────────────────────────────
    def _mask_np(self, obs: np.ndarray) -> np.ndarray:
        n = self.action_space.n
        m = obs[..., self.mask_start:self.mask_start + n] > 0.5
        empty = ~m.any(axis=-1, keepdims=True)
        return m | empty

    def _mask_th(self, obs: th.Tensor) -> th.Tensor:
        n = self.action_space.n
        m = obs[:, self.mask_start:self.mask_start + n] > 0.5
        empty = ~m.any(dim=1, keepdim=True)
        return m | empty

    def _masked_q(self, net, obs: th.Tensor) -> th.Tensor:
        return net(obs).masked_fill(~self._mask_th(obs), -th.inf)

    def _random_valid(self, mask: np.ndarray) -> np.ndarray:
        return np.array([np.random.choice(np.flatnonzero(row)) for row in mask])

    # ── acting ───────────────────────────────────────────────────────────────
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        obs = np.asarray(observation, dtype=np.float32)
        vectorized = obs.ndim == 2
        batch = obs if vectorized else obs[None]
        mask = self._mask_np(batch)
        if not deterministic and np.random.rand() < self.exploration_rate:
            action = self._random_valid(mask)
        else:
            self.policy.set_training_mode(False)
            with th.no_grad():
                obs_t = th.as_tensor(batch, device=self.device)
                action = self._masked_q(self.q_net, obs_t).argmax(dim=1).cpu().numpy()
        return (action if vectorized else action[0]), state

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            action = self._random_valid(self._mask_np(np.asarray(self._last_obs)))
        else:
            action, _ = self.predict(self._last_obs, deterministic=False)
        return action, action

    # ── learning ─────────────────────────────────────────────────────────────
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        def next_value(next_obs):
            return self._masked_q(self.q_net_target, next_obs).max(dim=1, keepdim=True)[0]
        _td_update(self, gradient_steps, batch_size, next_value)


class MaskableDDQN(MaskableDQN):
    """Double DQN with the same hard action mask: the online network picks
    argmax over valid next actions, the target network evaluates it."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        def next_value(next_obs):
            best = self._masked_q(self.q_net, next_obs).argmax(dim=1, keepdim=True)
            return self.q_net_target(next_obs).gather(1, best)
        _td_update(self, gradient_steps, batch_size, next_value)
