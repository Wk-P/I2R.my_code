"""v4.5.2: structure-aware Q-network for (Maskable) DQN / DDQN.

The same relation-biased attention network as the PPO policy (paper_rl/graph_net.GraphPolicyNet);
its unmasked scores are the Q-values: every ECU token gives Q(s, ECU j) through the shared ECU head
(ECU-permutation equivariant), the global token gives Q(s, EXIT). Masking is done by MaskableDQN
(mask computed from the raw state, see shared/dqn_variants.py). Needs PlacementEnv(obs_mode="raw").
"""
from __future__ import annotations

import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy

from paper_rl.graph_net import GraphPolicyNet, decode_raw


class GraphQNet(nn.Module):
    def __init__(self, n_ecu: int, n_svc: int, gkw: dict):
        super().__init__()
        self.n_ecu, self.n_svc = n_ecu, n_svc
        self.net = GraphPolicyNet(**gkw)

    def forward(self, obs):
        q, *_ = self.net.scores(*decode_raw(obs.float(), self.n_ecu, self.n_svc))
        return q

    def _predict(self, obs, deterministic: bool = True):
        return self(obs).argmax(dim=1).reshape(-1)

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)


class GraphDQNPolicy(DQNPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, n_ecu: int = 0, n_svc: int = 0,
                 d: int = 128, heads: int = 4, layers: int = 3, **kwargs):
        self.n_ecu, self.n_svc, self.gkw = n_ecu, n_svc, dict(d=d, heads=heads, layers=layers)
        kwargs.pop("net_arch", None)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_q_net(self):
        return GraphQNet(self.n_ecu, self.n_svc, self.gkw).to(self.device)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(n_ecu=self.n_ecu, n_svc=self.n_svc, **self.gkw)
        data.pop("net_arch", None)
        return data
