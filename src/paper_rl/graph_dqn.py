"""v4.5.2: structure-aware Q-network for (Maskable) DQN / DDQN.

The same relation-biased attention network as the PPO policy (paper_rl/graph_net.GraphPolicyNet);
its unmasked scores are the Q-values: every ECU token gives Q(s, ECU j) through the shared ECU head
(ECU-permutation equivariant), the global token gives Q(s, EXIT). Masking is done by MaskableDQN
(mask computed from the raw state, see shared/dqn_variants.py). Needs PlacementEnv(obs_mode="raw").
v4.5.4: also the Q-network of plain DQN / DDQN for the unconstrained / Lagrange / Repair mechanisms: the Q-values
are cut to the action space (N ECUs, no EXIT), and lam_in feeds Lagrange's lambda to the global token.
"""
from __future__ import annotations

import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy

from paper_rl.graph_net import GraphPolicyNet, decode_raw, raw_lambda, run_graphed


class GraphQNet(nn.Module):
    def __init__(self, n_ecu: int, n_svc: int, gkw: dict, n_out: int | None = None):
        super().__init__()
        self.n_ecu, self.n_svc, self.n_out = n_ecu, n_svc, n_out
        self.net = GraphPolicyNet(**gkw)

    def forward(self, obs):
        return self.q_and_mask(obs)[0]

    def q_and_mask(self, obs):
        """v5.0.0: Q-values and the raw feasibility mask [B, N+1] (ECUs + EXIT) from ONE pass of build(), so
        Maskable DQN does not rebuild the state features just for the mask (same values as before); on GPU
        replayed from a CUDA graph. Only (Q, mask) are graph outputs: the value head stays out of the graph, so
        it gets no gradient, exactly as in the eager pass."""
        return run_graphed(self, "q", self._q_and_mask, obs)

    def _q_and_mask(self, obs):
        obs = obs.float()
        lam = raw_lambda(obs, self.n_ecu, self.n_svc) if self.net.lam_in else None
        q, _, mask, _ = self.net.scores(*decode_raw(obs, self.n_ecu, self.n_svc), lam)
        return (q[:, :self.n_out] if self.n_out else q), mask

    def _predict(self, obs, deterministic: bool = True):
        return self(obs).argmax(dim=1).reshape(-1)

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)


class GraphDQNPolicy(DQNPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, n_ecu: int = 0, n_svc: int = 0,
                 d: int = 128, heads: int = 4, layers: int = 3, lam_in: bool = False, **kwargs):
        self.n_ecu, self.n_svc, self.gkw = n_ecu, n_svc, dict(d=d, heads=heads, layers=layers, lam_in=lam_in)
        kwargs.pop("net_arch", None)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_q_net(self):
        return GraphQNet(self.n_ecu, self.n_svc, self.gkw, self.action_space.n).to(self.device)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(n_ecu=self.n_ecu, n_svc=self.n_svc, **self.gkw)
        data.pop("net_arch", None)
        return data
