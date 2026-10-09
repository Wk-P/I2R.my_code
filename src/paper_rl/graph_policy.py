"""v4.4.3: MaskablePPO policy built on the structure-aware network (src/paper_rl/graph_net.py).

Shared encoder, separate heads: every ECU token is scored by the same actor head (ECU-
permutation equivariant), EXIT by the global token, and the value V(s) comes from the global
token plus the mean of all tokens (permutation invariant). Needs PlacementEnv(obs_mode="raw").
Everything else (distribution, masking, PPO update) is MaskablePPO's own.
separate=True (v4.4.8): actor and critic each get their own encoder of the same structure (no shared
parameters); the actor uses the ECU / EXIT heads of `gnet`, the critic the value head of `vnet`.
"""
from __future__ import annotations

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from paper_rl.graph_net import GraphPolicyNet, decode_raw


class GraphMaskablePolicy(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, n_ecu: int = 0, n_svc: int = 0,
                 d: int = 128, heads: int = 4, layers: int = 3, glob_std: bool = False, separate: bool = False,
                 pair_head: bool = False, plan: bool = False, plan_decay: bool = False, glob3: bool = False, **kwargs):
        self.n_ecu, self.n_svc = n_ecu, n_svc
        self.gkw = dict(d=d, heads=heads, layers=layers, glob_std=glob_std, pair_head=pair_head, plan=plan, plan_decay=plan_decay, glob3=glob3)
        self.separate = separate
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        self.gnet = GraphPolicyNet(**self.gkw)
        if self.separate:
            self.vnet = GraphPolicyNet(**self.gkw)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(n_ecu=self.n_ecu, n_svc=self.n_svc, separate=self.separate, **self.gkw)
        return data

    def _run(self, obs):
        x = decode_raw(obs.float(), self.n_ecu, self.n_svc)
        logits, value, _, _ = self.gnet(*x)
        if self.separate:
            value = self.vnet(*x)[1]
        return logits, value.unsqueeze(-1)

    def _dist(self, logits, action_masks):
        dist = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            dist.apply_masking(action_masks)
        return dist

    def forward(self, obs, deterministic: bool = False, action_masks=None):
        logits, values = self._run(obs)
        dist = self._dist(logits, action_masks)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, values, dist.log_prob(actions)

    def evaluate_actions(self, obs, actions, action_masks=None):
        logits, values = self._run(obs)
        dist = self._dist(logits, action_masks)
        return values, dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs, action_masks=None):
        return self._dist(self._run(obs)[0], action_masks)

    def predict_values(self, obs):
        return self._run(obs)[1]
