"""v4.4.3: MaskablePPO policy built on the structure-aware network (src/paper_rl/graph_net.py).

Shared encoder, separate heads: every ECU token is scored by the same actor head (ECU-
permutation equivariant), EXIT by the global token, and the value V(s) comes from the global
token plus the mean of all tokens (permutation invariant). Needs PlacementEnv(obs_mode="raw").
Everything else (distribution, masking, PPO update) is MaskablePPO's own.
separate=True (v4.4.8): actor and critic each get their own encoder of the same structure (no shared
parameters); the actor uses the ECU / EXIT heads of `gnet`, the critic the value head of `vnet`.
v4.5.4: also the policy of plain PPO for the unconstrained / Lagrange / Repair mechanisms -- mask_logits=False
uses the network's unmasked scores (no feasibility mask inside the network), the logits are cut to the
action space (N ECUs, no EXIT), and lam_in=True feeds Lagrange's lambda (last raw entry) to the global token.
With action_masks=None, MaskableActorCriticPolicy behaves as a plain categorical actor-critic.
"""
from __future__ import annotations

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from paper_rl.graph_net import GraphPolicyNet, decode_raw, raw_lambda, run_graphed


class GraphMaskablePolicy(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, n_ecu: int = 0, n_svc: int = 0,
                 d: int = 128, heads: int = 4, layers: int = 3, glob_std: bool = False, separate: bool = False,
                 pair_head: bool = False, plan: bool = False, plan_decay: bool = False, glob3: bool = False,
                 lam_in: bool = False, mask_logits: bool = True, feas_in: bool = True, **kwargs):
        self.n_ecu, self.n_svc = n_ecu, n_svc
        self.gkw = dict(d=d, heads=heads, layers=layers, glob_std=glob_std, pair_head=pair_head, plan=plan, plan_decay=plan_decay, glob3=glob3,
                        lam_in=lam_in, feas_in=feas_in)
        self.separate, self.mask_logits = separate, mask_logits
        assert not kwargs.pop("use_sde", False), "state-dependent exploration is for continuous actions"   # plain PPO passes it (v4.5.4)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        self.gnet = GraphPolicyNet(**self.gkw)
        if self.separate:
            self.vnet = GraphPolicyNet(**self.gkw)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(n_ecu=self.n_ecu, n_svc=self.n_svc, separate=self.separate, mask_logits=self.mask_logits, **self.gkw)
        return data

    def _run(self, obs):
        return run_graphed(self, "pi", self._run_eager, obs)    # v5.0.0: CUDA graph on GPU (graph_net.run_graphed)

    def _run_eager(self, obs):
        obs = obs.float()
        e = decode_raw(obs, self.n_ecu, self.n_svc)
        lam = raw_lambda(obs, self.n_ecu, self.n_svc) if self.gkw["lam_in"] else None
        logits, value, _, _ = (self.gnet if self.mask_logits else self.gnet.scores)(*e, lam)
        if self.separate:
            value = self.vnet.scores(*e, lam)[1]
        return logits[:, :self.action_space.n], value.unsqueeze(-1)

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
