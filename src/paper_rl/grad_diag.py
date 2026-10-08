"""v4.4.8: actor / critic gradient conflict on the shared encoder of the structure-aware policy.

After every rollout (returns and advantages computed, before the PPO update) the callback takes the
rollout buffer in minibatches exactly as MaskablePPO does (batch_size, advantages normalised per
minibatch) and computes, at the current parameters, the gradients on the shared encoder (input
layers, type embeddings, attention blocks, final LayerNorm) of
    L_pi = -mean(A_norm * ratio)            (ratio = 1 here, so grad = -mean(A_norm * grad log pi))
    L_V' = vf_coef * mse(returns, V(s))
and logs cos(grad L_pi, grad L_V') of the whole-buffer gradients, the mean minibatch cosine, the share
of minibatches with a negative cosine, and both gradient norms. Nothing is stepped; grads are zeroed.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch as th
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback


def encoder_params(policy):
    g = policy.gnet
    return [p for m in (g.inp, g.blocks, g.ln) for p in m.parameters()] + [g.type_emb]


def _flat(params):
    return th.cat([(p.grad if p.grad is not None else th.zeros_like(p)).reshape(-1) for p in params])


class GradDiagCallback(BaseCallback):
    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)
        self.rows = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        model, pol = self.model, self.model.policy
        params = encoder_params(pol)
        sum_pi = th.zeros(sum(p.numel() for p in params), device=pol.device)
        sum_v = th.zeros_like(sum_pi)
        coss, nb = [], 0
        for data in model.rollout_buffer.get(model.batch_size):
            values, log_prob, _ = pol.evaluate_actions(data.observations, data.actions.long().flatten(),
                                                       action_masks=data.action_masks)
            adv = data.advantages
            if model.normalize_advantage:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = th.exp(log_prob - data.old_log_prob)
            l_pi = -(adv * ratio).mean()
            l_v = model.vf_coef * F.mse_loss(data.returns, values.flatten())
            pol.zero_grad()
            l_pi.backward(retain_graph=True)
            g_pi = _flat(params).clone()
            pol.zero_grad()
            l_v.backward()
            g_v = _flat(params).clone()
            pol.zero_grad()
            coss.append(F.cosine_similarity(g_pi, g_v, dim=0).item())
            sum_pi += g_pi
            sum_v += g_v
            nb += 1
        self.rows.append({"timestep": self.num_timesteps,
                          "cos_full": F.cosine_similarity(sum_pi, sum_v, dim=0).item(),
                          "cos_mb_mean": sum(coss) / nb, "frac_mb_neg": sum(c < 0 for c in coss) / nb,
                          "norm_pi": (sum_pi / nb).norm().item(), "norm_v": (sum_v / nb).norm().item()})
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
