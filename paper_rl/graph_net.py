"""v4.4.3: permutation-equivariant policy network for the placement problem.

The state is given raw -- capacities caps [B,N], demands reqs [B,M] (placement order, i.e.
descending demand), conflict graph adj [B,M,M], assignment assign [B,M] (ECU index, -1 = not
placed), current step t [B] -- and every feature is derived inside `build`, so the same code
serves supervised tests and (later) PPO.

Tokens: 1 global + N ECUs + M services. Node features:
  global : AR so far, unplaced share, unplaced demand / total capacity, free capacity / total capacity
  ECU j  : c_j / c_max, free_j / c_max, active, feasible for the current service, hosted / M, load_j / c_j
  svc k  : d_k / c_max, placed, is current, feasible ECUs / N, conflicts with other unplaced / M
Relations (added to the attention logits through a learned weight per relation, layer and head):
  svc-svc conflict (both unplaced), svc hosted on ECU, unplaced svc conflicts with an ECU's
  services, unplaced svc fits the ECU's free capacity. Every ECU is scored by the same head;
  EXIT is scored from the global token. Swapping ECUs (or services) swaps the outputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_REL = 4


def build(caps, reqs, adj, assign, t):
    """Raw state -> (node features: glob [B,4], ecu [B,N,6], svc [B,M,5]), relations [B,R,L,L], mask [B,N+1], ar [B]."""
    B, N = caps.shape
    M = reqs.shape[1]
    ar_idx = torch.arange(B)
    cmax = caps.max(1, keepdim=True).values
    placed = assign >= 0
    unpl = ~placed
    host = assign.unsqueeze(-1) == torch.arange(N)                          # [B,M,N]
    hostf = host.float()
    load = (hostf * reqs.unsqueeze(-1)).sum(1)                              # [B,N]
    free = caps - load
    active = host.any(1)
    util = load / caps
    ar = (util * active).sum(1) / active.sum(1).clamp(min=1)
    adjf = adj.float()
    conf_ecu = (adjf @ hostf) > 0                                           # k conflicts with a service on j
    fits = reqs.unsqueeze(-1) <= free.unsqueeze(1)
    feas = fits & ~conf_ecu & unpl.unsqueeze(-1)                            # [B,M,N]
    tc = t.clamp(max=M - 1)
    cur = torch.arange(M) == tc.unsqueeze(1)
    m_ecu = feas[ar_idx, tc] & (t < M).unsqueeze(1)
    mask = torch.cat([m_ecu, ~m_ecu.any(1, keepdim=True)], 1)
    tot = caps.sum(1)
    glob = torch.stack([ar, unpl.float().mean(1), (reqs * unpl).sum(1) / tot, free.clamp(min=0).sum(1) / tot], 1)
    ecu = torch.stack([caps / cmax, free.clamp(min=0) / cmax, active.float(), m_ecu.float(),
                       hostf.sum(1) / M, util], -1)
    uu = unpl.unsqueeze(1) & unpl.unsqueeze(2)
    svc = torch.stack([reqs / cmax, placed.float(), cur.float(), feas.float().sum(-1) / N,
                       (adj & uu).float().sum(-1) / M * unpl], -1)
    L = 1 + N + M
    rel = torch.zeros(B, N_REL, L, L)
    s, e = slice(1 + N, L), slice(1, 1 + N)
    rel[:, 0, s, s] = (adj & uu).float()
    for r, x in ((1, hostf), (2, (conf_ecu & unpl.unsqueeze(-1)).float()), (3, (fits & unpl.unsqueeze(-1)).float())):
        rel[:, r, s, e] = x
        rel[:, r, e, s] = x.transpose(1, 2)
    return (glob, ecu, svc), rel, mask, ar


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.rel_w = nn.Parameter(torch.zeros(N_REL, heads))
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, rel):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        bias = torch.einsum("brij,rh->bhij", rel, self.rel_w)
        att = torch.softmax(q @ k.transpose(-1, -2) / self.dk ** 0.5 + bias, -1)
        x = x + self.o((att @ v).transpose(1, 2).reshape(B, L, d))
        return x + self.ff(self.ln2(x))


class GraphPolicyNet(nn.Module):
    def __init__(self, d: int = 128, heads: int = 4, layers: int = 3):
        super().__init__()
        self.inp = nn.ModuleList([nn.Linear(4, d), nn.Linear(6, d), nn.Linear(5, d)])
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.ecu_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.exit_head = nn.Linear(d, 1)
        self.value_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, caps, reqs, adj, assign, t):
        """-> masked logits [B,N+1], value [B], mask [B,N+1], ar [B]"""
        (glob, ecu, svc), rel, mask, ar = build(caps, reqs, adj, assign, t)
        N = caps.shape[1]
        x = torch.cat([self.inp[0](glob).unsqueeze(1) + self.type_emb[0], self.inp[1](ecu) + self.type_emb[1],
                       self.inp[2](svc) + self.type_emb[2]], 1)
        for b in self.blocks:
            x = b(x, rel)
        x = self.ln(x)
        logits = torch.cat([self.ecu_head(x[:, 1:1 + N]).squeeze(-1), self.exit_head(x[:, 0])], 1)
        value = self.value_head(torch.cat([x[:, 0], x[:, 1:].mean(1)], -1)).squeeze(-1)
        return logits.masked_fill(~mask, -1e9), value, mask, ar


def state_from_env(env):
    """Raw state tensors (batch of 1) of a PlacementEnv (ECU action, placement order)."""
    import numpy as np
    adj = np.zeros((env.M, env.M), dtype=bool)
    for i, ps in enumerate(env.partners):
        adj[i, list(ps)] = True
    assign = np.full(env.M, -1)
    for j, hs in enumerate(env.hosted):
        for i in hs:
            assign[i] = j
    return (torch.as_tensor(env.cap[None], dtype=torch.float32), torch.as_tensor(env.req[None], dtype=torch.float32),
            torch.as_tensor(adj[None]), torch.as_tensor(assign[None]), torch.as_tensor([env.t]))
