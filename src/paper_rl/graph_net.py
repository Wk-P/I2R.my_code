"""v4.4.3: permutation-equivariant policy network for the placement problem.

The state is given raw -- capacities caps [B,N], demands reqs [B,M] (placement order, i.e.
descending demand), conflict graph adj [B,M,M], assignment assign [B,M] (ECU index, -1 = not
placed), current step t [B] -- and every feature is derived inside `build`, so the same code
serves supervised tests and (later) PPO.

Tokens: 1 global + N ECUs + M services. Node features:
  global : AR so far, unplaced share, unplaced demand / total capacity, free capacity / total capacity
           [+ 2 sigma_util with glob_std (v4.4.6): std of u_j / c_j over the active ECUs, scaled to [0, 1]]
  ECU j  : c_j / c_max, free_j / c_max, active, feasible for the current service, hosted / M, load_j / c_j
  svc k  : d_k / c_max, placed, is current, feasible ECUs / N, conflicts with other unplaced / M
Relations (added to the attention logits through a learned weight per relation, layer and head):
  svc-svc conflict (both unplaced), svc hosted on ECU, unplaced svc conflicts with an ECU's
  services, unplaced svc fits the ECU's free capacity. Every ECU is scored by the same head;
  EXIT is scored from the global token. Swapping ECUs (or services) swaps the outputs.
  pair_head (v4.4.10): ECU j is scored from the current-service / ECU pair,
  MLP([h_svc(t), h_ecu(j), h_svc(t) * h_ecu(j), h_glob]) (4d -> d -> 1), one head shared by all ECUs.
  plan (v4.4.12): a separate plan encoder (same structure, own parameters) reads the initial state s_0 of
  the instance -- recovered exactly from any s_t by clearing the assignment and setting t = 0 -- and
  forms a service x ECU plan matrix P_ij = <W_s h_i(s_0), W_e h_j(s_0)> / sqrt(d), fixed for the whole
  episode. ECU logits become P[i_t, j] + (the usual per-step score). W_e starts at zero, so P = 0 and the
  policy equals the plain one at initialisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_REL = 4


def build(caps, reqs, adj, assign, t, glob_std: bool = False):
    """Raw state -> (node features: glob [B,4 or 5], ecu [B,N,6], svc [B,M,5]), relations [B,R,L,L], mask [B,N+1], ar [B]."""
    B, N = caps.shape
    M = reqs.shape[1]
    dev = caps.device
    ar_idx = torch.arange(B, device=dev)
    cmax = caps.max(1, keepdim=True).values
    placed = assign >= 0
    unpl = ~placed
    host = assign.unsqueeze(-1) == torch.arange(N, device=dev)              # [B,M,N]
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
    cur = torch.arange(M, device=dev) == tc.unsqueeze(1)
    m_ecu = feas[ar_idx, tc] & (t < M).unsqueeze(1)
    mask = torch.cat([m_ecu, ~m_ecu.any(1, keepdim=True)], 1)
    tot = caps.sum(1)
    gl = [ar, unpl.float().mean(1), (reqs * unpl).sum(1) / tot, free.clamp(min=0).sum(1) / tot]
    if glob_std:                                    # v4.4.6: spread of utilisation over the active ECUs
        var = (((util - ar.unsqueeze(1)) ** 2) * active).sum(1) / active.sum(1).clamp(min=1)
        gl.append(2 * var.clamp(min=0).sqrt())
    glob = torch.stack(gl, 1)
    ecu = torch.stack([caps / cmax, free.clamp(min=0) / cmax, active.float(), m_ecu.float(),
                       hostf.sum(1) / M, util], -1)
    uu = unpl.unsqueeze(1) & unpl.unsqueeze(2)
    svc = torch.stack([reqs / cmax, placed.float(), cur.float(), feas.float().sum(-1) / N,
                       (adj & uu).float().sum(-1) / M * unpl], -1)
    L = 1 + N + M
    rel = torch.zeros(B, N_REL, L, L, device=dev)
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


def _encode(inp, type_emb, blocks, ln, feats, rel):
    glob, ecu, svc = feats
    x = torch.cat([inp[0](glob).unsqueeze(1) + type_emb[0], inp[1](ecu) + type_emb[1], inp[2](svc) + type_emb[2]], 1)
    for b in blocks:
        x = b(x, rel)
    return ln(x)


class PlanEncoder(nn.Module):
    """v4.4.12: P = F(s_0), the service x ECU plan matrix of an instance (episode-constant)."""

    def __init__(self, d: int, heads: int, layers: int, glob_std: bool):
        super().__init__()
        self.d, self.glob_std = d, glob_std
        self.inp = nn.ModuleList([nn.Linear(5 if glob_std else 4, d), nn.Linear(6, d), nn.Linear(5, d)])
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.w_s, self.w_e = nn.Linear(d, d), nn.Linear(d, d)
        nn.init.zeros_(self.w_e.weight)
        nn.init.zeros_(self.w_e.bias)

    def forward(self, caps, reqs, adj):
        """-> P [B, M, N] from the initial state (nothing placed, t = 0)."""
        B, N = caps.shape
        M = reqs.shape[1]
        assign0 = torch.full((B, M), -1, dtype=torch.long, device=caps.device)
        t0 = torch.zeros(B, dtype=torch.long, device=caps.device)
        feats, rel, _, _ = build(caps, reqs, adj, assign0, t0, self.glob_std)
        x0 = _encode(self.inp, self.type_emb, self.blocks, self.ln, feats, rel)
        h_ecu, h_svc = x0[:, 1:1 + N], x0[:, 1 + N:]
        return self.w_s(h_svc) @ self.w_e(h_ecu).transpose(1, 2) / self.d ** 0.5


class GraphPolicyNet(nn.Module):
    def __init__(self, d: int = 128, heads: int = 4, layers: int = 3, glob_std: bool = False, pair_head: bool = False,
                 plan: bool = False):
        super().__init__()
        self.glob_std, self.pair_head, self.plan = glob_std, pair_head, plan
        if plan:                                        # v4.4.12: global plan matrix from s_0
            self.plan_enc = PlanEncoder(d, heads, layers, glob_std)
        self.inp = nn.ModuleList([nn.Linear(5 if glob_std else 4, d), nn.Linear(6, d), nn.Linear(5, d)])
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        if pair_head:                                   # v4.4.10: current-service x ECU pair scoring
            self.pair_mlp = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))
        else:
            self.ecu_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.exit_head = nn.Linear(d, 1)
        self.value_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, caps, reqs, adj, assign, t):
        """-> masked logits [B,N+1], value [B], mask [B,N+1], ar [B]"""
        feats, rel, mask, ar = build(caps, reqs, adj, assign, t, self.glob_std)
        N = caps.shape[1]
        x = _encode(self.inp, self.type_emb, self.blocks, self.ln, feats, rel)
        h_ecu = x[:, 1:1 + N]
        if self.pair_head:
            M = reqs.shape[1]
            h_svc = x[torch.arange(x.shape[0], device=x.device), 1 + N + t.clamp(max=M - 1)]   # current service
            h_svc = h_svc.unsqueeze(1).expand_as(h_ecu)
            z = torch.cat([h_svc, h_ecu, h_svc * h_ecu, x[:, :1].expand_as(h_ecu)], -1)
            ecu_logits = self.pair_mlp(z).squeeze(-1)
        else:
            ecu_logits = self.ecu_head(h_ecu).squeeze(-1)
        if self.plan:                                   # plan row of the current service: P[i_t, :]
            M = reqs.shape[1]
            P = self.plan_enc(caps, reqs, adj)
            ecu_logits = ecu_logits + P[torch.arange(P.shape[0], device=P.device), t.clamp(max=M - 1)]
        logits = torch.cat([ecu_logits, self.exit_head(x[:, 0])], 1)
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


def decode_raw(obs, n: int, m: int):
    """PlacementEnv raw observation [B, N+M+M*M+M+1] -> (caps, reqs, adj, assign, t)."""
    caps = obs[:, :n]
    reqs = obs[:, n:n + m]
    adj = obs[:, n + m:n + m + m * m].reshape(-1, m, m) > 0.5
    assign = obs[:, n + m + m * m:n + 2 * m + m * m].round().long() - 1
    t = obs[:, -1].round().long()
    return caps, reqs, adj, assign, t
