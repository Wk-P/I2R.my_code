"""v4.4.3: permutation-equivariant policy network for the placement problem.

The state is given raw -- capacities caps [B,N], demands reqs [B,M] (placement order, i.e.
descending demand), conflict graph adj [B,M,M], assignment assign [B,M] (ECU index, -1 = not
placed), current step t [B] -- and every feature is derived inside `build`, so the same code
serves supervised tests and (later) PPO.

Tokens: 1 global + N ECUs + M services. Node features:
  global : AR so far, unplaced share, unplaced demand / total capacity, free capacity / total capacity
           [+ 2 sigma_util with glob_std (v4.4.6): std of u_j / c_j over the active ECUs, scaled to [0, 1]]
           glob3 (v4.4.13, the professor's state summary): exactly [M_rem / M, AR_t, sigma_util,t] -- nothing else
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
  policy equals the plain one at initialisation. plan_decay (v4.4.13): the plan term is weighted by
  beta_t = (M - t) / M (1 at the first placement, 1/M at the last), so the per-step score takes over.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_REL = 4

# ── CUDA Graphs (v5.0.0) ────────────────────────────────────────────────────────────────────────────────
# The network is many small ops (build() + 3 attention layers ~ 150 kernels per pass); on GPU the time goes into
# launching them from Python, not into the GPU. run_graphed() records the whole pass (raw observation -> outputs)
# once per (owner, batch shape, grad mode) as a CUDA graph and afterwards replays it with one launch. The replayed
# kernels are the recorded ones, so the results are the same as the eager pass (checked bit for bit by
# src/scripts/check_v5.py). With grad enabled, torch.cuda.make_graphed_callables also records the backward pass.
# Off unless train.py turns it on for a CUDA run (V5_CUDA_GRAPHS=0 keeps it off).
CUDA_GRAPHS = False
_GRAPHS: dict[int, dict] = {}                   # id(owner) -> {(tag, shape, grad): runner}; owners stay alive


class _Fn(nn.Module):
    """make_graphed_callables needs a module: the owner's parameters, forward = fn."""

    def __init__(self, owner: nn.Module, fn):
        super().__init__()
        self.owner, self.fn = owner, fn

    def forward(self, obs):
        return self.fn(obs)


def _capture_nograd(fn, obs):
    static_in = obs.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
        for _ in range(3):                      # warm-up outside the graph (allocator, cuBLAS handles)
            fn(static_in)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        static_out = fn(static_in)

    def run(x):
        static_in.copy_(x)
        g.replay()
        return tuple(o.clone() for o in static_out)   # the next replay overwrites the static outputs
    return run


def _capture_grad(owner, fn, obs):
    return torch.cuda.make_graphed_callables(_Fn(owner, fn), (obs.clone(),), allow_unused_input=True)


def run_graphed(owner: nn.Module, tag: str, fn, obs):
    """fn(obs) -> tuple of tensors, replayed from a CUDA graph when CUDA_GRAPHS is on and obs is on a GPU.
    Everything fn uses must be fixed-shape and sync-free (true for build() / the encoder / the heads)."""
    if not (CUDA_GRAPHS and obs.is_cuda) or torch.cuda.is_current_stream_capturing():
        return fn(obs)
    key = (tag, tuple(obs.shape), obs.dtype, torch.is_grad_enabled())
    cache = _GRAPHS.setdefault(id(owner), {"owner": owner})
    if key not in cache:
        cache[key] = _capture_grad(owner, fn, obs) if key[-1] else _capture_nograd(fn, obs)
    return cache[key](obs)


def build(c, e, adj, assign, t, glob_std: bool = False, glob3: bool = False):
    """Raw state -> (node features: glob [B,4 or 5], ecu [B,N,6], svc [B,M,5]), relations [B,R,L,L], mask [B,N+1], ar [B]."""
    B, N = c.shape
    M = e.shape[1]
    dev = c.device
    ar_idx = torch.arange(B, device=dev)
    cmax = c.max(1, keepdim=True).values
    placed = assign >= 0
    unpl = ~placed
    host = assign.unsqueeze(-1) == torch.arange(N, device=dev)              # [B,M,N]
    hostf = host.float()
    load = (hostf * e.unsqueeze(-1)).sum(1)                              # [B,N]
    free = c - load
    active = host.any(1)
    util = load / c
    ar = (util * active).sum(1) / active.sum(1).clamp(min=1)
    adjf = adj.float()
    conf_ecu = (adjf @ hostf) > 0                                           # k conflicts with a service on j
    fits = e.unsqueeze(-1) <= free.unsqueeze(1)
    feas = fits & ~conf_ecu & unpl.unsqueeze(-1)                            # [B,M,N]
    tc = t.clamp(max=M - 1)
    cur = torch.arange(M, device=dev) == tc.unsqueeze(1)
    m_ecu = feas[ar_idx, tc] & (t < M).unsqueeze(1)
    mask = torch.cat([m_ecu, ~m_ecu.any(1, keepdim=True)], 1)
    tot = c.sum(1)
    if glob3:                                       # v4.4.13: [M_rem / M, AR_t, sigma_util,t]
        var = (((util - ar.unsqueeze(1)) ** 2) * active).sum(1) / active.sum(1).clamp(min=1)
        gl = [unpl.float().mean(1), ar, var.clamp(min=0).sqrt()]
    else:
        gl = [ar, unpl.float().mean(1), (e * unpl).sum(1) / tot, free.clamp(min=0).sum(1) / tot]
    if glob_std and not glob3:                                    # v4.4.6: spread of utilisation over the active ECUs
        var = (((util - ar.unsqueeze(1)) ** 2) * active).sum(1) / active.sum(1).clamp(min=1)
        gl.append(2 * var.clamp(min=0).sqrt())
    glob = torch.stack(gl, 1)
    ecu = torch.stack([c / cmax, free.clamp(min=0) / cmax, active.float(), m_ecu.float(),
                       hostf.sum(1) / M, util], -1)
    uu = unpl.unsqueeze(1) & unpl.unsqueeze(2)
    svc = torch.stack([e / cmax, placed.float(), cur.float(), feas.float().sum(-1) / N,
                       (adj & uu).float().sum(-1) / M * unpl], -1)
    L = 1 + N + M
    rel = torch.zeros(B, N_REL, L, L, device=dev)
    s, e_ = slice(1 + N, L), slice(1, 1 + N)
    rel[:, 0, s, s] = (adj & uu).float()
    for r, x in ((1, hostf), (2, (conf_ecu & unpl.unsqueeze(-1)).float()), (3, (fits & unpl.unsqueeze(-1)).float())):
        rel[:, r, s, e_] = x
        rel[:, r, e_, s] = x.transpose(1, 2)
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

    def forward(self, c, e, adj):
        """-> P [B, M, N] from the initial state (nothing placed, t = 0)."""
        B, N = c.shape
        M = e.shape[1]
        assign0 = torch.full((B, M), -1, dtype=torch.long, device=c.device)
        t0 = torch.zeros(B, dtype=torch.long, device=c.device)
        feats, rel, _, _ = build(c, e, adj, assign0, t0, self.glob_std)
        x0 = _encode(self.inp, self.type_emb, self.blocks, self.ln, feats, rel)
        h_ecu, h_svc = x0[:, 1:1 + N], x0[:, 1 + N:]
        return self.w_s(h_svc) @ self.w_e(h_ecu).transpose(1, 2) / self.d ** 0.5


class GraphPolicyNet(nn.Module):
    def __init__(self, d: int = 128, heads: int = 4, layers: int = 3, glob_std: bool = False, pair_head: bool = False,
                 plan: bool = False, plan_decay: bool = False, glob3: bool = False, lam_in: bool = False,
                 feas_in: bool = True):
        super().__init__()
        # v6.0.1: feas_in=False (unconstrained) -- every constraint-derived input is zeroed: ECU "feasible for the
        # current service", svc "feasible ECUs / N" and "conflicts with other unplaced / M", and the relations
        # svc-svc conflict, svc conflicts with an ECU, svc fits an ECU; utilisation and the global AR are clipped at 1,
        # so no violation can be read from the state. Shapes unchanged. Violations are still counted by the env.
        self.feas_in = feas_in
        assert feas_in or not (glob3 or glob_std), "feas_in=False: default global token only"
        self.lam_in = lam_in                            # v4.5.4: Lagrange -- lambda / LAMBDA_MAX joins the global token
        assert not (glob3 and glob_std), "glob3 already contains sigma_util"
        self.glob3 = glob3
        self.glob_std, self.pair_head, self.plan, self.plan_decay = glob_std, pair_head, plan, plan_decay
        assert plan or not plan_decay, "plan_decay needs plan"
        if plan:                                        # v4.4.12: global plan matrix from s_0
            self.plan_enc = PlanEncoder(d, heads, layers, glob_std)
        self.inp = nn.ModuleList([nn.Linear((3 if glob3 else 5 if glob_std else 4) + lam_in, d), nn.Linear(6, d), nn.Linear(5, d)])
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        if pair_head:                                   # v4.4.10: current-service x ECU pair scoring
            self.pair_mlp = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))
        else:
            self.ecu_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.exit_head = nn.Linear(d, 1)
        self.value_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, c, e, adj, assign, t, lam=None):
        """-> masked logits [B,N+1], value [B], mask [B,N+1], ar [B]"""
        logits, value, mask, ar = self.scores(c, e, adj, assign, t, lam)
        return logits.masked_fill(~mask, -1e9), value, mask, ar

    def scores(self, c, e, adj, assign, t, lam=None):
        """-> unmasked scores [B,N+1] (ECUs + EXIT; used as Q-values by graph DQN, v4.5.2), value [B], mask, ar.
        lam [B] (lambda / LAMBDA_MAX) is required iff the network was built with lam_in (v4.5.4)."""
        feats, rel, mask, ar = build(c, e, adj, assign, t, self.glob_std, self.glob3)
        if not self.feas_in:
            glob, ecu, svc = feats
            # no violation leaks either: utilisation clipped to 1 (free is already clamped at 0), AR from the clipped one
            util = ecu[..., 5].clamp(max=1.0)
            act = ecu[..., 2]
            glob = torch.cat([((util * act).sum(1) / act.sum(1).clamp(min=1)).unsqueeze(1), glob[:, 1:]], 1)
            ecu = torch.cat([ecu[..., :3], torch.zeros_like(ecu[..., 3:4]), ecu[..., 4:5], util.unsqueeze(-1)], -1)
            svc = torch.cat([svc[..., :3], torch.zeros_like(svc[..., 3:])], -1)
            feats = (glob, ecu, svc)
            rel = torch.stack([torch.zeros_like(rel[:, 0]), rel[:, 1], torch.zeros_like(rel[:, 2]), torch.zeros_like(rel[:, 3])], 1)
        if self.lam_in:
            assert lam is not None, "lam_in network needs lambda (Lagrange raw observation)"
            feats = (torch.cat([feats[0], lam.float().unsqueeze(1)], 1), *feats[1:])
        N = c.shape[1]
        x = _encode(self.inp, self.type_emb, self.blocks, self.ln, feats, rel)
        h_ecu = x[:, 1:1 + N]
        if self.pair_head:
            M = e.shape[1]
            h_svc = x[torch.arange(x.shape[0], device=x.device), 1 + N + t.clamp(max=M - 1)]   # current service
            h_svc = h_svc.unsqueeze(1).expand_as(h_ecu)
            z = torch.cat([h_svc, h_ecu, h_svc * h_ecu, x[:, :1].expand_as(h_ecu)], -1)
            ecu_logits = self.pair_mlp(z).squeeze(-1)
        else:
            ecu_logits = self.ecu_head(h_ecu).squeeze(-1)
        if self.plan:                                   # plan row of the current service: P[i_t, :]
            M = e.shape[1]
            P = self.plan_enc(c, e, adj)
            row = P[torch.arange(P.shape[0], device=P.device), t.clamp(max=M - 1)]
            if self.plan_decay:                         # v4.4.13: beta_t = (M - t) / M
                row = row * ((M - t.clamp(max=M - 1)).float() / M).unsqueeze(1)
            ecu_logits = ecu_logits + row
        logits = torch.cat([ecu_logits, self.exit_head(x[:, 0])], 1)
        value = self.value_head(torch.cat([x[:, 0], x[:, 1:].mean(1)], -1)).squeeze(-1)
        return logits, value, mask, ar


def state_from_env(env):
    """Raw state tensors (batch of 1) of a PlacementEnv (ECU action, placement order)."""
    import numpy as np
    adj = np.zeros((env.M, env.M), dtype=bool)
    for m, ps in enumerate(env.partners):
        adj[m, list(ps)] = True
    assign = np.full(env.M, -1)
    for n, hs in enumerate(env.hosted):
        for m in hs:
            assign[m] = n
    return (torch.as_tensor(env.c[None], dtype=torch.float32), torch.as_tensor(env.e[None], dtype=torch.float32),
            torch.as_tensor(adj[None]), torch.as_tensor(assign[None]), torch.as_tensor([env.t]))


def decode_raw(obs, n: int, m: int):
    """PlacementEnv raw observation [B, N+M+M*M+M+1 (+1)] -> (caps, reqs, adj, assign, t).
    t is read at its fixed position, so the optional trailing lambda (Lagrange, v4.5.4) does not move it."""
    c = obs[:, :n]
    e = obs[:, n:n + m]
    adj = obs[:, n + m:n + m + m * m].reshape(-1, m, m) > 0.5
    assign = obs[:, n + m + m * m:n + 2 * m + m * m].round().long() - 1
    t = obs[:, n + 2 * m + m * m].round().long()
    return c, e, adj, assign, t


def raw_lambda(obs, n: int, m: int):
    """v4.5.4: lambda / LAMBDA_MAX of a Lagrange raw observation [B] (the entry after t), None if absent."""
    k = n + 2 * m + m * m + 1
    return obs[:, k] if obs.shape[1] > k else None
