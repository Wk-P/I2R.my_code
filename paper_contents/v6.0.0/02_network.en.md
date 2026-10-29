# 2. Network: relation-biased Transformer encoder

Code: `src/paper_rl/graph_net.py` (`GraphPolicyNet`, `build`, `Block`), PPO wrapper `src/paper_rl/graph_policy.py`, DQN wrapper `src/paper_rl/graph_dqn.py`. All 12 variants use this one structure (`--net graph`); no MLP is used.

## 2.1 Sizes

| item | value |
|---|---|
| hidden size d | 128 |
| attention heads | 4 (d_k = 32) |
| Transformer layers | 3 |
| feed-forward | 128 → 256 → 128, GELU |
| normalisation | pre-LN (LayerNorm before each sub-layer) + a final LayerNorm |
| positional encoding | none (permutation-equivariant in ECUs and services); the current service is marked by an "is current" feature |
| tokens | 1 + N + M |
| parameters (PPO: one network; DQN: online and target network, one each) | 450,227 (lagrangian: 450,355, one extra global input λ) |

## 2.2 Input: tokens and features

The network reads the raw observation (04_environment) and derives every feature in `build()`:

- **global token (4)**: AR so far, share of unplaced services, unplaced demand / total capacity, free capacity / total capacity. lagrangian adds λ / (1 + λ).
- **ECU token (6 each)**: c_n / c_max, max(free capacity, 0) / c_max, in use, feasible for the current service, hosted services / M, load / c_n.
- **service token (5 each)**: e_m / c_max, placed, is current, feasible ECUs / N, conflicts with other unplaced services / M (0 once placed). c_max = the instance's largest capacity.
- Each token type is projected to d by its own linear layer (4→128, 6→128, 5→128; lagrangian global 5→128), plus a learned type embedding (3 × d, zero-initialised).

## 2.3 Relation-biased attention

In every layer the attention logits are QKᵀ / √d_k + Σ_r R_r · w_{r,h}, with R_r four 0/1 relation matrices and w a learned weight per layer and head (4 × heads, zero-initialised):

1. service–service conflict (both unplaced)
2. service hosted on ECU (symmetric)
3. unplaced service conflicts with a service on the ECU (symmetric)
4. unplaced service fits the ECU's free capacity (symmetric)

## 2.4 Heads

| head | structure | use |
|---|---|---|
| ecu_head | Linear(128→128) → GELU → Linear(128→1) | one score per ECU token, the same head for every ECU |
| exit_head | Linear(128→1) | scores EXIT from the global token (used by maskable only) |
| value_head | Linear(256→128) → GELU → Linear(128→1) | input [global token, mean of the other tokens], gives V(s) (PPO only) |

## 2.5 Use per algorithm

- PPO: actor and critic share the encoder (`separate_critic = False`). maskable uses N + 1 logits (EXIT included) and sets infeasible logits to −1e9 inside the network; the other strategies use the unmasked scores cut to the first N (no EXIT). The action distribution is categorical.
- DQN / DDQN: the unmasked scores are the Q-values (N + 1 for maskable, N otherwise); value_head is not used and gets no gradient. The online and the target network are each a full `GraphPolicyNet`.
- Optional parts switched off (every run): glob_std, glob3, pair_head, plan, plan_decay, separate_critic all False.
- Initialisation: PyTorch defaults (`_build` is overridden, so SB3's orthogonal initialisation does not apply); relation weights w and type embeddings start at zero.
- `PPO_NET_ARCH` (256-256) and `DQN_NET_ARCH` (128-128) in config.py are for the MLP network only and unused by these runs.

## 2.6 Parameter count (without the λ input)

| part | parameters |
|---|---|
| inp | 2,304 |
| blocks | 397,488 |
| ln | 256 |
| ecu_head | 16,641 |
| exit_head | 129 |
| value_head | 33,025 |
| type_emb | 384 |
| total | 450,227 |

Each Transformer layer has 132,496 parameters:

| part | parameters |
|---|---|
| ln1 + ln2 | 512 |
| qkv | 49,536 |
| o | 16,512 |
| ff | 65,920 |
| rel_w | 16 |
