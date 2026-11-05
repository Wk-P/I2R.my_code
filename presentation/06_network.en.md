# 6. The network: a Transformer with relation hints

Code: `GraphPolicyNet` in `src/paper_rl/graph_net.py`, used by all 12 variants.

## Why not a plain MLP

An MLP sees one long list of numbers and must guess how "ECU 3's free space" relates to "service 7's demand".
Our problem is naturally "a set of ECUs + a set of services + relations between them", like a graph.
So we borrow the Transformer: **every ECU and every service is a "word" (token), and they look at each other**.

## Step 1: split into tokens

For EQ (10 ECUs, 10 services) there are 21 tokens:

- **1 global token**, 4 numbers: AR so far, share of services not yet placed, unplaced demand / total capacity, free capacity / total capacity (Lagrangian adds λ).
- **N ECU tokens**, 6 numbers each: capacity, free space, in use, **fits the current service**, number hosted, load ratio.
- **M service tokens**, 5 numbers each: demand, placed, **is the current one**, number of ECUs that fit it, conflicts with unplaced services.

The network computes these itself from the raw observation (`build()`); each token then becomes a 128-number vector.

## Step 2: look at each other (attention), 3 rounds

Each token looks at all others and updates itself. For example:

- ECU 3 sees "the current service conflicts with one of mine" → it should not be chosen.
- Service 5 sees "the ECUs that fit me are nearly full" → that reaches the relevant ECUs.

**Relation hints**: we tell attention directly which tokens are specially related:

1. Two services conflict
2. A service sits on an ECU
3. A service conflicts with a service already on an ECU
4. A service fits on an ECU

Each relation has learnable weights (starting at 0; they grow as important as training finds them).

Each round uses 4 "heads" in parallel (each can focus on something different); afterwards each token is processed on its own (a small feed-forward net). 3 rounds in total.

## Step 3: scoring

- Each ECU token → the same small network → **one score**.
- Global token → **EXIT score** (Mask only).
- Global token + mean of all tokens → **V(s)** (PPO only).

## Step 4: using the scores

- PPO: scores → softmax → probabilities.
- DQN/DDQN: scores are Q-values.
- Mask: illegal ECUs get −∞.

## Picture

```
raw observation
   │  build(): ECU / service features + 4 relations
   ▼
[global] [ECU1 … ECUn] [svc1 … svcm]     ← each becomes 128-d
   │
   ├─ attention (+relation hints) → feed-forward  ┐
   ├─ attention (+relation hints) → feed-forward  ├ 3 rounds
   ├─ attention (+relation hints) → feed-forward  ┘
   ▼
ECU tokens → one score per ECU    global → EXIT score    global + mean → V(s)
```

## Numbers

| Term | Value | Meaning |
|---|---|---|
| d | 128 | Each token is 128 numbers |
| Heads | 4 (32 dims each) | 4 groups look in parallel |
| Layers | 3 | 3 rounds of looking |
| Feed-forward | 128→256→128, GELU | Per-token processing after looking |
| Pre-LN | | Normalize before each round; steadier training |
| Positional encoding | None | Reorder ECUs and the outputs reorder too; IDs don't matter |
| Parameters | 450,227 (Lagrangian 450,355) | About 450k learnable numbers |

**One network fits every scenario size**: the token count follows N and M; the parameters do not change.
