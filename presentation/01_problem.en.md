# 1. The problem: placing services on ECUs

## An analogy

N boxes (**ECUs**) of different sizes (capacity c_n); M items (**services**) of different sizes (demand e_m).
Every item must go into a box, under two rules:

1. **Capacity**: the items in a box must not exceed its size.
2. **Privacy**: some items conflict (they share a privacy conflict set); two conflicting items cannot share a box.

## What "packed well" means

We use **AR (average resource utilization)**: over the boxes actually used, how full each one is, averaged.

> Example: 2 boxes used, one 90% full, one 50% full → AR = (0.9 + 0.5) / 2 = 0.7.

Higher is better: **use few boxes and fill each one**, no half-empty boxes.

Formally: AR = Σ(e_m / c_n over legal placements) ÷ number of ECUs used. Violating placements score nothing.

## Three scenarios

| Scenario | ECUs N | Services M | Character |
|---|---|---|---|
| LT | 10 | 15 | Many items, few boxes: tightest, hardest |
| EQ | 10 | 10 | Balanced |
| GT | 15 | 10 | Many boxes, few items: loose |

## Data

- Capacity c_n uniform in {50, 55, …, 195}; demand e_m uniform in {10, 15, …, 95}.
- K = 10 conflict sets; any two services conflict with probability p = 0.6 (many conflicts).
- **Every instance is kept only after ILP solves it to optimality**, so each one is solvable and its best score is known.
- Training: 50,000 instances per scenario; testing: a separate 5,000 (no overlap).

## What we want to show

ILP finds the optimum but is slow (file 2). We want RL to place services "at a glance":
**close to ILP, much faster, and never violating constraints** (privacy is a hard constraint).
Closeness is **gap = 1 − mean RL AR / mean ILP AR**; smaller is better.
