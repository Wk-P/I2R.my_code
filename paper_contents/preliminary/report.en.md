# Preliminary results: Maskable vs Lagrangian (50k instances, 20M, Transformer)

- Compiled 2026-10-15 from the reports of finished batches; no new computation.
- Status: **preliminary**. Repair (v5.3.1, running) and unconstraint (v5.3.2, queued) have no results yet; Lagrangian has seed 1 only.

## 1. Setup

| Item | Setting |
|---|---|
| Problem | service → ECU placement with capacity and privacy-conflict constraints; scenarios LT / EQ / GT |
| Network | structure-aware network (Transformer), trained on GPU |
| Training data | pool of 50k instances, each solved to optimality by ILP (`results/sup_diag/<scen>.json`) |
| Steps | 20M |
| Reward / episode | ar_pen reward, γ = 1, full episode (a dead end counts as a violation), services in descending demand |
| Evaluation | test set: each seed's own 400 instances; deterministic replay; gap = relative gap to the ILP AR, over instances without violation / EXIT |

Sources:

| Mechanism | Batch | Seeds | Note |
|---|---|---|---|
| Maskable PPO | v4.4.15 → v4.4.16 → v4.4.17 (collected in v4.5.1) | 1–3 | 5M → 10M → 20M resumed chain |
| Maskable DQN / DDQN | v4.5.2 | 1–3 | ε 1.0→0 over the first 50% of steps (10M) |
| Lagrangian PPO / DQN / DDQN | v5.1.0 | 1 | λ init 0.001, Adam lr 0.035, λ ≥ 0 uncapped, J_C of the behaviour policy, d = 0; DQN / DDQN ε 1.0→0 over a fixed 2M steps |

## 2. Main results (20M, test set)

### 2.1 Relative gap

Maskable: mean ± sample std over 3 seeds; Lagrangian: seed 1 only.

| Mechanism | Learner | LT | EQ | GT | Mean of 3 scenarios |
|---|---|---|---|---|---|
| Maskable | PPO | 7.0% ± 0.3% | 4.9% ± 0.3% | 4.6% ± 0.2% | **5.5% ± 0.0%** |
| Maskable | DQN | 9.9% ± 0.5% | 4.1% ± 0.1% | 3.1% ± 0.4% | **5.7% ± 0.3%** |
| Maskable | DDQN | 9.9% ± 0.5% | 4.6% ± 0.3% | 3.8% ± 0.8% | **6.1% ± 0.1%** |
| Lagrangian | PPO | 6.7% | 6.0% | 7.2% | **6.6%** |
| Lagrangian | DQN | 25.9% | 8.1% | 3.4% | **12.5%** |
| Lagrangian | DDQN | 26.7% | 11.9% | 6.5% | **15.0%** |

### 2.2 Violations and failure rate

Failure = EXIT or any constraint violation. Maskable never picks an infeasible ECU; it fails only at a dead end (no feasible ECU left).

| Mechanism | Learner | LT failure | LT capacity viol. | LT privacy viol. | EQ failure | GT failure |
|---|---|---|---|---|---|---|
| Maskable | PPO | 1.8% ± 0.3% | 0 (mask) | 0 (mask) | 0.1% ± 0.1% | 0.0% |
| Maskable | DQN | 2.6% ± 0.6% | 0 (mask) | 0 (mask) | 0.1% ± 0.1% | 0.0% |
| Maskable | DDQN | 2.5% ± 0.5% | 0 (mask) | 0 (mask) | 0.0% | 0.0% |
| Lagrangian | PPO | 2.7% | 2.8% | 0.2% | 0.0% | 0.0% |
| Lagrangian | DQN | 6.8% | 2.0% | 5.2% | 0.0% | 0.0% |
| Lagrangian | DDQN | 6.0% | 3.8% | 3.8% | 0.0% | 0.0% |

### 2.3 Learning curves (test gap, mean of 3 scenarios)

| Steps | Mask PPO | Mask DQN | Mask DDQN | Lag PPO | Lag DQN | Lag DDQN |
|---|---|---|---|---|---|---|
| 5M | 7.4% | 9.4% | 9.2% | 10.0% | 20.2% | 16.4% |
| 10M | 6.3% | 7.0% | 8.7% | 8.3% | 14.8% | 19.3% |
| 15M | 5.9% | 7.2% | 7.3% | 7.6% | 13.7% | 17.0% |
| 20M | 5.5% | 5.7% | 6.1% | 6.6% | 12.5% | 15.0% |

## 3. Preliminary findings

1. **Maskable is the best mechanism for all three learners.** Mean gap 5.5% (PPO) / 5.7% (DQN) / 6.1% (DDQN), seed std ≤ 0.3 pp, zero constraint violations.
2. **Lagrangian PPO comes close to Maskable but still violates on LT.** Mean gap 6.6%, about 1.1 pp above Maskable PPO; no violations on EQ / GT; on LT 2.8% capacity and 0.2% privacy violations.
3. **Lagrangian DQN / DDQN break down on LT.** LT gap about 26% with 2–5% violations; EQ / GT are normal, and DQN on GT reaches 3.4%, the lowest gap of any method there.
4. **LT is the hardest scenario.** Every method has its highest gap and failure rate on LT; the Lagrangian λ on LT is still rising at 20M (PPO 2.97, DQN 5.60, DDQN 5.31) and J_C has not reached 0.
5. **Maskable curves still fall at 20M** (15M → 20M: 0.4–1.5 pp), so 20M is not yet the ceiling.

## 4. Limitations and pending work

- **Lagrangian has seed 1 only**: no variance, no statistical comparison yet.
- **Settings are not fully uniform yet**: Maskable DQN / DDQN decay ε over the first 10M steps, the others over a fixed 2M; Maskable PPO is a resumed chain, the others one 20M run.
- **Repair and unconstraint have no 20M results yet** (v5.3.1 running, v5.3.2 queued, seed 1).
- The gap is over passed instances only; where the failure rate is high (Lagrangian DQN / DDQN on LT) it reflects the easier instances.
- Only relative gaps are listed; absolute AR and ILP AR are not in these batch reports and will be added in the final version.
- Planned final comparison: 4 mechanisms × 3 learners × 3 scenarios × 3 seeds, all with 50k instances, 20M, and the same ε setting.

## 5. Sources

- Maskable PPO: `paper_contents/v4.5.1/report.md` (Transformer column)
- Maskable DQN / DDQN: `paper_contents/v4.5.2/report.md` (Transformer column)
- Lagrangian: `paper_contents/v5.1.0/report.md`, `paper_contents/v5.1.0/report_data.json`
