# Data-size diagnostic: is the RL gap due to the training signal or to too few training instances

- Generated 2026-10-10 06:10:59; script `src/scripts/diag_datasize.py`.
- Supervised reference: the same structure-aware network imitating ILP-optimal actions (setting of `diag_arch.py`: 25k steps × batch 256, AdamW, lr 5e-4, cosine schedule); only the number of training instances changes. 1600 = exactly the 1600 training instances used by RL seed 1 (same instances, different training signal); 5000 / 10k / 50k = fresh instances from the same generator. Test = the 400 test instances of RL seed 1, deterministic rollout; relative gap over non-EXIT instances.
- RL: the v4.4.5 three-seed GPU baseline (1M steps), each rolled out deterministically on its own 1600 training instances and 400 test instances.

## 1. Supervised reference: number of training instances vs test gap

| Scenario | Training instances | Source | Train accuracy | Test accuracy | Test relative gap | Test EXIT rate | Train-set relative gap |
|---|---|---|---|---|---|---|---|
| LT | 1600 | RL seed-1 training instances | 100.0% | 72.2% | 10.1% | 6.5% | 0.0% |
| LT | 5000 | fresh instances | 90.6% | 77.2% | 7.3% | 4.3% | 5.3% |
| LT | 10000 | fresh instances | 83.8% | 81.6% | 5.7% | 2.3% | — |
| LT | 50000 | fresh instances | 82.3% | 81.7% | 6.1% | 2.7% | — |
| EQ | 1600 | RL seed-1 training instances | 100.0% | 77.0% | 7.8% | 0.0% | 0.0% |
| EQ | 5000 | fresh instances | 99.8% | 78.8% | 6.1% | 0.0% | 0.1% |
| EQ | 10000 | fresh instances | 90.5% | 82.5% | 4.4% | 0.0% | — |
| EQ | 50000 | fresh instances | 85.8% | 85.5% | 3.5% | 0.0% | — |
| GT | 1600 | RL seed-1 training instances | 100.0% | 77.4% | 7.2% | 0.0% | -0.0% |
| GT | 5000 | fresh instances | 96.2% | 80.2% | 6.0% | 0.0% | 1.8% |
| GT | 10000 | fresh instances | 89.3% | 84.4% | 4.3% | 0.0% | — |
| GT | 50000 | fresh instances | 86.7% | 86.4% | 3.7% | 0.0% | — |

## 2. RL: gap on the training set vs the test set

| Scenario | Seed | Train-set relative gap (1600) | Test-set relative gap (400) | Train EXIT | Test EXIT |
|---|---|---|---|---|---|
| LT | 1 | 11.2% | 11.5% | 2.7% | 5.2% |
| LT | 2 | 10.9% | 11.1% | 2.7% | 3.0% |
| LT | 3 | 11.1% | 11.9% | 3.0% | 2.2% |
| LT | mean | 11.1% | 11.5% | 2.8% | 3.5% |
| EQ | 1 | 7.9% | 10.4% | 0.0% | 0.0% |
| EQ | 2 | 7.6% | 9.7% | 0.0% | 0.2% |
| EQ | 3 | 8.0% | 10.4% | 0.1% | 0.0% |
| EQ | mean | 7.8% | 10.2% | 0.0% | 0.1% |
| GT | 1 | 7.1% | 9.3% | 0.0% | 0.0% |
| GT | 2 | 7.2% | 8.9% | 0.0% | 0.0% |
| GT | 3 | 6.9% | 8.0% | 0.0% | 0.0% |
| GT | mean | 7.0% | 8.7% | 0.0% | 0.0% |
