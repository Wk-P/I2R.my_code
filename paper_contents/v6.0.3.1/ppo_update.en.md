# v6.0.3.1 PPO per-update step-size diagnostic

- Generated 2026-10-22 16:43:39; script `src/scripts/diag_ppo_update_v6.py`; took 47.9 min.
- The 5M / 10M / 15M checkpoints and the 20M final model of v6.0.0 Mask PPO (3 scenarios × seeds 1–3) each trained on for 61,440 steps (3 PPO updates) with exactly the training settings (pool, reward, hyper-parameters, optimiser state), only logging SB3's per-update statistics; nothing is saved, the original runs are untouched.
- clip_range = 0.1, 10 epochs per update. approx_kl = approximate KL between old and new policy; clip_fraction = share of samples clipped; explained_variance = return variance explained by the critic (1 perfect, 0 = constant prediction).
- Means over the three seeds (each seed first averaged over its updates).

| Scen | Training point | approx_kl | clip_fraction | entropy (−entropy_loss) | value_loss | explained_variance |
|---|---|---|---|---|---|---|
| EQ | 5M | 0.15669 | 34.546% | 0.4899 | 0.001534 | 0.8298 |
| EQ | 10M | 0.12919 | 31.536% | 0.4587 | 0.001407 | 0.8546 |
| EQ | 15M | 0.12211 | 28.513% | 0.4164 | 0.001283 | 0.8671 |
| EQ | 20M | 0.10748 | 26.585% | 0.4051 | 0.001230 | 0.8756 |
| GT | 5M | 0.13712 | 34.097% | 0.5504 | 0.001162 | 0.8377 |
| GT | 10M | 0.13281 | 29.371% | 0.4801 | 0.001061 | 0.8574 |
| GT | 15M | 0.11052 | 26.432% | 0.4441 | 0.000973 | 0.8727 |
| GT | 20M | 0.10251 | 24.767% | 0.4246 | 0.000877 | 0.8870 |
| LT | 5M | 0.08437 | 33.538% | 0.5490 | 0.006119 | 0.6725 |
| LT | 10M | 0.10062 | 31.918% | 0.5027 | 0.005547 | 0.6589 |
| LT | 15M | 0.09013 | 30.599% | 0.4805 | 0.004543 | 0.6873 |
| LT | 20M | 0.08464 | 29.071% | 0.4564 | 0.004386 | 0.7071 |
