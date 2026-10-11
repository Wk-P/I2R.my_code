# Per-instance time and gap: structure-aware network vs ILP

- Generated 2026-10-11 14:04:22; script `src/scripts/bench_infer.py`; took 48.8 min.
- Models: v4.4.16 structure-aware Mask PPO (50k instances, 10M) (`logs/v4.4.16/manifest.json`, key `mask_ppo_pool10m`), seeds 1, 2, 3; each model is timed on the 400 test instances of its own training seed and the ILP on the same instances, so time and gap refer to the same instances.
- Network: one CPU thread (`torch.set_num_threads(1)`), deterministic policy; "full episode" = reset + action mask + forward pass + environment step at every step; "forward only" = the forward passes (SB3 predict overhead included); 5 warm-up episodes.
- ILP: `solve_ilp_max_ar` (Dinkelbach, PuLP + CBC single thread), model building and all iterations included; 12 instances solved in parallel, each timed on its own.
- GPU training processes were running at the same time (about one CPU core each); the 56 cores were not saturated but the machine was not idle. Speed-up = ILP time / network time (mean / median).

| Scenario | Method | Relative gap | Mean ms | Median ms | P95 ms | Max ms | Speed-up (mean) | Speed-up (median) |
|---|---|---|---|---|---|---|---|---|
| LT | ILP (Dinkelbach, mean 2.92 iterations) | 0 | 3210.4 | 1425.4 | 10919.2 | 137335.2 | 1× | 1× |
| LT | network: full episode | 7.6% ± 0.3% | 73.2 | 72.2 | 82.4 | 113.5 | 44× | 20× |
| LT | network: forward only | 7.6% ± 0.3% | 69.9 | 69.0 | 78.8 | 108.3 | 46× | 21× |
| EQ | ILP (Dinkelbach, mean 2.98 iterations) | 0 | 456.2 | 274.1 | 1592.8 | 6262.6 | 1× | 1× |
| EQ | network: full episode | 5.9% ± 0.5% | 48.9 | 46.9 | 59.8 | 212.1 | 9× | 6× |
| EQ | network: forward only | 5.9% ± 0.5% | 46.9 | 44.9 | 57.6 | 209.7 | 10× | 6× |
| GT | ILP (Dinkelbach, mean 3.20 iterations) | 0 | 679.8 | 408.3 | 2082.0 | 6311.4 | 1× | 1× |
| GT | network: full episode | 5.4% ± 0.3% | 52.5 | 51.3 | 65.0 | 78.9 | 13× | 8× |
| GT | network: forward only | 5.4% ± 0.3% | 50.4 | 49.2 | 62.4 | 75.9 | 13× | 8× |
