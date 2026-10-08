# v4.4.3 Structure-aware policy network (pure RL)

Same as v4.3.8 except **the policy / value network** (data v4.3.1.4 p = 0.6, descending demand, ECU action, reward ar_pen, γ = 1, full episode, Mask with EXIT, 40 envs × 512 steps, batch 256, 10 epochs, clip 0.1, entropy 0.005, lr 3e-4 all unchanged).

- **Network** (`src/paper_rl/graph_net.py`): one token for the global state, every ECU and every service; four relations (service–service conflict, service hosted on ECU, service conflicts with an ECU's services, service fits the ECU's free capacity) enter the attention as learned biases; 3 layers, d = 128, 4 heads, ~450k parameters.
- **Actor / critic** (`src/paper_rl/graph_policy.py`): shared encoder; every ECU scored by the same head (equivariant to ECU permutations), EXIT scored from the global token; V(s) from the global token plus the token mean (permutation invariant). Distribution, masking and PPO update are MaskablePPO's own.
- **Observation** (`obs = raw`, `src/paper_rl/env.py::_raw_obs`): capacities, demands (descending), conflict graph, ECU of every placed service, current step — the same information as the base observation, features computed inside the network; mask / AR checked against the environment on 2053 states.
- **Motivation**: `arch_diag.md` — the same network trained on ILP actions reaches 3.5–6.1% relative gap vs 13–17% for the MLP (13–15% for a large MLP).
- **Report**: MLP Mask PPO (v4.3.8 pilot), structure-aware Mask PPO (this version) and the oracle-supervised reference (same network imitating ILP actions; a representational reference, neither RL nor an upper bound) side by side: relative gap, EXIT, active ECUs, forced openings, step-1 and overall optimal-action rates (regret diagnostic), per-step optimal rate and regret share.

Run: `src/scripts/run_v4.4.3.py --pilot` (3 scenarios × seed 1 × 1M steps), report `pilot_report.md`. Training option `src/paper_rl/train.py --net graph` (implies `--obs raw`).
