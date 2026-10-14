# Check: v5.0.3.6 EQ — "greedy J_C ≈ 0, yet 14.7% of test instances fail at 5M"

- Date: 2026-10-14. Subject: v5.0.3.6 (commit 46caadd), EQ PPO (5d6f2706) and DQN (37fc5532); GT DQN (15003b19) and LT PPO (cb432d64) for comparison.
- Read-only: existing checkpoints / logs, deterministic replay on CPU; script `paper_contents/v5.0.3.6/check_v5.0.3.6.py` (args: scenario algo_dir exp_id learner).
- No new training, no version number.

## 1. Problem

On EQ, PPO's λ reached 0 before 4M; the last λ update (5.01M steps) saw a greedy J_C of 0.00 (100 episodes), yet the 5M report shows a 14.7% test failure rate (13.0% on the training set). DQN on EQ likewise: the last greedy J_C values were 0–0.07, the 5M test failure rate 24.0%.

## 2. Not an evaluation error: the final model violates on every data set and λ observation

Each model is evaluated at the λ it was trained at (ckpt_lambda.json / final_lambda). "Pool random 500" samples like the in-training greedy J_C (random draws from the 50k pool).

| Model | λ | Pool random 500: J_C / violating eps | Train 1600: J_C / violating eps | Test 400: J_C / violating eps |
|---|---|---|---|---|
| PPO EQ 3M | 0.641 | 0.002 / 0.2% | 0.003 / 0.3% | 0.000 / 0.0% |
| PPO EQ 4M | 0.000 | 0.006 / 0.2% | 0.008 / 0.4% | 0.000 / 0.0% |
| PPO EQ final | 0.000 | 0.200 / 14.4% | 0.167 / 13.0% | 0.188 / 14.7% |
| DQN EQ 3M | 0.000 | 0.002 / 0.2% | 0.001 / 0.1% | 0.003 / 0.3% |
| DQN EQ 4M | 0.066 | 0.024 / 2.4% | 0.030 / 2.6% | 0.045 / 4.5% |
| DQN EQ final | 0.101 | 0.186 / 18.6% | 0.214 / 21.0% | 0.240 / 24.0% |

- With the observed λ set to 0.5 or 2.0, the final PPO model still violates on 14.5–15.0% of test instances: the violations are not caused by a λ-observation mismatch.
- The in-memory evaluation on GPU at the end of training (results.json): PPO privacy violations 13.5%, DQN 24.0% — identical to the CPU evaluation of the saved file: saving / loading / device are fine.
- Conclusion: **the evaluation is correct; the final model itself violates. The violations appear after 4M, within a very short final stretch of training.**

## 3. Violations appear abruptly at the end

Share of violating behaviour-policy episodes per 10k steps (after 4.9M), from training_curve.csv:

| Run | 4.90M | 4.93M | 4.94M | 4.96M | 4.98M | 4.99M | 5.00M | 5.01M |
|---|---|---|---|---|---|---|---|---|
| PPO EQ (sampled) | 0.1% | 0.1% | 0.5% | 0.1% | 0.1% | 0.0% | 0.3% | 0.5% |
| DQN EQ (ε = 0) | 1.8% | 0.3% | 2.0% | 3.4% | 1.0% | 7.7% | — | — |
| DQN GT (ε = 0) | 0.2% | 4.9% | 16.7% | 14.1% | 14.3% | 3.4% | — | — |
| PPO LT (sampled) | 8.7% | 11.8% | 12.0% | 7.6% | 7.2% | 5.9% | 5.5% | 6.0% |

- **DQN**: ε is 0 after 2M, so the behaviour policy is the greedy policy. Its violation rate jumps from 0.2% to 16.7% and back within 10k steps: the policy itself oscillates quickly, and the final model happens to sit on a rising phase on EQ (7.7% in the last 10k steps, 18.6% for the final model).
- **PPO**: the last greedy J_C (5.01M, the policy before the update) was 0, but PPO performs one more gradient update after its last rollout (up to 5,017,600 steps), and the saved model is the post-update policy. The sampled behaviour policy stays at ~0.3%, yet the deterministic policy violates on 14% after this update. **Inference**: with λ = 0 the reward carries no violation penalty, so the logits of violating and legal actions come close to a tie (the violating action often has a higher AR), and a single update can flip the argmax on many instances. This is consistent with the data, but there is no model from just before the update to compare against directly.

## 4. Cause

1. λ reaching 0 is not an equilibrium but the penalty disappearing: on EQ/GT the greedy J_C is mostly 0 (resolution 0.01 with 100 episodes), so the normalized update subtracts η every time and λ quickly falls to 0.
2. Once λ = 0, nothing pushes the policy away from violations; they are held back only by previously learned preferences, which drift to a tie with the legal actions.
3. When violations return, λ climbs back from 0 at +0.025 per 1000 episodes — much slower than the policy degrades (DQN degrades within 10k steps).
4. Evaluating one snapshot at the end of training makes the result depend on where in the oscillation the end falls: 0% for PPO GT, 14.7% for PPO EQ.

## 5. Implications for 20M

- At 20M, EQ/GT spend longer at λ ≈ 0, so such end-of-training flips may be more frequent; a single 20M snapshot's violation rate is unreliable.
- This is not an evaluation bug and the evaluation needs no fix; what needs handling is the loss of the constraint at λ ≈ 0 and the randomness of the final snapshot.
- Directions that address it directly (for the user to decide):
  - a lower bound on λ (e.g. λ_min = 0.1–0.5; on EQ/GT λ already oscillates within 0–0.3), so a violation always costs something;
  - more greedy J_C episodes (100 → 1000): resolution 0.001, the same order as d = 0.001, avoiding "J_C = 0 → λ only falls";
  - a slower decrease of λ (a smaller η for decreases, an asymmetric update).
