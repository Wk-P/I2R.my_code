# Formula reference: ILP and RL algorithms

Every formula used for the paper results, each checked line by line against the code.

| Part | Code version |
|---|---|
| Mask / Repair / Lagrangian / Unconstraint models of the main tables (v6.0.0-merge) | v6.0.0, commit cbf08e9, `src/scripts/run_campaign.py` |
| Standard Lagrangian (new definition, not trained yet) | branch `v6.0.1-lagrangian`, commit c2cc5e6 |
| ILP, instance generation, test set | `src/shared/ilp_utils.py` (`solve_ilp_max_ar`), `src/paper_rl/data.py`, `src/scripts/diag_supervised.py`, `src/scripts/gen_testset_v6_0_4.py` |
| Evaluation metrics | `src/scripts/eval_testset_v6_0_4.py` (`summarize`) |

Indices start at 1 in the formulas and at 0 in the code.

---

## 1. Problem and notation

| Symbol | Meaning |
|---|---|
| N, M | number of ECUs, services (LT: N = 10, M = 15; EQ: N = 10, M = 10; GT: N = 15, M = 10) |
| n ∈ {1, …, N} | ECU index |
| m ∈ {1, …, M} | service index |
| c_n | capacity of ECU n (containers), in {50, 55, …, 195} |
| e_m | demand of service m (containers), in {10, 15, …, 95} |
| 𝒞_k, k = 1, …, K | privacy conflict sets (K = 10); two services of the same set may not share an ECU |
| x_mn ∈ {0, 1} | service m is placed on ECU n |
| y_n ∈ {0, 1} | ECU n is active |

**AR (average resource utilisation)**: the mean utilisation of the active ECUs.

$$
\text{AR}(x, y) = \frac{F(x)}{G(y)},\qquad F(x) = \sum_{m=1}^{M}\sum_{n=1}^{N} x_{mn}\,\frac{e_m}{c_n},\qquad G(y) = \sum_{n=1}^{N} y_n
$$

**Constraints**:

$$
\begin{aligned}
&\textstyle\sum_{n} x_{mn} = 1 && \forall m &&\text{(each service placed exactly once)}\\
&\textstyle\sum_{m} x_{mn}\, e_m \le c_n\, y_n && \forall n &&\text{(capacity; an inactive ECU hosts nothing)}\\
&x_{mn} \le y_n && \forall m, n &&\text{(activation)}\\
&\textstyle\sum_{m \in \mathcal{C}_k} x_{mn} \le 1 && \forall k, \forall n &&\text{(privacy conflict)}\\
&x_{mn},\, y_n \in \{0, 1\}
\end{aligned}
$$

The optimum is AR\* (ILP AR):

$$
\text{AR}^* = \max_{(x, y)\ \text{feasible}}\ \frac{F(x)}{G(y)}
$$

---

## 2. ILP: Dinkelbach's method

The objective AR = F/G is a ratio, so it cannot be given to an ILP solver directly. Dinkelbach's method keeps a variable ρ, "the best AR found so far", and in each iteration solves a parametric ILP with a linear objective (constraints as in Section 1):

$$
\varphi(\rho) = \max_{(x, y)\ \text{feasible}}\ \big[\, F(x) - \rho\, G(y) \,\big]
$$

**What each iteration asks**: since G(y) ≥ 1,

$$
F(x) - \rho\, G(y) > 0 \iff \frac{F(x)}{G(y)} > \rho \iff \text{the AR of } (x, y) > \rho
$$

So φ(ρ) > 0 means some placement has an AR above ρ, and φ(ρ) = 0 means none has.

**Iteration**:

1. ρ₀ = 0.
2. Solve φ(ρ_t) with CBC (PuLP); the status must be `Optimal`. This gives the optimal solution (x_t, y_t).
3. ρ_{t+1} = AR(x_t, y_t) = F(x_t) / G(y_t), with G = the number of ECUs that actually host a service.
4. Stop when |ρ_{t+1} − ρ_t| < 10⁻⁶ and output AR\* = ρ_{t+1}; otherwise go back to step 2 (at most 30 iterations).

**Convergence**:

$$
\varphi(\rho) > 0 \iff \rho < \text{AR}^*,\qquad \varphi(\rho) = 0 \iff \rho = \text{AR}^*
$$

F(x_t) − ρ_{t+1} G(y_t) = 0 holds only for the previous solution; φ is the maximum over all feasible solutions, so it is positive as long as some placement has a higher AR. It is 0 only when ρ is already the largest AR. So φ = 0 at convergence is the stopping criterion, and the reported optimal AR is ρ, not φ. Every iterate is feasible, so ρ_t never decreases; there are finitely many feasible solutions, so the iteration stops after finitely many steps.

**Illustration** (suppose there are only 3 feasible placements, with AR 0.70, 0.80 and 0.90):

| Iteration | ρ | φ(ρ) | placement returned by the ILP (one of) | new ρ |
|---|---|---|---|---|
| 1 | 0 | > 0 | AR 0.70 | 0.70 |
| 2 | 0.70 | > 0 (0.80 and 0.90 exceed 0.70) | AR 0.90 | 0.90 |
| 3 | 0.90 | 0 (no placement exceeds 0.90) | — | stop, AR\* = 0.90 |

**Example** (EQ validation set, instance 1): φ(0) = 5.3754 (7 ECUs, AR = 0.7679); φ(0.7679) = 0.0000, stop, AR\* = 0.7679.

Note: ρ is used only by the ILP; the λ of the Lagrangian in Section 6.4 is a different quantity (the dual variable of the violation penalty).

---

## 3. Data

- **Generation**: c_n and e_m are drawn independently and uniformly from the values above. There are K = 10 conflict sets; each service joins each set independently with probability q, and sets with fewer than 2 members are dropped.

$$
q = \sqrt{1 - (1 - p)^{1/K}},\qquad p = 1 - (1 - q^2)^K = 0.6
$$

  p is the probability that two services share a conflict set, i.e. conflict with each other.

- **Filtering**: every instance is solved with the method of Section 2 and redrawn if the status is not `Optimal` (infeasible). So every instance is feasible and has a known AR\*.
- **Training**: `results/sup_diag/<scenario>.json`, 50,000 instances per scenario; every episode draws one uniformly at random.
- **Test**: `data/testset_v6.0.4/<scenario>.json`, 5000 instances per scenario, shared by every seed and model. Same generator as the training pool, disjoint seeds, de-duplicated by content.

---

## 4. MDP (shared by the four mechanisms)

### 4.1 Episode

The services are sorted by descending demand (stable for ties) and relabelled so that e_1 ≥ e_2 ≥ … ≥ e_M. Step t (t = 1, …, M) places service t.

Full episode: apart from Mask's EXIT there is no early termination, so every episode has M steps.

### 4.2 Action

$$
a_t \in \{1, \dots, N\}\ \ (\text{Unconstraint / Repair / Lagrangian}),\qquad a_t \in \{1, \dots, N, \text{EXIT}\}\ \ (\text{Mask})
$$

### 4.3 Violations and transition

Let n = ã_t be the ECU actually executed at step t (the repaired ECU for Repair, a_t otherwise). Let H_n be the set of services already on ECU n before step t, and r_n = c_n − Σ_{m∈H_n} e_m its remaining capacity. H_n includes violating placements, so r_n can be negative.

$$
v^{\text{cap}}_t = \mathbb{1}\big[r_n < e_t\big],\qquad
v^{\text{priv}}_t = \mathbb{1}\big[\exists\, m \in H_n,\ \exists\, k:\ \{t, m\} \subseteq \mathcal{C}_k\big]
$$

$$
c_t = v^{\text{cap}}_t + v^{\text{priv}}_t \in \{0, 1, 2\}
$$

Note: c_t is the number of violations at step t; it is a different quantity from the ECU capacity c_n.

After execution: H_n ← H_n ∪ {t}, r_n ← r_n − e_t. Violating placements are executed as well.

Service t is **feasible** on ECU n if and only if r_n ≥ e_t and there is no privacy conflict. The feasible set of the current service is

$$
\mathcal{A}_t = \{\, n : r_n \ge e_t,\ \text{service } t \text{ has no conflict with } H_n \,\}
$$

### 4.4 AR and number of legal placements

Only legal placements (steps with c_t = 0) are counted:

$$
M_v = \sum_{t} \mathbb{1}[c_t = 0],\qquad
\mathcal{E}_{\text{legal}} = \{\, \tilde a_t : c_t = 0 \,\},\qquad
\text{AR} = \frac{\sum_{t:\, c_t = 0} e_t / c_{\tilde a_t}}{|\mathcal{E}_{\text{legal}}|}
$$

(AR = 0 if 𝓔_legal is empty.) A violating placement counts in neither the numerator nor the denominator, and neither do the services left unplaced after an EXIT. On a violation-free completion (M_v = M), AR equals AR(x, y) of that placement and cannot exceed AR\*.

### 4.5 State

The environment returns the raw state: c (N entries), e (M entries, in placement order), the conflict graph A ∈ {0,1}^{M×M} (A_{mk} = 1 if and only if m ≠ k and they share some 𝒞_k), the ECU of every service (0 = not placed), and the current step t. Lagrangian appends λ/(1 + λ) (λ uncapped).

The network builds its features from the raw state (`graph_net.build`). Let load_n be the total demand placed on ECU n (violating placements included), free_n = c_n − load_n, and c_max = max_n c_n:

| Token | Features |
|---|---|
| global (1) | AR_net; (M − t)/M; demand of the unplaced services / Σ_n c_n; Σ_n max(free_n, 0) / Σ_n c_n; [Lagrangian: λ/(1 + λ)] |
| ECU n (N) | c_n / c_max; max(free_n, 0) / c_max; active; feasible for the current service; number of hosted services / M; load_n / c_n |
| service k (M) | e_k / c_max; placed; is the current service; number of feasible ECUs / N; conflicts with other unplaced services / M (0 once placed) |

AR_net = Σ_{active n} (load_n / c_n) / (number of active ECUs). It includes violating placements and is only an input feature of the network, not the AR of Section 4.4.

Four relation matrices enter the attention as biases: conflict between two unplaced services; service hosted on an ECU; unplaced service conflicting with a service already on an ECU; unplaced service whose demand is ≤ the ECU's free capacity. The network input is the same for the four mechanisms (Unconstraint also sees the feasibility features); Lagrangian only adds λ.

---

## 5. Reward (ar_pen, γ = 1)

Terminal reward:

$$
R_{\text{term}} =
\begin{cases}
\text{AR}, & M_v = M\ \ (\text{violation-free completion; AR} \in (0, 1])\\[2pt]
-\left(1 - \dfrac{M_v}{M}\right), & \text{otherwise (EXIT included; in } [-1, 0)\text{)}
\end{cases}
$$

Per-step reward (T = the last step of the episode: T = M, or the EXIT step):

$$
r_t = \begin{cases} 0, & t < T \\ R_{\text{term}}, & t = T \end{cases}
\qquad\qquad
G = \sum_{t=1}^{T} r_t = R_{\text{term}}
$$

Lagrangian additionally subtracts λ c_t at every step (Section 6.4).

---

## 6. The four constraint mechanisms

### 6.1 Unconstraint

Actions are executed as chosen, with no constraint handling. The reward is ar_pen of Section 5, so violations are penalised only indirectly, through the failure branch of the terminal reward.

### 6.2 Mask (with EXIT)

Action mask:

$$
\text{mask}_t(n) = \mathbb{1}[n \in \mathcal{A}_t],\qquad \text{mask}_t(\text{EXIT}) = \mathbb{1}[\mathcal{A}_t = \varnothing]
$$

- If some ECU is feasible, only feasible ECUs can be chosen and EXIT is masked.
- If no ECU is feasible, EXIT is the only valid action. The episode ends at once with reward −(1 − M_v/M); here M_v = t − 1.
- So Mask never executes a violating placement: c_t ≡ 0, and failures come only from EXIT.

PPO: the network sets the logits of invalid actions to −10⁹, and MaskablePPO samples with the same mask. DQN / DDQN: exploration, the greedy action and the TD target are all restricted to the mask (Section 8.3).

### 6.3 Repair

$$
\tilde a_t =
\begin{cases}
a_t, & a_t \in \mathcal{A}_t\\[2pt]
\arg\max_{n \in \mathcal{A}_t} \dfrac{e_t}{c_n}, & a_t \notin \mathcal{A}_t,\ \mathcal{A}_t \ne \varnothing \quad(\text{best fit: the smallest feasible ECU; ties to the lowest index})\\[2pt]
a_t, & \mathcal{A}_t = \varnothing \quad(\text{nothing to repair to: executed as chosen, the violation is recorded})
\end{cases}
$$

The reward is ar_pen. The learner stores and trains on the agent's action a_t, not on the repaired ã_t.

### 6.4 Lagrangian

#### (a) v6.0.0: the version in the main tables, in fact "ar_pen + adaptive violation penalty"

$$
r_t = -\lambda c_t\ \ (t < M),\qquad r_M = R_{\text{term}} - \lambda c_M,\qquad G = R_{\text{term}} - \lambda \sum_{t=1}^{M} c_t
$$

λ is fixed within an episode and takes its latest value at each reset. Dual update (Adam, lr = 0.035, λ₀ = 0.001, d = 0, λ uncapped):

$$
\mathcal{L}(\lambda) = -\lambda\,(J_C - d),\qquad \lambda \leftarrow \text{Adam}\big(\nabla_\lambda \mathcal{L}\big),\qquad \lambda \leftarrow \max(\lambda, 0)
$$

$$
J_C = \frac{1}{|\mathcal{B}|} \sum_{\text{episode} \in \mathcal{B}} \sum_{t} c_t\quad(\text{mean violations per episode of the behaviour policy})
$$

- PPO: one update at the end of every rollout (40 × 512 = 20480 steps); 𝓑 = the episodes finished in that rollout.
- DQN / DDQN: one update every 1000 episodes; 𝓑 = those 1000 episodes (non-overlapping).

Problems: R_term already penalises violations once and λc_t penalises them again; and with d = 0 and J_C ≥ 0 the gradient is always ≤ 0, so λ can only increase (0 decreases in 201 logged λ values). So this is not a standard Lagrangian; the paper calls it "ar_pen + adaptive violation penalty".

#### (b) v6.0.1-lagrangian: standard Lagrangian (new definition, not trained yet)

Constrained MDP:

$$
\max_\pi\ J_R(\pi) = \mathbb{E}_\pi\big[\text{AR}_{\text{obj}}\big]\qquad \text{s.t.}\qquad J_C(\pi) = \mathbb{E}_\pi\Big[\sum_{t=1}^{M} c_t\Big] \le d,\quad d = 0.001
$$

The unconstrained objective AR_obj includes violating placements, with each ECU's utilisation clipped to 1 (𝓔_exec = the ECUs that received a service):

$$
\text{AR}_{\text{obj}} = \frac{1}{|\mathcal{E}_{\text{exec}}|} \sum_{n \in \mathcal{E}_{\text{exec}}} \min\!\Big(\frac{\sum_{t:\, \tilde a_t = n} e_t}{c_n},\ 1\Big)
$$

$$
r_t = -\lambda c_t\ \ (t < M),\qquad r_M = \text{AR}_{\text{obj}} - \lambda c_M,\qquad G = \text{AR}_{\text{obj}} - \lambda \sum_{t} c_t
$$

G is exactly the π-dependent part of the Lagrangian L(π, λ) = J_R − λ(J_C − d). The dual update is as in (a) except d = 0.001: λ increases when J_C > d and decreases when J_C < d. Flags: `--mechanism lagrange --reward ar_obj --lambda-target 0.001`.

---

## 7. Network (structure-aware attention network, `graph_net.GraphPolicyNet`)

L = 1 + N + M tokens (global, ECUs, services), width d = 128, 4 heads, 3 layers.

**Input**: h⁰ = Linear_τ(f) + E_τ. τ ∈ {global, ECU, service} is the token type; each type has its own linear layer Linear_τ and type embedding E_τ (E initialised to 0).

**Each layer** (pre-LN):

$$
\alpha^{(h)} = \text{softmax}\!\Big( \frac{Q^{(h)} K^{(h)\top}}{\sqrt{d/4}} + \sum_{r=1}^{4} w^{(\ell)}_{r,h}\, R_r \Big),\qquad
x \leftarrow x + W_o\,[\alpha^{(h)} V^{(h)}]_h,\qquad
x \leftarrow x + \text{FF}(\text{LN}(x))
$$

Q, K, V are linear maps of LN(x); R_r are the 4 relation matrices of Section 4.5; w is a learned scalar per layer, head and relation (initialised to 0); FF is d → 2d → d with GELU. A final LN follows the last layer.

**Heads**:

$$
s_n = \text{MLP}_{\text{ecu}}(h_n)\ (d \to d \to 1,\ \text{shared by all ECUs}),\qquad
s_{\text{EXIT}} = w^\top h_{\text{glob}} + b,\qquad
V(s) = \text{MLP}_v\big([\,h_{\text{glob}},\ \tfrac{1}{N+M}\textstyle\sum_{\text{non-global tokens}} h\,]\big)
$$

- PPO: π(a | s) = softmax(s_a). Under Mask the scores of invalid actions are set to −10⁹; the other mechanisms take the N ECU scores without a mask.
- DQN / DDQN: Q(s, n) = s_n, Q(s, EXIT) = s_EXIT (EXIT for Mask only); the value head is not used.
- Permuting the ECUs (or the services) permutes the outputs accordingly, i.e. the network is permutation-equivariant.

---

## 8. Learning algorithms

All networks are trained with Adam at a constant learning rate; γ = 1; 2 × 10⁷ steps per run.

### 8.1 PPO (SB3 PPO for Unconstraint / Repair / Lagrangian; sb3-contrib MaskablePPO for Mask)

Advantages by GAE (γ = 1, λ_GAE = 0.95):

$$
\delta_t = r_t + \gamma (1 - d_t)\, V(s_{t+1}) - V(s_t),\qquad
\hat A_t = \sum_{l \ge 0} (\gamma \lambda_{\text{GAE}})^l\, \delta_{t+l},\qquad
\hat R_t = \hat A_t + V(s_t)
$$

d_t is the termination flag; Â_t is standardised within each minibatch. Loss:

$$
\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)},\qquad
L^{\text{clip}} = \mathbb{E}\big[\min\big(\rho_t \hat A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\, \hat A_t\big)\big]
$$

$$
L = -L^{\text{clip}} + c_v\, \mathbb{E}\big[(V_\theta(s_t) - \hat R_t)^2\big] - c_e\, \mathbb{E}\big[\mathcal{H}(\pi_\theta(\cdot \mid s_t))\big]
$$

| Parameter | Value |
|---|---|
| parallel envs × steps per env | 40 × 512 = 20480 steps per rollout |
| minibatch, epochs | 256, 10 epochs (800 gradient updates per rollout) |
| ε (clip), c_v, c_e | 0.1, 0.5, 0.005 |
| optimiser | Adam, lr = 3 × 10⁻⁴, eps = 10⁻⁵ (SB3's setting for PPO), gradient-norm clip 0.5 |

### 8.2 DQN / Double DQN (SB3 DQN and `shared/dqn_variants.py`)

TD target (Q̄ = target network):

$$
\text{DQN:}\quad y = r + \gamma (1 - d) \max_{a'} \bar Q(s', a')
$$

$$
\text{Double DQN:}\quad y = r + \gamma (1 - d)\, \bar Q\big(s',\ \arg\max_{a'} Q(s', a')\big)
$$

Loss: Huber (smooth L1, β = 1), ℒ = 𝔼[Huber(Q(s, a) − y)].

| Parameter | Value |
|---|---|
| parallel envs | 12 |
| replay buffer, batch, learning starts | 10⁵, 64, after 2000 steps |
| update frequency | one gradient update every 4 env calls (4 × 12 = 48 steps) |
| target network | full copy (τ = 1) every ⌊500/12⌋ = 41 env calls (≈ 492 steps) |
| ε-greedy | ε falls linearly from 1 to 0 over the first 2 × 10⁶ steps, then stays 0; one random draw is shared by the 12 envs of an env call |
| optimiser | Adam, lr = 10⁻³, PyTorch defaults (eps = 10⁻⁸), gradient-norm clip 10 |

### 8.3 Mask-DQN / Mask-DDQN

The mask 𝒜(s) is that of Section 6.2 (EXIT included), computed by the network from the raw state.

- Exploration and before learning starts: uniform over 𝒜(s).
- Greedy: a = argmax_{a ∈ 𝒜(s)} Q(s, a).
- TD target:

$$
\text{Mask-DQN:}\quad y = r + \gamma (1 - d) \max_{a' \in \mathcal{A}(s')} \bar Q(s', a')
$$

$$
\text{Mask-DDQN:}\quad y = r + \gamma (1 - d)\, \bar Q\big(s',\ \arg\max_{a' \in \mathcal{A}(s')} Q(s', a')\big)
$$

### 8.4 Lagrangian dual variable

See Section 6.4. λ has its own Adam (lr = 0.035), separate from the network's optimiser.

---

## 9. Evaluation metrics

Each test instance is run once with the deterministic policy (PPO: the most probable action; DQN: argmax Q; Mask with its mask; Lagrangian with the λ at the end of training). Let 𝒯 be the test set (|𝒯| = 5000) and 𝒯_ok = {i : M_v^{(i)} = M} the instances completed without violation:

$$
\text{gap} = 1 - \frac{\frac{1}{|\mathcal{T}|}\sum_{i \in \mathcal{T}} \text{AR}^{(i)}}{\frac{1}{|\mathcal{T}|}\sum_{i \in \mathcal{T}} \text{AR}^{*(i)}}
$$

$$
\text{violation-free share} = \frac{|\mathcal{T}_{\text{ok}}|}{|\mathcal{T}|},\qquad
\text{violation-free relative gap} = \frac{\overline{\text{AR}^*}_{\mathcal{T}_{\text{ok}}} - \overline{\text{AR}}_{\mathcal{T}_{\text{ok}}}}{\overline{\text{AR}^*}_{\mathcal{T}_{\text{ok}}}}
$$

$$
\text{capacity violation rate} = \frac{1}{|\mathcal{T}|}\sum_{i} \mathbb{1}\Big[\textstyle\sum_t v^{\text{cap}(i)}_t > 0\Big],\quad
\text{privacy violation rate} = \frac{1}{|\mathcal{T}|}\sum_{i} \mathbb{1}\Big[\textstyle\sum_t v^{\text{priv}(i)}_t > 0\Big],\quad
\text{EXIT rate} = \frac{1}{|\mathcal{T}|}\sum_{i} \mathbb{1}[\text{EXIT}^{(i)}]
$$

- AR counts legal placements only (Section 4.4). The gap is over all instances, failed ones included (with the AR of their legal part).
- The "AR gap" in the training logs is the absolute difference ILP AR − AR on 𝒯_ok, not a relative value.
- The report tables give the mean ± standard deviation over seeds.

---

## 10. Implementation details that affect interpretation

1. First Lagrangian episode: the training envs are created with λ = 0.1 (`config.LAMBDA_INIT`); λ₀ = 0.001 is set when training starts and takes effect at the next reset. So the first episode of every parallel env uses λ = 0.1.
2. Lagrangian DQN / DDQN: the replay buffer stores r_t as sampled (with the λc_t of that time already subtracted); it is not recomputed when λ changes.
3. Repair: the action stored in the replay buffer (DQN) or used for log π (PPO) is the agent's a_t, not the repaired ã_t.
4. AR_net and load in the network features include violating placements (Section 4.5). The AR in the reward and in the evaluation counts legal placements only (Section 4.4); the two differ.
