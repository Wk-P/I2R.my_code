# Related work: dead ends after masking / feasibility prediction

Context: the v6.0.2 diagnostic (`paper_contents/v6.0.2/exit_diag.en.md`) found that on LT the instances where a Mask policy EXITs have no feasible completion from step 4.4 on average, while the EXIT happens at step 12.0; a one-step necessary condition notices it only 0.4 steps before the EXIT. The works below relate to "letting RL learn at the step where it goes wrong".

## 1. Dead ends after masking in combinatorial optimisation

| Work | Key points | Relation to ours |
|---|---|---|
| Bi, Ma, Zhou, Song, Cao, Wu, Zhang. *Learning to Handle Complex Constraints for Vehicle Routing Problems.* NeurIPS 2024 (PIP / PIP-D). [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/a9d2a5fd12d34250c21b5e4fa8d906b0-Paper-Conference.pdf) · [arXiv 2410.21066](https://arxiv.org/abs/2410.21066) | Masking the current step is easy, masking actions that make the rest infeasible is hard (NP-hard); Lagrangian multiplier for constraint awareness + preventative infeasibility masking; PIP-D learns to predict that mask with an auxiliary decoder. Large drop in infeasible rate on TSPTW / TSPDL | Closest comparison: same "dead end despite masking" problem |
| Zhao, She, Zhu, Yang, Xu. *Online 3D Bin Packing with Constrained Deep Reinforcement Learning.* AAAI 2021. [arXiv 2006.14978](https://arxiv.org/abs/2006.14978) | A feasibility predictor outputs a placement mask that modulates the actor's action probabilities; feasibility-mask loss + infeasibility penalty, CMDP formulation | Packing problem, learned feasibility mask |
| Solozabal, Ceberio, Takáč. *Constrained Combinatorial Optimization with Reinforcement Learning.* [arXiv 2006.11984](https://arxiv.org/abs/2006.11984) | Prior neural CO only handled problems where masking guarantees feasibility; encodes constraints as reward penalties in a CMDP | Mask vs penalty |

## 2. Dead-end theory

| Work | Key points | Relation to ours |
|---|---|---|
| Fatemi, Sharma, van Seijen, Kahou. *Dead-ends and Secure Exploration in Reinforcement Learning.* ICML 2019. [PMLR](https://proceedings.mlr.press/v97/fatemi19a.html) | Defines dead-end states (any action leads to an undesired terminal); a "security" condition as an auxiliary value function bounding exploration | The 7.5 steps placed after the instance became unsolvable are steps inside a dead end |
| Fatemi, Killian, Subramanian, Ghassemi. *Medical Dead-ends and Learning to Identify High-risk States and Treatments.* NeurIPS 2021. [arXiv 2110.04186](https://arxiv.org/abs/2110.04186) | Offline identification of states and actions that lead into dead ends | Learned dead-end detection |

## 3. Safe RL: feasible sets / safety critics

| Work | Key points | Relation to ours |
|---|---|---|
| Yu, Ma, Li, Chen. *Reachability Constrained Reinforcement Learning.* ICML 2022. [PMLR](https://proceedings.mlr.press/v162/yu22d.html) · [arXiv 2205.07536](https://arxiv.org/abs/2205.07536) | Reachability analysis characterises the feasible set where constraints can be satisfied persistently; the safety value function is the constraint | Corresponds to "does the fixed prefix still have a feasible completion" |
| Thananjeyan, Balakrishna, Nair, Luo, Srinivasan, Hwang, Gonzalez, Ibarz, Finn, Goldberg. *Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones.* RA-L 2021. [arXiv 2010.15920](https://arxiv.org/abs/2010.15920) | A learned safety critic estimates the violation risk of state-action pairs; above a threshold a recovery policy takes over | Learned risk estimate |

## 4. Difference from our setting

- All works above use a **learned approximation** of feasibility or risk. Our instances are all ILP-verified feasible and small (N ≤ 15, M ≤ 15), so during training we can decide **exactly** whether the current prefix still has a feasible completion: about 44 ms per ILP call, about 0.9 ms with an exact backtracking search (agreeing with the ILP on 200/200 random LT prefixes).
- Only a few searches were run; no work was found that uses exact feasibility as a step-level signal or label in constructive RL training, but this is not a novelty claim. To use it as a contribution, search systematically with feasibility oracle, preventative masking, dead-end, constructive neural solver, etc.
