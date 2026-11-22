# 6 Conclusion and Future Work (material)

## Conclusion

[Prose by the author.] Facts:
- The ILP AR objective is transferred to the RL reward and the constraints to four constraint-handling strategies; 12 variants are compared on LT/EQ/GT.
- PPO-M: zero violations, three-scenario mean relative AR gap 4.36% (on violation-free instances), 28× faster than the ILP (47.5× on LT).
- Exact solution is strongly NP-hard (Proposition 1, Section 3.2); online inference is polynomial (Section 5.5).

## Future work

[Prose by the author.] Directions from open questions:
- Reduce EXIT on LT by detecting dead ends earlier (feasibility prediction / preventive masking, see Section 2.5; an exact feasibility check takes about 0.9 ms at this size).
- Explain and exploit why repair beats masking for DQN.
- Solve-time curves for larger N, M and cross-size generalization.
- More training instances (data size was the main bottleneck earlier: supervised gap dropped clearly from 1600 to 10k instances).
