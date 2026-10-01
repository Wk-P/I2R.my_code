"""paper_rl — unified implementation (v4.3.1.3+).

One environment, one configuration and one training entry point for all
12 models ({PPO, DQN, DDQN} x {none, mask, lagrange, repair}) and all
three scenarios; LT / EQ / GT differ only in the number of ECUs (N) and
services (M). See paper_contents/v4.3.1/v4.3.1.3/README.md.
"""
