# 2. The reference: ILP

## What it does

ILP (integer linear programming) writes the problem as equations and hands it to a solver (PuLP + CBC), which guarantees the **optimum**.
Here it is the "full marks" answer used to compute RL's gap.

## Variables

- x_mn = 1 if service m is on ECU n, else 0.
- y_n = 1 if ECU n is used.

Constraints: each service on exactly one ECU; ECU load ≤ capacity; two conflicting services never on the same ECU; placing requires the ECU to be used.

## The catch: the objective is a ratio

AR = total utilization ÷ number of ECUs used, a fraction that linear programming cannot maximize directly.
**Dinkelbach's method** turns it into a few ordinary problems:

1. Guess an AR value q (start at 0).
2. Solve an ordinary ILP: maximize total utilization − q × number of ECUs.
3. Compute the true AR of that solution; it becomes the new q.
4. Stop when q barely changes (change < 1e-6, i.e. nothing better than q exists); otherwise go back to step 2.

It converges in about **3 iterations** on average (3.06 measured). Code: `solve_ilp_max_ar` in `src/shared/ilp_utils.py`.

## The problem with it: speed

On one CPU thread an instance takes **0.5–3 s** on average (LT is slowest, about 3.2 s), growing with size.
RL needs tens of milliseconds per instance (file 8).
