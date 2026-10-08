#!/usr/bin/env python3
"""v4.4.9 check: ar_tel is return-preserving -- for every trajectory, sum_t r_t(ar_tel) == sum_t r_t(ar_pen).

Two environments (ar_pen, ar_tel) on the same instance follow the same action sequence. Checked per
trajectory: |G_tel - G_pen| < 1e-6; per non-terminal step: r_t == AR_t - AR_{t-1} and the paid total
== AR_t; on success: the last step == AR_M - AR_{M-1}. Policies: uniform over legal actions, and a
"greedy-bad" mix that prefers the ECU with the least free capacity (drives EXIT at many different
steps). Settings: the training one (Mask + EXIT + full episode) on all scenarios, plus none / repair /
lagrange (lambda > 0) with and without full episode, and Mask without EXIT.

    .venv/bin/python src/scripts/check_ar_tel.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from paper_rl.env import PlacementEnv                       # noqa: E402
from paper_rl.train import split_instances                  # noqa: E402

TOL = 1e-6
SETTINGS = [("mask", dict(full_episode=True, exit_action=True)), ("mask", dict(full_episode=False)),
            ("none", dict(full_episode=True)), ("repair", dict(full_episode=True)), ("repair", dict(full_episode=False)),
            ("lagrange", dict(full_episode=True))]


def run(scen, mech, kw, episodes, seed):
    _, train, test = split_instances(scen, 1)
    insts = train + test
    envs = [PlacementEnv(insts, mech, mode, lam=0.7 if mech == "lagrange" else 0.0, **kw) for mode in ("ar_pen", "ar_tel")]
    rng = np.random.default_rng(seed)
    worst, n_succ, exit_steps, fails = 0.0, 0, Counter(), []
    for e in range(episodes):
        k = int(rng.integers(len(insts)))
        for env in envs:
            env.use_instance(k)
            env.reset()
        bad = rng.random() < 0.5                                  # half the episodes: least-free-capacity choice
        g, done, ar_prev = [0.0, 0.0], False, 0.0
        while not done:
            pen, tel = envs
            if mech == "mask":
                legal = np.flatnonzero(pen.action_masks())
            else:
                legal = np.arange(pen.N)
            ecus = legal[legal < pen.N]
            if bad and len(ecus):
                a = int(ecus[np.argmin(pen.remaining[ecus])]) if rng.random() < 0.7 else int(rng.choice(legal))
            else:
                a = int(rng.choice(legal))
            out = [env.step(a) for env in envs]
            (_, r0, d0, _, i0), (_, r1, d1, _, i1) = out
            assert d0 == d1 and abs(i0["ar"] - i1["ar"]) < 1e-12
            g[0] += r0
            g[1] += r1
            done = d0
            if not done:
                if abs(r1 - (i1["ar"] - ar_prev)) > TOL or abs(tel.paid - i1["ar"]) > TOL:
                    fails.append((scen, mech, kw, k, "step", r1, i1["ar"] - ar_prev))
            elif i1["valid_placed"] == tel.M and not i1["exited"]:
                n_succ += 1
                if mech != "lagrange" and abs(r1 - (i1["ar"] - ar_prev)) > TOL:
                    fails.append((scen, mech, kw, k, "last", r1, i1["ar"] - ar_prev))
            ar_prev = i1["ar"]
        if i1["exited"]:
            exit_steps[i1["services_placed"]] += 1
        worst = max(worst, abs(g[1] - g[0]))
        if abs(g[1] - g[0]) > TOL:
            fails.append((scen, mech, kw, k, "return", g[0], g[1]))
    return worst, n_succ, exit_steps, fails


def main():
    ok = True
    for scen in ("lt", "eq", "gt"):
        for mech, kw in SETTINGS:
            worst, n_succ, exits, fails = run(scen, mech, kw, 3000 if kw.get("exit_action") else 600, seed=hash((scen, mech)) % 2**31)
            tag = f"{scen} {mech:8s} {'full' if kw.get('full_episode') else 'stop'}{'+EXIT' if kw.get('exit_action') else ''}"
            print(f"{tag:26s} max|G_tel-G_pen| = {worst:.2e}  success {n_succ:4d}  EXIT at steps "
                  f"{dict(sorted(exits.items())) if exits else '-'}  failures {len(fails)}")
            if fails:
                ok = False
                print("   first failure:", fails[0])
    print("ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
