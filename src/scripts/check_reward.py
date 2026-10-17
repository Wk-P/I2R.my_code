#!/usr/bin/env python3
"""Formula check of the ar_pen reward and the four strategies (2026-10-17, for src/scripts/run_campaign.py).

Replays episodes in PlacementEnv(reward_mode="ar_pen", full_episode=True, exit_action = Maskable only) and recomputes,
step by step and without the env's own bookkeeping, what every quantity must be:

  order        services in descending demand (stable)
  feasibility  capacity (remaining >= demand) and privacy (no hosted service shares a conflict set)
  executed ECU none / lagrange: the chosen ECU; mask: the action comes from action_masks(), which must be the
               feasible set + EXIT (EXIT valid iff no ECU is feasible); repair: an infeasible choice becomes the best-fit feasible ECU
               argmax demand / capacity, or stays when nothing is feasible
  violations   cap_v = remaining < demand, conf_v = privacy conflict, per placement
  reward       r_t = 0 before the end; terminal AR if all M placements are legal (M_v = M), else M_v/M - 1
               (no factor M); Mask's EXIT ends the episode with M_v/M - 1; Lagrangian additionally
               -lambda * (cap_v + conf_v) at every step
  AR           sum over legal placements n_i / e_j, divided by the number of ECUs hosting a legal placement
  episode      exactly M steps (Mask: until EXIT; Mask never violates); on a violation-free episode AR <= ILP AR (ar_star is the ILP optimum)

and the Lagrangian dual step used by v5.4.0 (Adam on -lambda (J_C - d), lambda >= 0). Writes a JSON summary and
exits non-zero on the first mismatch.

    .venv/bin/python src/scripts/check_reward.py --out logs/<batch>/check_reward.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
POOL = "results/sup_diag/{scen}.json"
TOL = 1e-6


class Mismatch(AssertionError):
    pass


def need(cond, msg):
    if not cond:
        raise Mismatch(msg)


def episode(env, inst, mech, policy, rng, lam):
    """One episode with independent recomputation; returns a dict of what happened."""
    caps, dem = list(inst["ECUs"]), list(inst["SVCs"])
    n, m = len(caps), len(dem)
    order = sorted(range(m), key=lambda i: -dem[i])
    sets = [set(s) for s in inst["conflict_sets"]]
    conflicts = lambda a, b: any(a in s and b in s for s in sets)
    rem = [float(c) for c in caps]
    hosted = [set() for _ in range(n)]
    legal_ru, legal_e, m_v = 0.0, set(), 0
    n_cap = n_conf = 0
    if mech == "lagrange":
        env.set_lambda(lam)
    env.use_instance(env.instances.index(inst))
    env.reset()
    need(list(env.order) == order, f"order {env.order} != descending demand {order}")
    rewards, dead_steps, exited, steps = [], 0, False, 0
    for t in range(m):
        s = order[t]
        feas = [rem[j] >= dem[s] and not any(conflicts(s, h) for h in hosted[j]) for j in range(n)]
        if mech == "mask":
            mask = env.action_masks()
            need(len(mask) == n + 1, f"mask length {len(mask)} != N + 1 {n + 1} (Maskable must have EXIT)")
            need(list(mask) == feas + [not any(feas)], f"t={t}: mask {list(mask)} != feasible {feas} + EXIT")
            a = int(rng.choice(np.flatnonzero(mask)))
        else:
            a = policy(feas, rng, n)
        steps += 1
        if mech == "mask" and a == n:                                  # EXIT
            dead_steps += 1
            _, r, done, trunc, info = env.step(a)
            exp_r = m_v / m - 1.0
            need(done and not trunc and info["exited"], f"t={t}: EXIT must end the episode")
            need(abs(r - exp_r) < 1e-5, f"t={t} EXIT: reward {r} != M_v/M - 1 = {exp_r}")
            rewards.append(r)
            exited = True
            break
        if mech == "repair" and not feas[a] and any(feas):
            cand = [j for j in range(n) if feas[j]]
            want = max(cand, key=lambda j: (np.float32(dem[s]) / np.float32(caps[j]), -cand.index(j)))
        else:
            want = a
        dead_steps += not any(feas)
        _, r, done, trunc, info = env.step(a)
        got = int(env._assign[t]) - 1
        need(got == want, f"t={t} {mech}: executed ECU {got} != expected {want} (action {a}, feasible {feas})")
        cv, fv = rem[want] < dem[s], any(conflicts(s, h) for h in hosted[want])
        n_cap += cv
        n_conf += fv
        rem[want] -= dem[s]
        hosted[want].add(s)
        if not (cv or fv):
            m_v += 1
            legal_ru += dem[s] / caps[want]
            legal_e.add(want)
        need(done == (t == m - 1), f"t={t}: done={done} (full episode must have exactly M={m} steps)")
        need(not trunc, "truncated")
        need(bool(info["cap_violated"]) == cv and bool(info["conflict_violated"]) == fv,
             f"t={t}: step violation flags {info['cap_violated']},{info['conflict_violated']} != {cv},{fv}")
        if done:                                    # ar_pen terminal, no factor M
            exp_r = legal_ru / len(legal_e) if m_v == m else m_v / m - 1.0
        else:
            exp_r = 0.0
        if mech == "lagrange":
            exp_r -= lam * (int(cv) + int(fv))
        need(abs(r - exp_r) < 1e-5 * max(1.0, abs(exp_r)), f"t={t} {mech}: reward {r} != expected {exp_r}")
        rewards.append(r)
    ar = legal_ru / len(legal_e) if legal_e else 0.0
    need(abs(env.ar - ar) < 1e-5, f"AR {env.ar} != recomputed {ar}")
    need(info["capacity_violations"] == n_cap and info["conflict_violations"] == n_conf, "episode violation counts")
    clean = n_cap == 0 and n_conf == 0 and not exited
    term = rewards[-1] + (lam * (int(info["cap_violated"]) + int(info["conflict_violated"])) if mech == "lagrange" else 0.0)
    need((term > 0) == clean and -1.0 <= term <= 1.0, f"terminal {term}: must be AR in (0,1] on success, [-1,0) otherwise")
    if clean:
        need(ar <= inst["ar_star"] + TOL, f"violation-free AR {ar} > ILP AR {inst['ar_star']}")
    if mech == "mask":
        need(n_cap == 0 and n_conf == 0, "Maskable placed a violating service (EXIT must take the dead end)")
        need(exited == (dead_steps > 0), "Maskable reached a dead end without EXIT")
    else:
        need(steps == m, f"{mech}: {steps} steps != M")
    if mech == "repair":
        need(clean or dead_steps > 0, "Repair violated with a feasible ECU available")
    return {"clean": clean, "ar": ar, "ar_star": inst["ar_star"], "ret": float(sum(rewards)), "cap": n_cap, "conf": n_conf,
            "dead": dead_steps}


def check_dual():
    """v5.4.0 / campaign Lagrangian dual step: Adam (lr 0.035) on -lambda (J_C - d), d = 0, lambda >= 0, init 0.001."""
    from paper_rl.train import make_callback

    class VE:
        def env_method(self, *a, **k):
            return None
    out = {}
    for name, seq in (("J>d", [2.0] * 5), ("J=d=0 after violations", [2.0] * 3 + [0.0] * 5), ("J=0 from start", [0.0] * 5)):
        cb = make_callback("lagrange", 1, ROOT / "logs", "x", lam_max=float("inf"), lam_update="adam", lam_lr=0.035,
                           lam_init=0.001, lam_target=0.0)
        import types
        cb.model = types.SimpleNamespace(get_env=lambda: VE())   # SB3: training_env is model.get_env()
        cb.num_timesteps = 0
        lams = []
        for j in seq:
            cb._step_lambda(j, j)
            lams.append(round(cb.lam, 6))
        need(all(x >= 0 for x in lams), "lambda < 0")
        out[name] = lams
    need(out["J>d"][0] > 0.001 and all(b > a for a, b in zip(out["J>d"], out["J>d"][1:])), "lambda must rise while J_C > d")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="instances per scenario")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from paper_rl.env import PlacementEnv
    rng = random.Random(0)
    nrng = np.random.default_rng(0)
    policies = {"random": lambda f, r, n: int(r.integers(n)),
                "feasible-first": lambda f, r, n: int(r.choice([j for j in range(n) if f[j]] or list(range(n))))}
    summary, total = {}, 0
    for scen in ("eq", "gt", "lt"):
        pool = json.loads((ROOT / POOL.format(scen=scen)).read_text())
        insts = rng.sample(pool, a.n)
        for mech in ("none", "lagrange", "mask", "repair"):
            env = PlacementEnv(insts, mech, "ar_pen", lam=0.0, obs_mode="raw", full_episode=True, exit_action=mech == "mask",
                               lam_max=float("inf"))
            need(env.action_space.n == env.N + (mech == "mask"), "action space must be N (+ EXIT for Maskable only)")
            for pname, pol in policies.items():
                res = [episode(env, x, mech, pol, nrng, 0.7) for x in insts]
                total += len(res)
                summary[f"{scen}/{mech}/{pname}"] = {
                    "episodes": len(res), "violation_free": float(np.mean([r["clean"] for r in res])),
                    "mean_return": float(np.mean([r["ret"] for r in res])),
                    "episodes_with_dead_end": float(np.mean([r["dead"] > 0 for r in res]))}
                print(f"[ok] {scen} {mech:8s} {pname:14s} {len(res)} episodes | violation-free "
                      f"{summary[f'{scen}/{mech}/{pname}']['violation_free']:.3f}", flush=True)
    dual = check_dual()
    print(f"[ok] Lagrangian dual (Adam, d = 0): {dual}", flush=True)
    out = {"ok": True, "episodes_checked": total, "per_case": summary, "dual": dual,
           "note": "with d = 0 the dual gradient -(J_C - d) is never positive, so lambda never decreases"}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"ALL CHECKS PASSED ({total} episodes)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Mismatch as e:
        print(f"CHECK FAILED: {e}", flush=True)
        sys.exit(1)
