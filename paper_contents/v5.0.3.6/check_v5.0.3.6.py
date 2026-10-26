import sys, json, random
import numpy as np, torch
torch.set_num_threads(4)
from pathlib import Path
from paper_rl.env import PlacementEnv
from paper_rl.train import model_class, split_instances
ROOT = Path("/home/soar009/github/my_code")
scen, algo, eid, learner = sys.argv[1:5]
run = ROOT/"results/unified"/scen/algo/eid
argv = json.loads((run/"run.json").read_text())["argv"]
print("argv", " ".join(argv))
pool = json.loads((ROOT/f"results/sup_diag/{scen}.json").read_text())
sub = [pool[m] for m in sorted(random.Random(0).sample(range(len(pool)), 1600))]
_, _, test = split_instances(scen, 1)
ck = json.loads((run/"ckpt_lambda.json").read_text())
fin = json.loads((run/"results.json").read_text())["training"]["final_lambda"]
models = [(f"ckpt_{s}", run/f"ckpt_{s}_{eid}.zip", ck[str(s)]) for s in (3000000, 4000000)] + [("final", next(run.glob("model_*")), fin)]
def cost(model, insts, lam, rand=None, n=None):
    env = PlacementEnv(insts, "lagrange", "ar_pen", lam=lam, obs_mode="raw", full_episode=True, lam_max=5.0,
                       **({"rng_seed": rand} if rand is not None else {}))
    env.set_lambda(lam)
    c = []
    for k in range(n or len(insts)):
        if rand is None: env.use_instance(k)
        o, _ = env.reset(); d = False
        while not d:
            a, _ = model.predict(o, deterministic=True); o, _, d, _, info = env.step(int(a))
        c.append(info["cost_ep"])
    c = np.array(c); return f"J_C={c.mean():.4f} fail={np.mean(c>0):.3f}"
for name, f, lam in models:
    m_ = model_class(learner, "lagrange").load(str(f), device="cpu")
    print(name, "lam", round(lam,4))
    print("  pool random 500 (as greedy_env):", cost(m_, pool, lam, rand=999, n=500))
    print("  train-1600 subset             :", cost(m_, sub, lam))
    print("  test 400                      :", cost(m_, test, lam))
    for l2 in (0.5, 2.0):
        print(f"  test 400 with obs lam={l2}      :", cost(m_, test, l2))
