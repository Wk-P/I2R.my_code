import json, os, sys, time
from pathlib import Path
scen = sys.argv[1]
os.environ["TRAIN_SEED"] = "1"
ROOT = Path("/home/soar009/github/my_code")
here = ROOT / "scenarios" / scen / "ppo"
for p in (str(ROOT), str(here.parent), str(here)):
    sys.path.insert(0, p)
os.chdir(here)
import config as C
from ilp.objects import ECU, SVC
from shared.ilp_utils import solve_ilp
times, ars = [], []
for sc in C.TEST_SCENARIOS:
    caps, reqs, cs = sc[0], sc[1], sc[2] if len(sc) > 2 else []
    e = [ECU(f"ECU{i}", c) for i, c in enumerate(caps)]
    s = [SVC(f"SVC{i}", r) for i, r in enumerate(reqs)]
    t0 = time.perf_counter(); res = solve_ilp(e, s, cs); times.append(time.perf_counter() - t0)
    ars.append(res["avg_utilization"])
import numpy as np
print(json.dumps({"scen": scen, "n_test": len(times), "ms_mean": float(np.mean(times)) * 1000,
                  "ms_std": float(np.std(times, ddof=1)) * 1000, "ms_median": float(np.median(times)) * 1000,
                  "ms_max": float(np.max(times)) * 1000, "ar": float(np.mean(ars))}))
