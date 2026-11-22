"""Recompute the headline numbers of the thesis from the two source files in this folder.

table_2d.json is written by build_tables.py: AR, ILP AR and gap are computed on violation-free test instances only.

Usage: python experiments/verify.py   (no training, no ILP; reads table_2d.json and time_full.json)
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "table_2d.json").read_text())
t = json.loads((HERE / "time_full.json").read_text())

by_strategy, by_scenario = defaultdict(list), defaultdict(list)
for c in d["cells"]:
    by_strategy[c["strategy"]].append(c["mean"])
    for s, g in c["gap"].items():
        by_scenario[s].append(g)
    if c["strategy"] == "maskable" and c["RL_algorithm"] == "PPO":
        ppo_m = c

ilp_ms = {s: st.mean(v["ms"]) for s, v in t["ilp"].items()}
rl_ms = {s: st.mean(v["ms"]) for s, v in t["rl"]["mask|ppo"].items()}

print(f"PPO-M mean gap        {ppo_m['mean']*100:.2f}% ± {ppo_m['mean_std']*100:.2f}%  "
      f"(LT {ppo_m['gap']['lt']*100:.2f} / EQ {ppo_m['gap']['eq']*100:.2f} / GT {ppo_m['gap']['gt']*100:.2f})")
print(f"PPO-M AR / ILP AR    {st.mean(ppo_m['ar'].values()):.4f} / {st.mean(ppo_m['ilp_ar'].values()):.4f}  (violation-free instances)")
print("Strategy row mean     " + "  ".join(f"{k} {st.mean(v)*100:.2f}%" for k, v in by_strategy.items()))
print("Scenario mean (12 v.) " + "  ".join(f"{k} {st.mean(v)*100:.2f}%" for k, v in by_scenario.items()))
print(f"ILP time              mean {st.mean(ilp_ms.values()):.0f} ms  " + "  ".join(f"{k} {v:.0f}" for k, v in ilp_ms.items()))
print(f"PPO-M speed-up        {st.mean(ilp_ms.values())/st.mean(rl_ms.values()):.1f}x  "
      + "  ".join(f"{k} {ilp_ms[k]/rl_ms[k]:.1f}x" for k in rl_ms))
