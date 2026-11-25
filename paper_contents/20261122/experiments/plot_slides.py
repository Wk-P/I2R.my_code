"""Slide figures (16:9, sans-serif 15 pt, 300 dpi PNG) -> ../slides_figs/

Every figure is drawn on ONE scope only: the main line uses the mean of LT / EQ / GT (the paper's headline numbers,
e.g. PPO-M 4.36%); backup/ holds the identical figure for each scenario (<name>_lt / _eq / _gt), nothing is mixed.

  main (three-scenario mean)                 backup (one per scenario)
  1_gap.png               gap, mechanism x learner, error bar = std over seeds        backup/1_gap_{lt,eq,gt}.png
  2a_viol.png             executed violation rate, mechanism x learner                backup/2a_viol_{lt,eq,gt}.png
  2b_viol_breakdown.png   failures per variant: EXIT / privacy / capacity / both     backup/2b_viol_breakdown_{lt,eq,gt}.png
  2c_gap_vs_viol.png      gap vs violation rate (closes groups 1 and 2)              -- (per scenario: see 1 and 2a)
  3_time.png              online time per instance, ILP + the 12 variants (linear)   backup/3_time_{lt,eq,gt}.png

Sources: gap = table_2d.json (Tables 7, 8); violations = the 108 test runs of src/scripts/merge_2d_table.load()
(Tables 9-11); time = time_full.json (Table 6). A three-scenario mean is the plain mean of the three scenario values
(gap: Table 7's per-seed mean over scenarios, then mean over seeds). Violation = 1 - violation-free - EXIT (EXIT is
not an executed violation). Breakdown is exact: privacy only = viol - cap, capacity only = viol - priv,
both = cap + priv - viol (per-instance shares). Colours: learner PPO / DQN / DDQN = blue / orange / aqua; ILP and
EXIT grey.

    .venv/bin/python paper_contents/20261122/experiments/plot_slides.py
"""
import json
import shutil
import statistics as st
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent / "slides_figs"
sys.path.insert(0, str(ROOT / "src" / "scripts"))

MECHS = [("maskable", "mask", "M", "Mask"), ("repair", "repair", "R", "Repair"),
         ("lagrangian", "lagrange", "L", "Lagrangian"), ("unconstraint", "none", "U", "Unconstraint")]
LEARNERS = [("PPO", "ppo", "#2a78d6"), ("DQN", "dqn", "#eb6834"), ("DDQN", "ddqn", "#1baf7a")]
SCENS = ["lt", "eq", "gt"]
SCOPE = {"mean": "mean of LT / EQ / GT", "lt": "LT", "eq": "EQ", "gt": "GT"}
INK, MUTED, GRID, C_ILP = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
SIZE, MS, LW = (12.0, 6.75), 11, 1.6

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.linewidth": 0.8, "savefig.facecolor": "white", "legend.frameon": False})


def fname(base, scope):
    return base if scope == "mean" else f"backup/{base}_{scope}"


def save(fig, name, tight=True):
    (OUT / name).parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300, transparent=False)
    plt.close(fig)


def recessive(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=LW * 0.6)
    ax.set_axisbelow(True)


def scope_tag(ax, scope):
    ax.text(1.0, 1.02, SCOPE[scope] if scope == "mean" else f"Scenario: {SCOPE[scope]}", transform=ax.transAxes,
            ha="right", va="bottom", color=MUTED, fontsize=14, fontweight="normal" if scope == "mean" else "bold")


# ── data ────────────────────────────────────────────────────────────────────
CELLS = {(c["strategy"], c["RL_algorithm"]): c for c in json.loads((HERE / "table_2d.json").read_text())["cells"]}
KEY = {m: mk for m, mk, _, _ in MECHS} | {l: lk for l, lk, _ in LEARNERS}


def gap(m, l, scope):
    c = CELLS[m, l]
    return (100 * c["mean"], 100 * c["mean_std"]) if scope == "mean" else (100 * c["gap"][scope],
                                                                             100 * c["gap_std"][scope])


def runs():
    """{(mech key, learner key, scope): {viol, exit, cap, priv}} in %, mean over seeds; scope 'mean' = mean of 3"""
    import merge_2d_table as M
    rows = M.load()
    assert len(rows) == 108
    out = {}
    for _, mk, _, _ in MECHS:
        for _, lk, _ in LEARNERS:
            for s in SCENS:
                sel = [r["new"] for r in rows if (r["mech"], r["learner"], r["scen"]) == (mk, lk, s)]
                assert len(sel) == 3
                f = lambda g: 100 * st.mean(g(n) for n in sel)
                out[mk, lk, s] = dict(viol=max(0.0, f(lambda n: 1 - n["vf"] - n["exit"])),
                                      exit=f(lambda n: n["exit"]), cap=f(lambda n: n["cap"]),
                                      priv=f(lambda n: n["conf"]))
            out[mk, lk, "mean"] = {k: st.mean(out[mk, lk, s][k] for s in SCENS) for k in out[mk, lk, "lt"]}
    return out


# ── shared layouts ──────────────────────────────────────────────────────────
def grouped(ax, vals, errs=None):
    """x = mechanism, one bar per learner, value label on every bar"""
    x, w = np.arange(len(MECHS)), 0.26
    hi = max(v + (0 if errs is None else errs[k]) for k, v in vals.items())
    ax.set_ylim(0, hi * 1.13)
    for j, (lname, _, col) in enumerate(LEARNERS):
        xs = x + (j - 1) * w
        v = [vals[m[0], lname] for m in MECHS]
        e = [0] * len(v) if errs is None else [errs[m[0], lname] for m in MECHS]
        ax.bar(xs, v, w - 0.02, color=col, label=lname, yerr=None if errs is None else e,
               error_kw=dict(ecolor=MUTED, elinewidth=LW * 0.8, capsize=4))
        for xi, vi, ei in zip(xs, v, e):
            ax.text(xi, vi + ei + hi * 0.012, "0" if vi < 1e-9 else f"{vi:.2f}%", ha="center", va="bottom",
                    fontsize=12.5)
    ax.set_xticks(x, [m[3] for m in MECHS])
    ax.tick_params(axis="x", length=0, labelsize=16)
    recessive(ax)


def variant_rows():
    rows = []
    for i, (m, mk, tag, _) in enumerate(MECHS):
        rows += ([None] if i else []) + [(f"{l}-{tag}", mk, lk) for l, lk, _ in LEARNERS]
    return rows


def row_axis(ax, rows):
    top, ticks = len(rows) - 1, {}
    for k, r in enumerate(rows):
        if r is None:
            ax.axhline(top - k, color=GRID, lw=LW * 0.8)
        else:
            ticks[r[0]] = top - k
    ax.set_yticks(list(ticks.values()), list(ticks.keys()))
    for t in ax.get_yticklabels():
        t.set_fontweight("bold" if t.get_text() == "PPO-M" else "normal")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, top + 0.7)
    return ticks


# ── 1. GAP ──────────────────────────────────────────────────────────────────
def fig_gap(scope):
    vals, errs = {}, {}
    for m, *_ in MECHS:
        for l, *_ in LEARNERS:
            vals[m, l], errs[m, l] = gap(m, l, scope)
    fig, ax = plt.subplots(figsize=SIZE)
    grouped(ax, vals, errs)
    ax.set_ylabel("Relative gap (%)")
    ax.legend(loc="upper left", ncol=3)
    scope_tag(ax, scope)
    save(fig, fname("1_gap", scope))


# ── 2. Violation ────────────────────────────────────────────────────────────
def fig_viol(R, scope):
    vals = {(m, l): R[mk, lk, scope]["viol"] for m, mk, _, _ in MECHS for l, lk, _ in LEARNERS}
    fig, ax = plt.subplots(figsize=SIZE)
    grouped(ax, vals)
    hi = ax.get_ylim()[1]
    ax.set_ylabel("Executed constraint violations (%)")
    ax.legend(loc="upper left", ncol=3)
    ax.annotate("0 by construction:\ninfeasible ECUs are masked", xy=(0, hi * 0.07), xytext=(0, hi * 0.32),
                ha="center", fontsize=14, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    scope_tag(ax, scope)
    save(fig, fname("2a_viol", scope))


SEGS = [("EXIT (stop, not executed)", C_ILP, lambda d: d["exit"]),
        ("Privacy only", "#e34948", lambda d: d["viol"] - d["cap"]),
        ("Capacity only", "#eda100", lambda d: d["viol"] - d["priv"]),
        ("Both", "#6250d6", lambda d: d["cap"] + d["priv"] - d["viol"])]


def fig_breakdown(R, scope):
    rows = variant_rows()
    totals = {r[0]: R[r[1], r[2], scope]["exit"] + R[r[1], r[2], scope]["viol"] for r in filter(None, rows)}
    small = [t for t in totals.values() if t < 20]
    xmax = max(1.0, float(np.ceil(max(small) * 1.18)))     # bars above 20% are cut at xmax, value written
    fig, ax = plt.subplots(figsize=SIZE)
    ticks = row_axis(ax, rows)
    for name, mk, lk in filter(None, rows):
        d, yy, left = R[mk, lk, scope], ticks[name], 0.0
        for _, col, f in SEGS:
            v = max(0.0, f(d))
            vis = max(0.0, min(v, xmax - left))
            if vis > 0:
                ax.barh(yy, vis, 0.62, left=left, color=col, edgecolor="white", linewidth=1.5)
            left += v
        t = totals[name]
        if t > xmax:
            ax.text(xmax * 0.99, yy, f"{t:.1f}%  →", ha="right", va="center", color="white", fontweight="bold")
        else:
            ax.text(t + xmax * 0.012, yy, "0" if t < 1e-9 else f"{t:.2f}%", ha="left", va="center",
                    fontweight="bold" if name == "PPO-M" else "normal")
    ax.set_xlim(0, xmax)
    cut = any(t > xmax for t in totals.values())
    ax.set_xlabel("Share of test instances (%)" + (f"   —   bars beyond {xmax:g}% are cut, value shown" if cut else ""))
    recessive(ax, "x")
    ax.legend(handles=[Patch(color=c, label=n) for n, c, _ in SEGS], loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=4, borderaxespad=0.2, handlelength=1.2, columnspacing=1.2)
    scope_tag(ax, scope)
    save(fig, fname("2b_viol_breakdown", scope))


def fig_gap_vs_viol(R):
    mshape = {"M": "o", "R": "s", "L": "D", "U": "X"}
    pts = [(f"{l}-{tag}", R[mk, lk, "mean"]["viol"], gap(m, l, "mean")[0], col, mshape[tag])
           for m, mk, tag, _ in MECHS for l, lk, col in LEARNERS]
    fig, (a, b) = plt.subplots(1, 2, figsize=SIZE, sharey=True, gridspec_kw=dict(width_ratios=[4, 1], wspace=0.05))
    off = {"PPO-M": (10, -10), "DQN-M": (10, 6), "DDQN-M": (10, -6), "PPO-R": (6, 14), "DQN-R": (10, -10),
           "DDQN-R": (10, 6), "PPO-L": (-10, -14), "DQN-L": (10, 0), "DDQN-L": (10, 0), "PPO-U": (8, -16),
           "DQN-U": (-10, 0), "DDQN-U": (-10, 0)}
    for ax in (a, b):
        for n, x, y, col, mk in pts:
            ax.plot(x, y, mk, color=col, ms=MS * 1.15, mec="white", mew=1.2, zorder=3)
            dx, dy = off[n]
            ax.annotate(n, (x, y), xytext=(dx, dy), textcoords="offset points", va="center",
                        ha="left" if dx > 0 else "right", fontsize=13,
                        fontweight="bold" if n == "PPO-M" else "normal", annotation_clip=True)
        recessive(ax, "both")
    a.set_xlim(-0.15, 2.45); b.set_xlim(29.5, 35)
    a.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    a.set_ylim(0, 27)
    b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
    for ax, xs in ((a, 1), (b, 0)):                   # break marks on the x-axis
        ax.plot([xs - 0.012 * (1 if xs else 4), xs + 0.012 * (1 if xs else 4)], [-0.02, 0.02],
                transform=ax.transAxes, color=INK, lw=1, clip_on=False)
    a.set_ylabel("Relative gap (%)")
    fig.supxlabel("Executed constraint violations (%)", fontsize=15)
    a.text(0.02, 0.97, "lower-left = better", transform=a.transAxes, color=MUTED, va="top")
    b.text(1.0, 1.01, SCOPE["mean"], transform=b.transAxes, ha="right", va="bottom", color=MUTED, fontsize=14)
    hs = [Line2D([], [], ls="", marker="o", color=col, ms=MS, label=l) for l, _, col in LEARNERS]
    hs += [Line2D([], [], ls="", marker=mshape[t], color=MUTED, ms=MS, label=name) for _, _, t, name in MECHS]
    fig.legend(handles=hs, loc="upper left", ncol=7, borderaxespad=0.3, handletextpad=0.2, columnspacing=1.0)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.12)
    save(fig, "2c_gap_vs_viol", tight=False)


# ── 3. Executed time ────────────────────────────────────────────────────────
def fig_time(scope):
    d = json.loads((HERE / "time_full.json").read_text())
    if scope == "mean":                                 # all 15,000 instances (equal size per scenario)
        mean = lambda v: float(np.mean(np.concatenate([v[s]["ms"] for s in SCENS])))
    else:
        mean = lambda v: float(np.mean(v[scope]["ms"]))
    ilp = mean(d["ilp"])
    names, vals, cols = ["ILP"], [ilp], [C_ILP]
    for _, mk, tag, _ in MECHS:
        for l, lk, col in LEARNERS:
            names.append(f"{l}-{tag}"); vals.append(mean(d["rl"][f"{mk}|{lk}"])); cols.append(col)
    rl = vals[1:]
    x = np.arange(len(names)) + np.array([0] + [0.6 + (i // 3) * 0.35 for i in range(12)])   # gap after ILP, mechs
    fig, ax = plt.subplots(figsize=SIZE)
    ax.bar(x, vals, 0.8, color=cols)
    for xi, v in zip(x, vals):
        ax.text(xi, v + ilp * 0.012, f"{v:,.0f}", ha="center", va="bottom", fontsize=12.5)
    ax.text(x[1:].mean(), ilp * 0.55, f"RL: {ilp / max(rl):.1f}–{ilp / min(rl):.1f}× faster than ILP",
            ha="center", va="center", fontsize=20, fontweight="bold")
    ax.set_xticks(x, names, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, ilp * 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_ylabel("Mean online time per instance (ms)")
    recessive(ax)
    ax.legend(handles=[Patch(color=C_ILP, label="ILP")] + [Patch(color=c, label=l) for l, _, c in LEARNERS],
              loc="upper right", ncol=4)
    scope_tag(ax, scope)
    save(fig, fname("3_time", scope))


if __name__ == "__main__":
    if OUT.exists():                                    # everything in slides_figs/ is generated here
        shutil.rmtree(OUT)
    R = runs()
    for scope in ["mean"] + SCENS:
        fig_gap(scope)
        fig_viol(R, scope)
        fig_breakdown(R, scope)
        fig_time(scope)
    fig_gap_vs_viol(R)
    print("\n".join(sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*.png"))))
