"""Main table of the thesis with the gap computed on violation-free (successful) test instances only.

Gap definition (author's rule): AR, ILP AR and the AR gap are computed only on test instances that were completed
without any violation (and, for masking, without EXIT). Relative gap = 1 - mean_ok(AR) / mean_ok(ILP AR) over those
instances. Violation and EXIT rates are still reported over all 5000 instances.

Reuses src/scripts/merge_2d_table.py (same 108 runs, same tables and figures) and only replaces the gap fields.
Writes into this folder: table_2d.json, table_2d.md, table_2d.en.md, table_main.tex, table_2d.png and
../figs/fig_gap_bars.{pdf,eps}, ../figs/fig_gap_viol.{pdf,eps}.
Usage (repo root): python paper_contents/20261122/experiments/build_tables.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src" / "scripts"))
import merge_2d_table as M  # noqa: E402

DEF_ZH_OLD = "- **gap** = 1 − mean(AR) / mean(ILP AR)，AR 只计合法放置"
DEF_ZH_NEW = "- **gap** = 1 − mean(AR) / mean(ILP AR)，只在无违约完成的实例上计算（Mask 另排除 EXIT 实例），AR 只计合法放置"
DEF_EN_OLD = "- **gap** = 1 − mean(AR) / mean(ILP AR), AR over legal placements"
DEF_EN_NEW = "- **gap** = 1 − mean(AR) / mean(ILP AR) over test instances completed without violation (and without EXIT for masking), AR over legal placements"


def success_rows():
    rows = M.load()
    for r in rows:
        n = r["new"]
        star_ok = n["vf_abs"] / n["vf_rel"]          # mean ILP AR on violation-free instances
        n.update(gap=n["vf_rel"], ilp=star_ok, ar=star_ok - n["vf_abs"])
    return rows


def main():
    tmp = HERE / "_build"
    tmp.mkdir(exist_ok=True)
    shutil.copy(M.OUT / "time_bench.json", tmp / "time_bench.json")
    M.OUT = tmp
    rows = success_rows()
    M.plot(rows); M.paper(rows); M.dump(rows)
    (tmp / "table_2d.md").write_text(M.build(rows, "zh").replace(DEF_ZH_OLD, DEF_ZH_NEW))
    (tmp / "table_2d.en.md").write_text(M.build(rows, "en").replace(DEF_EN_OLD, DEF_EN_NEW))
    tex = (tmp / "table_main.tex").read_text().replace("on the test set,", "on violation-free test instances,")
    (tmp / "table_main.tex").write_text(tex)
    for f in ["table_2d.json", "table_2d.md", "table_2d.en.md", "table_main.tex", "table_2d.png"]:
        shutil.copy(tmp / f, HERE / f)
    figs = HERE.parent / "figs"
    for f in ["fig_gap_bars", "fig_gap_viol"]:
        shutil.copy(tmp / f"{f}.pdf", figs / f"{f}.pdf")
        subprocess.run(["pdftops", "-eps", str(figs / f"{f}.pdf"), str(figs / f"{f}.eps")], check=True)
    shutil.rmtree(tmp)
    print("written:", HERE)


if __name__ == "__main__":
    main()
