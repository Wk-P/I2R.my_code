"""实时合并 scenario_count_sweep 的结果，边跑边更新，不用等 24 组全部跑完。

每次运行：扫 logs/scenario_count_sweep/*.log（文件名自带
scenario/train_count/steps/exp_id，格式 {sc}_tc{tc}_{steps}_{exp_id}.log），
对已经产出 results.json 的：
  1) 把 train_scenario_count / requested_total_steps 写进 results.json 本身
  2) 在 results/add_states/<sc>/ppo_mask/ 下建可读软链接 tc<tc>_<steps> -> <exp_id>
  3) 汇总进单张表 logs/scenario_count_sweep/live_summary.{json,txt}

可重复运行（幂等），配合 watch loop 每隔一段时间跑一次。
"""
import json
import re
import statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/soar009/github/my_code")
LOG_DIR = ROOT / "logs/scenario_count_sweep"
PATTERN = re.compile(r"^(lt|eq|gt)_tc(\d+)_(\d+)_([0-9a-f]+)\.log$")


def main():
    groups = defaultdict(list)
    pending = []

    for log in sorted(LOG_DIR.glob("*.log")):
        m = PATTERN.match(log.name)
        if not m:
            continue
        sc, tc, steps, exp_id = m.groups()
        tc, steps = int(tc), int(steps)

        result_path = ROOT / f"results/add_states/{sc}/ppo_mask/{exp_id}/results.json"
        algo_dir = ROOT / f"results/add_states/{sc}/ppo_mask"
        link = algo_dir / f"tc{tc}_{steps}"

        if not result_path.exists():
            pending.append(f"{sc} tc={tc} steps={steps} exp_id={exp_id}")
            continue

        d = json.loads(result_path.read_text())
        if d.get("train_scenario_count") != tc or d.get("training", {}).get("requested_total_steps") != steps:
            d["train_scenario_count"] = tc
            d.setdefault("training", {})["requested_total_steps"] = steps
            result_path.write_text(json.dumps(d, indent=2, ensure_ascii=False))

        if not link.is_symlink() or link.resolve() != (algo_dir / exp_id).resolve():
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(Path(exp_id))

        a = d["maskable_ppo"]
        groups[sc].append({
            "train_count": tc, "steps": steps, "exp_id": exp_id,
            "success_rate": a.get("success_rate"), "ar_mean": a.get("ar_mean"),
            "ar_std": a.get("ar_std"), "ilp_ar": d["ilp"]["ar"],
        })

    lines = [f"live merge — {sum(len(v) for v in groups.values())}/24 done, {len(pending)} pending", ""]
    lines.append(f"{'scenario':<10}{'train_count':<13}{'steps':<12}{'success%':>10}{'ar_mean':>10}{'ar_gap':>10}")
    out = {}
    for sc in ("lt", "eq", "gt"):
        rows = groups.get(sc, [])
        out[sc] = rows
        for r in sorted(rows, key=lambda x: (x["steps"], x["train_count"])):
            gap = r["ilp_ar"] - r["ar_mean"] if r["ar_mean"] is not None else None
            lines.append(
                f"{sc:<10}{r['train_count']:<13}{r['steps']:<12}"
                f"{(r['success_rate'] or 0)*100:>9.1f}%{r['ar_mean'] or 0:>10.4f}{gap or 0:>10.4f}"
            )
    if pending:
        lines.append("")
        lines.append("pending: " + "; ".join(pending))

    (LOG_DIR / "live_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    text = "\n".join(lines)
    (LOG_DIR / "live_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
