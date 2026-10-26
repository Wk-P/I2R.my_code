"""实时合并 multiseed_all_2m5m 的结果，边跑边更新。

扫 logs/multiseed_2m5m/*.log（文件名 {sc}_{steps}_seed{seed}_{exp_id}.log），
对已经产出 results.json 的：写入 train_seed/requested_total_steps 元数据、
建可读软链接 seed<seed>_<steps> -> <exp_id>、汇总进 live_summary.{json,txt}
（按 (scenario, steps) 分组算 success_rate/ar_mean 的 mean±std）。
可重复运行（幂等）。
"""
import json
import re
import statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/soar009/github/my_code")
LOG_DIR = ROOT / "logs/multiseed_2m5m"
PATTERN = re.compile(r"^(lt|eq|gt)_(\d+)_seed(\d+)_([0-9a-f]+)\.log$")


def main():
    groups = defaultdict(list)
    pending = []

    for log in sorted(LOG_DIR.glob("*.log")):
        m = PATTERN.match(log.name)
        if not m:
            continue
        sc, steps, seed, exp_id = m.groups()
        steps, seed = int(steps), int(seed)

        result_path = ROOT / f"results/add_states/{sc}/ppo_mask/{exp_id}/results.json"
        algo_dir = ROOT / f"results/add_states/{sc}/ppo_mask"
        link = algo_dir / f"seed{seed}_{steps}"

        if not result_path.exists():
            pending.append(f"{sc} steps={steps} seed={seed} exp_id={exp_id}")
            continue

        d = json.loads(result_path.read_text())
        if d.get("train_seed") != seed or d.get("training", {}).get("requested_total_steps") != steps:
            d["train_seed"] = seed
            d.setdefault("training", {})["requested_total_steps"] = steps
            result_path.write_text(json.dumps(d, indent=2, ensure_ascii=False))

        if not link.is_symlink() or link.resolve() != (algo_dir / exp_id).resolve():
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(Path(exp_id))

        a = d["maskable_ppo"]
        groups[(sc, steps)].append({
            "seed": seed, "exp_id": exp_id,
            "success_rate": a.get("success_rate"), "ar_mean": a.get("ar_mean"),
            "ilp_ar": d["ilp"]["ar"],
        })

    total_done = sum(len(v) for v in groups.values())
    lines = [f"live merge — {total_done}/30 done, {len(pending)} pending", ""]
    lines.append(f"{'scenario':<10}{'steps':<12}{'n':<3}{'success_mean±std':<22}{'ar_mean±std':<20}seeds")
    out = {}
    for (sc, steps), rows in sorted(groups.items()):
        sr = [r["success_rate"] for r in rows if r["success_rate"] is not None]
        ar = [r["ar_mean"] for r in rows if r["ar_mean"] is not None]
        sr_mean = statistics.mean(sr) if sr else None
        sr_std = statistics.pstdev(sr) if len(sr) > 1 else 0.0
        ar_mean = statistics.mean(ar) if ar else None
        ar_std = statistics.pstdev(ar) if len(ar) > 1 else 0.0
        out[f"{sc}_{steps}"] = {"runs": rows, "success_rate_mean": sr_mean, "success_rate_std": sr_std,
                                 "ar_mean_mean": ar_mean, "ar_mean_std": ar_std, "n": len(rows)}
        lines.append(
            f"{sc:<10}{steps:<12}{len(rows):<3}"
            f"{(sr_mean or 0)*100:>6.1f}%±{(sr_std or 0)*100:<6.1f}      "
            f"{ar_mean or 0:>7.4f}±{ar_std or 0:<7.4f}    "
            + ",".join(str(r["seed"]) for r in sorted(rows, key=lambda x: x["seed"]))
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
