#!/usr/bin/env python3
"""Auto-report generator for final_paper_experiments campaigns.

Reads a campaign manifest (written by run_v4.1.0_2M_campaign.py, mapping
scenario -> algo -> seed -> exp_id), aggregates each run's summary.csv into
a per-(scenario, algo) mean+/-std table, and writes a self-contained
Markdown report to paper_contents/v4.1.0_reports/.

This script is invoked automatically at the end of every full campaign run
on the final_paper_experiments branch (per project requirement) but can
also be run manually against a partial manifest to check progress:

    .venv/bin/python scripts/generate_v4.1.0_report.py --manifest scripts/logs/v4.1.0_2M_campaign/manifest.json
"""
import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

REPORT_DIR = PROJECT_ROOT / "paper_contents" / "v4.1.0_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ALGO_LABELS = {
    "ppo": "PPO (P3, unconstrained baseline — unchanged in v4.1.0)",
    "ppo_mask": "MaskablePPO (P4)",
    "ppo_lagrangian": "Lagrange PPO (P5)",
    "ppo_opt": "PPO + Repair (P6)",
    "dqn": "DQN",
    "ddqn": "DDQN",
}

FIXED_ALGOS = {"ppo_mask", "ppo_lagrangian", "ppo_opt", "dqn", "ddqn"}

# v4.0.0 frozen baseline (results/add_states, 5M steps, 5 seeds) exp_ids are
# not tracked by a manifest -- read whatever exp_id directories exist there
# for reference-only comparison. Different step budget/seed count, so this
# is directional context, not an apples-to-apples ablation.
V4_0_0_RESULTS_ROOT = PROJECT_ROOT / "results" / "add_states"


def read_summary_row(exp_dir: Path) -> dict | None:
    """Second row of summary.csv is the trained method (first row is always
    the ILP optimum)."""
    csv_path = exp_dir / "summary.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        return None
    return rows[1]


def read_ilp_row(exp_dir: Path) -> dict | None:
    csv_path = exp_dir / "summary.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def fmt_mean_std(values: list[float]) -> str:
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f}±{statistics.stdev(values):.4f}"


def collect_v4_0_0_reference(scenario: str, algo: str) -> dict | None:
    """Best-effort: scan results/add_states/<scenario>/<algo>/*/summary.csv,
    pool all exp_id dirs found (no manifest for the frozen run), used only
    as directional before/after context in the report."""
    algo_dir = V4_0_0_RESULTS_ROOT / scenario / algo
    if not algo_dir.is_dir():
        return None
    ars, success, cap_v, conflict_v = [], [], [], []
    for exp_dir in algo_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        row = read_summary_row(exp_dir)
        if row is None:
            continue
        try:
            ars.append(float(row["ar_mean"]))
            success.append(float(row["success_rate"]))
            cap_v.append(float(row["cap_viol_rate"]))
            conflict_v.append(float(row["conflict_viol_rate"]))
        except (KeyError, ValueError):
            continue
    if not ars:
        return None
    return {
        "ar": fmt_mean_std(ars),
        "success_rate": fmt_mean_std(success),
        "cap_viol_rate": fmt_mean_std(cap_v),
        "conflict_viol_rate": fmt_mean_std(conflict_v),
        "n_runs_pooled": len(ars),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    exp_ids = manifest["exp_ids"]
    scenarios = manifest["scenarios"]
    algos = manifest["algos"]
    seeds = manifest["seeds"]
    steps = manifest["steps"]
    branch = manifest.get("branch", "final_paper_experiments")

    results_root = PROJECT_ROOT / "results" / branch

    lines = []
    lines.append(f"# v4.1.0 补充实验报告 — 自动生成\n")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(
        f"- 规模：{len(seeds)} 种子 × {len(scenarios)} 场景 × {len(algos)} 算法 "
        f"= {len(seeds)*len(scenarios)*len(algos)} 次训练，每次 {steps:,} 步\n"
        f"- 分支：`{branch}`，结果目录：`results/{branch}/`\n"
        f"- 代码改动：见 `paper_contents/v4.1.0_changelog.md`（"
        f"ppo_mask/ppo_lagrangian/ppo_opt/dqn/ddqn 接入此前从未生效的逐步违规惩罚；"
        f"`ppo`(P3) 未改动，作为不变基线）\n"
        f"- **⚠️ 与冻结的 v4.0.0（results/add_states，5M步/5种子）不是同一预算，"
        f"下表『v4.0.0参考』列仅供方向性对比，不是严格消融实验**\n"
    )
    lines.append("---\n")

    missing = []
    for scenario in scenarios:
        lines.append(f"\n## 场景：{scenario}\n")
        # ILP reference from any one completed exp_id in this scenario
        ilp_val = None
        lines.append(
            "| 算法 | v4.1.0 AR(均值±标准差) | v4.1.0 success_rate | "
            "v4.1.0 cap_viol_rate | v4.1.0 conflict_viol_rate | "
            "v4.0.0参考 AR | v4.0.0参考 success_rate | v4.0.0参考违规率(cap/conflict) |\n"
        )
        lines.append("|---|---|---|---|---|---|---|---|\n")

        for algo in algos:
            seed_exp_ids = exp_ids.get(scenario, {}).get(algo, {})
            ars, success, cap_v, conflict_v = [], [], [], []
            for seed in seeds:
                exp_id = seed_exp_ids.get(str(seed))
                if exp_id is None:
                    missing.append(f"{scenario}/{algo}/seed{seed} (未在manifest中找到exp_id)")
                    continue
                exp_dir = results_root / scenario / algo / exp_id
                row = read_summary_row(exp_dir)
                if row is None:
                    missing.append(f"{scenario}/{algo}/seed{seed} exp_id={exp_id} (summary.csv缺失或未完成)")
                    continue
                if ilp_val is None:
                    ilp_row = read_ilp_row(exp_dir)
                    if ilp_row:
                        ilp_val = ilp_row.get("ar_mean")
                try:
                    ars.append(float(row["ar_mean"]))
                    success.append(float(row["success_rate"]))
                    cap_v.append(float(row["cap_viol_rate"]))
                    conflict_v.append(float(row["conflict_viol_rate"]))
                except (KeyError, ValueError):
                    missing.append(f"{scenario}/{algo}/seed{seed} exp_id={exp_id} (字段解析失败)")

            v4_0_0_ref = collect_v4_0_0_reference(scenario, algo)
            fixed_tag = " 🔧" if algo in FIXED_ALGOS else ""
            label = ALGO_LABELS.get(algo, algo) + fixed_tag

            if v4_0_0_ref:
                ref_ar = v4_0_0_ref["ar"]
                ref_success = v4_0_0_ref["success_rate"]
                ref_viol = f"{v4_0_0_ref['cap_viol_rate']} / {v4_0_0_ref['conflict_viol_rate']}"
            else:
                ref_ar = ref_success = ref_viol = "N/A"

            lines.append(
                f"| {label} | {fmt_mean_std(ars)} | {fmt_mean_std(success)} | "
                f"{fmt_mean_std(cap_v)} | {fmt_mean_std(conflict_v)} | "
                f"{ref_ar} | {ref_success} | {ref_viol} |\n"
            )

        if ilp_val:
            lines.append(f"\nILP最优参考（本轮场景，单次）：AR={ilp_val}\n")

    lines.append("\n---\n\n## 数据完整性\n\n")
    if missing:
        lines.append(f"⚠️ {len(missing)} 个 run 缺失或未完成：\n\n")
        for m in missing:
            lines.append(f"- {m}\n")
    else:
        lines.append("全部 run 数据完整，无缺失。\n")

    lines.append(
        "\n🔧 = 本次修复了死代码惩罚(v4.1.0)的算法；`ppo`(P3) 未标记，"
        "代码与v4.0.0完全一致，只是重跑在了2M步/3种子预算下作为同一批次的参照。\n"
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORT_DIR / f"report_{ts}.md"
    out_path.write_text("".join(lines), encoding="utf-8")

    latest_path = REPORT_DIR / "latest.md"
    latest_path.write_text("".join(lines), encoding="utf-8")

    print(f"Report written to {out_path}")
    print(f"Latest symlink-equivalent updated: {latest_path}")
    if missing:
        print(f"WARNING: {len(missing)} runs missing/incomplete — see report for details")


if __name__ == "__main__":
    main()
