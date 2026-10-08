#!/usr/bin/env bash
# 论文验证实验驱动脚本 —— 在 paper-verfication 分支上跑四个累加式版本：
#   v1.0.1  熵系数 bug 修复（统一为静态 0.005，无调度）
#   v1.0.2  在 v1.0.1 基础上加熵系数调度（前高后低）
#   v1.0.3  在 v1.0.2 基础上加负优势样本剪枝（prune_weight=0.1）
#   v1.0.4  在 v1.0.3 基础上调优剪枝力度（prune_weight=0.3，更温和）
#
# 每个版本跑 3 场景(lt/eq/gt) × 2 算法(ppo_mask/ppo_lagrangian) = 6 次训练，
# 顺序执行（与 shared host 上其它编排脚本一致，避免抢核）。
# 每个版本跑完后：写 results/paper-verfication/<version>_summary.json、
# 打 git tag、发邮件通知（含 AR 结果 vs ILP 摘要）。
#
# 用法: nohup src/scripts/run_paper_verification.sh > logs/paper_verification.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SCENARIOS=(lt eq gt)
ALGOS=(ppo_mask ppo_lagrangian)

# version_tag : ENT_COEF_INIT : ENT_COEF_FINAL : ADV_PRUNE_WEIGHT : description
VERSIONS=(
  "v1.0.1:0.005:0.005:1.0:熵系数bug修复（静态0.005对齐，无调度）"
  "v1.0.2:0.02:0.002:1.0:熵系数调度（前高后低，缓解PPO探索过早收敛）"
  "v1.0.3:0.02:0.002:0.1:负优势样本剪枝（CMA-ES风格，权重0.1）"
  "v1.0.4:0.02:0.002:0.3:剪枝力度调优（权重0.3，更温和）"
)

for entry in "${VERSIONS[@]}"; do
  IFS=":" read -r VTAG ENT_INIT ENT_FINAL PRUNE DESC <<< "$entry"
  PAPER_VERSION="${VTAG#v}"
  export ENT_COEF_INIT="$ENT_INIT" ENT_COEF_FINAL="$ENT_FINAL" ADV_PRUNE_WEIGHT="$PRUNE" PAPER_VERSION

  EXP_ID="$(.venv/bin/python -c 'from shared.paths import new_exp_id; print(new_exp_id())')"
  export EXP_ID

  echo "=== [$VTAG] 开始  ($DESC)  exp_id=$EXP_ID  ent=[$ENT_INIT->$ENT_FINAL] prune=$PRUNE  $(date) ==="

  for scenario in "${SCENARIOS[@]}"; do
    for algo in "${ALGOS[@]}"; do
      script="$ROOT/scenarios/$scenario/$algo/run_all.py"
      echo "  -- [$VTAG] $scenario/$algo 开始 $(date)"
      PYTHONUNBUFFERED=1 "$ROOT/.venv/bin/python" -u "$script"
      echo "  -- [$VTAG] $scenario/$algo 完成 $(date)"
    done
  done

  # 汇总本版本 6 次训练的 AR 结果 vs ILP
  SUMMARY="$("$ROOT/.venv/bin/python" - "$VTAG" "$DESC" "$EXP_ID" <<'PY'
import json, sys
from pathlib import Path

vtag, desc, exp_id = sys.argv[1], sys.argv[2], sys.argv[3]
root = Path("results/paper-verfication")
lines = [f"{vtag}  {desc}", f"exp_id={exp_id}", ""]
for scenario in ("lt", "eq", "gt"):
    for algo in ("ppo_mask", "ppo_lagrangian"):
        p = root / scenario / algo / exp_id / "results.json"
        if not p.exists():
            lines.append(f"{scenario}/{algo}: 结果文件缺失 ({p})")
            continue
        d = json.loads(p.read_text())
        ilp_ar = d["ilp"]["ar"]
        key = "maskable_ppo" if algo == "ppo_mask" else "lagrange_ppo"
        ar = d[key]["ar_mean"]
        gap = ilp_ar - ar
        lines.append(f"{scenario}/{algo}: AR={ar:.4f}  ILP={ilp_ar:.4f}  gap={gap:.4f}")

summary_path = root / f"{vtag}_summary.txt"
summary_path.parent.mkdir(parents=True, exist_ok=True)
text = "\n".join(lines)
summary_path.write_text(text + "\n")
print(text)
PY
)"

  echo "$SUMMARY"

  # 打 tag（同一份代码，tag 标记"截至此版本的结果集"）
  git tag -a "$VTAG" -m "$DESC

$SUMMARY" 2>&1 || echo "  [警告] git tag $VTAG 失败（可能已存在），跳过"

  "$ROOT/.venv/bin/python" -c "
from shared.notify import send_notification
send_notification(
    subject='[paper-verfication] $VTAG 已完成',
    body='''$SUMMARY''',
)
"

  echo "=== [$VTAG] 完成  $(date) ==="
done

# 全部完成后的总汇总邮件
FINAL_SUMMARY="$("$ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path
root = Path("results/paper-verfication")
parts = []
for f in sorted(root.glob("v1.0.*_summary.txt")):
    parts.append(f.read_text())
print("\n\n".join(parts))
PY
)"

"$ROOT/.venv/bin/python" -c "
from shared.notify import send_notification
send_notification(
    subject='[paper-verfication] 四个版本全部训练完成',
    body='''$FINAL_SUMMARY''',
)
"

echo "=== 全部版本训练完成 $(date) ==="
