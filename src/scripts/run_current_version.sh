#!/usr/bin/env bash
# 论文验证实验驱动脚本 —— 跑"当前版本"（由 src/shared/version_config.py 的
# CURRENT_VERSION 统一定义，不在本脚本或任何调用点里写死版本号）。
#
# 与 run_paper_verification.sh 的区别：那个脚本专门用来复现 v1.0.1..v1.0.4
# 四个历史消融版本，每个版本的熵系数/剪枝力度都不同，因此各自的版本号是
# 该脚本本身的研究设计，刻意写死；这个脚本跑的是"当下这一版代码"，版本号
# 唯一来源是 src/shared/version_config.py::CURRENT_VERSION，改版本只需要改那一
# 个文件，不需要动这里或 src/shared/paths.py。
#
# 跑 3 场景(lt/eq/gt) × 2 算法(ppo_mask/ppo_lagrangian) 的 BC pipeline
# (run_all_bc.py)，用 config.py 里的默认超参（不覆盖 ENT_COEF/ADV_PRUNE_WEIGHT/
# TOTAL_STEPS，全部走各自 config.py 当前的默认值）。跑完后：写
# results/paper-verfication/<version>_summary.json、打 git tag、发邮件通知。
#
# 用法: nohup src/scripts/run_current_version.sh > logs/current_version.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SCENARIOS=(lt eq gt)
ALGOS=(ppo_mask ppo_lagrangian)

VERSION="$("$ROOT/.venv/bin/python" -c 'from shared.version_config import CURRENT_VERSION; print(CURRENT_VERSION)')"
VTAG="v${VERSION}"
DESC="当前版本（版本号来自 src/shared/version_config.py::CURRENT_VERSION）"

EXP_ID="$("$ROOT/.venv/bin/python" -c 'from shared.paths import new_exp_id; print(new_exp_id())')"
export EXP_ID

echo "=== [$VTAG] 开始  ($DESC)  exp_id=$EXP_ID  $(date) ==="

for scenario in "${SCENARIOS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    script="$ROOT/scenarios/$scenario/$algo/run_all_bc.py"
    echo "  -- [$VTAG] $scenario/$algo 开始 $(date)"
    PYTHONUNBUFFERED=1 "$ROOT/.venv/bin/python" -u "$script"
    echo "  -- [$VTAG] $scenario/$algo 完成 $(date)"
  done
done

# 汇总本版本各 scenario/algo 的 AR 结果 vs ILP
SUMMARY="$("$ROOT/.venv/bin/python" - "$VTAG" "$DESC" "$EXP_ID" <<'PY'
import json, sys
from pathlib import Path

vtag, desc, exp_id = sys.argv[1], sys.argv[2], sys.argv[3]
root = Path("results/paper-verfication")
lines = [f"{vtag}  {desc}", f"exp_id={exp_id}", ""]
for scenario in ("lt", "eq", "gt"):
    for algo in ("ppo_mask", "ppo_lagrangian"):
        p = root / scenario / algo / f"{exp_id}_bc" / "results.json"
        if not p.exists():
            lines.append(f"{scenario}/{algo}: 结果文件缺失 ({p})")
            continue
        d = json.loads(p.read_text())
        ilp_ar = d["ilp"]["ar"]
        key = "maskable_ppo_bc" if algo == "ppo_mask" else "lagrange_ppo_bc"
        ar = d[key]["ar_mean"] if key in d else None
        if ar is None:
            lines.append(f"{scenario}/{algo}: 结果字段缺失 (key={key})")
            continue
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

# 打 tag（若该版本已经打过 tag，跳过而不报错中断）
git tag -a "$VTAG" -m "$DESC

$SUMMARY" 2>&1 || echo "  [警告] git tag $VTAG 失败（可能已存在），跳过"

"$ROOT/.venv/bin/python" -c "
from shared.notify import send_notification
send_notification(
    subject='[paper-verfication] $VTAG 当前版本实验已完成',
    body='''$SUMMARY''',
)
"

echo "=== [$VTAG] 完成  $(date) ==="
