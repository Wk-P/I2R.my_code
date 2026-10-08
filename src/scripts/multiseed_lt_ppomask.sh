#!/usr/bin/env bash
# v1.2.4 诊断实验：lt/ppo_mask 用同一套配置换 5 个不同 TRAIN_SEED 各跑一遍
# run_all_bc.py（BC pipeline，与 src/scripts/run_current_version.sh 跑的流程一致），
# 目的是量化 success_rate 的 run-to-run 方差 —— 之前所有数字都来自单一
# seed=42，不知道"卡在某个数字上不去"是系统性瓶颈还是 seed 运气。
#
# TRAIN_SEED 同时控制 train/test 划分和 RL 训练随机性（见
# scenarios/lt/ppo_mask/config.py），所以每个 seed 是一次独立的端到端复现，
# 不是同一份数据换个随机种子这么窄。
#
# 5 个 seed 顺序跑（不并行），避免和其他实验抢 CPU/GPU；每个 seed 用独立
# exp_id（不设 EXP_ID，让 new_exp_id() 自己起一个），结果落在
# results/<branch>/lt/ppo_mask/<exp_id>_bc/ 下，互不覆盖。
#
# 用法（和 run_current_version.sh 同一个约定，脚本本身不自我后台化）:
#   nohup src/scripts/multiseed_lt_ppomask.sh > logs/multiseed_lt_ppomask.log 2>&1 &
#   disown

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SEEDS=(42 1 2 3 4)
declare -a RESULT_PAIRS=()

echo "=== [v1.2.4] lt/ppo_mask multi-seed diagnostic 开始  seeds=${SEEDS[*]}  $(date) ==="

for seed in "${SEEDS[@]}"; do
  unset EXP_ID
  export TRAIN_SEED="$seed"
  EXP_ID="$("$ROOT/.venv/bin/python" -c 'from shared.paths import new_exp_id; print(new_exp_id())')"
  export EXP_ID
  echo "  -- [v1.2.4] seed=$seed exp_id=$EXP_ID starting $(date)"
  PYTHONUNBUFFERED=1 "$ROOT/.venv/bin/python" -u "$ROOT/scenarios/lt/ppo_mask/run_all_bc.py"
  echo "  -- [v1.2.4] seed=$seed exp_id=$EXP_ID finished $(date)"
  RESULT_PAIRS+=("$seed:$EXP_ID")
done

SUMMARY="$("$ROOT/.venv/bin/python" - "${RESULT_PAIRS[@]}" <<'PY'
import json, sys
from pathlib import Path

pairs = [r.split(":") for r in sys.argv[1:]]
root = Path("results/paper-verfication/lt/ppo_mask")
lines = ["v1.2.4 lt/ppo_mask multi-seed diagnostic", ""]
for seed, exp_id in pairs:
    p = root / f"{exp_id}_bc" / "results.json"
    if not p.exists():
        lines.append(f"seed={seed} exp_id={exp_id}: 结果文件缺失 ({p})")
        continue
    d = json.loads(p.read_text())
    ev = d.get("maskable_ppo_bc", {})
    lines.append(
        f"seed={seed} exp_id={exp_id}: success_rate={ev.get('success_rate')} "
        f"ar_mean={ev.get('ar_mean')} ar_std={ev.get('ar_std')}"
    )

text = "\n".join(lines)
out = root / "v1.2.4_multiseed_summary.txt"
out.write_text(text + "\n")
print(text)
PY
)"

echo "$SUMMARY"

"$ROOT/.venv/bin/python" -c "
from shared.notify import send_notification
send_notification(
    subject='[training] v1.2.4 lt/ppo_mask multi-seed 诊断实验已完成',
    body='''$SUMMARY''',
)
"

echo "=== [v1.2.4] lt/ppo_mask multi-seed diagnostic 全部完成  $(date) ==="
