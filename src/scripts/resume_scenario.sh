#!/usr/bin/env bash
# 依次跑完一个 scenario 剩下的算法（默认从 ppo_lagrangian 开始，因为
# ppo / ppo_mask 通常已经跑完），整个序列跑在一个和会话解耦的后台进程里。
#
# 用法: src/scripts/resume_scenario.sh <scenario> [algo1 algo2 ...]
#   不传 algo 列表时默认: ppo_lagrangian ppo_opt dqn ddqn
#
# algo 名字加 "_bc" 后缀（如 ppo_mask_bc）会改跑该算法目录下的
# run_all_bc.py（ILP 行为克隆预训练变体，见 src/shared/bc_pretrain.py），
# 而不是普通的 run_all.py —— 两者仍然共用同一个 EXP_ID，同一批实验里
# baseline 和 BC 结果会正确归到一张卡片下。

set -euo pipefail

SCENARIO="${1:?用法: $0 <scenario> [algo1 algo2 ...]}"
shift || true
ALGOS=("$@")
if [ "${#ALGOS[@]}" -eq 0 ]; then
  ALGOS=(ppo_lagrangian ppo_opt dqn ddqn)
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$ROOT/scripts/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${SCENARIO}_sequence_${TS}.log"

# One exp_id per experiment batch, shared by every scenario/algo launched
# together. If a caller (src/scripts/start_experiment.sh) already exported
# EXP_ID, every algo in this scenario's sequence inherits that same id;
# otherwise mint one here so a lone `resume_scenario.sh eq ...` call still
# has all 6 of its algos share a single id instead of each getting its own.
if [ -z "${EXP_ID:-}" ]; then
  EXP_ID="$(cd "$ROOT" && .venv/bin/python -c 'from shared.paths import new_exp_id; print(new_exp_id())')"
fi

runner() {
  cd "$ROOT"
  for algo in "${ALGOS[@]}"; do
    if [[ "$algo" == *_bc ]]; then
      script="$ROOT/scenarios/$SCENARIO/${algo%_bc}/run_all_bc.py"
    else
      script="$ROOT/scenarios/$SCENARIO/$algo/run_all.py"
    fi
    if [ ! -f "$script" ]; then
      echo "=== [$algo] skip: $script not found ==="
      continue
    fi
    echo "=== [$SCENARIO] starting $algo at $(date) ==="
    PYTHONUNBUFFERED=1 "$ROOT/.venv/bin/python" -u "$script"
    echo "=== [$SCENARIO] finished $algo at $(date) ==="
  done
  "$ROOT/.venv/bin/python" -c "
from shared.notify import send_notification
send_notification(
    subject='[training] $SCENARIO 场景全部算法已完成',
    body='scenario=$SCENARIO exp_id=$EXP_ID algos=${ALGOS[*]} 于 $(date) 完成。日志: $LOG_FILE',
)
"
}

export ROOT SCENARIO EXP_ID
setsid nohup bash -c "$(declare -f runner); ALGOS=($(printf '%q ' "${ALGOS[@]}")); runner" \
  > "$LOG_FILE" 2>&1 < /dev/null &
disown
PID=$!

echo "started pid=$PID  scenario=$SCENARIO  exp_id=$EXP_ID  algos=${ALGOS[*]}  log=$LOG_FILE"

python3 - "$SCENARIO" "$LOG_FILE" <<'PY'
import json, sys
from pathlib import Path

scenario, log_file = sys.argv[1], sys.argv[2]
src = Path("app/backend/log_sources.json")
data = json.loads(src.read_text()) if src.exists() else {}
data[scenario] = log_file
src.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
PY

echo "log_sources.json 已更新: $SCENARIO -> $LOG_FILE"
