#!/usr/bin/env bash
# 后台启动单个 scenario/algo 的 run_all.py，与终端/SSH/VSCode 会话生命周期解耦。
#
# 用法: src/scripts/train_bg.sh <scenario> <algo>
#   scenario: eq | gt | lt
#   algo:     ppo | ppo_mask | ppo_lagrangian | ppo_opt | dqn | ddqn
#
# 用 setsid 起新 session + nohup 忽略 SIGHUP + disown，
# 这样父进程（Claude 会话 / SSH / VSCode）退出不会连带杀死训练进程。
#
# 日志写到 logs/<scenario>_<algo>_<timestamp>.log，
# 并自动更新 app/backend/log_sources.json 供看板读取实时进度。

set -euo pipefail

SCENARIO="${1:?用法: $0 <scenario> <algo>}"
ALGO="${2:?用法: $0 <scenario> <algo>}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT/scenarios/$SCENARIO/$ALGO/run_all.py"
[ -f "$SCRIPT" ] || { echo "找不到脚本: $SCRIPT" >&2; exit 1; }

LOG_DIR="$ROOT/scripts/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${SCENARIO}_${ALGO}_${TS}.log"

cd "$ROOT"
setsid nohup env PYTHONUNBUFFERED=1 "$ROOT/.venv/bin/python" -u "$SCRIPT" > "$LOG_FILE" 2>&1 < /dev/null &
disown
PID=$!

echo "started pid=$PID  scenario=$SCENARIO algo=$ALGO  log=$LOG_FILE"

# 更新看板的 log_sources.json（每个 scenario 一个 key，指向其最新日志）
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
