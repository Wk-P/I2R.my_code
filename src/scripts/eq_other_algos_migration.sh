#!/usr/bin/env bash
# eq场景下ppo/ppo_lagrangian/ppo_opt/dqn/ddqn移植分级reward(lt的M*(2AR-1)公式)后的
# 首轮验证跑，seed=42单seed，先确认迁移有效再补多seed。仿照
# src/scripts/lt_other_algos_5seed.sh的写法。eq默认TOTAL_STEPS=2M(见
# src/shared/training_steps_config.py，未被SCENARIO_TOTAL_STEPS覆盖)。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ALGOS=(ppo ppo_lagrangian ppo_opt dqn ddqn)
SEEDS=(42)
MAX_CONCURRENT=5

LOG_DIR="$ROOT/scripts/logs/eq_other_algos_2M"
mkdir -p "$LOG_DIR"

declare -a RUNS=()
running=0

launch() {
  local algo="$1" seed="$2"
  local exp_id
  exp_id="$("$ROOT/.venv/bin/python" -c 'from shared.paths import new_exp_id; print(new_exp_id())')"
  local log_file="$LOG_DIR/${algo}_2000000_seed${seed}_${exp_id}.log"
  env PYTHONUNBUFFERED=1 TRAIN_SEED="$seed" EXP_ID="$exp_id" \
    "$ROOT/.venv/bin/python" -u "scenarios/eq/$algo/run_all.py" --total-timesteps 2000000 \
    > "$log_file" 2>&1 < /dev/null &
  RUNS+=("$algo:$seed:$exp_id")
  echo "  started algo=$algo seed=$seed exp_id=$exp_id pid=$! ($((${#RUNS[@]})))"
}

echo "=== eq_other_algos_migration 开始 $(date)  max_concurrent=$MAX_CONCURRENT ==="

for seed in "${SEEDS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    if [ "$running" -ge "$MAX_CONCURRENT" ]; then
      wait -n
      running=$((running - 1))
    fi
    launch "$algo" "$seed"
    running=$((running + 1))
  done
done

echo "=== 全部 ${#RUNS[@]} 个任务已提交，等待剩余完成 ... $(date) ==="
wait
echo "=== 全部完成 $(date) ==="
