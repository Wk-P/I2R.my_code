#!/usr/bin/env bash
# lt场景下ppo/ppo_lagrangian/ppo_opt/dqn/ddqn（移植了ppo_mask v2.6.0分级reward后）
# 补齐 seed 1,2,3,4（seed42 已经跑过，见 logs/lt_other_algos_5M/），
# 凑齐5-seed统计，验证"success_rate从0跳到27%~80%"不是单次运气。
#
# 并发节流：每进程内部约10线程，机器56核，最多同时跑 MAX_CONCURRENT=5 个，
# 用 wait -n 节流，不一次性全丢出去。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ALGOS=(ppo ppo_lagrangian ppo_opt dqn ddqn)
SEEDS=(1 2 3 4)
MAX_CONCURRENT=5

LOG_DIR="$ROOT/scripts/logs/lt_other_algos_5M"
mkdir -p "$LOG_DIR"

declare -a RUNS=()
running=0

launch() {
  local algo="$1" seed="$2"
  local exp_id
  exp_id="$("$ROOT/.venv/bin/python" -c 'from shared.paths import new_exp_id; print(new_exp_id())')"
  local log_file="$LOG_DIR/${algo}_5000000_seed${seed}_${exp_id}.log"
  env PYTHONUNBUFFERED=1 TRAIN_SEED="$seed" EXP_ID="$exp_id" \
    "$ROOT/.venv/bin/python" -u "scenarios/lt/$algo/run_all.py" --total-timesteps 5000000 \
    > "$log_file" 2>&1 < /dev/null &
  RUNS+=("$algo:$seed:$exp_id")
  echo "  started algo=$algo seed=$seed exp_id=$exp_id pid=$! ($((${#RUNS[@]})) / 20)"
}

echo "=== lt_other_algos_5seed 开始 $(date)  max_concurrent=$MAX_CONCURRENT ==="

# round-robin: seed外层、algo内层，让5个算法从一开始就同时占并发槽
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
