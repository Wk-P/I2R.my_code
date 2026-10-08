#!/usr/bin/env bash
# 统一的一次性完整实验:eq/gt场景下ppo/ppo_lagrangian/ppo_opt/dqn/ddqn
# (刚migration完lt的分级reward M*(2AR-1))的5-seed补全跑。
# 口径对齐 src/scripts/lt_other_algos_5seed.sh:同样5个算法、同样seed=1..5、
# 同样MAX_CONCURRENT=5节流。steps用eq/gt各自的默认配置(2M,见
# src/shared/training_steps_config.py，未被SCENARIO_TOTAL_STEPS覆盖)。
# 不再跑单独的单seed(42)验证,之前eq/seed42那5次结果作废,统一用这次的
# seed 1-5数据,避免口径不一致。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SCENARIOS=(eq gt)
ALGOS=(ppo ppo_lagrangian ppo_opt dqn ddqn)
SEEDS=(1 2 3 4 5)
MAX_CONCURRENT=5

LOG_DIR="$ROOT/scripts/logs/eq_gt_migration_5seed"
mkdir -p "$LOG_DIR"

declare -a RUNS=()
running=0

launch() {
  local scenario="$1" algo="$2" seed="$3"
  local exp_id
  exp_id="$("$ROOT/.venv/bin/python" -c 'from shared.paths import new_exp_id; print(new_exp_id())')"
  local log_file="$LOG_DIR/${scenario}_${algo}_2000000_seed${seed}_${exp_id}.log"
  env PYTHONUNBUFFERED=1 TRAIN_SEED="$seed" EXP_ID="$exp_id" \
    "$ROOT/.venv/bin/python" -u "scenarios/$scenario/$algo/run_all.py" --total-timesteps 2000000 \
    > "$log_file" 2>&1 < /dev/null &
  RUNS+=("$scenario:$algo:$seed:$exp_id")
  echo "  started scenario=$scenario algo=$algo seed=$seed exp_id=$exp_id pid=$! ($((${#RUNS[@]})) / 50)"
}

echo "=== eq_gt_migration_5seed 开始 $(date)  max_concurrent=$MAX_CONCURRENT ==="

# round-robin: seed最外层，scenario/algo内层，让并发槽从一开始就打满
for seed in "${SEEDS[@]}"; do
  for scenario in "${SCENARIOS[@]}"; do
    for algo in "${ALGOS[@]}"; do
      if [ "$running" -ge "$MAX_CONCURRENT" ]; then
        wait -n
        running=$((running - 1))
      fi
      launch "$scenario" "$algo" "$seed"
      running=$((running + 1))
    done
  done
done

echo "=== 全部 ${#RUNS[@]} 个任务已提交，等待剩余完成 ... $(date) ==="
wait
echo "=== 全部完成 $(date) ==="
