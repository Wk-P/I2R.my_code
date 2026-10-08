#!/usr/bin/env bash
# 启动一整批实验（eq + gt + lt 一起跑），三个 scenario 共用同一个 exp_id ——
# 这代表"同一次实验"，而不是每个 scenario/algo 各自一个随机 id。
#
# 用法: src/scripts/start_experiment.sh [algo1 algo2 ... | bc | bc-compare]
#   不传 algo 列表时默认: ppo ppo_mask ppo_lagrangian ppo_opt dqn ddqn（全跑，18 模型）
#   algo 名字加 "_bc" 后缀（如 ppo_mask_bc dqn_bc）会跑 ILP 行为克隆预训练
#   变体（run_all_bc.py），见 src/scripts/resume_scenario.sh
#
#   两个简便参数（省得手打一长串 algo 名）：
#     bc          用 BC 变体替换掉 5 个支持 BC 的算法（ppo 保持不变），18 模型
#     bc-compare  baseline 和 BC 都跑，两两对比，33 模型

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ALGOS=("$@")
if [ "${#ALGOS[@]}" -eq 0 ]; then
  ALGOS=(ppo ppo_mask ppo_lagrangian ppo_opt dqn ddqn)
elif [ "${#ALGOS[@]}" -eq 1 ] && [ "${ALGOS[0]}" = "bc" ]; then
  ALGOS=(ppo ppo_mask_bc ppo_lagrangian_bc ppo_opt_bc dqn_bc ddqn_bc)
elif [ "${#ALGOS[@]}" -eq 1 ] && [ "${ALGOS[0]}" = "bc-compare" ]; then
  ALGOS=(ppo ppo_mask ppo_mask_bc ppo_lagrangian ppo_lagrangian_bc ppo_opt ppo_opt_bc dqn dqn_bc ddqn ddqn_bc)
fi

export EXP_ID="$(cd "$ROOT" && .venv/bin/python -c 'from shared.paths import new_exp_id; print(new_exp_id())')"
echo "=== 本次实验 EXP_ID = $EXP_ID（eq/gt/lt 共用） ==="

for scenario in eq gt lt; do
  "$ROOT/scripts/resume_scenario.sh" "$scenario" "${ALGOS[@]}"
done
