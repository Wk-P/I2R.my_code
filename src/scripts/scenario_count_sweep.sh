#!/usr/bin/env bash
# 回答"训练场景数量是否越多越好"：lt/eq/gt 各在 train_count ∈ {40,80,120,160}
# （测试集固定不变，见 config.py 的 TRAIN_SCENARIO_COUNT）× steps ∈ {2M,5M}
# 下各跑一次（单 seed=42，seed 统计留到这一步问题回答完之后再补），
# 一共 3×4×2=24 组。
#
# 并发节流：每个训练进程内部 TORCH_NUM_THREADS=10（config.py 里写死），
# 机器 56 核，所以最多同时跑 MAX_CONCURRENT=5 个（5*10=50 核，留余量），
# 用 `wait -n` 做节流，不会重演一次性起 30 个进程导致 CPU 严重超订、
# 训练速度掉了 50 倍、跑了 16 小时都没跑完的问题。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SCENARIOS=(lt eq gt)
TRAIN_COUNTS=(40 80 120 160)
STEP_SETS=(2000000 5000000)
SEED=42
MAX_CONCURRENT=5

LOG_DIR="$ROOT/scripts/logs/scenario_count_sweep"
mkdir -p "$LOG_DIR"

declare -a RUNS=()
running=0

launch() {
  local sc="$1" tc="$2" steps="$3"
  local exp_id
  exp_id="$("$ROOT/.venv/bin/python" -c 'from shared.paths import new_exp_id; print(new_exp_id())')"
  local log_file="$LOG_DIR/${sc}_tc${tc}_${steps}_${exp_id}.log"
  env PYTHONUNBUFFERED=1 TRAIN_SEED="$SEED" TRAIN_SCENARIO_COUNT="$tc" EXP_ID="$exp_id" \
    "$ROOT/.venv/bin/python" -u "scenarios/$sc/ppo_mask/run_all.py" --total-timesteps "$steps" \
    > "$log_file" 2>&1 < /dev/null &
  RUNS+=("$sc:$tc:$steps:$SEED:$exp_id")
  echo "  started scenario=$sc train_count=$tc steps=$steps exp_id=$exp_id pid=$! ($((${#RUNS[@]})) / 24)"
}

echo "=== scenario_count_sweep 开始 $(date)  max_concurrent=$MAX_CONCURRENT ==="

# round-robin 交叉遍历（先 scenario、再 steps、再 train_count 变化最慢），
# 这样并发槽从一开始就同时分给 lt/eq/gt，不会先把一个 scenario 的组合
# 全占满槽位、其它 scenario 迟迟排不上。
for tc in "${TRAIN_COUNTS[@]}"; do
  for steps in "${STEP_SETS[@]}"; do
    for sc in "${SCENARIOS[@]}"; do
      if [ "$running" -ge "$MAX_CONCURRENT" ]; then
        wait -n
        running=$((running - 1))
      fi
      launch "$sc" "$tc" "$steps"
      running=$((running + 1))
    done
  done
done

echo "=== 全部 24 个任务已提交，等待剩余完成 ... $(date) ==="
wait
echo "=== 全部完成 $(date)，开始汇总 ==="

SUMMARY_JSON="$LOG_DIR/scenario_count_sweep_summary.json"
"$ROOT/.venv/bin/python" - "${RUNS[@]}" <<'PY' > "$LOG_DIR/scenario_count_sweep_summary.txt"
import json, sys
from pathlib import Path
from collections import defaultdict

runs = [r.split(":") for r in sys.argv[1:]]
groups = defaultdict(list)

for sc, tc, steps, seed, exp_id in runs:
    p = Path(f"results/add_states/{sc}/ppo_mask/{exp_id}/results.json")
    if not p.exists():
        print(f"[missing] {sc} train_count={tc} steps={steps} exp_id={exp_id}: {p} not found")
        continue
    d = json.loads(p.read_text())
    a = d["maskable_ppo"]
    d["train_seed"] = int(seed)
    d["train_scenario_count"] = int(tc)
    d.setdefault("training", {})["requested_total_steps"] = int(steps)
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    groups[sc].append({
        "train_count": int(tc), "steps": int(steps), "exp_id": exp_id,
        "success_rate": a.get("success_rate"), "ar_mean": a.get("ar_mean"),
        "ar_std": a.get("ar_std"), "ilp_ar": d["ilp"]["ar"],
    })

out = {}
lines = ["scenario  train_count  steps       success%   ar_mean   ar_gap"]
for sc, rows in sorted(groups.items()):
    out[sc] = rows
    for r in sorted(rows, key=lambda x: (x["steps"], x["train_count"])):
        gap = r["ilp_ar"] - r["ar_mean"] if r["ar_mean"] is not None else None
        lines.append(
            f"{sc:<10}{r['train_count']:<13}{r['steps']:<12}"
            f"{(r['success_rate'] or 0)*100:>7.1f}%  {r['ar_mean'] or 0:>7.4f}  {gap or 0:>7.4f}"
        )

Path("logs/scenario_count_sweep/scenario_count_sweep_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("\n".join(lines))
PY

cat "$LOG_DIR/scenario_count_sweep_summary.txt"

"$ROOT/.venv/bin/python" -c "
from shared.notify import send_notification
body = open('$LOG_DIR/scenario_count_sweep_summary.txt').read()
send_notification(
    subject='[training] scenario_count_sweep 全部完成',
    body=body,
)
"

echo "=== scenario_count_sweep 全部完成 $(date) ==="
