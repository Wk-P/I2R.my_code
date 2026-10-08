#!/usr/bin/env bash
# lt/eq/gt × {2M, 5M} 全量多seed诊断：每个 (scenario, steps) 组合跑5个seed
# (42,1,2,3,4，沿用 multiseed_lt_ppomask.sh 的约定)，量化 run-to-run 方差，
# 因为单seed对比时观察到"steps多success反而更低"，需要多seed才能判断是
# 真实趋势还是噪声。见 project_v2.5.0_multiseed_2m5m 讨论。
#
# 每个 (scenario, steps, seed) 三元组用独立 exp_id，互不覆盖，
# 结果落在 results/add_states/<scenario>/ppo_mask/<exp_id>/results.json。
# 并发节流：每个训练进程内部 TORCH_NUM_THREADS=10（config.py 里写死），机器
# 56 核，所以最多同时跑 MAX_CONCURRENT=5 个（5*10=50核，留余量），用 `wait -n`
# 节流——不能像早期版本那样一次性把 30 个全丢出去，那样会把 CPU 严重超订，
# 训练速度掉了 50 倍、跑 16 小时都跑不完。
#
# 遍历顺序 steps→seed→scenario（scenario 放最内层）：并发槽从一开始就
# 同时分给 lt/eq/gt，不会先把一个 scenario 的组合占满槽位、其它 scenario
# 迟迟排不上。
#
# 注意：子进程不能 disown——disown 会把进程从当前 shell 的 job table 摘掉，
# 之后的 `wait`/`wait -n` 就等不到它们了（会立刻返回，汇总在训练还没跑完时
# 就执行，拿到的全是"文件不存在"）。这个脚本本身应该整体用 nohup 包起来
# 后台跑（runbook 见 multiseed_lt_ppomask.sh 用法），不需要再给子进程加
# setsid/disown。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SEEDS=(42 1 2 3 4)
SCENARIOS=(lt eq gt)
STEP_SETS=(2000000 5000000)
MAX_CONCURRENT=5

LOG_DIR="$ROOT/scripts/logs/multiseed_2m5m"
mkdir -p "$LOG_DIR"

declare -a RUNS=()   # "scenario:steps:seed:exp_id"
running=0

launch() {
  local sc="$1" steps="$2" seed="$3"
  local exp_id
  exp_id="$("$ROOT/.venv/bin/python" -c 'from shared.paths import new_exp_id; print(new_exp_id())')"
  local log_file="$LOG_DIR/${sc}_${steps}_seed${seed}_${exp_id}.log"
  env PYTHONUNBUFFERED=1 TRAIN_SEED="$seed" EXP_ID="$exp_id" \
    "$ROOT/.venv/bin/python" -u "scenarios/$sc/ppo_mask/run_all.py" --total-timesteps "$steps" \
    > "$log_file" 2>&1 < /dev/null &
  RUNS+=("$sc:$steps:$seed:$exp_id")
  echo "  started scenario=$sc steps=$steps seed=$seed exp_id=$exp_id pid=$! ($((${#RUNS[@]})) / 30)"
}

echo "=== multiseed 2M/5M 全量诊断开始 $(date)  max_concurrent=$MAX_CONCURRENT ==="

for steps in "${STEP_SETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for sc in "${SCENARIOS[@]}"; do
      if [ "$running" -ge "$MAX_CONCURRENT" ]; then
        wait -n
        running=$((running - 1))
      fi
      launch "$sc" "$steps" "$seed"
      running=$((running + 1))
    done
  done
done

echo "=== 全部 ${#RUNS[@]} 个任务已提交，等待剩余完成 ... $(date) ==="
wait
echo "=== 全部任务完成 $(date)，开始汇总 ==="

SUMMARY_JSON="$LOG_DIR/multiseed_2m5m_summary.json"
"$ROOT/.venv/bin/python" - "${RUNS[@]}" <<PY > "$LOG_DIR/multiseed_2m5m_summary.txt"
import json, statistics, sys
from pathlib import Path
from collections import defaultdict

runs = [r.split(":") for r in sys.argv[1:]]
groups = defaultdict(list)

for sc, steps, seed, exp_id in runs:
    p = Path(f"results/add_states/{sc}/ppo_mask/{exp_id}/results.json")
    if not p.exists():
        print(f"[missing] {sc} steps={steps} seed={seed} exp_id={exp_id}: {p} not found")
        continue
    d = json.loads(p.read_text())
    d["train_seed"] = int(seed)
    d.setdefault("training", {})["requested_total_steps"] = int(steps)
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    # 可读软链接：results/add_states/<sc>/ppo_mask/seed<seed>_<steps> -> <exp_id>
    algo_dir = Path(f"results/add_states/{sc}/ppo_mask")
    link = algo_dir / f"seed{seed}_{steps}"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(exp_id))
    a = d["maskable_ppo"]
    groups[(sc, steps)].append({
        "seed": seed, "exp_id": exp_id,
        "success_rate": a.get("success_rate"),
        "ar_mean": a.get("ar_mean"),
        "ar_std": a.get("ar_std"),
        "attempts_mean": a.get("attempts_mean"),
        "ilp_ar": d["ilp"]["ar"],
    })

out = {}
lines = ["scenario  steps      n  success_mean±std      ar_mean±std          seeds"]
for (sc, steps), rows in sorted(groups.items()):
    sr = [r["success_rate"] for r in rows if r["success_rate"] is not None]
    ar = [r["ar_mean"] for r in rows if r["ar_mean"] is not None]
    sr_mean = statistics.mean(sr) if sr else None
    sr_std = statistics.pstdev(sr) if len(sr) > 1 else 0.0
    ar_mean = statistics.mean(ar) if ar else None
    ar_std = statistics.pstdev(ar) if len(ar) > 1 else 0.0
    out[f"{sc}_{steps}"] = {"runs": rows, "success_rate_mean": sr_mean, "success_rate_std": sr_std,
                             "ar_mean_mean": ar_mean, "ar_mean_std": ar_std, "n": len(rows)}
    lines.append(
        f"{sc:<9}{steps:<11}{len(rows):<3}"
        f"{(sr_mean or 0)*100:>6.1f}%±{(sr_std or 0)*100:<6.1f}"
        f"{ar_mean or 0:>8.4f}±{ar_std or 0:<8.4f}   "
        + ",".join(str(r["seed"]) for r in rows)
    )

Path("$SUMMARY_JSON").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("\n".join(lines))
PY

cat "$LOG_DIR/multiseed_2m5m_summary.txt"

"$ROOT/.venv/bin/python" -c "
from shared.notify import send_notification
body = open('$LOG_DIR/multiseed_2m5m_summary.txt').read()
send_notification(
    subject='[training] lt/eq/gt 2M/5M multi-seed 诊断实验全部完成',
    body=body,
)
"

echo "=== multiseed 2M/5M 全量诊断全部完成 $(date) ==="
