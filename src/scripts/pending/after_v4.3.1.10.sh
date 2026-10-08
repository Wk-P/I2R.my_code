#!/usr/bin/env bash
# Runs once v4.3.1.10 is over: wait for its driver to exit, then
#   1. stop here if any v4.3.1.10 job failed (needs a look / --resume first);
#   2. regenerate the v4.3.1.10 report (AR / ILP AR / AR gap, re-evaluated per instance) and commit it;
#   3. apply the v4.3.4 patch (obs_mode = conflict), commit, tag v4.3.4;
#   4. launch v4.3.4 (108 jobs) on the otherwise idle machine.
# Every step is logged to scripts/logs/after_v4.3.1.10.log; any error stops the chain.
#
#   nohup bash scripts/pending/after_v4.3.1.10.sh > scripts/logs/after_v4.3.1.10.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
DRIVER_PID=${DRIVER_PID:?pid of the v4.3.1.10 driver}
log() { echo "[$(date '+%F %T')] $*"; }
TRAILER=$'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'

log "waiting for v4.3.1.10 driver (pid $DRIVER_PID)"
while kill -0 "$DRIVER_PID" 2>/dev/null; do sleep 60; done
log "v4.3.1.10 driver exited"
tail -3 scripts/logs/run_v4.3.1.10_driver.log

if ! grep -q "completed, 0 failed" scripts/logs/run_v4.3.1.10_driver.log; then
    log "v4.3.1.10 has failed jobs (or no 'done' line) -- NOT starting v4.3.4"
    exit 1
fi

log "regenerating v4.3.1.10 report"
$PY scripts/run_v4.3.1.10.py --report
git add paper_contents/v4.3.1/v4.3.1.10/report.md
git commit -q -m "v4.3.1.10 结果：report.md（10M 与 5M、贪心对比；AR / ILP AR / AR gap）${TRAILER}"
log "committed v4.3.1.10 report: $(git log --oneline -1)"

log "applying v4.3.4 patch"
git apply scripts/pending/v4.3.4_obs_conflict.patch
git add paper_rl/env.py paper_rl/train.py paper_contents/v4.3.4/README.md
git add -f scripts/run_v4.3.4.py scripts/pending/after_v4.3.1.10.sh scripts/pending/v4.3.4_obs_conflict.patch
git commit -q -m "v4.3.4: 冲突感知观测（state ablation），观测加入未放置服务之间的冲突图，108 个任务启动脚本${TRAILER}"
git tag v4.3.4
log "committed and tagged: $(git log --oneline -1)"

log "launching v4.3.4"
nohup $PY scripts/run_v4.3.4.py > scripts/logs/run_v4.3.4_driver.log 2>&1 &
sleep 30
head -5 scripts/logs/run_v4.3.4_driver.log
log "done"
