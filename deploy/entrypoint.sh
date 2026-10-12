#!/usr/bin/env bash
# Lab container entry point.
#   serve   (default) panel + queue + monitor under supervisord
#   verify  training-identity check: this image's code vs the pre-v5 reference code, CPU + GPU (+ MPS), ~15 min
#   smoke   v5.0.2 end to end with tiny budgets (both stages, interim + final report, ILP timing), ~20 min
#   <cmd>   anything else is run as given (e.g. bash, or .venv/bin/python src/scripts/queue_runner.py list)
set -euo pipefail
cd /app

# mounted data: create the layout, seed reports from the image the first time, link the mail credentials
mkdir -p results logs paper_contents config
rsync -a --ignore-existing /opt/seed/paper_contents/ paper_contents/
[ -f config/.env.email ] && ln -sf /app/config/.env.email /app/.env.email

case "${1:-serve}" in
  serve)
    exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf ;;
  verify)
    .venv/bin/python src/scripts/check_v5.py identical
    exec .venv/bin/python src/scripts/check_v5.py identical --mps ;;
  smoke)
    export RESULTS_SPACE=smoke_v502 V5_SMOKE=1
    exec .venv/bin/python src/scripts/run_v5.py --version 5.0.2 ;;
  *)
    exec "$@" ;;
esac
