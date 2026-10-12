#!/usr/bin/env bash
# One-click deployment of the my_code lab on a Linux + NVIDIA machine (see deploy/README.md).
#
#   cd my_code && deploy/install.sh [--restore] [--verify] [--tunnel]
#     --restore  pull the experiment data from RCLONE_REMOTE first (deploy/data_sync.sh restore)
#     --verify   after starting: training-identity check (~15 min) and v5.0.2 smoke run (~20 min) in the container
#     --tunnel   create / reuse the Cloudflare tunnel (deploy/cloudflare_setup.py) and start cloudflared
#
# Needs on the host: an NVIDIA driver >= 535, Docker with the compose plugin, the NVIDIA Container Toolkit.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; repo="$(dirname "$here")"; cd "$repo"
restore=0; verify=0; tunnel=0
for a in "$@"; do case "$a" in --restore) restore=1;; --verify) verify=1;; --tunnel) tunnel=1;;
  *) sed -n '2,10p' "$0"; exit 1;; esac; done
say() { printf '\n== %s\n' "$*"; }
dc() { docker compose -f "$here/docker-compose.yml" --env-file "$here/.env" "$@"; }

say "1/6 host checks"
command -v nvidia-smi >/dev/null || { echo "no NVIDIA driver: sudo ubuntu-drivers install && sudo reboot"; exit 1; }
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
command -v docker >/dev/null || { echo "no Docker: https://docs.docker.com/engine/install/ubuntu/"; exit 1; }
docker compose version >/dev/null || { echo "no docker compose plugin: sudo apt install docker-compose-plugin"; exit 1; }
if ! docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L >/dev/null 2>&1; then
  cat <<'EOF'
Docker cannot use the GPUs. Install the NVIDIA Container Toolkit:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
EOF
  exit 1
fi
echo "Docker sees the GPUs"

say "2/6 settings (deploy/.env)"
if [ ! -f "$here/.env" ]; then
  cp "$here/.env.example" "$here/.env"; chmod 600 "$here/.env"
  sed -i "s#^DATA_ROOT=.*#DATA_ROOT=$HOME/mycode-data#" "$here/.env"
  echo "created deploy/.env (DATA_ROOT=$HOME/mycode-data); edit it for rclone / Cloudflare if needed"
fi
set -a; . "$here/.env"; set +a
mkdir -p "$DATA_ROOT"/{results,logs,paper_contents,config}
echo "DATA_ROOT=$DATA_ROOT  IMAGE_TAG=${IMAGE_TAG:-latest}"

say "3/6 data"
if [ "$restore" = 1 ]; then "$here/data_sync.sh" restore "$DATA_ROOT"; fi
[ -f "$DATA_ROOT/results/sup_diag/lt.json" ] && echo "50k training pool: present" \
  || echo "WARNING: results/sup_diag/{lt,eq,gt}.json missing in DATA_ROOT -- copy them (deploy/README.md) before training"

say "4/6 build the image (first time ~10-20 min: PyTorch downloads)"
rm -rf "$here/ref-src"; mkdir -p "$here/ref-src"
git archive 9aa6bc7 src | tar -x -C "$here/ref-src" --strip-components=1     # pre-v5 reference code (identity check)
dc build lab

say "5/6 start"
dc up -d lab
for i in $(seq 1 30); do curl -sf "localhost:${PANEL_PORT:-8081}/api/system" >/dev/null && break; sleep 2; done
curl -sf "localhost:${PANEL_PORT:-8081}/api/system" >/dev/null && echo "panel: http://localhost:${PANEL_PORT:-8081}" \
  || { echo "panel did not come up: docker logs mycode-lab"; exit 1; }

if [ "$verify" = 1 ]; then
  say "verify (training identity + smoke, ~35 min)"
  dc exec lab /usr/local/bin/entrypoint.sh verify
  dc exec lab /usr/local/bin/entrypoint.sh smoke
  echo "identity reports: $DATA_ROOT/logs/check_v5/identical.md, identical_mps.md"
fi

say "6/6 public access"
if [ "$tunnel" = 1 ]; then
  python3 "$here/cloudflare_setup.py" --apply
  set -a; . "$here/.env"; set +a
  dc --profile tunnel up -d cloudflared
  echo "tunnel up: https://$CF_HOSTNAME"
else
  echo "skipped (run with --tunnel; settings in deploy/.env)"
fi
say "done -- queue: docker exec mycode-lab .venv/bin/python src/scripts/queue_runner.py list"
