#!/usr/bin/env bash
# Back up / restore the experiment data with rclone (Cloudflare R2, Google Drive, S3, another disk, ...).
#
#   deploy/data_sync.sh backup  [SOURCE]   SOURCE -> RCLONE_REMOTE   (default SOURCE = DATA_ROOT from deploy/.env)
#   deploy/data_sync.sh restore [TARGET]   RCLONE_REMOTE -> TARGET   (default TARGET = DATA_ROOT)
#   deploy/data_sync.sh check   [SOURCE]   only show what differs
#
# SOURCE / TARGET hold results/ logs/ paper_contents/ and config/.env.email (the Docker layout). The repository
# checkout of the pre-Docker machine has the same results/ logs/ paper_contents/ and .env.email at its root,
# so `deploy/data_sync.sh backup ~/github/my_code` works there too.
# backup uses `rclone copy` (never deletes on the remote); restore never overwrites newer local files.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
[ -f "$here/.env" ] && set -a && . "$here/.env" && set +a
: "${RCLONE_REMOTE:?set RCLONE_REMOTE in deploy/.env (an rclone remote, e.g. r2:mycode-data)}"
command -v rclone >/dev/null || { echo "rclone not found: https://rclone.org/install/  (then: rclone config)"; exit 1; }

mode="${1:-}"; root="${2:-${DATA_ROOT:-}}"
[ -n "$root" ] || { echo "no data directory: give one or set DATA_ROOT"; exit 1; }
filters=(--exclude '**/.progress*' --exclude 'logs/mps/**' --exclude '**/__pycache__/**' --exclude '**/.cache/**')
dirs=(results logs paper_contents)

case "$mode" in
  backup)
    for d in "${dirs[@]}"; do [ -d "$root/$d" ] && rclone copy "$root/$d" "$RCLONE_REMOTE/$d" "${filters[@]}" -P --transfers 16; done
    for f in "$root/config/.env.email" "$root/.env.email"; do
      [ -f "$f" ] && rclone copyto "$f" "$RCLONE_REMOTE/config/.env.email" && break
    done ;;
  restore)
    mkdir -p "$root/config"
    for d in "${dirs[@]}"; do rclone copy "$RCLONE_REMOTE/$d" "$root/$d" --update "${filters[@]}" -P --transfers 16; done
    rclone copy "$RCLONE_REMOTE/config" "$root/config" --update -P ;;
  check)
    for d in "${dirs[@]}"; do [ -d "$root/$d" ] && rclone check "$root/$d" "$RCLONE_REMOTE/$d" "${filters[@]}" --one-way || true; done ;;
  *)
    sed -n '2,13p' "$0"; exit 1 ;;
esac
