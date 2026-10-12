# syntax=docker/dockerfile:1
# my_code lab image (v5.0.0): training (CPU + GPU environments), the panel, the serial queue and the monitor.
# Code and Python environments are baked in; experiment data (results/, logs/, paper_contents/) is mounted from
# the host (see docker-compose.yml), so the image is the same on every machine and the data survives rebuilds.
# Build context = the repository root:  docker compose -f deploy/docker-compose.yml build

# ── 1. panel frontend (its build output is git-ignored, so build it here) ────────────────────────────────
FROM node:22.11.0-bookworm-slim AS web
WORKDIR /w/app/frontend
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY app/frontend/ ./
COPY app/backend/static/ ../backend/static/
RUN npm run build                                   # -> /w/app/backend/static

# ── 2. lab ────────────────────────────────────────────────────────────────────────────────────────────────
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    V5_REF_SRC=/opt/ref/src
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv git procps numactl supervisor rsync curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv
WORKDIR /app

# Python environments first (cached unless requirements change). Same versions as the reference machine:
# .venv     = CPU (panel, queue, reports, MLP) - requirements.txt pins torch==2.10.0+cpu (PyTorch CPU index)
# .venv-gpu = GPU (structure-aware network)   - torch 2.5.1 + CUDA 12.1 runtime, the rest of requirements.txt
COPY requirements.txt pyproject.toml ./
RUN python3.12 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN uv venv .venv-gpu --python /usr/bin/python3.12 \
    && uv pip install --python .venv-gpu/bin/python --no-cache torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 \
    && grep -v -i '^torch==' requirements.txt > /tmp/req-gpu.txt \
    && uv pip install --python .venv-gpu/bin/python --no-cache -r /tmp/req-gpu.txt

# code, the built frontend, editable installs (no dependencies: they are above)
COPY . .
COPY --from=web /w/app/backend/static/ app/backend/static/
RUN .venv/bin/pip install --no-cache-dir --no-deps -e . \
    && uv pip install --python .venv-gpu/bin/python --no-deps -e .

# reference (pre-v5) code for the training-identity check; deploy/install.sh exports it with git archive
COPY deploy/ref-src/ /opt/ref/src/
# git-tracked reports seed an empty paper_contents volume on first start
RUN mkdir -p /opt/seed && cp -a paper_contents /opt/seed/paper_contents

COPY deploy/supervisord.conf /etc/supervisor/conf.d/mycode.conf
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
EXPOSE 8081
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
