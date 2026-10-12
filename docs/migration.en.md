# Migration guide: moving the experiment environment to another Linux machine

This guide moves `my_code` (the ILP vs RL service-placement experiments) to a new Linux + NVIDIA GPU machine so that v5.x experiments can continue there. It goes "required → verification → optional"; every step has an estimated time.

> Based on the current machine as of 2026-10-12 (Ubuntu 24.04.2, NVIDIA driver 535.309.01, Python 3.12.3, 3 × GTX 1080 Ti, 2 × Xeon E5-2660 v4).

## 0. Total time

| Case | Steps | Estimated time |
|---|---|---|
| **Minimum to train** (NVIDIA driver already installed) | steps 1–5 + verification (step 6) | **about 1–1.5 h** (about 30 min of it is the verification training) |
| Plus panel, queue service, public access | steps 7–9 as well | **about 2–3 h** |
| No NVIDIA driver yet | step 1 installs it and reboots | **add 30–60 min** |
| Copying all past results (~26 GB) | optional part of step 5 | ~5–10 min on gigabit LAN; possibly hours over the internet |

Most of the time goes into downloading GPU PyTorch (~2.5 GB), copying data, and the verification training.

## 1. Prerequisites on the new machine (about 0–60 min)

- **OS**: Ubuntu 22.04 / 24.04 (other distributions work with adjusted commands). WSL2 is not recommended: it has no MPS, which the PPO stage relies on (measured 1.56×).
- **GPU**: NVIDIA, driver **≥ 535** (CUDA 12.x). GPU PyTorch ships its own CUDA runtime, so **no CUDA Toolkit is needed**.
  ```bash
  nvidia-smi                      # GPUs and driver version shown = fine; otherwise install the driver first:
  sudo ubuntu-drivers install     # recommended driver, then reboot
  ```
- **Memory**: about 2–3 GB per training process. 27–36 at once need ~90 GB; with less memory, lower the concurrency (`V5_PARALLEL`, step 6).
- **Disk**: about 10 GB for code and environments; about 15 GB of results per 100 runs.

## 2. System packages (about 5 min, needs sudo)

```bash
sudo apt update
sudo apt install -y git python3.12 python3.12-venv build-essential numactl
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv: used for the GPU environment
```
The ILP solver CBC comes with PuLP; nothing else to install.

## 3. Get the code (about 5 min)

The new machine needs access to the GitHub repository (add its SSH key to the GitHub account, or use HTTPS + a token).

```bash
mkdir -p ~/github && cd ~/github
git clone git@github.com:Wk-P/I2R.my_code.git my_code
cd my_code
git checkout main && git log --oneline -1          # should match the current machine (push there first)
```
**Run `git push` and `git push --tags` on the current machine before migrating**, otherwise the new machine misses the latest code and the v5.0.0 / v5.0.2 tags.

## 4. The two Python environments (about 15–30 min, mostly downloads)

`.venv` (CPU: panel, queue, reports, MLP training) and `.venv-gpu` (GPU: structure-aware network training).

```bash
cd ~/github/my_code

# 4.1 CPU environment
python3.12 -m venv .venv
# requirements.txt pins torch==2.10.0+cpu, which is only on PyTorch's own index, hence --extra-index-url
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e .                          # makes the packages under src/ importable

# 4.2 GPU environment (PyTorch 2.5.1 + CUDA 12.1 runtime)
uv venv .venv-gpu --python 3.12
uv pip install --python .venv-gpu/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
grep -v -i "^torch==" requirements.txt > /tmp/req-gpu.txt
uv pip install --python .venv-gpu/bin/python -r /tmp/req-gpu.txt
uv pip install --python .venv-gpu/bin/python -e .

# 4.3 Check
.venv-gpu/bin/python -c "import torch, stable_baselines3 as s, sb3_contrib as c; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), s.__version__, c.__version__)"
# expected: 2.5.1+cu121 True <number of GPUs> 2.7.1 2.7.1
```

The versions must match the current machine (stable-baselines3 / sb3-contrib 2.7.1, gymnasium 1.2.3, PyTorch 2.5.1), otherwise results are not comparable.

## 5. Copy the data that is not in git (about 5–15 min)

| What | Size | Required? | Used for |
|---|---|---|---|
| `results/sup_diag/{lt,eq,gt}.json` | ~50 MB | **required** | the 50k ILP-verified training pool |
| `logs/v4.5.2/`, `logs/v4.4.15`–`v4.4.17/` etc. (manifests) | a few MB | **required** | reports find their reference data through them |
| the 18 structure-aware Maskable DQN/DDQN run directories of v4.5.2 | ~2.4 GB | **required for v5.0.2** | v5.0.2 reuses them as Maskable DQN/DDQN |
| `.env.email` | a few hundred bytes | required for e-mail | credentials for report / failure mails (**never commit it**) |
| the rest of `results/unified/` | ~18 GB | optional | past results on the panel, reproducing old versions |

On the **current** machine (replace `NEW` with `user@new-host`):

```bash
cd ~/github/my_code
NEW=user@new-host
rsync -avP results/sup_diag  $NEW:~/github/my_code/results/
rsync -avP logs/             $NEW:~/github/my_code/logs/ --exclude 'mps/'
rsync -avP .env.email        $NEW:~/github/my_code/
# the 18 reference runs of v4.5.2 (needed by v5.0.2):
python3 - <<'EOF' > /tmp/ref_dirs.txt
import json
m = json.load(open("logs/v4.5.2/manifest.json"))
for sd in m["exp_ids"].values():
    for s, d in sd.items():
        for k, v in d.items():
            if k.endswith("graph"):
                print(f"results/unified/{s}/{'mask_dqn' if k.startswith('dqn') else 'mask_ddqn'}/{v}")
EOF
rsync -avP --files-from=/tmp/ref_dirs.txt -r . $NEW:~/github/my_code/
# optional: all past results
# rsync -avP results/unified $NEW:~/github/my_code/results/
```

## 6. Verification (about 30 min, **required**)

The new machine's GPU model or driver may differ, so first make sure training works correctly there.

```bash
cd ~/github/my_code

# 6.1 Identity: on the new machine, v5 code and pre-v5 code train bit-identically (CPU + GPU, ~10 min)
git worktree add ../my_code-ref 9aa6bc7             # the pre-v5.0.0 reference code
.venv/bin/python src/scripts/check_v5.py identical --old-src ../my_code-ref/src
cat logs/check_v5/identical.en.md                   # every row should say "yes"
git worktree remove ../my_code-ref

# 6.2 Smoke: v5.0.2's two stages, interim report, final report and ILP timing with tiny budgets (~20 min)
RESULTS_SPACE=smoke_v502 V5_SMOKE=1 .venv/bin/python src/scripts/run_v5.py --version 5.0.2
ls results/smoke_v5.0.2/report.md                   # present = passed
```

If both pass, the new machine is ready for real experiments.

**Comparability across machines**: on one machine, a seed reproduces bit for bit; **on a different GPU model or driver the same seed is no longer bit-identical, but statistically equivalent**. Keep one comparison (e.g. all runs of one v5.0.x) on one machine; if runs must be split, record in the report which machine ran which run.

**Different resources**: `run_v5.py` runs 27 / 36 at a time per stage by default; adjust for the new machine:
```bash
V5_PARALLEL=12 .venv/bin/python src/scripts/run_v5.py --version 5.0.2   # e.g. less memory or fewer GPUs
```
The solve-time comparison (models vs ILP) must be measured entirely on one machine, otherwise the times are not comparable.

## 7. Optional: machine tuning (about 10 min, needs root)

The same three settings as on the current machine (GPU persistence mode, CPU performance governor, fan speed at boot). Fan control uses NVML from a small environment:

```bash
sudo python3 -m venv /root/nvfan && sudo /root/nvfan/bin/pip install -q nvidia-ml-py
sudo tee /etc/systemd/system/gpu-tune.service > /dev/null <<'EOF'
[Unit]
Description=GPU persistence mode + CPU performance governor
After=multi-user.target
[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/bin/sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $g; done'
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now gpu-tune.service
```
Whether the fans need a fixed speed depends on the new machine's cooling (the current machine sits in a corner, so GPU 0 hits 84 °C and throttles; that is why its fans are fixed).

## 8. Optional: panel and experiment queue (about 15 min)

The panel (monitor, charts, version reports) and the serial experiment queue are user-level systemd services. Adjust the paths for the new user name and directory.

```bash
cd ~/github/my_code/app/frontend && npm ci && npm run build && cd ~/github/my_code   # needs Node.js 18+
mkdir -p ~/.config/systemd/user
# copy the 4 unit files from the current machine, or recreate them from it:
#   my-code-panel.service    panel (uvicorn, port 8081)
#   my-code-queue.service    serial experiment queue (KillMode=process: restarting it never kills experiments)
#   my-code-monitor.service  stuck / finished detection
#   cloudflared-my-code.service  public access (step 9)
sed -i "s#/home/soar009#$HOME#g" ~/.config/systemd/user/my-code-*.service
loginctl enable-linger $USER                        # services keep running after logout (prefix sudo if needed)
systemctl --user daemon-reload
systemctl --user enable --now my-code-panel my-code-queue my-code-monitor
curl -s localhost:8081/api/system | head -c 200     # output = panel is up
```

## 9. Optional: public access (about 15–30 min)

The current machine publishes the panel at `my-code.knettf.com` through a Cloudflare Tunnel. On the new machine, either:
- **reuse the same tunnel**: copy `~/.cloudflared/my-code.yml` and the credentials json it names (`~/.cloudflared/<tunnel id>.json`), install `cloudflared`, enable `cloudflared-my-code.service`. If the same tunnel runs on both machines, requests are split between them, so **stop it on the old machine after migrating**;
- **create a new tunnel**: `cloudflared tunnel login` → `cloudflared tunnel create` → DNS. Needs Cloudflare account access.

## 10. Continue the experiments

```bash
.venv/bin/python src/scripts/queue_runner.py list                      # the queue
.venv/bin/python src/scripts/queue_runner.py add v5.0.2 --log logs/run_v5.0.2_driver.log -- \
    .venv/bin/python src/scripts/run_v5.py --version 5.0.2
```
Reports go to `paper_contents/v<version>/report.md` and `report.en.md`; an e-mail goes out when they are done (needs `.env.email` from step 5).

## Appendix: key versions on the current machine

| Item | Version |
|---|---|
| OS | Ubuntu 24.04.2 LTS, kernel 6.17 |
| NVIDIA driver | 535.309.01 (CUDA 12.2) |
| Python | 3.12.3 |
| PyTorch (GPU environment) | 2.5.1 + cu121 |
| stable-baselines3 / sb3-contrib | 2.7.1 / 2.7.1 |
| gymnasium | 1.2.3 |
| PuLP (CBC) | 3.3.0 |
| Node.js (panel build) | 22.21.1 |
